use bytemuck::{Pod, Zeroable};
use std::{fmt::Debug, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, AcquireError, PresentInfo, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn create_instance() -> Arc<Instance> {
    // Load Library
    let library = VulkanLibrary::new().unwrap();
    // Get Required Extensions to draw to window
    let required_extensions = vulkano_win::required_extensions(&library);
    // Create the Instance
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            // Enable enumerating devices that use non-conformant vulkan implementations. (eg. MoltenVK)
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .expect("Could not create instance");
    instance
}

fn create_surface<T>(
    event_loop: &EventLoop<T>,
    instance: Arc<Instance>,
    title: &str,
) -> Arc<Surface<Window>> {
    let surface = WindowBuilder::new()
        .with_title(title)
        .with_resizable(false)
        .with_inner_size(LogicalSize::new(1200, 800))
        .build_vk_surface(event_loop, instance)
        .unwrap();
    surface
}

fn get_device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    }
}

fn get_physical_device_and_queue_family(
    instance: Arc<Instance>,
    surface: Arc<Surface<Window>>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("Could not enumerate physical devices")
        .filter(|physical_device| {
            physical_device
                .supported_extensions()
                .contains(&device_extensions)
        })
        .filter_map(|physical_device| {
            physical_device
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, queue)| {
                    queue.queue_flags.graphics
                        && physical_device
                            .surface_support(i as u32, &surface)
                            .unwrap_or(false)
                })
                .map(|i| (physical_device, i as u32))
        })
        .min_by_key(|(physical_device, _)| {
            // Score devices based on performace. Lower is better.
            match physical_device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("Could not find suitable physical device")
}

fn get_logical_device_and_queues(
    physical_device: Arc<PhysicalDevice>,
    device_extensions: DeviceExtensions,
    queue_family_index: u32,
) -> (Arc<Device>, Arc<Queue>) {
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("Could not get logical device");
    let queue = queues.next().expect("Could not get first queue");
    (device, queue)
}

fn create_swapchain(
    device: Arc<Device>,
    surface: Arc<Surface<Window>>,
) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .unwrap();
    let image_format = Some(
        device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0,
    );
    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count,
            image_format,
            image_extent: surface.window().inner_size().into(),
            image_usage: ImageUsage {
                color_attachment: true,
                ..ImageUsage::empty()
            },
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .iter()
                .next()
                .unwrap(),
            ..Default::default()
        },
    )
    .expect("Could not create swapchain")
}

fn create_framebuffers(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).expect("Could not create imageview");
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .expect("Could not create framebuffer")
        })
        .collect::<Vec<_>>()
}

fn main() {
    let event_loop = EventLoop::new();

    let instance = create_instance();
    let surface = create_surface(&event_loop, instance.clone(), "Badlands VK");
    let device_extensions = get_device_extensions();
    let (physical_device, queue_family_index) =
        get_physical_device_and_queue_family(instance.clone(), surface.clone(), &device_extensions);

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, queue) =
        get_logical_device_and_queues(physical_device, device_extensions, queue_family_index);
    let (mut swapchain, images) = create_swapchain(device.clone(), surface.clone());

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            // Own name for the attachment
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            // The 'color' in the array refers to our own name of attachment above
            color: [color],
            depth_stencil: {}
        }
    )
    .expect("Could not create render pass");

    // Unique per draw call type
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    // There are multiple framebuffers
    let mut framebuffers = create_framebuffers(&images, render_pass.clone(), &mut viewport);

    // Set up render layers
    let mut layers = vec![
        Box::new(BasicTriangleDrawLayer::new(
            device.clone(),
            render_pass.clone(),
        )),
        Box::new(BasicTriangleDrawLayer::new(
            device.clone(),
            render_pass.clone(),
        )),
    ];

    // Loop
    let mut is_swapchain_invalid = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                is_swapchain_invalid = true;
            }
            Event::RedrawEventsCleared => {
                let dimensions = surface.window().inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                // Periodic cleanup
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if is_swapchain_invalid {
                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(result) => result,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(error) => panic!("Failed to recreate swapchain: {:?}", error),
                        };

                    swapchain = new_swapchain;
                    framebuffers =
                        create_framebuffers(&new_images, render_pass.clone(), &mut viewport);
                    is_swapchain_invalid = false;
                }

                let (framebuffer_image_index, is_swapchain_suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(result) => result,
                        Err(AcquireError::OutOfDate) => {
                            is_swapchain_invalid = true;
                            return;
                        }
                        Err(error) => panic!("Failed to acquire next image: {:?}", error),
                    };

                if is_swapchain_suboptimal {
                    is_swapchain_invalid = true;
                }

                let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .expect("Could not create command buffer");

                command_buffer_builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            // Clear value for 'color' attachment
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[framebuffer_image_index as usize].clone(),
                            )
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()]);

                for layer in layers.iter_mut() {
                    layer.draw(&mut command_buffer_builder);
                }

                command_buffer_builder.end_render_pass().unwrap();

                let command_buffer = command_buffer_builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        PresentInfo {
                            index: framebuffer_image_index,
                            ..PresentInfo::swapchain(swapchain.clone())
                        },
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        is_swapchain_invalid = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(error) => {
                        panic!("Failed to flush future: {:?}", error);
                    }
                }
            }
            _ => (),
        }
    });
}

trait DrawLayer {
    fn draw(
        &mut self,
        command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    );
}

struct BasicTriangleDrawLayer {
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
}
impl_vertex!(Vertex, position);
impl BasicTriangleDrawLayer {
    pub fn new(device: Arc<Device>, render_pass: Arc<RenderPass>) -> Self {
        let vertices = [
            Vertex {
                position: [-0.5, -0.25],
            },
            Vertex {
                position: [0.0, 0.5],
            },
            Vertex {
                position: [0.25, -0.1],
            },
        ];
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            vertices,
        )
        .expect("Could not create buffer from iter");

        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: "
				#version 450

				layout(location = 0) in vec2 position;

				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
				}
			"
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: "
				#version 450

				layout(location = 0) out vec4 f_color;

				void main() {
					f_color = vec4(1.0, 0.0, 0.0, 1.0);
				}
			"
            }
        }

        let vs = vs::load(device.clone()).expect("Could not load vertex shader");
        let fs = fs::load(device.clone()).expect("Could not load fragment shader");

        let pipeline = GraphicsPipeline::start()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .build(device.clone())
            .expect("Could not build pipeline");

        Self {
            pipeline,
            vertex_buffer,
        }
    }
}
impl DrawLayer for BasicTriangleDrawLayer {
    fn draw(
        &mut self,
        command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        command_buffer_builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
    }
}

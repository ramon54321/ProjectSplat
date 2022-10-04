use crate::{
    layers::basic::BasicTriangleDrawLayer,
    util::{
        create_framebuffers, create_instance, create_surface, create_swapchain,
        get_device_extensions, get_logical_device_and_queues, get_physical_device_and_queue_family,
    },
};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassContents,
    },
    pipeline::graphics::viewport::Viewport,
    swapchain::{
        acquire_next_image, AcquireError, PresentInfo, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

mod layers;
mod util;

pub trait DrawLayer {
    fn draw(
        &mut self,
        command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    );
}

pub fn render() {
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

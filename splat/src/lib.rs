use crate::util::{
    create_framebuffers, create_instance, create_surface, create_swapchain, get_device_extensions,
    get_logical_device_and_queues, get_physical_device_and_queue_family,
};
use nalgebra_glm::Vec2;
use std::{
    cell::RefCell,
    collections::HashSet,
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassContents,
    },
    device::{Device, Queue},
    pipeline::graphics::viewport::Viewport,
    render_pass::RenderPass,
    swapchain::{
        acquire_next_image, AcquireError, PresentInfo, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use winit::{
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

mod layers;
mod util;

pub use layers::{
    basic::BasicTriangleDrawLayer,
    text::{TextDrawLayer, TextEnqueueRequest},
};
pub use winit::event::VirtualKeyCode;

pub trait DrawLayer<T> {
    fn setup(&mut self, device: Arc<Device>, queue: Arc<Queue>, render_pass: Arc<RenderPass>);
    fn draw(&mut self, gpu_interface: &mut GpuInterface, meta: &mut Meta, state: &mut T);
}

pub enum AlignHorizontal {
    Left,
    Center,
    Right,
}
pub enum AlignVertical {
    Top,
    Center,
    Bottom,
}

pub struct Meta<'a> {
    pub frames_per_second: usize,
    pub layer_durations: Vec<Duration>,
    pub delta_time: Duration,
    pub mouse_position: Vec2,
    pub is_mouse_down: bool,
    pub is_mouse_released: bool,
    pub keys_hold: &'a HashSet<VirtualKeyCode>,
    pub keys_up: &'a HashSet<VirtualKeyCode>,
    pub keys_down: &'a HashSet<VirtualKeyCode>,
}

pub struct GpuInterface<'a> {
    pub command_buffer_builder: &'a mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    pub viewport: Viewport,
}

pub fn render<T: 'static>(mut state: T, mut layers: Vec<Rc<RefCell<dyn DrawLayer<T>>>>) {
    let event_loop = EventLoop::new();

    let instance = create_instance();
    let surface = create_surface(&event_loop, instance.clone(), "Badlands VK");
    let device_extensions = get_device_extensions();
    let (physical_device, queue_family_index) =
        get_physical_device_and_queue_family(instance.clone(), surface.clone(), &device_extensions);

    println!(
        "Using graphics device: {} ({:?})",
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
    for layer in layers.iter_mut() {
        layer
            .borrow_mut()
            // TODO: Add this to gpu interface
            .setup(device.clone(), queue.clone(), render_pass.clone());
    }

    // Loop
    let mut is_swapchain_invalid = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    // Framerate management
    let mut layer_durations = Vec::new();
    let mut frame_timer = Instant::now();
    let mut frame_second_timer = Instant::now();
    let mut frame_counter = 0;
    let mut frames_per_second = 0;

    let mut mouse_position = Vec2::new(0.0, 0.0);
    let mut is_mouse_down = false;
    let mut is_mouse_released = false;
    let mut keys_hold = HashSet::new();
    let mut keys_down = HashSet::new();
    let mut keys_up = HashSet::new();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                window_id: _,
                event,
            } => match event {
                WindowEvent::KeyboardInput {
                    device_id: _,
                    input,
                    is_synthetic: _,
                } => {
                    if input.virtual_keycode.is_some() {
                        let key = input.virtual_keycode.unwrap();
                        let value = input.state == ElementState::Pressed;
                        if value {
                            keys_down.insert(key);
                            keys_hold.insert(key);
                        } else {
                            keys_up.insert(key);
                            keys_hold.remove(&key);
                        }
                    }
                }
                WindowEvent::CursorMoved {
                    device_id: _,
                    position,
                    ..
                } => {
                    mouse_position.x = position.x as f32;
                    mouse_position.y = position.y as f32;
                }
                WindowEvent::MouseInput {
                    device_id: _,
                    state,
                    button,
                    ..
                } => match state {
                    ElementState::Pressed => match button {
                        MouseButton::Left => {
                            is_mouse_down = true;
                        }
                        _ => (),
                    },
                    ElementState::Released => match button {
                        MouseButton::Left => {
                            is_mouse_down = false;
                            is_mouse_released = true;
                        }
                        _ => (),
                    },
                },
                WindowEvent::Resized(_) => {
                    is_swapchain_invalid = true;
                }
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => (),
            },
            Event::RedrawEventsCleared => {
                // Frame timers
                frame_counter = frame_counter + 1;
                let time_since_frame_counter_reset = Instant::now() - frame_second_timer;
                if time_since_frame_counter_reset.as_millis() >= 1000 {
                    frame_second_timer = Instant::now();
                    frames_per_second = frame_counter;
                    frame_counter = 0;
                }
                let delta_time = Instant::now() - frame_timer;
                frame_timer = Instant::now();

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

                // Construct state for current frame
                let mut meta = Meta {
                    frames_per_second,
                    layer_durations: layer_durations.clone(),
                    delta_time,
                    mouse_position,
                    is_mouse_down,
                    is_mouse_released,
                    keys_hold: &keys_hold,
                    keys_down: &keys_down,
                    keys_up: &keys_up,
                };

                let mut gpu_interface = GpuInterface {
                    command_buffer_builder: &mut command_buffer_builder,
                    viewport: viewport.clone(),
                };

                // Draw each layer
                layer_durations.clear();
                for layer in layers.iter_mut() {
                    let timer = Instant::now();

                    layer
                        .borrow_mut()
                        .draw(&mut gpu_interface, &mut meta, &mut state);

                    let duration = Instant::now() - timer;
                    layer_durations.push(duration);
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

                // Reset Meta
                is_mouse_released = false;
                keys_up.clear();
                keys_down.clear();
            }
            _ => (),
        }
    });
}

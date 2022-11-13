use crate::util::{
    create_framebuffers_from_swapchain, create_instance, create_surface, create_swapchain,
    get_device_extensions, get_device_features, get_logical_device_and_queues,
    get_physical_device_and_queue_family,
};
use nalgebra_glm::Vec2;
use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant},
};
use vulkano::{
    device::{Device, Queue},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, RenderPass},
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
        SwapchainCreationError,
    },
    sync::{self, GpuFuture},
};
use winit::{
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

mod layers;
mod util;

pub use layers::{
    basic::LayerBuildBasicTriangle,
    text::{LayerBuildText, TextEnqueueRequest},
};
pub use winit::event::VirtualKeyCode;

pub struct SetupContext<'a, T, S> {
    pub state: &'a mut T,
    pub setup_state: &'a mut S,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub swapchain: Arc<Swapchain<Window>>,
}

pub struct SetupResponse {
    pub swapchain_render_pass: Arc<RenderPass>,
}

pub struct BuildContext<'a, 'b, T, S> {
    pub state: &'a mut T,
    pub setup_state: &'a mut S,
    pub meta: &'a mut Meta<'b>,
    pub viewport: Viewport,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub previous_frame_end_future: &'a mut Option<Box<dyn GpuFuture>>,
    pub acquire_future: Option<SwapchainAcquireFuture<Window>>,
    pub swapchain: Arc<Swapchain<Window>>,
    pub swapchain_framebuffer: Arc<Framebuffer>,
    pub swapchain_framebuffer_image_index: usize,
}

#[derive(Hash)]
pub enum AlignHorizontal {
    Left,
    Center,
    Right,
}
#[derive(Hash)]
pub enum AlignVertical {
    Top,
    Center,
    Bottom,
}

pub struct Meta<'a> {
    pub frames_per_second: usize,
    pub was_swapchain_rebuilt: bool,
    pub delta_time: Duration,
    pub mouse_position: Vec2,
    pub is_mouse_down: bool,
    pub is_mouse_released: bool,
    pub keys_hold: &'a HashSet<VirtualKeyCode>,
    pub keys_up: &'a HashSet<VirtualKeyCode>,
    pub keys_down: &'a HashSet<VirtualKeyCode>,
}

#[derive(Debug, Clone)]
pub struct SplatCreateInfo {
    pub title: String,
    pub size: [u16; 2],
    pub is_resizable: bool,
    pub is_maximized: bool,
    pub is_fullscreen: bool,
    pub clear_color: [f32; 4],
}
impl Default for SplatCreateInfo {
    fn default() -> Self {
        Self {
            title: "Splat App".to_string(),
            size: [800, 600],
            is_resizable: false,
            is_maximized: false,
            is_fullscreen: false,
            clear_color: [0.349, 0.314, 0.294, 1.0],
        }
    }
}

pub type BuildResponse = Option<Box<dyn GpuFuture>>;

pub fn render<T: 'static, S: 'static>(
    splat_create_info: SplatCreateInfo,
    mut state: T,
    mut setup_state: S,
    setup: fn(setup_context: &mut SetupContext<T, S>) -> SetupResponse,
    build: fn(draw_context: &mut BuildContext<T, S>) -> BuildResponse,
) {
    let event_loop = EventLoop::new();

    let instance = create_instance();
    let surface = create_surface(&splat_create_info, &event_loop, instance.clone());
    let device_features = get_device_features();
    let device_extensions = get_device_extensions();
    let (physical_device, queue_family_index) = get_physical_device_and_queue_family(
        instance.clone(),
        surface.clone(),
        &device_features,
        &device_extensions,
    );

    println!(
        "Using graphics device: {} ({:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, queue) = get_logical_device_and_queues(
        physical_device,
        &device_features,
        &device_extensions,
        queue_family_index,
    );
    let (mut swapchain, images) = create_swapchain(device.clone(), surface.clone());

    let mut setup_context = SetupContext {
        device: device.clone(),
        queue: queue.clone(),
        swapchain: swapchain.clone(),
        state: &mut state,
        setup_state: &mut setup_state,
    };
    let setup_response = setup(&mut setup_context);

    // Unique per draw call type
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    // Swapchain contains multiple framebuffers
    let mut framebuffers = create_framebuffers_from_swapchain(
        &images,
        setup_response.swapchain_render_pass.clone(),
        &mut viewport,
    );

    // Loop
    let mut is_swapchain_invalid = false;
    let mut previous_frame_end_future = Some(sync::now(device.clone()).boxed());

    // Framerate management
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
                previous_frame_end_future
                    .as_mut()
                    .unwrap()
                    .cleanup_finished();

                let mut was_swapchain_rebuilt = false;
                if is_swapchain_invalid {
                    was_swapchain_rebuilt = true;
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
                    framebuffers = create_framebuffers_from_swapchain(
                        &new_images,
                        setup_response.swapchain_render_pass.clone(),
                        &mut viewport,
                    );
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

                let mut meta = Meta {
                    frames_per_second,
                    was_swapchain_rebuilt,
                    delta_time,
                    mouse_position,
                    is_mouse_down,
                    is_mouse_released,
                    keys_hold: &keys_hold,
                    keys_down: &keys_down,
                    keys_up: &keys_up,
                };
                let mut draw_context = BuildContext {
                    previous_frame_end_future: &mut previous_frame_end_future,
                    acquire_future: Some(acquire_future),
                    state: &mut state,
                    setup_state: &mut setup_state,
                    meta: &mut meta,
                    viewport: viewport.clone(),
                    device: device.clone(),
                    queue: queue.clone(),
                    swapchain: swapchain.clone(),
                    swapchain_framebuffer: framebuffers[framebuffer_image_index as usize].clone(),
                    swapchain_framebuffer_image_index: framebuffer_image_index,
                };

                let last_future = build(&mut draw_context);

                match last_future {
                    Some(future) => previous_frame_end_future = Some(future.boxed()),
                    None => panic!("Last future error"),
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

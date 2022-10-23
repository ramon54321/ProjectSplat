use std::sync::Arc;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::LogicalSize,
    event_loop::EventLoop,
    window::{Fullscreen, Window, WindowBuilder},
};

use crate::SplatCreateInfo;

pub fn create_instance() -> Arc<Instance> {
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

pub fn create_surface<T>(
    splat_create_info: &SplatCreateInfo,
    event_loop: &EventLoop<T>,
    instance: Arc<Instance>,
) -> Arc<Surface<Window>> {
    let mut surface = WindowBuilder::new()
        .with_title(&splat_create_info.title)
        .with_resizable(splat_create_info.is_resizable)
        .with_maximized(splat_create_info.is_maximized);

    if splat_create_info.is_fullscreen {
        surface = surface.with_fullscreen(Some(Fullscreen::Borderless(None)));
    } else {
        surface = surface.with_inner_size(LogicalSize::new(
            splat_create_info.size[0],
            splat_create_info.size[1],
        ));
    }

    surface.build_vk_surface(event_loop, instance).unwrap()
}

pub fn get_device_features() -> Features {
    Features {
        wide_lines: false, // Apple M1 vulkan backend does not yet support wide lines
        ..Features::empty()
    }
}

pub fn get_device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    }
}

pub fn get_physical_device_and_queue_family(
    instance: Arc<Instance>,
    surface: Arc<Surface<Window>>,
    device_features: &Features,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("Could not enumerate physical devices")
        .filter(|physical_device| {
            physical_device
                .supported_features()
                .contains(&device_features)
                && physical_device
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

pub fn get_logical_device_and_queues(
    physical_device: Arc<PhysicalDevice>,
    device_features: &Features,
    device_extensions: &DeviceExtensions,
    queue_family_index: u32,
) -> (Arc<Device>, Arc<Queue>) {
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_features: device_features.clone(),
            enabled_extensions: device_extensions.clone(),
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

pub fn create_swapchain(
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

pub fn create_framebuffers_from_swapchain(
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

use bytemuck::{Pod, Zeroable};
use splat::{
    render, BuildContext, BuildResponse, LayerBuildBasicTriangle, LayerBuildContext,
    LayerSetupContext, SetupContext, SetupResponse, SplatCreateInfo,
};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, ImageBlit,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    format::Format,
    image::{
        view::ImageView, AttachmentImage, ImageAccess, ImageDimensions, ImageLayout, ImageUsage,
        StorageImage,
    },
    impl_vertex,
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode},
    swapchain::PresentInfo,
    sync::GpuFuture,
};

fn main() {
    render(
        SplatCreateInfo::default(),
        MyState {
            render_passes: MyStateRenderPasses::default(),
        },
        MySetupState::default(),
        setup,
        build,
    );
}

#[derive(Default)]
pub struct MyState {
    pub render_passes: MyStateRenderPasses,
}

#[derive(Default)]
pub struct MySetupState {
    pub layers: MySetupStateLayers,
}
#[derive(Default)]
pub struct MySetupStateLayers {
    pub basic_triangle_draw_layer: Option<LayerBuildBasicTriangle>,
    pub image_debug_draw_layer: Option<ImageDebugDrawLayer>,
}
#[derive(Default)]
pub struct MyStateRenderPasses {
    pub offscreen_render_pass: MyStateOffscreenRenderPass,
    pub swapchain_render_pass: MyStateSwapchainRenderPass,
}
#[derive(Default)]
pub struct MyStateOffscreenRenderPass {
    pub render_pass: Option<Arc<RenderPass>>,
    pub framebuffer: Option<Arc<Framebuffer>>,
    pub attachment_image: Option<Arc<AttachmentImage>>,
    pub result_image: Option<Arc<StorageImage>>,
}
#[derive(Default)]
pub struct MyStateSwapchainRenderPass {
    pub render_pass: Option<Arc<RenderPass>>,
}

fn setup(setup_context: &mut SetupContext<MyState, MySetupState>) -> SetupResponse {
    setup_offscreen_render_pass(setup_context);
    setup_swapchain_render_pass(setup_context);

    SetupResponse {
        swapchain_render_pass: setup_context
            .state
            .render_passes
            .swapchain_render_pass
            .render_pass
            .as_ref()
            .unwrap()
            .clone(),
    }
}

fn setup_offscreen_render_pass(setup_context: &mut SetupContext<MyState, MySetupState>) {
    let attachment_image = AttachmentImage::with_usage(
        setup_context.device.clone(),
        [500, 500],
        Format::R8G8B8A8_UNORM,
        ImageUsage {
            color_attachment: true,
            transfer_src: true,
            ..ImageUsage::empty()
        },
    )
    .unwrap();
    let attachment_image_view = ImageView::new_default(attachment_image.clone()).unwrap();
    let render_pass = vulkano::single_pass_renderpass!(
        setup_context.device.clone(),
        attachments: {
            // Own name for the attachment
            color: {
                load: Clear,
                store: Store,
                format: Format::R8G8B8A8_UNORM,
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
    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![attachment_image_view],
            ..Default::default()
        },
    )
    .unwrap();
    let result_image = StorageImage::new(
        setup_context.device.clone(),
        ImageDimensions::Dim2d {
            width: 512,
            height: 512,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        [setup_context.queue.queue_family_index()],
    )
    .unwrap();

    // Setup setup state
    let offscreen_render_pass = &mut setup_context.state.render_passes.offscreen_render_pass;
    offscreen_render_pass.render_pass = Some(render_pass.clone());
    offscreen_render_pass.framebuffer = Some(framebuffer.clone());
    offscreen_render_pass.attachment_image = Some(attachment_image.clone());
    offscreen_render_pass.result_image = Some(result_image.clone());

    // Setup draw layers
    let mut layer_setup_context = LayerSetupContext {
        state: setup_context.state,
        setup_state: setup_context.setup_state,
        device: setup_context.device.clone(),
        queue: setup_context.queue.clone(),
        render_pass: render_pass.clone(),
    };

    let mut basic_triangle_draw_layer = LayerBuildBasicTriangle::default();
    basic_triangle_draw_layer.setup(&mut layer_setup_context);
    setup_context.setup_state.layers.basic_triangle_draw_layer = Some(basic_triangle_draw_layer);
}

fn setup_swapchain_render_pass(setup_context: &mut SetupContext<MyState, MySetupState>) {
    let render_pass = vulkano::single_pass_renderpass!(
        setup_context.device.clone(),
        attachments: {
            // Own name for the attachment
            color: {
                load: Clear,
                store: Store,
                format: setup_context.swapchain.image_format(),
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

    // Setup setup state
    let swapchain_render_pass = &mut setup_context.state.render_passes.swapchain_render_pass;
    swapchain_render_pass.render_pass = Some(render_pass.clone());

    // Setup draw layers
    let mut layer_setup_context = LayerSetupContext {
        state: setup_context.state,
        setup_state: setup_context.setup_state,
        device: setup_context.device.clone(),
        queue: setup_context.queue.clone(),
        render_pass: render_pass.clone(),
    };

    let mut image_debug_draw_layer = ImageDebugDrawLayer::default();
    image_debug_draw_layer.setup(&mut layer_setup_context);
    setup_context.setup_state.layers.image_debug_draw_layer = Some(image_debug_draw_layer);
}

fn build(build_context: &mut BuildContext<MyState, MySetupState>) -> BuildResponse {
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        build_context.device.clone(),
        build_context.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // Offscreen pass
    build_offscreen_render_pass(build_context, &mut command_buffer_builder);

    // Compute pass
    build_blit_offscreen_to_result_image(build_context, &mut command_buffer_builder);

    // Swapchain pass
    build_swapchain_render_pass(build_context, &mut command_buffer_builder);

    let command_buffer = command_buffer_builder.build().unwrap();

    let future = build_context
        .previous_frame_end_future
        .take()
        .unwrap()
        .join(build_context.acquire_future.take().unwrap())
        .then_execute(build_context.queue.clone(), command_buffer)
        .unwrap()
        .then_swapchain_present(
            build_context.queue.clone(),
            PresentInfo {
                index: build_context.swapchain_framebuffer_image_index,
                ..PresentInfo::swapchain(build_context.swapchain.clone())
            },
        )
        .then_signal_fence_and_flush();

    Some(Box::new(future.unwrap()))
}

fn build_offscreen_render_pass(
    build_context: &mut BuildContext<MyState, MySetupState>,
    command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) {
    command_buffer_builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 1.0, 0.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(
                    build_context
                        .state
                        .render_passes
                        .offscreen_render_pass
                        .framebuffer
                        .as_ref()
                        .unwrap()
                        .clone(),
                )
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .set_viewport(0, [build_context.viewport.clone()]);

    // Draw
    let mut layer_draw_context = LayerBuildContext {
        state: build_context.state,
        meta: build_context.meta,
        viewport: build_context.viewport.clone(),
        device: build_context.device.clone(),
        queue: build_context.queue.clone(),
        command_buffer_builder,
    };
    build_context
        .setup_state
        .layers
        .basic_triangle_draw_layer
        .as_mut()
        .unwrap()
        .build(&mut layer_draw_context);

    command_buffer_builder.end_render_pass().unwrap();
}

fn build_blit_offscreen_to_result_image(
    build_context: &mut BuildContext<MyState, MySetupState>,
    command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) {
    let source_image = build_context
        .state
        .render_passes
        .offscreen_render_pass
        .attachment_image
        .as_ref()
        .unwrap()
        .clone();
    let destination_image = build_context
        .state
        .render_passes
        .offscreen_render_pass
        .result_image
        .as_ref()
        .unwrap()
        .clone();
    command_buffer_builder
        .blit_image(BlitImageInfo {
            src_image_layout: ImageLayout::General,
            dst_image_layout: ImageLayout::General,
            regions: [ImageBlit {
                src_subresource: source_image.subresource_layers(),
                src_offsets: [[0, 0, 0], [500, 500, 1]],
                dst_subresource: destination_image.subresource_layers(),
                dst_offsets: [[0, 0, 0], [250, 250, 1]],
                ..Default::default()
            }]
            .into(),
            filter: Filter::Nearest,
            ..BlitImageInfo::images(source_image.clone(), destination_image.clone())
        })
        .unwrap();
}

fn build_swapchain_render_pass(
    build_context: &mut BuildContext<MyState, MySetupState>,
    command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) {
    command_buffer_builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([1.0, 1.0, 0.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(build_context.swapchain_framebuffer.clone())
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .set_viewport(0, [build_context.viewport.clone()]);

    // Draw
    let mut layer_draw_context = LayerBuildContext {
        state: build_context.state,
        meta: build_context.meta,
        viewport: build_context.viewport.clone(),
        device: build_context.device.clone(),
        queue: build_context.queue.clone(),
        command_buffer_builder,
    };
    build_context
        .setup_state
        .layers
        .image_debug_draw_layer
        .as_mut()
        .unwrap()
        .build(&mut layer_draw_context);

    command_buffer_builder.end_render_pass().unwrap();
}

#[derive(Default)]
pub struct ImageDebugDrawLayer {
    device: Option<Arc<Device>>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: Option<Arc<CpuAccessibleBuffer<[Vertex]>>>,
    descriptor_set: Option<Arc<PersistentDescriptorSet>>,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
}
impl_vertex!(Vertex, position, uv);
impl ImageDebugDrawLayer {
    fn setup(&mut self, setup_context: &mut LayerSetupContext<MyState, MySetupState>) {
        let vertices = [
            Vertex {
                position: [-0.5, -0.5],
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [-0.5, 0.5],
                uv: [0.0, 1.0],
            },
            Vertex {
                position: [0.5, 0.5],
                uv: [1.0, 1.0],
            },
            Vertex {
                position: [-0.5, -0.5],
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.5],
                uv: [1.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5],
                uv: [1.0, 0.0],
            },
        ];
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            setup_context.device.clone(),
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
                layout(location = 1) in vec2 uv; 

                layout(location = 0) out vec2 v_uv;

				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
                    v_uv = uv;
				}
			"
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: "
				#version 450

                layout(set = 0, binding = 0) uniform sampler2D tex;

                layout(location = 0) in vec2 v_uv;

				layout(location = 0) out vec4 f_color;

				void main() {
                    f_color = texture(tex, v_uv);
				}
			"
            }
        }

        let vs = vs::load(setup_context.device.clone()).expect("Could not load vertex shader");
        let fs = fs::load(setup_context.device.clone()).expect("Could not load fragment shader");

        let pipeline = GraphicsPipeline::start()
            .render_pass(Subpass::from(setup_context.render_pass.clone(), 0).unwrap())
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .build(setup_context.device.clone())
            .expect("Could not build pipeline");

        let sampler = Sampler::new(
            setup_context.device.clone(),
            SamplerCreateInfo {
                min_filter: Filter::Nearest,
                mag_filter: Filter::Nearest,
                mipmap_mode: SamplerMipmapMode::Nearest,
                ..Default::default()
            },
        )
        .expect("Could not create sampler");

        let layout = pipeline
            .layout()
            .set_layouts()
            .get(0)
            .expect("Could not get layout");

        let texture = ImageView::new_default(
            setup_context
                .state
                .render_passes
                .offscreen_render_pass
                .result_image
                .as_ref()
                .unwrap()
                .clone(),
        )
        .unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(0, texture, sampler)],
        )
        .expect("Could not create descriptor set");

        self.device = Some(setup_context.device.clone());
        self.pipeline = Some(pipeline);
        self.vertex_buffer = Some(vertex_buffer);
        self.descriptor_set = Some(descriptor_set);
    }
    fn build(&mut self, draw_context: &mut LayerBuildContext<MyState>) {
        draw_context
            .command_buffer_builder
            .bind_pipeline_graphics(self.pipeline.as_ref().unwrap().clone())
            .bind_vertex_buffers(0, self.vertex_buffer.as_ref().unwrap().clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.as_ref().as_mut().unwrap().layout().clone(),
                0,
                self.descriptor_set.as_ref().unwrap().clone(),
            )
            .draw(self.vertex_buffer.as_ref().unwrap().len() as u32, 1, 0, 0)
            .unwrap();
    }
}

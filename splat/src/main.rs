use splat::{
    render, BasicTriangleDrawLayer, CopyDrawLayer, DrawContext, DrawLayer, LayerDrawContext,
    LayerSetupContext, Meta, MySetupState, MyState, SetupContext, SetupResponse, SplatCreateInfo,
    TextDrawLayer,
};
use std::{cell::RefCell, rc::Rc, sync::Arc};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, ImageBlit,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    format::Format,
    image::{
        view::ImageView, AttachmentImage, ImageAccess, ImageDimensions, ImageLayout, ImageUsage,
        StorageImage,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    sampler::Filter,
};

fn main() {
    render(
        SplatCreateInfo::default(),
        MyState {},
        MySetupState::default(),
        setup,
        draw,
    );
}

fn setup(setup_context: &mut SetupContext<MyState, MySetupState>) -> SetupResponse {
    setup_offscreen_render_pass(setup_context);
    setup_swapchain_render_pass(setup_context);

    SetupResponse {
        swapchain_render_pass: setup_context
            .setup_state
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
    let offscreen_render_pass = &mut setup_context
        .setup_state
        .render_passes
        .offscreen_render_pass;
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

    let mut basic_triangle_draw_layer = BasicTriangleDrawLayer::default();
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
    let swapchain_render_pass = &mut setup_context
        .setup_state
        .render_passes
        .swapchain_render_pass;
    swapchain_render_pass.render_pass = Some(render_pass.clone());

    // Setup draw layers
    let mut layer_setup_context = LayerSetupContext {
        state: setup_context.state,
        setup_state: setup_context.setup_state,
        device: setup_context.device.clone(),
        queue: setup_context.queue.clone(),
        render_pass: render_pass.clone(),
    };

    let mut copy_draw_layer = CopyDrawLayer::default();
    copy_draw_layer.setup(&mut layer_setup_context);
    setup_context.setup_state.layers.copy_draw_layer = Some(copy_draw_layer);
}

fn draw(draw_context: &mut DrawContext<MyState, MySetupState>) -> PrimaryAutoCommandBuffer {
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        draw_context.device.clone(),
        draw_context.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // Offscreen pass
    draw_offscreen_render_pass(draw_context, &mut command_buffer_builder);

    // Compute pass
    compute_blit_offscreen_to_result_image(draw_context, &mut command_buffer_builder);

    // Swapchain pass
    draw_swapchain_render_pass(draw_context, &mut command_buffer_builder);

    command_buffer_builder.build().unwrap()
}

fn draw_offscreen_render_pass(
    draw_context: &mut DrawContext<MyState, MySetupState>,
    command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) {
    command_buffer_builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 1.0, 0.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(
                    draw_context
                        .setup_state
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
        .set_viewport(0, [draw_context.viewport.clone()]);

    // Draw
    let mut layer_draw_context = LayerDrawContext {
        state: draw_context.state,
        meta: draw_context.meta,
        viewport: draw_context.viewport.clone(),
        command_buffer_builder,
    };
    <BasicTriangleDrawLayer as DrawLayer<MyState, MySetupState>>::draw(
        &mut draw_context
            .setup_state
            .layers
            .basic_triangle_draw_layer
            .as_mut()
            .unwrap(),
        &mut layer_draw_context,
    );

    command_buffer_builder.end_render_pass().unwrap();
}

fn compute_blit_offscreen_to_result_image(
    draw_context: &mut DrawContext<MyState, MySetupState>,
    command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) {
    let source_image = draw_context
        .setup_state
        .render_passes
        .offscreen_render_pass
        .attachment_image
        .as_ref()
        .unwrap()
        .clone();
    let destination_image = draw_context
        .setup_state
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

fn draw_swapchain_render_pass(
    draw_context: &mut DrawContext<MyState, MySetupState>,
    command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) {
    command_buffer_builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([1.0, 1.0, 0.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(draw_context.swapchain_framebuffer.clone())
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .set_viewport(0, [draw_context.viewport.clone()]);

    // Draw
    let mut layer_draw_context = LayerDrawContext {
        state: draw_context.state,
        meta: draw_context.meta,
        viewport: draw_context.viewport.clone(),
        command_buffer_builder,
    };
    draw_context
        .setup_state
        .layers
        .copy_draw_layer
        .as_mut()
        .unwrap()
        .draw(&mut layer_draw_context);

    command_buffer_builder.end_render_pass().unwrap();
}

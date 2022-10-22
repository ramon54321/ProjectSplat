use splat::{
    render, BasicTriangleDrawLayer, CopyDrawLayer, DrawContext, DrawLayer, Meta, MyState,
    SetupContext, SplatCreateInfo, TextDrawLayer,
};
use std::{cell::RefCell, rc::Rc};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    image::StorageImage,
};

fn main() {
    render(
        SplatCreateInfo::default(),
        MyState {
            other_pass_result_image: None,
        },
        setup,
        draw,
    );
}

fn setup(setup_context: &mut SetupContext<MyState>) {
    let text_draw_layer = Rc::new(RefCell::new(TextDrawLayer::default()));
    let basic_triangle_draw_layer_pre = Rc::new(RefCell::new(BasicTriangleDrawLayer::default()));
    let copy_draw_layer = Rc::new(RefCell::new(CopyDrawLayer::default()));
    let pre_layers: Vec<Rc<RefCell<dyn DrawLayer<MyState>>>> = vec![basic_triangle_draw_layer_pre];
    let layers: Vec<Rc<RefCell<dyn DrawLayer<MyState>>>> = vec![copy_draw_layer, text_draw_layer];

    //let mut setup_info = SetupInfo {
    //state: &mut state,
    //device: device.clone(),
    //queue: queue.clone(),
    //render_pass: pre_render_pass.clone(),
    //pre_destination_image: pre_destination_image.clone(),
    //};
    //for pre_layer in pre_layers.iter_mut() {
    //pre_layer.borrow_mut().setup(&mut setup_info);
    //}
    //for layer in layers.iter_mut() {
    //layer.borrow_mut().setup(&mut setup_info);
    //}
}

fn draw(draw_context: &mut DrawContext<MyState>) -> PrimaryAutoCommandBuffer {
    println!("Building my own command buffers");

    let device = draw_context.device.clone();
    let queue = draw_context.queue.clone();

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    command_buffer_builder.build().unwrap()
}

use splat::{
    render, BasicTriangleDrawLayer, CopyDrawLayer, DrawLayer, SplatCreateInfo, TextDrawLayer,
};
use std::{cell::RefCell, rc::Rc};

fn main() {
    struct MyState {}
    let text_draw_layer = Rc::new(RefCell::new(TextDrawLayer::default()));
    let basic_triangle_draw_layer_pre = Rc::new(RefCell::new(BasicTriangleDrawLayer::default()));
    let copy_draw_layer = Rc::new(RefCell::new(CopyDrawLayer::default()));
    let pre_layers: Vec<Rc<RefCell<dyn DrawLayer<MyState>>>> = vec![basic_triangle_draw_layer_pre];
    let layers: Vec<Rc<RefCell<dyn DrawLayer<MyState>>>> = vec![copy_draw_layer, text_draw_layer];
    render(SplatCreateInfo::default(), MyState {}, pre_layers, layers);
}

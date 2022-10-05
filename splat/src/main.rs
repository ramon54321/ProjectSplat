use splat::{render, BasicTriangleDrawLayer, DrawLayer, TextDrawLayer};
use std::{cell::RefCell, rc::Rc};

fn main() {
    struct MyState {}
    let text_draw_layer = Rc::new(RefCell::new(TextDrawLayer::default()));
    let basic_triangle_draw_layer = Rc::new(RefCell::new(BasicTriangleDrawLayer::default()));
    let layers: Vec<Rc<RefCell<dyn DrawLayer<MyState>>>> =
        vec![basic_triangle_draw_layer, text_draw_layer];
    render(MyState {}, layers);
}

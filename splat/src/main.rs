use splat::{render, BasicTriangleDrawLayer, DrawLayer};

fn main() {
    let layers: Vec<Box<dyn DrawLayer>> = vec![Box::new(BasicTriangleDrawLayer::default())];
    render(layers);
}

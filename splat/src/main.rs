use splat::{render, BasicTriangleDrawLayer, DrawLayer, TextDrawLayer};

fn main() {
    let layers: Vec<Box<dyn DrawLayer>> = vec![
        //Box::new(BasicTriangleDrawLayer::default()),
        Box::new(TextDrawLayer::default()),
    ];
    render(layers);
}

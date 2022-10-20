use crate::{DrawInfo, DrawLayer, SetupInfo};
use bytemuck::{Pod, Zeroable};
use std::{fmt::Debug, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    impl_vertex,
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline,
    },
    render_pass::Subpass,
};

#[derive(Default)]
pub struct BasicTriangleDrawLayer {
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: Option<Arc<CpuAccessibleBuffer<[Vertex]>>>,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
}
impl_vertex!(Vertex, position);
impl<T> DrawLayer<T> for BasicTriangleDrawLayer {
    fn setup(&mut self, setup_info: &mut SetupInfo<T>) {
        let vertices = [
            Vertex {
                position: [-0.5, -0.25],
            },
            Vertex {
                position: [0.0, 0.5],
            },
            Vertex {
                position: [0.25, -0.1],
            },
        ];
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            setup_info.device.clone(),
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

				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
				}
			"
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: "
				#version 450

				layout(location = 0) out vec4 f_color;

				void main() {
					f_color = vec4(1.0, 0.0, 0.0, 1.0);
				}
			"
            }
        }

        let vs = vs::load(setup_info.device.clone()).expect("Could not load vertex shader");
        let fs = fs::load(setup_info.device.clone()).expect("Could not load fragment shader");

        let pipeline = GraphicsPipeline::start()
            .render_pass(Subpass::from(setup_info.render_pass.clone(), 0).unwrap())
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .build(setup_info.device.clone())
            .expect("Could not build pipeline");

        self.pipeline = Some(pipeline);
        self.vertex_buffer = Some(vertex_buffer);
    }
    fn draw(&mut self, draw_info: &mut DrawInfo<T>) {
        draw_info
            .gpu_interface
            .command_buffer_builder
            .bind_pipeline_graphics(self.pipeline.as_ref().unwrap().clone())
            .bind_vertex_buffers(0, self.vertex_buffer.as_ref().unwrap().clone())
            .draw(self.vertex_buffer.as_ref().unwrap().len() as u32, 1, 0, 0)
            .unwrap();
    }
}

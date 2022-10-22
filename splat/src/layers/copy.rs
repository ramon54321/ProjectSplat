use crate::{DrawInfo, DrawLayer, SetupInfo};
use bytemuck::{Pod, Zeroable};
use std::{fmt::Debug, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{BlitImageInfo, ImageBlit},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageAccess, ImageCreateFlags, ImageDimensions, ImageLayout, ImageUsage, ImmutableImage,
        MipmapsCount,
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
    render_pass::Subpass,
    sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode},
};

#[derive(Default)]
pub struct CopyDrawLayer {
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
impl<T> DrawLayer<T> for CopyDrawLayer {
    fn setup(&mut self, setup_info: &mut SetupInfo<T>) {
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

        let sampler = Sampler::new(
            setup_info.device.clone(),
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

        let texture = ImageView::new_default(setup_info.pre_destination_image.clone()).unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(0, texture, sampler)],
        )
        .expect("Could not create descriptor set");

        self.device = Some(setup_info.device.clone());
        self.pipeline = Some(pipeline);
        self.vertex_buffer = Some(vertex_buffer);
        self.descriptor_set = Some(descriptor_set);
    }
    fn draw(&mut self, draw_info: &mut DrawInfo<T>) {
        draw_info
            .gpu_interface
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

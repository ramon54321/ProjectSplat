use crate::DrawLayer;
use bytemuck::{Pod, Zeroable};
use rusttype::{gpu_cache::Cache, point, Font, Glyph, PositionedGlyph, Rect, Scale};
use std::{fmt::Debug, io::Cursor, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferToImageInfo, PrimaryAutoCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageCreateFlags, ImageDimensions, ImageLayout, ImageUsage, ImmutableImage, MipmapsCount,
    },
    impl_vertex,
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode},
};

const CACHE_WIDTH: usize = 1000;
const CACHE_HEIGHT: usize = 1000;

struct TextData {
    glyphs: Vec<PositionedGlyph<'static>>,
    color: [f32; 4],
}

#[derive(Default)]
pub struct TextDrawLayer {
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    font: Option<Font<'static>>,
    cache: Option<Cache<'static>>,
    cache_pixels: Vec<u8>,
    texts: Vec<TextData>,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
    tex_position: [f32; 2],
    color: [f32; 4],
}
impl_vertex!(Vertex, position, tex_position, color);
impl TextDrawLayer {
    fn queue_text(&mut self, x: f32, y: f32, size: f32, color: [f32; 4], text: &str) {
        let glyphs = self
            .font
            .as_ref()
            .unwrap()
            .layout(text, Scale::uniform(size), point(x, y))
            .map(|glyph| glyph.standalone())
            .collect::<Vec<_>>();
        for glyph in &glyphs {
            self.cache.as_mut().unwrap().queue_glyph(0, glyph.clone());
        }
        self.texts.push(TextData {
            glyphs: glyphs.clone(),
            color,
        });
    }
}
impl DrawLayer for TextDrawLayer {
    fn setup(&mut self, device: Arc<Device>, queue: Arc<Queue>, render_pass: Arc<RenderPass>) {
        let font_data = include_bytes!("DejaVuSans.ttf");
        let font = Font::from_bytes(font_data as &[u8]).unwrap();
        let cache = Cache::builder()
            .dimensions(CACHE_WIDTH as u32, CACHE_HEIGHT as u32)
            .build();
        let cache_pixels = vec![0; CACHE_WIDTH * CACHE_HEIGHT];

        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: "
                    #version 450
                    
                    layout(location = 0) in vec2 position;
                    layout(location = 1) in vec2 tex_position;
                    layout(location = 2) in vec4 color;
                    layout(location = 0) out vec2 v_tex_position;
                    layout(location = 1) out vec4 v_color;
                    
                    void main() {
                        gl_Position = vec4(position, 0.0, 1.0);
                        v_tex_position = tex_position;
                        v_color = color;
                    }
			"
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: "
                    #version 450

                    layout(location = 0) in vec2 v_tex_position;
                    layout(location = 1) in vec4 v_color;
                    layout(location = 0) out vec4 f_color;
                    
                    layout(set = 0, binding = 0) uniform sampler2D tex;
                    
                    void main() {
                        f_color = v_color * texture(tex, v_tex_position)[0];
                    }
			"
            }
        }

        let vs = vs::load(device.clone()).expect("Could not load vertex shader");
        let fs = fs::load(device.clone()).expect("Could not load fragment shader");

        let pipeline = GraphicsPipeline::start()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .blend_alpha_blending()
            .build(device.clone())
            .expect("Could not build pipeline");

        self.device = Some(device);
        self.queue = Some(queue);
        self.pipeline = Some(pipeline);
        self.font = Some(font);
        self.cache = Some(cache);
        self.cache_pixels = cache_pixels;
    }
    fn draw(
        &mut self,
        command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        viewport: Viewport,
    ) {
        self.queue_text(
            0.0,
            100.0,
            80.0,
            [0.0, 1.0, 0.0, 1.0],
            "hello, how are you?",
        );
        self.queue_text(50.0, 50.0, 30.0, [1.0, 1.0, 0.0, 1.0], "I am very good...");

        println!("Drawing {} texts", self.texts.len());

        // update texture cache
        self.cache
            .as_mut()
            .unwrap()
            .cache_queued(|rect, src_data| {
                let width = (rect.max.x - rect.min.x) as usize;
                let height = (rect.max.y - rect.min.y) as usize;
                let mut dst_index = rect.min.y as usize * CACHE_WIDTH + rect.min.x as usize;
                let mut src_index = 0;

                for _ in 0..height {
                    let dst_slice = &mut self.cache_pixels[dst_index..dst_index + width];
                    let src_slice = &src_data[src_index..src_index + width];
                    dst_slice.copy_from_slice(src_slice);

                    dst_index += CACHE_WIDTH;
                    src_index += width;
                }
            })
            .unwrap();

        let (cache_texture, future) = ImmutableImage::from_iter(
            self.cache_pixels.clone(),
            ImageDimensions::Dim2d {
                width: CACHE_WIDTH as u32,
                height: CACHE_HEIGHT as u32,
                array_layers: 1,
            },
            MipmapsCount::One,
            Format::R8_UNORM,
            self.queue.as_ref().unwrap().clone(),
        )
        .unwrap();

        let sampler = Sampler::new(
            self.device.as_ref().as_mut().unwrap().clone(),
            SamplerCreateInfo {
                min_filter: Filter::Linear,
                mag_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let cache_texture_view = ImageView::new_default(cache_texture).unwrap();

        let layout = self
            .pipeline
            .as_ref()
            .as_mut()
            .unwrap()
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();

        //// Use `image_view` instead of `image_view_sampler`, since the sampler is already in the layout.
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                cache_texture_view,
                sampler,
            )],
        )
        .unwrap();

        let vertices = vec![
            Vertex {
                position: [0.0, 1.0],
                tex_position: [0.0, 1.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                position: [0.0, 0.0],
                tex_position: [0.0, 0.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 0.0],
                tex_position: [1.0, 0.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 0.0],
                tex_position: [1.0, 0.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0],
                tex_position: [1.0, 1.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                position: [0.0, 1.0],
                tex_position: [0.0, 1.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
        ];
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            self.device.as_ref().as_mut().unwrap().clone(),
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            vertices,
        )
        .expect("Could not create buffer from iter");

        let screen_width = viewport.dimensions[0];
        let screen_height = viewport.dimensions[1];

        command_buffer_builder
            .bind_pipeline_graphics(self.pipeline.as_ref().unwrap().clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.as_ref().as_mut().unwrap().layout().clone(),
                0,
                set.clone(),
            );

        for text in self.texts.iter() {
            let mut text_vertices = Vec::new();
            for glyph in text.glyphs.iter() {
                if let Ok(Some((uv_rect, screen_rect))) =
                    self.cache.as_ref().unwrap().rect_for(0, glyph)
                {
                    let gl_rect = Rect {
                        min: point(
                            (screen_rect.min.x as f32 / screen_width as f32 - 0.5) * 2.0,
                            (screen_rect.min.y as f32 / screen_height as f32 - 0.5) * 2.0,
                        ),
                        max: point(
                            (screen_rect.max.x as f32 / screen_width as f32 - 0.5) * 2.0,
                            (screen_rect.max.y as f32 / screen_height as f32 - 0.5) * 2.0,
                        ),
                    };
                    let glyph_verticies = vec![
                        Vertex {
                            position: [gl_rect.min.x, gl_rect.max.y],
                            tex_position: [uv_rect.min.x, uv_rect.max.y],
                            color: text.color,
                        },
                        Vertex {
                            position: [gl_rect.min.x, gl_rect.min.y],
                            tex_position: [uv_rect.min.x, uv_rect.min.y],
                            color: text.color,
                        },
                        Vertex {
                            position: [gl_rect.max.x, gl_rect.min.y],
                            tex_position: [uv_rect.max.x, uv_rect.min.y],
                            color: text.color,
                        },
                        Vertex {
                            position: [gl_rect.max.x, gl_rect.min.y],
                            tex_position: [uv_rect.max.x, uv_rect.min.y],
                            color: text.color,
                        },
                        Vertex {
                            position: [gl_rect.max.x, gl_rect.max.y],
                            tex_position: [uv_rect.max.x, uv_rect.max.y],
                            color: text.color,
                        },
                        Vertex {
                            position: [gl_rect.min.x, gl_rect.max.y],
                            tex_position: [uv_rect.min.x, uv_rect.max.y],
                            color: text.color,
                        },
                    ];
                    for glyph_vertex in glyph_verticies {
                        text_vertices.push(glyph_vertex);
                    }
                };
            }
            let vertex_buffer = CpuAccessibleBuffer::from_iter(
                self.device.as_ref().as_mut().unwrap().clone(),
                BufferUsage {
                    vertex_buffer: true,
                    ..BufferUsage::empty()
                },
                false,
                text_vertices,
            )
            .expect("Could not create buffer from iter");
            command_buffer_builder
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap();
        }

        self.texts.clear();
    }
}

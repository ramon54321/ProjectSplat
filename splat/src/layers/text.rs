use crate::{AlignHorizontal, AlignVertical, DrawLayer, GpuInterface, Meta};
use bytemuck::{Pod, Zeroable};
use rusttype::{gpu_cache::Cache, point, Font, PositionedGlyph, Rect, Scale};
use std::{collections::HashMap, fmt::Debug, rc::Rc, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    impl_vertex,
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode},
};

const TEXT_GLYPH_CACHE_LIMIT: usize = 128;
const PIXEL_CACHE_WIDTH: usize = 256;
const PIXEL_CACHE_HEIGHT: usize = 256;

type GlyphCacheEntry = (Rc<Vec<PositionedGlyph<'static>>>, f32, f32);

struct TextData {
    x: f32,
    y: f32,
    glyphs: Rc<Vec<PositionedGlyph<'static>>>,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
    tex_position: [f32; 2],
    color: [f32; 4],
}
impl_vertex!(Vertex, position, tex_position, color);

pub struct TextEnqueueRequest {
    pub x: f32,
    pub y: f32,
    pub align_horizontal: AlignHorizontal,
    pub align_vertical: AlignVertical,
    pub color: [f32; 4],
    pub text: String,
    pub should_cache: bool,
}

#[derive(Default)]
pub struct TextDrawLayer {
    device: Option<Arc<Device>>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    descriptor_set: Option<Arc<PersistentDescriptorSet>>,
    font: Option<Font<'static>>,
    cache: Option<Cache<'static>>,
    requests: Vec<TextEnqueueRequest>,
    text_glyph_cache: HashMap<String, GlyphCacheEntry>,
}
impl TextDrawLayer {
    pub fn enqueue_text(&mut self, request: TextEnqueueRequest) {
        self.requests.push(request);
    }
}
impl<T> DrawLayer<T> for TextDrawLayer {
    fn setup(&mut self, device: Arc<Device>, queue: Arc<Queue>, render_pass: Arc<RenderPass>) {
        let font_data = include_bytes!("DejaVuSans.ttf");
        let font = Font::from_bytes(font_data as &[u8]).unwrap();
        let (cache, cache_pixels) = create_glyph_cache_and_pixels(font.clone());

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
            .color_blend_state(ColorBlendState::default().blend_alpha())
            .build(device.clone())
            .expect("Could not build pipeline");

        let (cache_texture, _) = ImmutableImage::from_iter(
            cache_pixels.clone(),
            ImageDimensions::Dim2d {
                width: PIXEL_CACHE_WIDTH as u32,
                height: PIXEL_CACHE_HEIGHT as u32,
                array_layers: 1,
            },
            MipmapsCount::One,
            Format::R8_UNORM,
            queue.clone(),
        )
        .expect("Could not create text glyph cache texture");

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                min_filter: Filter::Linear,
                mag_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .expect("Could not create sampler");

        let cache_texture_view =
            ImageView::new_default(cache_texture.clone()).expect("Could not create image view");

        let layout = pipeline
            .layout()
            .set_layouts()
            .get(0)
            .expect("Could not get layout");

        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                cache_texture_view,
                sampler,
            )],
        )
        .expect("Could not create descriptor set");

        self.device = Some(device);
        self.pipeline = Some(pipeline);
        self.font = Some(font);
        self.cache = Some(cache);
        self.descriptor_set = Some(set);
    }
    fn draw(&mut self, gpu_interface: &mut GpuInterface, _meta: &mut Meta, _state: &mut T) {
        let screen_width = gpu_interface.viewport.dimensions[0];
        let screen_height = gpu_interface.viewport.dimensions[1];

        // Prepare command buffer for text draw calls
        gpu_interface
            .command_buffer_builder
            .bind_pipeline_graphics(self.pipeline.as_ref().unwrap().clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.as_ref().as_mut().unwrap().layout().clone(),
                0,
                self.descriptor_set.as_ref().unwrap().clone(),
            );

        // Debug cache texture
        #[cfg(debug_vulkan_text)]
        submit_debug_cache_texture_draw(gpu_interface, self.device.as_ref().unwrap().clone());

        let text_datas = build_text_data_from_requests(
            &self.requests,
            &mut self.text_glyph_cache,
            self.font.as_ref().unwrap(),
        );
        self.requests.clear();

        let vertices = build_vertices_from_text_datas(
            &text_datas,
            self.cache.as_ref().unwrap(),
            screen_width,
            screen_height,
        );

        submit_vertices_draw(
            gpu_interface,
            self.device.as_ref().unwrap().clone(),
            vertices,
        );
    }
}

fn create_glyph_cache_and_pixels<'a>(font: Font<'a>) -> (Cache<'a>, Vec<u8>) {
    let mut cache = Cache::builder()
        .dimensions(PIXEL_CACHE_WIDTH as u32, PIXEL_CACHE_HEIGHT as u32)
        .scale_tolerance(5.0)
        .position_tolerance(5.0)
        .build();

    let glyphs = font.glyphs_for(
        " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:!@#$%^&*()[]{}?-=1234567890`~/"
            .chars(),
    );

    let scale = Scale::uniform(24.0);
    for glyph in glyphs {
        let glyph = glyph.scaled(scale);
        let glyph = glyph.positioned(point(0.0, 0.0));
        cache.queue_glyph(0, glyph.clone());
    }

    let mut pixels = vec![0; PIXEL_CACHE_WIDTH * PIXEL_CACHE_HEIGHT];
    cache
        .cache_queued(|rect, src_data| {
            let width = (rect.max.x - rect.min.x) as usize;
            let height = (rect.max.y - rect.min.y) as usize;
            let mut dst_index = rect.min.y as usize * PIXEL_CACHE_WIDTH + rect.min.x as usize;
            let mut src_index = 0;

            for _ in 0..height {
                let dst_slice = &mut pixels[dst_index..dst_index + width];
                let src_slice = &src_data[src_index..src_index + width];
                dst_slice.copy_from_slice(src_slice);

                dst_index += PIXEL_CACHE_WIDTH;
                src_index += width;
            }
        })
        .expect("Could not cache glyphs");

    (cache, pixels)
}

fn build_text_data_from_requests(
    requests: &Vec<TextEnqueueRequest>,
    text_glyph_cache: &mut HashMap<String, GlyphCacheEntry>,
    font: &Font<'static>,
) -> Vec<TextData> {
    let mut texts = Vec::with_capacity(requests.len());
    for i in 0..requests.len() {
        let request = &requests[i];
        let scale = Scale::uniform(24.0);

        // Check if glyphs for this text is cached
        if request.should_cache {
            if let Some((glyphs, horizontal_offset, vertical_offset)) =
                text_glyph_cache.get(&request.text)
            {
                texts.push(TextData {
                    x: request.x - horizontal_offset,
                    y: request.y + vertical_offset,
                    glyphs: glyphs.clone(),
                    color: request.color,
                });
                continue;
            }
        }

        // Alignment
        let v_metrics = font.v_metrics(scale);
        let line_height = v_metrics.ascent;
        let vertical_offset = match request.align_vertical {
            AlignVertical::Top => line_height,
            AlignVertical::Center => line_height / 2.0,
            AlignVertical::Bottom => 0.0,
        };

        // Place glyphs
        let mut glyphs = Vec::with_capacity(request.text.len());
        let mut x_current = 0.0;
        let mut width_total = 0.0;
        for char in request.text.chars() {
            let glyph = font.glyph(char);
            let glyph = glyph.scaled(scale);
            let advance_width = glyph.h_metrics().advance_width;
            let next_glyph = glyph.positioned(point(x_current, 0.0));
            x_current += advance_width;
            width_total = width_total + advance_width;
            glyphs.push(next_glyph);
        }

        // Alignment
        let horizontal_offset = match request.align_horizontal {
            AlignHorizontal::Left => 0.0,
            AlignHorizontal::Center => width_total / 2.0,
            AlignHorizontal::Right => width_total,
        };

        // Reference count glyphs to avoid clone in cache
        let glyphs = Rc::new(glyphs);

        // Store
        texts.push(TextData {
            x: request.x - horizontal_offset,
            y: request.y + vertical_offset,
            glyphs: glyphs.clone(),
            color: request.color,
        });

        if request.should_cache {
            // Reset cache if too large
            if text_glyph_cache.len() >= TEXT_GLYPH_CACHE_LIMIT {
                text_glyph_cache.clear();
                println!("Resetting Cache");
            }

            // Cache glyphs
            text_glyph_cache.insert(
                request.text.clone(),
                (glyphs, horizontal_offset, vertical_offset),
            );
        }
    }
    texts
}

fn build_vertices_from_text_datas(
    texts: &Vec<TextData>,
    cache: &Cache,
    screen_width: f32,
    screen_height: f32,
) -> Vec<Vertex> {
    let mut vertices: Vec<Vertex> = Vec::with_capacity(1000);
    for text_index in 0..texts.len() {
        let text = &texts[text_index];
        for glyph_index in 0..text.glyphs.len() {
            let glyph = &text.glyphs[glyph_index];
            if let Ok(Some((uv_rect, screen_rect))) = cache.rect_for(0, glyph) {
                let gl_rect = Rect {
                    min: point(
                        ((screen_rect.min.x as f32 + text.x) / screen_width - 0.5) * 2.0,
                        ((screen_rect.min.y as f32 + text.y) / screen_height - 0.5) * 2.0,
                    ),
                    max: point(
                        ((screen_rect.max.x as f32 + text.x) / screen_width - 0.5) * 2.0,
                        ((screen_rect.max.y as f32 + text.y) / screen_height - 0.5) * 2.0,
                    ),
                };
                vertices.push(Vertex {
                    position: [gl_rect.min.x, gl_rect.max.y],
                    tex_position: [uv_rect.min.x, uv_rect.max.y],
                    color: text.color,
                });
                vertices.push(Vertex {
                    position: [gl_rect.min.x, gl_rect.min.y],
                    tex_position: [uv_rect.min.x, uv_rect.min.y],
                    color: text.color,
                });
                vertices.push(Vertex {
                    position: [gl_rect.max.x, gl_rect.min.y],
                    tex_position: [uv_rect.max.x, uv_rect.min.y],
                    color: text.color,
                });
                vertices.push(Vertex {
                    position: [gl_rect.max.x, gl_rect.min.y],
                    tex_position: [uv_rect.max.x, uv_rect.min.y],
                    color: text.color,
                });
                vertices.push(Vertex {
                    position: [gl_rect.max.x, gl_rect.max.y],
                    tex_position: [uv_rect.max.x, uv_rect.max.y],
                    color: text.color,
                });
                vertices.push(Vertex {
                    position: [gl_rect.min.x, gl_rect.max.y],
                    tex_position: [uv_rect.min.x, uv_rect.max.y],
                    color: text.color,
                });
            };
        }
    }
    vertices
}

fn submit_vertices_draw(
    gpu_interface: &mut GpuInterface,
    device: Arc<Device>,
    vertices: Vec<Vertex>,
) {
    if vertices.len() > 0 {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            vertices,
        )
        .expect("Could not create buffer from iter");
        gpu_interface
            .command_buffer_builder
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
    }
}

#[allow(dead_code)]
fn submit_debug_cache_texture_draw(gpu_interface: &mut GpuInterface, device: Arc<Device>) {
    let mut vertices = Vec::new();
    vertices.push(Vertex {
        position: [0.0, 1.0],
        tex_position: [0.0, 1.0],
        color: [1.0, 1.0, 1.0, 1.0],
    });
    vertices.push(Vertex {
        position: [0.0, 0.0],
        tex_position: [0.0, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
    });
    vertices.push(Vertex {
        position: [1.0, 0.0],
        tex_position: [1.0, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
    });
    vertices.push(Vertex {
        position: [1.0, 0.0],
        tex_position: [1.0, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
    });
    vertices.push(Vertex {
        position: [1.0, 1.0],
        tex_position: [1.0, 1.0],
        color: [1.0, 1.0, 1.0, 1.0],
    });
    vertices.push(Vertex {
        position: [0.0, 1.0],
        tex_position: [0.0, 1.0],
        color: [1.0, 1.0, 1.0, 1.0],
    });
    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device,
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        vertices,
    )
    .expect("Could not create buffer from iter");
    gpu_interface
        .command_buffer_builder
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .draw(vertex_buffer.len() as u32, 1, 0, 0)
        .unwrap();
}

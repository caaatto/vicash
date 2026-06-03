use crate::audio::AudioRuntime;
use crate::frame::{Frame, SharedFrame, UiEvent};
use crate::settings::{CaptureInfo, FitMode, Settings};
use anyhow::{Context, Result, anyhow};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Icon, Window, WindowId};

const ICON_PNG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/icon-256.png"));

pub fn build_event_loop() -> Result<EventLoop<UiEvent>> {
    EventLoop::<UiEvent>::with_user_event()
        .build()
        .context("failed to create event loop")
}

pub fn run(
    event_loop: EventLoop<UiEvent>,
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
    capture_info: CaptureInfo,
    audio: Option<Arc<AudioRuntime>>,
) -> Result<()> {
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        shared,
        settings,
        capture_info,
        audio,
        gpu: None,
        last_seq: 0,
        last_log: Instant::now(),
        frames_since_log: 0,
        preview_fps: 0.0,
        tex_size: (0, 0),
    };
    event_loop.run_app(&mut app).context("event loop exited with error")?;
    Ok(())
}

struct App {
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
    capture_info: CaptureInfo,
    audio: Option<Arc<AudioRuntime>>,
    gpu: Option<Gpu>,
    last_seq: u64,
    last_log: Instant,
    frames_since_log: u64,
    preview_fps: f32,
    tex_size: (u32, u32),
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture: Option<wgpu::Texture>,
    bind_group: Option<wgpu::BindGroup>,
    vertex_buffer: wgpu::Buffer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

const SHADER: &str = r#"
struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv:  vec2<f32>,
};
struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"#;

impl ApplicationHandler<UiEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let icon = load_icon();
        let mut attrs = Window::default_attributes()
            .with_title("vicash")
            .with_inner_size(PhysicalSize::new(1280u32, 720u32));
        if let Some(i) = icon.clone() {
            attrs = attrs.with_window_icon(Some(i));
        }
        #[cfg(target_os = "windows")]
        if let Some(i) = icon {
            use winit::platform::windows::WindowAttributesExtWindows;
            attrs = attrs.with_taskbar_icon(Some(i));
        }
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };
        // Apply the icon to the live window too. Windows in particular
        // sometimes ignores the attribute and only honours set_window_icon
        // called after the HWND exists.
        if let Some(i) = load_icon() {
            window.set_window_icon(Some(i));
        }
        match pollster::block_on(init_gpu(window.clone())) {
            Ok(gpu) => {
                gpu.window.request_redraw();
                self.gpu = Some(gpu);
            }
            Err(e) => {
                log::error!("wgpu init failed: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UiEvent) {
        match event {
            UiEvent::FrameReady => {
                if let Some(gpu) = self.gpu.as_ref() {
                    gpu.window.request_redraw();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else { return };

        // egui sees the event first. We do NOT unconditionally request a
        // redraw on response.repaint, because egui can return repaint=true
        // for synthetic events and that cascades into a render-every-tick
        // loop. The capture thread drives our redraws via FrameReady, which
        // gives egui a chance to re-run at capture fps and stay responsive.
        let _ = gpu.egui_state.on_window_event(&gpu.window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    let max_dim = gpu.device.limits().max_texture_dimension_2d;
                    gpu.config.width = new_size.width.clamp(1, max_dim);
                    gpu.config.height = new_size.height.clamp(1, max_dim);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::F1),
                    repeat: false,
                    ..
                },
                ..
            } => {
                let mut s = self.settings.lock();
                s.show_panel = !s.show_panel;
                gpu.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Some(f) = self.shared.get() {
                    if f.seq != self.last_seq {
                        self.last_seq = f.seq;
                        upload_frame(gpu, &f, &mut self.tex_size);
                        self.frames_since_log += 1;
                    }
                }
                let mut settings = self.settings.lock().clone();
                if let Err(e) = render_frame(
                    gpu,
                    &mut settings,
                    &self.capture_info,
                    self.preview_fps,
                    self.tex_size,
                    self.audio.as_ref(),
                ) {
                    log::warn!("render: {e:#}");
                }
                // Write back any settings the user changed in the panel.
                *self.settings.lock() = settings;

                tick_fps_log(&mut self.last_log, &mut self.frames_since_log, &mut self.preview_fps);
            }
            _ => {}
        }
    }
}

async fn init_gpu(window: Arc<Window>) -> Result<Gpu> {
    let size = window.inner_size();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let surface = instance.create_surface(window.clone())
        .context("failed to create wgpu surface")?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow!("no suitable GPU adapter"))?;
    // Use the adapter's full limits so 1080p+ windows do not exceed the
    // texture cap that downlevel defaults set to 2048.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("vcshare-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .context("failed to get GPU device")?;
    let max_dim = device.limits().max_texture_dimension_2d;

    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
        wgpu::PresentMode::Mailbox
    } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
        wgpu::PresentMode::Immediate
    } else {
        wgpu::PresentMode::Fifo
    };
    log::info!(
        "wgpu adapter: {:?}, format: {:?}, present: {:?}",
        adapter.get_info().name,
        format,
        present_mode
    );

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.clamp(1, max_dim),
        height: size.height.clamp(1, max_dim),
        present_mode,
        desired_maximum_frame_latency: 1,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &config);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vcshare-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vcshare-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vcshare-pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("vcshare-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("vcshare-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // Empty placeholder; updated per frame to reflect the chosen fit mode.
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vcshare-vbuf"),
        contents: bytemuck::cast_slice(&quad(1.0, 1.0)),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    let egui_ctx = egui::Context::default();
    let egui_state = egui_winit::State::new(
        egui_ctx.clone(),
        egui_ctx.viewport_id(),
        &window,
        Some(window.scale_factor() as f32),
        None,
        None,
    );
    let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1, false);

    Ok(Gpu {
        window,
        surface,
        device,
        queue,
        config,
        pipeline,
        bind_group_layout,
        sampler,
        texture: None,
        bind_group: None,
        vertex_buffer,
        egui_ctx,
        egui_state,
        egui_renderer,
    })
}

fn upload_frame(gpu: &mut Gpu, frame: &Frame, current: &mut (u32, u32)) {
    if *current != (frame.width, frame.height) || gpu.texture.is_none() {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vcshare-frame-tex"),
            size: wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vcshare-bg"),
            layout: &gpu.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&gpu.sampler),
                },
            ],
        });
        gpu.texture = Some(texture);
        gpu.bind_group = Some(bind_group);
        *current = (frame.width, frame.height);
    }
    let rgba = rgb_to_rgba(&frame.rgb);
    let texture = gpu.texture.as_ref().expect("texture initialized above");
    gpu.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * frame.width),
            rows_per_image: Some(frame.height),
        },
        wgpu::Extent3d {
            width: frame.width,
            height: frame.height,
            depth_or_array_layers: 1,
        },
    );
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let pixels = rgb.len() / 3;
    let mut out = Vec::with_capacity(pixels * 4);
    for chunk in rgb.chunks_exact(3) {
        out.extend_from_slice(chunk);
        out.push(255);
    }
    out
}

fn render_frame(
    gpu: &mut Gpu,
    settings: &mut Settings,
    capture_info: &CaptureInfo,
    preview_fps: f32,
    tex_size: (u32, u32),
    audio: Option<&Arc<AudioRuntime>>,
) -> Result<()> {
    // Update vertex buffer for current fit mode.
    let (qx, qy) = quad_scale(
        settings.fit_mode,
        tex_size.0.max(1) as f32 / tex_size.1.max(1) as f32,
        gpu.config.width.max(1) as f32 / gpu.config.height.max(1) as f32,
    );
    let new_quad = quad(qx, qy);
    gpu.queue.write_buffer(&gpu.vertex_buffer, 0, bytemuck::cast_slice(&new_quad));

    let surface_texture = match gpu.surface.get_current_texture() {
        Ok(t) => t,
        Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
            gpu.surface.configure(&gpu.device, &gpu.config);
            return Ok(());
        }
        Err(e) => return Err(anyhow!("surface acquire: {e:?}")),
    };
    let view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // Build the egui UI.
    let raw_input = gpu.egui_state.take_egui_input(&gpu.window);
    let full_output = gpu.egui_ctx.run(raw_input, |ctx| {
        build_ui(ctx, settings, capture_info, preview_fps, audio);
    });
    gpu.egui_state
        .handle_platform_output(&gpu.window, full_output.platform_output);
    let pixels_per_point = full_output.pixels_per_point;
    let primitives = gpu.egui_ctx.tessellate(full_output.shapes, pixels_per_point);

    for (id, image_delta) in &full_output.textures_delta.set {
        gpu.egui_renderer
            .update_texture(&gpu.device, &gpu.queue, *id, image_delta);
    }
    let screen_desc = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [gpu.config.width, gpu.config.height],
        pixels_per_point,
    };

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vcshare-enc") });

    gpu.egui_renderer.update_buffers(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        &primitives,
        &screen_desc,
    );

    let clear_color = wgpu::Color {
        r: settings.background_color[0] as f64,
        g: settings.background_color[1] as f64,
        b: settings.background_color[2] as f64,
        a: 1.0,
    };

    {
        let mut rp = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vcshare-rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime();
        if let Some(bind_group) = gpu.bind_group.as_ref() {
            rp.set_pipeline(&gpu.pipeline);
            rp.set_bind_group(0, bind_group, &[]);
            rp.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
            rp.draw(0..6, 0..1);
        }
        gpu.egui_renderer.render(&mut rp, &primitives, &screen_desc);
    }
    gpu.queue.submit(Some(encoder.finish()));
    surface_texture.present();

    for id in &full_output.textures_delta.free {
        gpu.egui_renderer.free_texture(id);
    }
    Ok(())
}

fn build_ui(
    ctx: &egui::Context,
    settings: &mut Settings,
    capture_info: &CaptureInfo,
    preview_fps: f32,
    audio: Option<&Arc<AudioRuntime>>,
) {
    if settings.show_stats {
        egui::Area::new(egui::Id::new("stats"))
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
            .show(ctx, |ui| {
                egui::Frame::popup(&ctx.style())
                    .fill(egui::Color32::from_black_alpha(160))
                    .show(ui, |ui| {
                        ui.colored_label(
                            egui::Color32::WHITE,
                            format!(
                                "Ziel {} fps  {}",
                                capture_info.fps_target, capture_info.format_label
                            ),
                        );
                        ui.colored_label(
                            egui::Color32::WHITE,
                            format!("Vorschau {:.1} fps", preview_fps),
                        );
                        ui.colored_label(
                            egui::Color32::from_rgb(200, 200, 200),
                            "F1 für Einstellungen",
                        );
                    });
            });
    }

    if settings.show_panel {
        egui::Window::new("Einstellungen")
            .default_pos(egui::pos2(20.0, 80.0))
            .resizable(false)
            .collapsible(true)
            .show(ctx, |ui| {
                ui.label(format!(
                    "Ziel: {} fps  {}",
                    capture_info.fps_target, capture_info.format_label
                ));
                ui.separator();
                ui.label("Anzeige");
                egui::ComboBox::from_label("Bildanpassung")
                    .selected_text(match settings.fit_mode {
                        FitMode::Stretch => "Strecken",
                        FitMode::Fit => "Einpassen (Letterbox)",
                        FitMode::Fill => "Füllen (zuschneiden)",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut settings.fit_mode, FitMode::Stretch, "Strecken");
                        ui.selectable_value(&mut settings.fit_mode, FitMode::Fit, "Einpassen (Letterbox)");
                        ui.selectable_value(&mut settings.fit_mode, FitMode::Fill, "Füllen (zuschneiden)");
                    });
                ui.checkbox(&mut settings.show_stats, "Statistik-Overlay zeigen");
                ui.horizontal(|ui| {
                    ui.label("Hintergrund");
                    ui.color_edit_button_rgb(&mut settings.background_color);
                });
                ui.separator();
                ui.label("Relay");
                ui.add(
                    egui::Slider::new(&mut settings.jpeg_quality, 1..=100)
                        .text("JPEG-Qualität"),
                );

                if let Some(rt) = audio {
                    ui.separator();
                    ui.label("Audio");
                    let state = &rt.state;
                    let in_name = state.input_name();
                    let out_name = state.output_name();
                    ui.label(format!(
                        "{} Hz, {} Kanäle, gepuffert {} ms",
                        state.sample_rate(),
                        state.channels(),
                        state.buffered_ms()
                    ));

                    ui.horizontal(|ui| {
                        ui.label("Eingang ");
                        let mut chosen = in_name.clone();
                        egui::ComboBox::from_id_salt("audio_in")
                            .selected_text(short(&chosen))
                            .show_ui(ui, |ui| {
                                for name in crate::audio::list_input_devices() {
                                    ui.selectable_value(&mut chosen, name.clone(), short(&name));
                                }
                            });
                        if chosen != in_name {
                            if let Err(e) = rt.set_input(&chosen) {
                                log::warn!("input switch failed: {e:#}");
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Ausgang");
                        let mut chosen = out_name.clone();
                        egui::ComboBox::from_id_salt("audio_out")
                            .selected_text(short(&chosen))
                            .show_ui(ui, |ui| {
                                for name in crate::audio::list_output_devices() {
                                    ui.selectable_value(&mut chosen, name.clone(), short(&name));
                                }
                            });
                        if chosen != out_name {
                            if let Err(e) = rt.set_output(&chosen) {
                                log::warn!("output switch failed: {e:#}");
                            }
                        }
                    });

                    let mut volume = state.volume();
                    if ui
                        .add(
                            egui::Slider::new(&mut volume, 0..=200)
                                .text("Lautstärke (%)")
                                .integer(),
                        )
                        .changed()
                    {
                        state.set_volume(volume);
                    }
                    let mut muted = state.is_muted();
                    if ui.checkbox(&mut muted, "Stumm").changed() {
                        state.set_muted(muted);
                    }
                    let mut delay = state.delay_ms();
                    if ui
                        .add(
                            egui::Slider::new(&mut delay, 0..=500)
                                .text("Sync-Verzögerung (ms)")
                                .integer(),
                        )
                        .changed()
                    {
                        state.set_delay_ms(delay);
                    }
                } else {
                    ui.separator();
                    ui.colored_label(
                        egui::Color32::from_rgb(180, 180, 180),
                        "Audio aus (beim Start --audio mitgeben)",
                    );
                }

                ui.separator();
                ui.colored_label(
                    egui::Color32::from_rgb(180, 180, 180),
                    "Gerät, Auflösung und fps gelten ab dem nächsten Start",
                );
                if ui.button("Schließen").clicked() {
                    settings.show_panel = false;
                }
            });
    }
}

/// Returns the quad scale (x, y) in clip space for the chosen fit mode given
/// the source and destination aspect ratios.
fn quad_scale(mode: FitMode, src_aspect: f32, dst_aspect: f32) -> (f32, f32) {
    match mode {
        FitMode::Stretch => (1.0, 1.0),
        FitMode::Fit => {
            if src_aspect > dst_aspect {
                (1.0, dst_aspect / src_aspect)
            } else {
                (src_aspect / dst_aspect, 1.0)
            }
        }
        FitMode::Fill => {
            if src_aspect > dst_aspect {
                (src_aspect / dst_aspect, 1.0)
            } else {
                (1.0, dst_aspect / src_aspect)
            }
        }
    }
}

fn load_icon() -> Option<Icon> {
    log::info!("loading icon, png size = {} bytes", ICON_PNG.len());
    let img = match image::load_from_memory(ICON_PNG) {
        Ok(i) => i.to_rgba8(),
        Err(e) => {
            log::error!("icon png decode failed: {e}");
            return None;
        }
    };
    let (w, h) = img.dimensions();
    log::info!("icon decoded {}x{}, building winit icon", w, h);
    match Icon::from_rgba(img.into_raw(), w, h) {
        Ok(i) => {
            log::info!("icon ready");
            Some(i)
        }
        Err(e) => {
            log::error!("icon from_rgba failed: {e}");
            None
        }
    }
}

/// Shorten device names so they fit the dropdown without horizontal scroll.
fn short(s: &str) -> String {
    const MAX: usize = 48;
    if s.len() <= MAX {
        s.to_string()
    } else {
        let head = &s[..MAX.saturating_sub(3)];
        format!("{head}...")
    }
}

fn quad(sx: f32, sy: f32) -> [Vertex; 6] {
    [
        Vertex { pos: [-sx, -sy], uv: [0.0, 1.0] },
        Vertex { pos: [ sx, -sy], uv: [1.0, 1.0] },
        Vertex { pos: [-sx,  sy], uv: [0.0, 0.0] },
        Vertex { pos: [ sx, -sy], uv: [1.0, 1.0] },
        Vertex { pos: [ sx,  sy], uv: [1.0, 0.0] },
        Vertex { pos: [-sx,  sy], uv: [0.0, 0.0] },
    ]
}

fn tick_fps_log(last: &mut Instant, frames: &mut u64, out_fps: &mut f32) {
    let now = Instant::now();
    let elapsed = now.duration_since(*last);
    if elapsed >= Duration::from_secs(1) {
        let fps = *frames as f64 / elapsed.as_secs_f64();
        *out_fps = fps as f32;
        if elapsed >= Duration::from_secs(5) {
            log::info!("preview {:.1} fps", fps);
            *last = now;
            *frames = 0;
        }
    }
}

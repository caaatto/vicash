use crate::audio::AudioRuntime;
use crate::capture::{CaptureController, CaptureRequest};
use crate::i18n::{self, Language};
use crate::perf::PerfMetrics;
use crate::settings::PresentMode;
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
use winit::window::{Fullscreen, Icon, Window, WindowId, WindowLevel};

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
    capture: Arc<CaptureController>,
    metrics: Arc<PerfMetrics>,
) -> Result<()> {
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        shared,
        settings,
        capture_info,
        audio,
        capture,
        metrics,
        gpu: None,
        last_seq: 0,
        last_log: Instant::now(),
        frames_since_log: 0,
        preview_fps: 0.0,
        tex_size: (0, 0),
        pending_capture: None,
        applied_window: AppliedWindow::default(),
    };
    event_loop.run_app(&mut app).context("event loop exited with error")?;
    Ok(())
}

struct App {
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
    capture_info: CaptureInfo,
    audio: Option<Arc<AudioRuntime>>,
    capture: Arc<CaptureController>,
    metrics: Arc<PerfMetrics>,
    gpu: Option<Gpu>,
    last_seq: u64,
    last_log: Instant,
    frames_since_log: u64,
    preview_fps: f32,
    tex_size: (u32, u32),
    /// Resolution / fps the user is staging in the F1 panel before Apply.
    pending_capture: Option<PendingCapture>,
    /// Last window-mode settings applied; used to detect changes and call
    /// the right winit method only when something actually flipped.
    applied_window: AppliedWindow,
}

#[derive(Default, Clone, Copy, PartialEq)]
struct AppliedWindow {
    fullscreen: bool,
    borderless: bool,
    always_on_top: bool,
    hide_cursor: bool,
}

#[derive(Clone, Copy)]
struct PendingCapture {
    width: u32,
    height: u32,
    fps: u32,
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    /// Present modes the adapter actually supports; we offer only these in
    /// the UI and clamp to Fifo if a chosen mode is unavailable.
    supported_present_modes: Vec<wgpu::PresentMode>,
    sampler: wgpu::Sampler,
    /// RGB pipeline: single Rgba8UnormSrgb texture, used when the capture
    /// thread already gave us RGB pixels.
    rgb_pipeline: wgpu::RenderPipeline,
    rgb_bgl: wgpu::BindGroupLayout,
    rgb_state: Option<RgbTexState>,
    /// NV12 pipeline: two textures (Y as R8Unorm, UV as Rg8Unorm) and a
    /// fragment shader that converts BT.709 YUV to RGB. Used when the device
    /// hands us NV12 directly so we skip a CPU colour conversion.
    nv12_pipeline: wgpu::RenderPipeline,
    nv12_bgl: wgpu::BindGroupLayout,
    nv12_state: Option<Nv12TexState>,
    /// Which pipeline produced the last successful texture upload, so render
    /// knows what to draw. Reset when the frame size changes.
    active: ActivePipeline,
    vertex_buffer: wgpu::Buffer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum ActivePipeline {
    #[default]
    None,
    Rgb,
    Nv12,
}

struct RgbTexState {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

struct Nv12TexState {
    y_texture: wgpu::Texture,
    uv_texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

const RGB_SHADER: &str = r#"
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

/// NV12 fragment shader. Two textures: Y plane (R8Unorm at full resolution)
/// and UV plane (Rg8Unorm at half resolution). Converts BT.709 limited-range
/// YUV to RGB right at the fragment, then applies the sRGB EOTF so the
/// sRGB-aware surface storage matches the RGB pipeline. Without the gamma
/// step the colours come out washed because the surface re-applies a sRGB
/// curve on linear-treated gamma-encoded data.
const NV12_SHADER: &str = r#"
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

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var uv_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

fn srgb_to_linear(c: f32) -> f32 {
    let cl = clamp(c, 0.0, 1.0);
    if (cl <= 0.04045) {
        return cl / 12.92;
    }
    return pow((cl + 0.055) / 1.055, 2.4);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let y = textureSample(y_tex, samp, in.uv).r;
    let uv = textureSample(uv_tex, samp, in.uv).rg;
    // BT.709 limited range YUV -> sRGB-encoded RGB
    let yt = (y - 16.0 / 255.0) * (255.0 / 219.0);
    let ut = (uv.r - 128.0 / 255.0) * (255.0 / 224.0);
    let vt = (uv.g - 128.0 / 255.0) * (255.0 / 224.0);
    let r = yt + 1.5748 * vt;
    let g = yt - 0.1873 * ut - 0.4681 * vt;
    let b = yt + 1.8556 * ut;
    // Decode sRGB to linear; the sRGB-aware surface will re-encode on store.
    return vec4<f32>(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), 1.0);
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
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::F11),
                    repeat: false,
                    ..
                },
                ..
            } => {
                let mut s = self.settings.lock();
                s.fullscreen = !s.fullscreen;
                gpu.window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                    repeat: false,
                    ..
                },
                ..
            } => {
                // Esc leaves fullscreen so the user always has a way out.
                let mut s = self.settings.lock();
                if s.fullscreen {
                    s.fullscreen = false;
                    gpu.window.request_redraw();
                }
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
                apply_window_mode(&gpu.window, &settings, &mut self.applied_window);
                let mut pending = self.pending_capture;
                if let Err(e) = render_frame(
                    gpu,
                    &mut settings,
                    &self.capture_info,
                    self.preview_fps,
                    self.tex_size,
                    self.audio.as_ref(),
                    &self.capture,
                    &mut pending,
                    &self.metrics,
                ) {
                    log::warn!("render: {e:#}");
                }
                // Write back any settings the user changed in the panel.
                *self.settings.lock() = settings;
                self.pending_capture = pending;

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
    let supported_present_modes: Vec<wgpu::PresentMode> = caps.present_modes.iter().copied().collect();
    let present_mode = pick_present_mode(PresentMode::Mailbox, &supported_present_modes);
    log::info!(
        "wgpu adapter: {:?}, format: {:?}, present modes: {:?}, default: {:?}",
        adapter.get_info().name,
        format,
        supported_present_modes,
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

    let rgb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vcshare-rgb-shader"),
        source: wgpu::ShaderSource::Wgsl(RGB_SHADER.into()),
    });
    let nv12_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vcshare-nv12-shader"),
        source: wgpu::ShaderSource::Wgsl(NV12_SHADER.into()),
    });

    let rgb_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vcshare-rgb-bgl"),
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
    let nv12_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vcshare-nv12-bgl"),
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
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let rgb_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vcshare-rgb-pl"),
        bind_group_layouts: &[&rgb_bgl],
        push_constant_ranges: &[],
    });
    let nv12_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vcshare-nv12-pl"),
        bind_group_layouts: &[&nv12_bgl],
        push_constant_ranges: &[],
    });

    let rgb_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("vcshare-rgb-pipeline"),
        layout: Some(&rgb_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &rgb_shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &rgb_shader,
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
    install_cjk_fallback(&egui_ctx);
    let egui_state = egui_winit::State::new(
        egui_ctx.clone(),
        egui_ctx.viewport_id(),
        &window,
        Some(window.scale_factor() as f32),
        None,
        None,
    );
    let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1, false);

    let nv12_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("vcshare-nv12-pipeline"),
        layout: Some(&nv12_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &nv12_shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &nv12_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
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

    Ok(Gpu {
        window,
        surface,
        device,
        queue,
        config,
        supported_present_modes,
        sampler,
        rgb_pipeline,
        rgb_bgl,
        rgb_state: None,
        nv12_pipeline,
        nv12_bgl,
        nv12_state: None,
        active: ActivePipeline::None,
        vertex_buffer,
        egui_ctx,
        egui_state,
        egui_renderer,
    })
}

fn pick_present_mode(want: PresentMode, supported: &[wgpu::PresentMode]) -> wgpu::PresentMode {
    let candidate = match want {
        PresentMode::Immediate => wgpu::PresentMode::Immediate,
        PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
        PresentMode::Fifo => wgpu::PresentMode::Fifo,
    };
    if supported.contains(&candidate) {
        candidate
    } else {
        // Fall back through the cheapest still-supported alternative.
        for fb in [
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Fifo,
        ] {
            if supported.contains(&fb) {
                return fb;
            }
        }
        wgpu::PresentMode::Fifo
    }
}

fn upload_frame(gpu: &mut Gpu, frame: &Frame, current: &mut (u32, u32)) {
    match &frame.data {
        crate::frame::FrameData::Rgb(rgb) => {
            upload_rgb(gpu, frame.width, frame.height, rgb.as_slice(), current);
            gpu.active = ActivePipeline::Rgb;
        }
        crate::frame::FrameData::Nv12(nv12) => {
            upload_nv12(gpu, frame.width, frame.height, nv12.as_slice(), current);
            gpu.active = ActivePipeline::Nv12;
        }
    }
}

fn upload_rgb(gpu: &mut Gpu, w: u32, h: u32, rgb: &[u8], current: &mut (u32, u32)) {
    let needs_realloc = gpu.rgb_state.as_ref().map(|s| s.size != (w, h)).unwrap_or(true);
    if needs_realloc {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vcshare-rgb-tex"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vcshare-rgb-bg"),
            layout: &gpu.rgb_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
            ],
        });
        gpu.rgb_state = Some(RgbTexState { texture, bind_group, size: (w, h) });
    }
    let rgba = rgb_to_rgba(rgb);
    let state = gpu.rgb_state.as_ref().expect("rgb texture initialized above");
    gpu.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &state.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    *current = (w, h);
}

fn upload_nv12(gpu: &mut Gpu, w: u32, h: u32, nv12: &[u8], current: &mut (u32, u32)) {
    let needs_realloc = gpu.nv12_state.as_ref().map(|s| s.size != (w, h)).unwrap_or(true);
    if needs_realloc {
        let y_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vcshare-nv12-y-tex"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let uv_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vcshare-nv12-uv-tex"),
            size: wgpu::Extent3d { width: w / 2, height: h / 2, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let y_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uv_view = uv_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vcshare-nv12-bg"),
            layout: &gpu.nv12_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
            ],
        });
        gpu.nv12_state = Some(Nv12TexState { y_texture, uv_texture, bind_group, size: (w, h) });
    }
    let y_len = (w as usize) * (h as usize);
    let state = gpu.nv12_state.as_ref().expect("nv12 textures initialized above");
    // Y plane: full resolution, one byte per sample.
    gpu.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &state.y_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &nv12[..y_len],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(w),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    // UV plane: half resolution, two bytes per sample (interleaved U,V).
    gpu.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &state.uv_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &nv12[y_len..],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(w),
            rows_per_image: Some(h / 2),
        },
        wgpu::Extent3d { width: w / 2, height: h / 2, depth_or_array_layers: 1 },
    );
    *current = (w, h);
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
    capture: &Arc<CaptureController>,
    pending: &mut Option<PendingCapture>,
    metrics: &Arc<PerfMetrics>,
) -> Result<()> {
    // Apply present-mode changes from the F1 panel by reconfiguring the
    // surface only when the chosen mode actually differs.
    let desired = pick_present_mode(settings.present_mode, &gpu.supported_present_modes);
    if desired != gpu.config.present_mode {
        gpu.config.present_mode = desired;
        gpu.surface.configure(&gpu.device, &gpu.config);
    }

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
        build_ui(ctx, settings, capture_info, preview_fps, audio, capture, pending, metrics);
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
        match gpu.active {
            ActivePipeline::Rgb => {
                if let Some(state) = gpu.rgb_state.as_ref() {
                    rp.set_pipeline(&gpu.rgb_pipeline);
                    rp.set_bind_group(0, &state.bind_group, &[]);
                    rp.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                    rp.draw(0..6, 0..1);
                }
            }
            ActivePipeline::Nv12 => {
                if let Some(state) = gpu.nv12_state.as_ref() {
                    rp.set_pipeline(&gpu.nv12_pipeline);
                    rp.set_bind_group(0, &state.bind_group, &[]);
                    rp.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                    rp.draw(0..6, 0..1);
                }
            }
            ActivePipeline::None => {}
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
    capture: &Arc<CaptureController>,
    pending: &mut Option<PendingCapture>,
    metrics: &Arc<PerfMetrics>,
) {
    let t = i18n::strings(settings.language);

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
                                "{} {} fps  {}",
                                t.stats_target,
                                capture_info.fps_target,
                                capture_info.format_label
                            ),
                        );
                        ui.colored_label(
                            egui::Color32::WHITE,
                            format!("{} {:.1} fps", t.stats_preview, preview_fps),
                        );
                        ui.colored_label(
                            egui::Color32::from_rgb(200, 200, 200),
                            t.stats_hint,
                        );
                    });
            });
    }

    if settings.show_panel {
        egui::Window::new(t.window_settings)
            .default_pos(egui::pos2(20.0, 80.0))
            .resizable(false)
            .collapsible(true)
            .show(ctx, |ui| {
                egui::CollapsingHeader::new(t.section_language)
                    .default_open(true)
                    .show(ui, |ui| {
                        egui::ComboBox::from_id_salt("language")
                            .selected_text(settings.language.label_native())
                            .show_ui(ui, |ui| {
                                for lang in Language::all() {
                                    ui.selectable_value(
                                        &mut settings.language,
                                        lang,
                                        lang.label_native(),
                                    );
                                }
                            });
                    });

                egui::CollapsingHeader::new(t.section_monitor)
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.checkbox(&mut settings.fullscreen, t.fullscreen);
                        ui.checkbox(&mut settings.borderless, t.borderless);
                        ui.checkbox(&mut settings.always_on_top, t.always_on_top);
                        ui.checkbox(&mut settings.hide_cursor, t.hide_cursor);
                    });

                egui::CollapsingHeader::new(t.section_display)
                    .default_open(true)
                    .show(ui, |ui| {
                        egui::ComboBox::from_label(t.fit_mode)
                            .selected_text(match settings.fit_mode {
                                FitMode::Stretch => t.fit_stretch,
                                FitMode::Fit => t.fit_fit,
                                FitMode::Fill => t.fit_fill,
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut settings.fit_mode, FitMode::Stretch, t.fit_stretch);
                                ui.selectable_value(&mut settings.fit_mode, FitMode::Fit, t.fit_fit);
                                ui.selectable_value(&mut settings.fit_mode, FitMode::Fill, t.fit_fill);
                            });
                        egui::ComboBox::from_label(t.present_mode)
                            .selected_text(match settings.present_mode {
                                PresentMode::Immediate => t.present_immediate,
                                PresentMode::Mailbox => t.present_mailbox,
                                PresentMode::Fifo => t.present_fifo,
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut settings.present_mode,
                                    PresentMode::Immediate,
                                    t.present_immediate,
                                );
                                ui.selectable_value(
                                    &mut settings.present_mode,
                                    PresentMode::Mailbox,
                                    t.present_mailbox,
                                );
                                ui.selectable_value(
                                    &mut settings.present_mode,
                                    PresentMode::Fifo,
                                    t.present_fifo,
                                );
                            });
                        ui.checkbox(&mut settings.show_stats, t.show_stats);
                        ui.horizontal(|ui| {
                            ui.label(t.background);
                            ui.color_edit_button_rgb(&mut settings.background_color);
                        });
                    });

                egui::CollapsingHeader::new(t.section_capture)
                    .default_open(true)
                    .show(ui, |ui| {
                        capture_section(ui, &t, capture, pending);
                    });

                egui::CollapsingHeader::new(t.section_audio)
                    .default_open(true)
                    .show(ui, |ui| {
                        audio_section(ui, &t, audio);
                    });

                egui::CollapsingHeader::new(t.section_relay)
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.add(
                            egui::Slider::new(&mut settings.jpeg_quality, 1..=100)
                                .text(t.jpeg_quality),
                        );
                    });

                egui::CollapsingHeader::new(t.section_performance)
                    .default_open(false)
                    .show(ui, |ui| {
                        let sys = metrics.system();
                        ui.label(format!(
                            "{} {:>5.1} %   {} {} MB",
                            t.perf_app_cpu,
                            metrics.cpu_percent(),
                            t.perf_app_ram,
                            metrics.memory_mb()
                        ));
                        ui.label(format!(
                            "{} {:>5.1} %   {} {} / {} MB",
                            t.perf_system_cpu,
                            sys.total_cpu_percent,
                            t.perf_system_ram,
                            sys.used_memory_mb,
                            sys.total_memory_mb
                        ));
                        ui.label(format!("{} {:.1} fps", t.perf_preview, preview_fps));
                    });

                ui.separator();
                ui.colored_label(
                    egui::Color32::from_rgb(180, 180, 180),
                    t.footer_note,
                );
                if ui.button(t.close).clicked() {
                    settings.show_panel = false;
                }
            });
    }
}

fn capture_section(
    ui: &mut egui::Ui,
    t: &i18n::Strings,
    capture: &Arc<CaptureController>,
    pending: &mut Option<PendingCapture>,
) {
    let available = capture.state.available.lock().clone();
    let current = capture.state.current.lock().clone();
    if let Some(c) = &current {
        ui.label(format!(
            "{}: {}x{} @ {} fps  {:?}",
            t.capture_active,
            c.resolution().width(),
            c.resolution().height(),
            c.frame_rate(),
            c.format()
        ));
    }
    if available.is_empty() {
        return;
    }

    let mut resolutions: Vec<(u32, u32)> = available
        .iter()
        .map(|f| (f.resolution().width(), f.resolution().height()))
        .collect();
    resolutions.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    resolutions.dedup();

    let cur_res = current
        .as_ref()
        .map(|c| (c.resolution().width(), c.resolution().height()))
        .unwrap_or((0, 0));
    let cur_fps = current.as_ref().map(|c| c.frame_rate()).unwrap_or(0);

    let p = pending.get_or_insert(PendingCapture {
        width: cur_res.0,
        height: cur_res.1,
        fps: cur_fps,
    });

    let mut chosen_res = (p.width, p.height);
    ui.horizontal(|ui| {
        ui.label(t.resolution);
        egui::ComboBox::from_id_salt("capture_res")
            .selected_text(format!("{}x{}", chosen_res.0, chosen_res.1))
            .show_ui(ui, |ui| {
                for (w, h) in &resolutions {
                    let label = format!("{w}x{h}");
                    ui.selectable_value(&mut chosen_res, (*w, *h), label);
                }
            });
    });
    p.width = chosen_res.0;
    p.height = chosen_res.1;

    let mut fps_options: Vec<u32> = available
        .iter()
        .filter(|f| {
            f.resolution().width() == p.width && f.resolution().height() == p.height
        })
        .map(|f| f.frame_rate())
        .collect();
    fps_options.sort_by(|a, b| b.cmp(a));
    fps_options.dedup();
    if !fps_options.contains(&p.fps) {
        if let Some(&best) = fps_options.first() {
            p.fps = best;
        }
    }
    let mut chosen_fps = p.fps;
    ui.horizontal(|ui| {
        ui.label(t.fps);
        egui::ComboBox::from_id_salt("capture_fps")
            .selected_text(format!("{chosen_fps}"))
            .show_ui(ui, |ui| {
                for f in &fps_options {
                    ui.selectable_value(&mut chosen_fps, *f, format!("{f}"));
                }
            });
    });
    p.fps = chosen_fps;

    let dirty = (p.width, p.height) != cur_res || p.fps != cur_fps;
    let apply_label = if dirty { t.apply } else { t.applied };
    ui.add_enabled_ui(dirty, |ui| {
        if ui.button(apply_label).clicked() {
            let req = CaptureRequest {
                device_index: capture.last_device_index(),
                width: Some(p.width),
                height: Some(p.height),
                fps: Some(p.fps),
                force_mjpeg: false,
            };
            capture.restart(req);
        }
    });
}

fn audio_section(
    ui: &mut egui::Ui,
    t: &i18n::Strings,
    audio: Option<&Arc<AudioRuntime>>,
) {
    let Some(rt) = audio else {
        ui.colored_label(
            egui::Color32::from_rgb(180, 180, 180),
            t.audio_off_hint,
        );
        return;
    };
    let state = &rt.state;
    let in_name = state.input_name();
    let out_name = state.output_name();
    ui.label(format!(
        "{} Hz, {} {}, {} {} ms",
        state.sample_rate(),
        state.channels(),
        t.audio_status_channels,
        t.audio_status_buffered,
        state.buffered_ms()
    ));

    ui.horizontal(|ui| {
        ui.label(t.audio_in);
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
        ui.label(t.audio_out);
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
                .text(t.volume)
                .integer(),
        )
        .changed()
    {
        state.set_volume(volume);
    }
    let mut muted = state.is_muted();
    if ui.checkbox(&mut muted, t.muted).changed() {
        state.set_muted(muted);
    }
    let mut delay = state.delay_ms();
    if ui
        .add(
            egui::Slider::new(&mut delay, 0..=500)
                .text(t.sync_delay)
                .integer(),
        )
        .changed()
    {
        state.set_delay_ms(delay);
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

/// Try to register a CJK font with egui so Chinese / Japanese / Korean
/// glyphs render. Looks for Microsoft YaHei on Windows first, falls back to
/// other common system fonts. If none are found Chinese text shows as boxes,
/// which is recoverable by switching the language back.
fn install_cjk_fallback(ctx: &egui::Context) {
    let candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ];
    let mut font_bytes: Option<Vec<u8>> = None;
    for path in candidates {
        if let Ok(bytes) = std::fs::read(path) {
            log::info!("CJK fallback font loaded from {path}");
            font_bytes = Some(bytes);
            break;
        }
    }
    let Some(bytes) = font_bytes else {
        log::warn!("no CJK system font found; Chinese text will render as boxes");
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "cjk-fallback".into(),
        egui::FontData::from_owned(bytes).into(),
    );
    if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        family.push("cjk-fallback".into());
    }
    if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
        family.push("cjk-fallback".into());
    }
    ctx.set_fonts(fonts);
}

fn apply_window_mode(window: &Window, settings: &Settings, applied: &mut AppliedWindow) {
    let want = AppliedWindow {
        fullscreen: settings.fullscreen,
        borderless: settings.borderless,
        always_on_top: settings.always_on_top,
        hide_cursor: settings.hide_cursor && settings.fullscreen,
    };
    if want.fullscreen != applied.fullscreen {
        window.set_fullscreen(if want.fullscreen {
            Some(Fullscreen::Borderless(None))
        } else {
            None
        });
    }
    if want.borderless != applied.borderless {
        // Decorations only matter in windowed mode; in fullscreen we suppress
        // them anyway, but tracking the user's preference keeps the toggle
        // honest when fullscreen is turned off.
        window.set_decorations(!want.borderless);
    }
    if want.always_on_top != applied.always_on_top {
        window.set_window_level(if want.always_on_top {
            WindowLevel::AlwaysOnTop
        } else {
            WindowLevel::Normal
        });
    }
    if want.hide_cursor != applied.hide_cursor {
        window.set_cursor_visible(!want.hide_cursor);
    }
    *applied = want;
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

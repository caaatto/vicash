// Draws the vicash app icon at build time, packages it into a multi-resolution
// .ico and embeds it via windres on Windows. Design: very dark navy square
// with rounded corners, two overlapping rounded monitor outlines stroked in a
// cyan-to-blue linear gradient, and the word "vicash" inside the front
// monitor.

use std::env;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use ab_glyph::{Font, FontVec, PxScale, ScaleFont, point};
use image::imageops::FilterType;
use image::{ImageBuffer, Rgba, RgbaImage};
use tiny_skia::{
    Color, FillRule, GradientStop, LinearGradient, Paint, PathBuilder, Pixmap, Point, Rect,
    Shader, SpreadMode, Stroke, Transform,
};

const BASE: u32 = 256;
const SIZES: &[u32] = &[16, 32, 48, 64, 128, 256];

const FONT_BYTES: &[u8] = include_bytes!("assets/JetBrainsMono-Regular.ttf");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/JetBrainsMono-Regular.ttf");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let png_256 = out_dir.join("icon-256.png");
    let ico_path = out_dir.join("icon.ico");

    let base_pixmap = draw_icon();
    base_pixmap
        .save_png(&png_256)
        .expect("write icon-256.png");

    let base_rgba = pixmap_to_rgba_image(&base_pixmap);
    write_ico(&base_rgba, &ico_path);

    #[cfg(target_os = "windows")]
    embed_windows_icon(&ico_path);
}

fn draw_icon() -> Pixmap {
    let mut pixmap = Pixmap::new(BASE, BASE).expect("256x256 pixmap");

    paint_background(&mut pixmap);

    // Two overlapping rounded monitors, stroked only. The back monitor sits
    // up-right of the front and is partly hidden by it. Both use the same
    // cyan-to-blue gradient so they read as one set.
    let back = rounded_rect(112.0, 64.0, 116.0, 78.0, 14.0);
    let front = rounded_rect(28.0, 102.0, 168.0, 110.0, 16.0);

    stroke_gradient(&mut pixmap, &back, 8.0, 112.0, 64.0, 228.0, 142.0);
    // Mask away the part of the back monitor that overlaps the front, so the
    // front truly looks in front. Done by re-drawing the front interior (the
    // background colour) over the back's stroke where they intersect, then
    // stroking the front on top.
    knockout_front_area(&mut pixmap, &front);
    stroke_gradient(&mut pixmap, &front, 9.0, 28.0, 102.0, 196.0, 212.0);

    draw_label(&mut pixmap, "vicash", 28.0, 102.0, 168.0, 110.0);

    pixmap
}

fn paint_background(pixmap: &mut Pixmap) {
    // Whole-icon rounded square base.
    let bg = rounded_rect(0.0, 0.0, BASE as f32, BASE as f32, 40.0);
    let mut bg_paint = Paint::default();
    bg_paint.set_color(Color::from_rgba8(0x0a, 0x0e, 0x18, 0xff));
    bg_paint.anti_alias = true;
    pixmap.fill_path(&bg, &bg_paint, FillRule::Winding, Transform::identity(), None);

    // Subtle diagonal sheen.
    if let Some(shader) = LinearGradient::new(
        Point::from_xy(0.0, 0.0),
        Point::from_xy(BASE as f32, BASE as f32),
        vec![
            GradientStop::new(0.0, Color::from_rgba8(0x12, 0x18, 0x26, 0xff)),
            GradientStop::new(1.0, Color::from_rgba8(0x05, 0x07, 0x0e, 0xff)),
        ],
        SpreadMode::Pad,
        Transform::identity(),
    ) {
        let mut p = Paint::default();
        p.shader = shader;
        p.anti_alias = true;
        pixmap.fill_path(&bg, &p, FillRule::Winding, Transform::identity(), None);
    }
}

fn knockout_front_area(pixmap: &mut Pixmap, front: &tiny_skia::Path) {
    // Fill the front monitor with the background colour so the back monitor's
    // stroke that goes behind it is hidden.
    let mut p = Paint::default();
    p.set_color(Color::from_rgba8(0x0a, 0x0e, 0x18, 0xff));
    p.anti_alias = true;
    pixmap.fill_path(front, &p, FillRule::Winding, Transform::identity(), None);
    // Re-apply the diagonal sheen so the inside of the front monitor matches
    // the rest of the background.
    if let Some(shader) = LinearGradient::new(
        Point::from_xy(0.0, 0.0),
        Point::from_xy(BASE as f32, BASE as f32),
        vec![
            GradientStop::new(0.0, Color::from_rgba8(0x12, 0x18, 0x26, 0xff)),
            GradientStop::new(1.0, Color::from_rgba8(0x05, 0x07, 0x0e, 0xff)),
        ],
        SpreadMode::Pad,
        Transform::identity(),
    ) {
        let mut p = Paint::default();
        p.shader = shader;
        p.anti_alias = true;
        pixmap.fill_path(front, &p, FillRule::Winding, Transform::identity(), None);
    }
}

fn stroke_gradient(
    pixmap: &mut Pixmap,
    path: &tiny_skia::Path,
    width: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) {
    let shader = LinearGradient::new(
        Point::from_xy(x0, y0),
        Point::from_xy(x1, y1),
        vec![
            // cyan
            GradientStop::new(0.0, Color::from_rgba8(0x4d, 0xe6, 0xff, 0xff)),
            // electric blue
            GradientStop::new(1.0, Color::from_rgba8(0x29, 0x79, 0xff, 0xff)),
        ],
        SpreadMode::Pad,
        Transform::identity(),
    )
    .unwrap_or_else(|| Shader::SolidColor(Color::from_rgba8(0x4d, 0xe6, 0xff, 0xff)));

    let mut paint = Paint::default();
    paint.shader = shader;
    paint.anti_alias = true;
    let mut stroke = Stroke::default();
    stroke.width = width;
    stroke.line_cap = tiny_skia::LineCap::Round;
    stroke.line_join = tiny_skia::LineJoin::Round;
    pixmap.stroke_path(path, &paint, &stroke, Transform::identity(), None);
}

fn draw_label(pixmap: &mut Pixmap, text: &str, mx: f32, my: f32, mw: f32, mh: f32) {
    let font = FontVec::try_from_vec(FONT_BYTES.to_vec()).expect("font parse");
    // Size the text to fit the monitor with some padding.
    let target_height = mh * 0.34;
    let scale = PxScale::from(target_height);
    let scaled = font.as_scaled(scale);

    let glyphs: Vec<_> = text
        .chars()
        .map(|c| (c, font.glyph_id(c)))
        .collect();

    let total_width: f32 = glyphs.iter().map(|(_, g)| scaled.h_advance(*g)).sum();
    let ascent = scaled.ascent();
    let descent = scaled.descent();
    let line_h = ascent - descent;

    let start_x = mx + (mw - total_width) / 2.0;
    let baseline_y = my + (mh + line_h) / 2.0 - descent;

    let w = pixmap.width() as i32;
    let h = pixmap.height() as i32;
    let data = pixmap.data_mut();

    let mut cursor_x = start_x;
    for (_, gid) in glyphs {
        let glyph = gid.with_scale_and_position(scale, point(cursor_x, baseline_y));
        if let Some(outlined) = font.outline_glyph(glyph) {
            let bounds = outlined.px_bounds();
            outlined.draw(|gx, gy, coverage| {
                let px = bounds.min.x as i32 + gx as i32;
                let py = bounds.min.y as i32 + gy as i32;
                if px < 0 || py < 0 || px >= w || py >= h {
                    return;
                }
                let idx = ((py * w + px) * 4) as usize;
                blend_cyan(&mut data[idx..idx + 4], coverage);
            });
        }
        cursor_x += scaled.h_advance(gid);
    }
}

/// Source-over alpha blend a cyan text pixel into the tiny-skia pixmap.
/// tiny-skia stores premultiplied RGBA (not BGRA) byte order in Pixmap::data().
/// We blend src over dst with `coverage` as the source alpha.
fn blend_cyan(dst: &mut [u8], coverage: f32) {
    let a = coverage.clamp(0.0, 1.0);
    // Bright cyan target colour.
    let src_r = 0x6c;
    let src_g = 0xe6;
    let src_b = 0xff;
    let inv = 1.0 - a;
    // Premultiplied: each channel times alpha. Existing dst channels are
    // already premultiplied, so straight Porter-Duff over works:
    //   out = src * a + dst * (1 - a)
    dst[0] = (src_r as f32 * a + dst[0] as f32 * inv) as u8;
    dst[1] = (src_g as f32 * a + dst[1] as f32 * inv) as u8;
    dst[2] = (src_b as f32 * a + dst[2] as f32 * inv) as u8;
    dst[3] = (255.0 * a + dst[3] as f32 * inv) as u8;
}

fn rounded_rect(x: f32, y: f32, w: f32, h: f32, r: f32) -> tiny_skia::Path {
    let r = r.min(w / 2.0).min(h / 2.0);
    let mut b = PathBuilder::new();
    b.move_to(x + r, y);
    b.line_to(x + w - r, y);
    b.quad_to(x + w, y, x + w, y + r);
    b.line_to(x + w, y + h - r);
    b.quad_to(x + w, y + h, x + w - r, y + h);
    b.line_to(x + r, y + h);
    b.quad_to(x, y + h, x, y + h - r);
    b.line_to(x, y + r);
    b.quad_to(x, y, x + r, y);
    b.close();
    b.finish().unwrap()
}

fn pixmap_to_rgba_image(pixmap: &Pixmap) -> RgbaImage {
    let w = pixmap.width();
    let h = pixmap.height();
    let mut buf = pixmap.data().to_vec();
    for px in buf.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    ImageBuffer::<Rgba<u8>, _>::from_raw(w, h, buf).expect("rgba buffer matches dims")
}

fn write_ico(base: &RgbaImage, path: &Path) {
    let mut icon_dir = ico::IconDir::new(ico::ResourceType::Icon);
    for &sz in SIZES {
        let resized = image::imageops::resize(base, sz, sz, FilterType::Lanczos3);
        let mut png_buf = Vec::new();
        {
            let mut cursor = Cursor::new(&mut png_buf);
            image::DynamicImage::ImageRgba8(resized)
                .write_to(&mut cursor, image::ImageFormat::Png)
                .expect("encode png");
        }
        let entry = ico::IconImage::read_png(&png_buf[..]).expect("decode png for ico");
        let dir_entry = ico::IconDirEntry::encode(&entry).expect("ico encode");
        icon_dir.add_entry(dir_entry);
    }
    let mut file = fs::File::create(path).expect("create icon.ico");
    icon_dir.write(&mut file).expect("write icon.ico");
}

#[cfg(target_os = "windows")]
fn embed_windows_icon(ico_path: &Path) {
    let safe_dir = PathBuf::from(r"C:\Local\tools\vcshare-build");
    let _ = fs::create_dir_all(&safe_dir);
    let safe_ico = safe_dir.join("icon.ico");
    if let Err(e) = fs::copy(ico_path, &safe_ico) {
        println!("cargo:warning=could not copy icon to safe path: {e}");
        return;
    }
    let rc_path = safe_dir.join("resource.rc");
    let res_path = safe_dir.join("vicash-icon.res");
    let rc = format!(
        "#pragma code_page(65001)\n1 ICON \"{}\"\n",
        safe_ico.display().to_string().replace('\\', "\\\\")
    );
    if let Err(e) = fs::write(&rc_path, rc) {
        println!("cargo:warning=could not write resource.rc: {e}");
        return;
    }
    let windres = r"C:\Local\tools\mingw64\bin\windres.exe";
    let status = std::process::Command::new(windres)
        .arg("--input-format=rc")
        .arg("--output-format=coff")
        .arg(&rc_path)
        .arg(&res_path)
        .status();
    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-arg-bins={}", res_path.display());
        }
        Ok(s) => println!("cargo:warning=windres exited with {s}"),
        Err(e) => println!("cargo:warning=could not invoke windres: {e}"),
    }
}

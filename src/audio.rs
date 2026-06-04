use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use parking_lot::Mutex;
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

/// Audio passthrough runtime. Holds the input and output streams alive and
/// can swap them when the user picks a different device.
pub struct AudioRuntime {
    streams: Mutex<Streams>,
    pub state: Arc<AudioState>,
}

// Fields are held only for their drop side effect (closing the streams) and
// to keep the ring buffer alive for the callbacks. Never accessed directly.
#[allow(dead_code)]
struct Streams {
    in_stream: Stream,
    out_stream: Stream,
    prod: Arc<Mutex<HeapProd<f32>>>,
    cons: Arc<Mutex<HeapCons<f32>>>,
}

pub struct AudioState {
    pub input_name: Mutex<String>,
    pub output_name: Mutex<String>,
    pub sample_rate: AtomicU32,
    /// Channels of the input stream.
    pub channels: AtomicU32,
    /// Channels the output device runs at.
    pub output_channels: AtomicU32,
    pub target_delay_ms: AtomicU32,
    pub buffered_samples: AtomicUsize,
    pub volume_percent: AtomicU32,
    pub muted: AtomicBool,
    /// MS2109/MS2130 quirk: the card reports a 2-channel stereo input but
    /// only fills the left channel. When this is true the output stage
    /// averages all input channels into one signal then broadcasts that to
    /// every output channel, so the audio shows up on both speakers.
    pub mix_to_mono: AtomicBool,
    /// Optional second sink that the input callback also writes to. Used by
    /// the fMP4 relay to consume raw f32 samples in parallel with the local
    /// speaker output, without rebuilding the audio stream. The relay
    /// install/remove this producer when its lifecycle starts/stops.
    pub relay_sink: Mutex<Option<HeapProd<f32>>>,
    /// Fast-path gate read by the input callback. When false the callback
    /// must NOT touch `relay_sink` at all - locking even an uncontended
    /// Mutex inside a real-time audio callback can push WASAPI past its
    /// frame deadline and force the local speaker output to drift.
    pub relay_active: AtomicBool,
}

pub fn list_input_devices() -> Vec<String> {
    cpal::default_host()
        .input_devices()
        .map(|iter| iter.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default()
}

pub fn list_output_devices() -> Vec<String> {
    cpal::default_host()
        .output_devices()
        .map(|iter| iter.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default()
}

/// Start passthrough. `input_hint` is a substring matched against input
/// device names (case insensitive). Pass None for system default input.
pub fn start(input_hint: Option<&str>, delay_ms: u32) -> Result<AudioRuntime> {
    let host = cpal::default_host();
    let input = pick_input(&host, input_hint)?;
    let output = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no default audio output device"))?;

    let state = Arc::new(AudioState {
        input_name: Mutex::new(String::new()),
        output_name: Mutex::new(String::new()),
        sample_rate: AtomicU32::new(0),
        channels: AtomicU32::new(0),
        output_channels: AtomicU32::new(0),
        target_delay_ms: AtomicU32::new(delay_ms),
        buffered_samples: AtomicUsize::new(0),
        volume_percent: AtomicU32::new(100),
        muted: AtomicBool::new(false),
        mix_to_mono: AtomicBool::new(false),
        relay_sink: Mutex::new(None),
        relay_active: AtomicBool::new(false),
    });

    let streams = build_streams(&input, &output, &state)?;
    Ok(AudioRuntime {
        streams: Mutex::new(streams),
        state,
    })
}

impl AudioRuntime {
    /// Swap to a different input device by name. Pre/post volume, mute and
    /// delay are preserved.
    pub fn set_input(&self, name: &str) -> Result<()> {
        let host = cpal::default_host();
        let input = find_device(&host.input_devices()?.collect::<Vec<_>>(), name)
            .ok_or_else(|| anyhow!("input device '{name}' not found"))?;
        let output = find_device(&host.output_devices()?.collect::<Vec<_>>(), &self.state.output_name())
            .or_else(|| host.default_output_device())
            .ok_or_else(|| anyhow!("no output device available"))?;
        let mut guard = self.streams.lock();
        *guard = build_streams(&input, &output, &self.state)?;
        Ok(())
    }

    /// Swap to a different output device by name.
    pub fn set_output(&self, name: &str) -> Result<()> {
        let host = cpal::default_host();
        let output = find_device(&host.output_devices()?.collect::<Vec<_>>(), name)
            .ok_or_else(|| anyhow!("output device '{name}' not found"))?;
        let input = find_device(&host.input_devices()?.collect::<Vec<_>>(), &self.state.input_name())
            .ok_or_else(|| anyhow!("input device gone"))?;
        let mut guard = self.streams.lock();
        *guard = build_streams(&input, &output, &self.state)?;
        Ok(())
    }
}

impl AudioState {
    pub fn input_name(&self) -> String {
        self.input_name.lock().clone()
    }

    pub fn output_name(&self) -> String {
        self.output_name.lock().clone()
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate.load(Ordering::Relaxed)
    }

    pub fn channels(&self) -> u16 {
        self.channels.load(Ordering::Relaxed) as u16
    }

    pub fn buffered_ms(&self) -> u32 {
        let sr = self.sample_rate();
        let ch = self.channels();
        if sr == 0 || ch == 0 {
            return 0;
        }
        let samples = self.buffered_samples.load(Ordering::Relaxed);
        ((samples * 1000) / (sr as usize * ch as usize)) as u32
    }

    pub fn delay_ms(&self) -> u32 {
        self.target_delay_ms.load(Ordering::Relaxed)
    }

    pub fn set_delay_ms(&self, ms: u32) {
        self.target_delay_ms.store(ms, Ordering::Relaxed);
    }

    pub fn volume(&self) -> u32 {
        self.volume_percent.load(Ordering::Relaxed)
    }

    pub fn set_volume(&self, percent: u32) {
        self.volume_percent.store(percent.min(200), Ordering::Relaxed);
    }

    pub fn is_muted(&self) -> bool {
        self.muted.load(Ordering::Relaxed)
    }

    pub fn set_muted(&self, m: bool) {
        self.muted.store(m, Ordering::Relaxed);
    }

    pub fn is_mix_to_mono(&self) -> bool {
        self.mix_to_mono.load(Ordering::Relaxed)
    }

    pub fn set_mix_to_mono(&self, m: bool) {
        self.mix_to_mono.store(m, Ordering::Relaxed);
    }

    /// Attach a relay-side producer that the input callback will also push
    /// each captured sample into. Returns a fresh consumer the caller can
    /// drain on its own thread. The producer is dropped automatically when
    /// the caller releases its consumer (consumer drop disconnects).
    pub fn install_relay_sink(&self, capacity: usize) -> HeapCons<f32> {
        let rb = HeapRb::<f32>::new(capacity);
        let (prod, cons) = rb.split();
        *self.relay_sink.lock() = Some(prod);
        // Order matters: install the producer first, THEN flip the gate
        // so the callback sees a consistent state (gate true => sink
        // already populated).
        self.relay_active.store(true, Ordering::Release);
        cons
    }

    /// Detach the relay sink so the input callback stops fanning out. Any
    /// samples already buffered in the consumer half stay there until that
    /// consumer is itself dropped.
    pub fn remove_relay_sink(&self) {
        // Flip the gate off first so subsequent callbacks skip the path,
        // THEN tear the producer down. This guarantees no callback is
        // mid-push when the producer disappears.
        self.relay_active.store(false, Ordering::Release);
        *self.relay_sink.lock() = None;
    }

    fn delay_samples(&self) -> usize {
        let sr = self.sample_rate();
        let ch = self.channels();
        if sr == 0 || ch == 0 {
            return 0;
        }
        (sr as usize) * (ch as usize) * (self.delay_ms() as usize) / 1000
    }
}

fn find_device(devices: &[cpal::Device], name: &str) -> Option<cpal::Device> {
    devices.iter().find(|d| d.name().ok().as_deref() == Some(name)).cloned()
}

fn pick_input(host: &cpal::Host, hint: Option<&str>) -> Result<cpal::Device> {
    if let Some(needle) = hint {
        let needle = needle.to_lowercase();
        for d in host
            .input_devices()
            .context("failed to list audio inputs")?
        {
            if let Ok(name) = d.name() {
                if name.to_lowercase().contains(&needle) {
                    return Ok(d);
                }
            }
        }
        log::warn!("no audio input matched '{needle}', falling back to default");
    }
    host.default_input_device()
        .ok_or_else(|| anyhow!("no default audio input device"))
}

fn build_streams(
    input: &cpal::Device,
    output: &cpal::Device,
    state: &Arc<AudioState>,
) -> Result<Streams> {
    let in_cfg = input
        .default_input_config()
        .context("failed to query default input config")?;
    let out_cfg = output
        .default_output_config()
        .context("failed to query default output config")?;

    let sample_rate = in_cfg.sample_rate().0;
    let channels = in_cfg.channels();
    let fmt = in_cfg.sample_format();
    let output_channels = out_cfg.channels();
    let output_rate = out_cfg.sample_rate().0;

    if output_rate != sample_rate {
        log::warn!(
            "audio: input is {}Hz but output is {}Hz; no resampling. Sync may drift.",
            sample_rate, output_rate
        );
    }

    *state.input_name.lock() = input.name().unwrap_or_else(|_| "<unknown>".into());
    *state.output_name.lock() = output.name().unwrap_or_else(|_| "<unknown>".into());
    state.sample_rate.store(sample_rate, Ordering::Relaxed);
    state.channels.store(channels as u32, Ordering::Relaxed);
    state.output_channels.store(output_channels as u32, Ordering::Relaxed);

    log::info!(
        "audio: {} ({}Hz, {} ch, {:?}) -> {} ({}Hz, {} ch, {:?})",
        state.input_name(), sample_rate, channels, fmt,
        state.output_name(), output_rate, output_channels, out_cfg.sample_format()
    );

    let delay_samples = state.delay_samples();
    let capacity = (sample_rate as usize) * (channels as usize) * 2;
    let rb = HeapRb::<f32>::new(capacity.max(delay_samples * 2 + 4096));
    let (prod, cons) = rb.split();
    let prod = Arc::new(Mutex::new(prod));
    let cons = Arc::new(Mutex::new(cons));

    let in_cfg_real: StreamConfig = in_cfg.config();
    let out_cfg_real: StreamConfig = StreamConfig {
        channels: output_channels,
        sample_rate: cpal::SampleRate(output_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let in_stream = build_input(input, &in_cfg_real, fmt, prod.clone(), state.clone())?;
    let out_stream = build_output(output, &out_cfg_real, cons.clone(), state.clone())?;

    in_stream.play().context("failed to start input stream")?;
    out_stream.play().context("failed to start output stream")?;

    Ok(Streams { in_stream, out_stream, prod, cons })
}

fn build_input(
    device: &cpal::Device,
    config: &StreamConfig,
    fmt: SampleFormat,
    prod: Arc<Mutex<HeapProd<f32>>>,
    state: Arc<AudioState>,
) -> Result<Stream> {
    let err_fn = |e| log::warn!("audio input error: {e}");
    let prod_for_cb = prod.clone();
    let state_for_cb = state.clone();
    let push = move |samples: &[f32]| {
        // Push to the local-speaker ringbuf; release the lock as soon as
        // we are done so we do NOT hold it while taking the relay lock.
        // Holding two locks across the second push tripled the worst-case
        // callback time and made WASAPI drift.
        {
            let mut p = prod_for_cb.lock();
            for s in samples {
                let _ = p.try_push(*s);
            }
            state_for_cb.buffered_samples.store(p.occupied_len(), Ordering::Relaxed);
        }
        // Fast-path gate: the lock and per-sample push for the relay
        // sink only run when the fMP4 relay is actually active. When the
        // toggle is off, this is one atomic load + a branch and the
        // input callback finishes within its real-time deadline.
        if state_for_cb.relay_active.load(Ordering::Relaxed) {
            let mut relay = state_for_cb.relay_sink.lock();
            if let Some(rp) = relay.as_mut() {
                for s in samples {
                    let _ = rp.try_push(*s);
                }
            }
        }
    };

    let stream = match fmt {
        SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| push(data),
            err_fn,
            None,
        ),
        SampleFormat::I16 => {
            let push = std::sync::Mutex::new(Box::new(push) as Box<dyn FnMut(&[f32]) + Send>);
            let mut tmp = Vec::<f32>::new();
            device.build_input_stream(
                config,
                move |data: &[i16], _| {
                    tmp.clear();
                    tmp.extend(data.iter().map(|s| *s as f32 / i16::MAX as f32));
                    push.lock().unwrap()(&tmp);
                },
                err_fn,
                None,
            )
        }
        SampleFormat::U16 => {
            let push = std::sync::Mutex::new(Box::new(push) as Box<dyn FnMut(&[f32]) + Send>);
            let mut tmp = Vec::<f32>::new();
            device.build_input_stream(
                config,
                move |data: &[u16], _| {
                    tmp.clear();
                    tmp.extend(data.iter().map(|s| (*s as f32 - 32768.0) / 32768.0));
                    push.lock().unwrap()(&tmp);
                },
                err_fn,
                None,
            )
        }
        other => bail!("unsupported audio input sample format: {other:?}"),
    }
    .context("failed to build audio input stream")?;
    Ok(stream)
}

fn build_output(
    device: &cpal::Device,
    config: &StreamConfig,
    cons: Arc<Mutex<HeapCons<f32>>>,
    state: Arc<AudioState>,
) -> Result<Stream> {
    let err_fn = |e| log::warn!("audio output error: {e}");
    let in_channels = state.channels() as usize;
    let out_channels = state.output_channels.load(Ordering::Relaxed) as usize;
    let stream = device
        .build_output_stream(
            config,
            move |out: &mut [f32], _| {
                let target = state.delay_samples();
                let mut c = cons.lock();
                let buffered = c.occupied_len();
                let frames_out = out.len() / out_channels.max(1);
                let samples_needed = frames_out * in_channels;
                let drainable = buffered.saturating_sub(target);

                let overshoot = drainable.saturating_sub(samples_needed + target / 2);
                if overshoot > 0 {
                    let drop_now = if in_channels > 0 { overshoot - overshoot % in_channels } else { 0 };
                    let mut dropped = 0;
                    while dropped < drop_now && c.try_pop().is_some() {
                        dropped += 1;
                    }
                }

                let mut frame_in: [f32; 16] = [0.0; 16];
                let in_ch_clamped = in_channels.min(frame_in.len());

                for f in 0..frames_out {
                    let recheck = c.occupied_len();
                    let can_pop = recheck >= in_channels
                        && recheck > target.saturating_sub(in_channels);
                    if can_pop {
                        for slot in &mut frame_in[..in_ch_clamped] {
                            *slot = c.try_pop().unwrap_or(0.0);
                        }
                    } else {
                        for slot in &mut frame_in[..in_ch_clamped] {
                            *slot = 0.0;
                        }
                    }
                    let out_frame = &mut out[f * out_channels..(f + 1) * out_channels];
                    if state.mix_to_mono.load(Ordering::Relaxed) {
                        map_frame_mono(&frame_in[..in_ch_clamped], out_frame);
                    } else {
                        map_frame(&frame_in[..in_ch_clamped], out_frame);
                    }
                }

                let gain = if state.muted.load(Ordering::Relaxed) {
                    0.0
                } else {
                    state.volume_percent.load(Ordering::Relaxed) as f32 / 100.0
                };
                if gain != 1.0 {
                    for slot in out.iter_mut() {
                        *slot *= gain;
                    }
                }

                state.buffered_samples.store(c.occupied_len(), Ordering::Relaxed);
            },
            err_fn,
            None,
        )
        .context("failed to build audio output stream")?;
    Ok(stream)
}

/// Average all input channels into a single sample and broadcast it to
/// every output channel. Use this for the MS2109/MS2130 "stereo input
/// but only the left channel carries audio" case.
fn map_frame_mono(in_frame: &[f32], out_frame: &mut [f32]) {
    if in_frame.is_empty() {
        for slot in out_frame.iter_mut() {
            *slot = 0.0;
        }
        return;
    }
    let avg: f32 = in_frame.iter().sum::<f32>() / in_frame.len() as f32;
    for slot in out_frame.iter_mut() {
        *slot = avg;
    }
}

fn map_frame(in_frame: &[f32], out_frame: &mut [f32]) {
    let inc = in_frame.len();
    let outc = out_frame.len();
    for slot in out_frame.iter_mut() {
        *slot = 0.0;
    }
    match inc {
        0 => {}
        1 => {
            let s = in_frame[0];
            if outc >= 1 {
                out_frame[0] = s;
            }
            if outc >= 2 {
                out_frame[1] = s;
            }
        }
        _ => {
            let n = inc.min(outc);
            out_frame[..n].copy_from_slice(&in_frame[..n]);
        }
    }
}

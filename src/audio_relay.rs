//! Uncompressed PCM audio for the LAN relay. No codec, no ffmpeg: 48 kHz
//! stereo s16 is ~1.5 Mbit/s, which is noise next to the MJPEG stream, so
//! there is nothing to gain from encoding and a lot of latency to lose.
//!
//! One pump thread drains the cpal input callback through the relay sink,
//! converts to interleaved s16le in ~10 ms chunks and publishes them into a
//! short ring. Every HTTP client (`/audio.wav`, the `/live` page, the fMP4
//! relay's ffmpeg feed) subscribes and reads chunks from that ring on its
//! own pace. The sink is only installed while at least one subscriber
//! exists, so the real-time callback pays nothing when nobody listens.

use parking_lot::{Condvar, Mutex};
use ringbuf::HeapCons;
use ringbuf::traits::Consumer;
use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::audio::AudioState;

/// Chunk length published to subscribers. Small enough to keep the end to
/// end delay low, large enough that per-chunk overhead stays negligible.
const CHUNK_MS: usize = 10;
/// How many chunks the ring keeps (~0.5 s). A subscriber that falls further
/// behind than this is snapped forward to the live edge.
const RING_CHUNKS: usize = 48;
/// A subscriber that lags more than this many chunks behind the newest one
/// skips ahead instead of replaying stale audio. Keeps latency bounded when
/// a client stalls briefly.
const MAX_LAG_CHUNKS: u64 = 12;

/// Beyond this the smoothed audio clock is snapped to the measured one
/// instead of slewed (device restart, long stall).
const PTS_RESYNC_US: f64 = 50_000.0;

/// One slice of interleaved s16le PCM plus the format it was captured in.
pub struct AudioChunk {
    pub sample_rate: u32,
    pub channels: u16,
    /// Presentation time of the first sample on the relay clock (see
    /// `pts_us`), already shifted by the user's audio sync delay so it lines
    /// up with the video frame timestamps.
    pub pts_us: u64,
    pub pcm: Vec<u8>,
}

/// Shared relay clock in microseconds. Video frames are stamped with their
/// capture instant and audio chunks with the arrival of their first sample,
/// both on this clock, so a client can line the two up without knowing
/// anything about the host's wall clock.
pub fn pts_us(t: Instant) -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = *EPOCH.get_or_init(Instant::now);
    t.saturating_duration_since(epoch).as_micros() as u64
}

struct Ring {
    chunks: VecDeque<(u64, Arc<AudioChunk>)>,
    next_seq: u64,
}

pub struct AudioRelay {
    /// The live passthrough state, owned by `AudioControl`. Watched rather
    /// than captured once so the relay follows "enable audio" in the F1
    /// panel and survives audio being restarted.
    source: Arc<Mutex<Option<Arc<AudioState>>>>,
    ring: Mutex<Ring>,
    cv: Condvar,
    subscribers: AtomicUsize,
    shutdown: Arc<AtomicBool>,
}

impl AudioRelay {
    pub fn spawn(
        source: Arc<Mutex<Option<Arc<AudioState>>>>,
        shutdown: Arc<AtomicBool>,
    ) -> std::io::Result<Arc<Self>> {
        let relay = Arc::new(Self {
            source,
            ring: Mutex::new(Ring { chunks: VecDeque::with_capacity(RING_CHUNKS), next_seq: 1 }),
            cv: Condvar::new(),
            subscribers: AtomicUsize::new(0),
            shutdown,
        });
        let pump = relay.clone();
        std::thread::Builder::new()
            .name("relay-audio".into())
            .spawn(move || pump.pump_loop())?;
        Ok(relay)
    }

    /// True when the passthrough is running, i.e. there is something to send.
    pub fn is_available(&self) -> bool {
        self.source.lock().is_some()
    }

    pub fn subscribers(&self) -> usize {
        self.subscribers.load(Ordering::Relaxed)
    }

    /// Start listening at the live edge. Dropping the subscription releases
    /// the sink again once the last listener is gone.
    pub fn subscribe(self: &Arc<Self>) -> AudioSubscription {
        self.subscribers.fetch_add(1, Ordering::Relaxed);
        let next = self.ring.lock().next_seq;
        AudioSubscription { relay: self.clone(), next }
    }

    fn publish(&self, chunk: AudioChunk) {
        let mut ring = self.ring.lock();
        let seq = ring.next_seq;
        ring.next_seq += 1;
        if ring.chunks.len() == RING_CHUNKS {
            ring.chunks.pop_front();
        }
        ring.chunks.push_back((seq, Arc::new(chunk)));
        drop(ring);
        self.cv.notify_all();
    }

    fn pump_loop(self: Arc<Self>) {
        let mut attached: Option<(Arc<AudioState>, HeapCons<f32>)> = None;
        let mut pending: Vec<f32> = Vec::with_capacity(4096);
        // Smoothed timestamp of the next chunk's first sample. Measuring
        // each chunk on its own would carry the 2 ms polling jitter of this
        // loop into the clients' sync; instead the clock advances by the
        // exact sample count and is only slewed towards the measurement.
        let mut next_pts: Option<f64> = None;
        while !self.shutdown.load(Ordering::Relaxed) {
            let current = if self.subscribers() > 0 { self.source.lock().clone() } else { None };

            // (Re)attach when audio appeared, got restarted, or went away.
            let same = match (&attached, &current) {
                (Some((a, _)), Some(c)) => Arc::ptr_eq(a, c),
                (None, None) => true,
                _ => false,
            };
            if !same {
                if let Some((old, _)) = attached.take() {
                    old.remove_relay_sink();
                    log::info!("relay audio: sink detached");
                }
                if let Some(state) = current {
                    let cap = (state.sample_rate().max(8000) as usize)
                        * (state.channels().max(1) as usize);
                    let cons = state.install_relay_sink(cap);
                    log::info!(
                        "relay audio: sink attached ({} Hz, {} ch)",
                        state.sample_rate(),
                        state.channels()
                    );
                    attached = Some((state, cons));
                }
                pending.clear();
                next_pts = None;
            }

            let Some((state, cons)) = attached.as_mut() else {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            };

            let rate = state.sample_rate().max(1);
            let channels = state.channels().max(1);
            let chunk_samples = rate as usize * channels as usize * CHUNK_MS / 1000;
            while let Some(s) = cons.try_pop() {
                pending.push(s);
            }
            if pending.len() < chunk_samples {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            // The oldest pending sample arrived roughly `pending` frames ago.
            let pending_us = (pending.len() / channels as usize) as f64 * 1e6 / rate as f64;
            let measured = pts_us(Instant::now()) as f64 - pending_us;
            let pts = match next_pts {
                Some(p) if (measured - p).abs() < PTS_RESYNC_US => p + (measured - p) * 0.02,
                _ => measured,
            };
            // Ship everything that is whole frames; keep the remainder so
            // channel alignment never breaks across chunks.
            let usable = pending.len() - pending.len() % channels as usize;
            let mono = state.is_mix_to_mono() && channels > 1;
            let mut pcm = Vec::with_capacity(usable * 2);
            for frame in pending[..usable].chunks_exact(channels as usize) {
                if mono {
                    let avg = frame.iter().sum::<f32>() / channels as f32;
                    for _ in 0..channels {
                        pcm.extend_from_slice(&to_s16(avg).to_le_bytes());
                    }
                } else {
                    for s in frame {
                        pcm.extend_from_slice(&to_s16(*s).to_le_bytes());
                    }
                }
            }
            pending.drain(..usable);
            next_pts = Some(pts + (usable / channels as usize) as f64 * 1e6 / rate as f64);
            // Same shift the local passthrough applies: capture cards hand
            // out audio earlier than the matching picture.
            let delay_us = state.delay_ms() as f64 * 1000.0;
            self.publish(AudioChunk {
                sample_rate: rate,
                channels,
                pts_us: (pts + delay_us).max(0.0) as u64,
                pcm,
            });
        }
        if let Some((old, _)) = attached.take() {
            old.remove_relay_sink();
        }
        self.cv.notify_all();
        log::info!("relay audio pump exiting");
    }
}

fn to_s16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

pub struct AudioSubscription {
    relay: Arc<AudioRelay>,
    next: u64,
}

impl AudioSubscription {
    /// Wait up to `timeout` for the next chunk. None on timeout or shutdown.
    pub fn recv(&mut self, timeout: Duration) -> Option<Arc<AudioChunk>> {
        let deadline = Instant::now() + timeout;
        let mut ring = self.relay.ring.lock();
        loop {
            if self.relay.shutdown.load(Ordering::Relaxed) {
                return None;
            }
            if let (Some((oldest, _)), Some((newest, _))) = (ring.chunks.front(), ring.chunks.back()) {
                if *newest >= self.next {
                    if newest - self.next >= MAX_LAG_CHUNKS || self.next < *oldest {
                        self.next = newest.saturating_sub(1).max(*oldest);
                    }
                    let idx = (self.next - oldest) as usize;
                    let chunk = ring.chunks[idx].1.clone();
                    self.next += 1;
                    return Some(chunk);
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            self.relay.cv.wait_for(&mut ring, deadline - now);
        }
    }
}

impl Drop for AudioSubscription {
    fn drop(&mut self) {
        self.relay.subscribers.fetch_sub(1, Ordering::Relaxed);
    }
}

/// RIFF/WAVE header for an endless stream. Size fields are maxed out so
/// players treat it as "keep reading until the connection closes".
pub fn wav_header(sample_rate: u32, channels: u16) -> [u8; 44] {
    let block_align = channels * 2;
    let byte_rate = sample_rate * block_align as u32;
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    h[22..24].copy_from_slice(&channels.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&(u32::MAX - 36).to_le_bytes());
    h
}

use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

/// Live performance metrics for the F1 panel. CPU and memory are sampled in
/// a background thread; latency and frame-interval counters are updated from
/// the capture and preview hot paths via the LatencyTracker.
pub struct PerfMetrics {
    pub cpu_percent: AtomicU64,
    pub memory_mb: AtomicU64,
    pub system: Mutex<SystemSnapshot>,
    pub latency: LatencyTracker,
}

/// Two exponentially-smoothed float counters: end-to-end pipeline latency
/// (capture publish to render present) and the capture-side frame interval
/// (jitter between successive frames arriving from the device). Both are
/// stored as f32::to_bits in atomics so the renderer and capture threads can
/// update them without locking.
pub struct LatencyTracker {
    pipeline_ms: AtomicU32,
    capture_interval_ms: AtomicU32,
    last_capture_mark: Mutex<Option<Instant>>,
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self {
            pipeline_ms: AtomicU32::new(0),
            capture_interval_ms: AtomicU32::new(0),
            last_capture_mark: Mutex::new(None),
        }
    }

    pub fn pipeline_ms(&self) -> f32 {
        f32::from_bits(self.pipeline_ms.load(Ordering::Relaxed))
    }

    pub fn capture_interval_ms(&self) -> f32 {
        f32::from_bits(self.capture_interval_ms.load(Ordering::Relaxed))
    }

    /// Called from the render thread on each present with the age of the
    /// frame that was just shown.
    pub fn record_pipeline(&self, age: Duration) {
        let ms = age.as_secs_f32() * 1000.0;
        let prev = self.pipeline_ms();
        let next = if prev == 0.0 { ms } else { prev * 0.85 + ms * 0.15 };
        self.pipeline_ms.store(next.to_bits(), Ordering::Relaxed);
    }

    /// Called from the capture thread on each frame so we can compute the
    /// inter-frame interval and smooth it.
    pub fn mark_capture(&self) {
        let now = Instant::now();
        let mut guard = self.last_capture_mark.lock();
        if let Some(prev) = *guard {
            let interval = now.saturating_duration_since(prev).as_secs_f32() * 1000.0;
            let cur = self.capture_interval_ms();
            let next = if cur == 0.0 { interval } else { cur * 0.85 + interval * 0.15 };
            self.capture_interval_ms.store(next.to_bits(), Ordering::Relaxed);
        }
        *guard = Some(now);
    }
}

#[derive(Default, Clone, Copy)]
pub struct SystemSnapshot {
    pub total_cpu_percent: f32,
    pub used_memory_mb: u64,
    pub total_memory_mb: u64,
}

impl PerfMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            cpu_percent: AtomicU64::new(0),
            memory_mb: AtomicU64::new(0),
            system: Mutex::new(SystemSnapshot::default()),
            latency: LatencyTracker::new(),
        })
    }

    pub fn cpu_percent(&self) -> f32 {
        f32::from_bits(self.cpu_percent.load(Ordering::Relaxed) as u32)
    }

    pub fn memory_mb(&self) -> u64 {
        self.memory_mb.load(Ordering::Relaxed)
    }

    pub fn system(&self) -> SystemSnapshot {
        *self.system.lock()
    }
}

pub fn spawn_sampler(metrics: Arc<PerfMetrics>) {
    let pid = Pid::from_u32(std::process::id());
    let _ = std::thread::Builder::new()
        .name("perf-sampler".into())
        .spawn(move || {
            let mut sys = System::new_with_specifics(
                RefreshKind::new()
                    .with_processes(ProcessRefreshKind::everything())
                    .with_memory(sysinfo::MemoryRefreshKind::everything())
                    .with_cpu(sysinfo::CpuRefreshKind::everything()),
            );
            // sysinfo needs two reads to compute CPU%, so prime once.
            sys.refresh_all();
            std::thread::sleep(Duration::from_millis(500));
            loop {
                sys.refresh_cpu_all();
                sys.refresh_memory();
                sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);

                let snapshot = SystemSnapshot {
                    total_cpu_percent: sys.global_cpu_usage(),
                    used_memory_mb: sys.used_memory() / 1024 / 1024,
                    total_memory_mb: sys.total_memory() / 1024 / 1024,
                };
                *metrics.system.lock() = snapshot;

                if let Some(p) = sys.process(pid) {
                    let cpu = p.cpu_usage();
                    metrics
                        .cpu_percent
                        .store(cpu.to_bits() as u64, Ordering::Relaxed);
                    let mem_mb = p.memory() / 1024 / 1024;
                    metrics.memory_mb.store(mem_mb, Ordering::Relaxed);
                }

                std::thread::sleep(Duration::from_secs(1));
            }
        });
}


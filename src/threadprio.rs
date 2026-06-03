/// Windows thread priority hints. Used to keep the capture and render
/// threads from being preempted when OBS / a game / browser is also fighting
/// for CPU. No-op on non-Windows.

#[cfg(windows)]
pub fn bump_capture_thread() {
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL,
        };
        let h = GetCurrentThread();
        let ok = SetThreadPriority(h, THREAD_PRIORITY_TIME_CRITICAL);
        if ok == 0 {
            log::warn!("could not raise capture thread priority");
        } else {
            log::info!("capture thread set to TIME_CRITICAL");
        }
    }
    bump_mmcss("Capture");
}

#[cfg(windows)]
pub fn bump_render_thread() {
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST,
        };
        let h = GetCurrentThread();
        let ok = SetThreadPriority(h, THREAD_PRIORITY_HIGHEST);
        if ok == 0 {
            log::warn!("could not raise render thread priority");
        } else {
            log::info!("render thread set to HIGHEST");
        }
    }
    bump_mmcss("Games");
}

#[cfg(not(windows))]
pub fn bump_capture_thread() {}

#[cfg(not(windows))]
pub fn bump_render_thread() {}

/// Register this thread with the Windows Multimedia Class Scheduler Service.
/// MMCSS gets the scheduler to favour media work over background tasks for
/// short, predictable bursts. The handle is intentionally leaked so the
/// classification stays in place for the thread's whole life.
#[cfg(windows)]
fn bump_mmcss(task: &str) {
    use std::ffi::OsStr;
    use std::iter::once;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;

    let wide: Vec<u16> = OsStr::new(task).encode_wide().chain(once(0)).collect();
    let mut task_index: u32 = 0;
    // AvSetMmThreadCharacteristicsW lives in avrt.dll; load and call dynamically
    // so we never hard-fail on systems that lack it.
    unsafe {
        use windows_sys::Win32::Foundation::HMODULE;
        type AvSetFn = unsafe extern "system" fn(
            *const u16,
            *mut u32,
        ) -> windows_sys::Win32::Foundation::HANDLE;

        let lib = load_library_w("avrt.dll");
        if lib.is_null() {
            return;
        }
        let proc = get_proc_address(lib, b"AvSetMmThreadCharacteristicsW\0");
        if proc.is_null() {
            return;
        }
        let func: AvSetFn = std::mem::transmute(proc);
        let handle = func(wide.as_ptr(), &mut task_index);
        if handle.is_null() {
            log::debug!("MMCSS classification '{task}' refused");
        } else {
            log::info!("thread classified as MMCSS '{task}' (index {task_index})");
        }
        // Intentionally leak `handle` so the classification persists; we
        // also leak `lib` since the process holds avrt.dll for its lifetime.
        let _ = handle;
        let _ = lib;
        // Suppress unused warning.
        let _ = ptr::null::<HMODULE>();
    }
}

#[cfg(windows)]
unsafe fn load_library_w(name: &str) -> *mut std::ffi::c_void {
    use std::ffi::OsStr;
    use std::iter::once;
    use std::os::windows::ffi::OsStrExt;
    extern "system" {
        fn LoadLibraryW(filename: *const u16) -> *mut std::ffi::c_void;
    }
    let wide: Vec<u16> = OsStr::new(name).encode_wide().chain(once(0)).collect();
    LoadLibraryW(wide.as_ptr())
}

#[cfg(windows)]
unsafe fn get_proc_address(
    lib: *mut std::ffi::c_void,
    name: &[u8],
) -> *mut std::ffi::c_void {
    extern "system" {
        fn GetProcAddress(module: *mut std::ffi::c_void, name: *const u8) -> *mut std::ffi::c_void;
    }
    GetProcAddress(lib, name.as_ptr())
}

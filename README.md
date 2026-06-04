# vicash

Use your capture card as a second monitor for your console, without an actual second monitor. Single 10 MB exe, no installer, no setup. VIdeo CApture SHare.

![vicash preview window with F1 settings panel open](assets/screenshots/hero.png)

## Download

[Latest release](https://github.com/caaatto/vicash/releases/latest) - one file, double-click, pick your capture card from the dropdown, you are done. Press F1 inside the window for live settings.

## What it solves

You plug a console (Switch, PS, Xbox) into an HDMI splitter, one leg goes to a TV, the other into a USB capture card on your PC. You want to actually use that capture feed as a playable monitor: low latency, with audio, full screen, without buying anything else.

The usual options all hurt:

- OBS preview adds 100-400 ms of lag and burns CPU you wanted for the game.
- The Windows Camera app treats your capture card like a webcam, no audio, no fullscreen toggle.
- Elgato's tools only work with Elgato hardware.
- VLC and PotPlayer can open a DirectShow device but default to hundreds of ms of caching.

vicash is built for the cheap "fake USB3" capture cards (MS2109, MS2130, generic AliExpress HDMI grabbers) that everyone actually owns. It defaults to settings those cards can actually deliver, ships latency numbers you can verify in the F1 panel, and does not try to be OBS.

## Features

- GPU rendered preview, redraws only when capture publishes a new frame so CPU and GPU stay idle the rest of the time
- NV12 zero-copy upload: Y and UV planes go straight to GPU textures, BT.709 conversion in a shader, no CPU colour conversion
- Borderless fullscreen + always on top + hide cursor (F11) so the window behaves like a real second monitor
- Audio passthrough from the capture card audio device to your default output, with a live volume slider, mute, and a sync delay slider so you can match audio to picture by ear
- **Mono mix toggle** for MS2109 / MS2130 cards that only carry audio on the left channel - averages all input channels and broadcasts to every speaker
- Live device switching from the F1 panel: pick audio input, audio output, capture resolution and fps without restarting
- **Device-by-name persistence** so Windows reshuffling capture indices after a reboot does not silently swap you onto a webcam
- Image adjustment shader: **brightness / contrast / saturation / hue** with named presets, **4:3 / 16:9 / 16:10 / custom aspect ratio** lock, **Ctrl + mouse-wheel zoom**, and a **CRT scanline** strength for retro consoles
- **F9** screenshot to PNG, **F10** video recording to MP4 (optional `ffmpeg.exe` on PATH)
- MJPEG over HTTP relay (`--serve 0.0.0.0:7777`) so a second PC can pull the feed in OBS as a browser source, with a **localhost-only bind toggle** for privacy on public networks
- "No signal" overlay when frame delivery stops, so you can tell at a glance whether the cable, console or app went away
- Honest performance dashboard: capture-to-present pipeline latency in milliseconds and capture-side frame interval, so you can see exactly which part of the chain is slow
- **Self-updater** that checks GitHub Releases on startup and offers a one-click install + restart for new versions, so critical fixes (like the v0.1.4 DWM crash hotfix) reach you without you having to think about it
- **Audio + video relay over fMP4 (beta)** with optional `ffmpeg.exe`. Open `/player` for a low-latency MSE viewer (~0.5-1s) or `/stream.mp4` directly for OBS Browser Source. Marked beta because audio capture-card sync varies by hardware
- Settings persist to `%APPDATA%\caaatto\vicash\config.toml` and reload on next launch
- Three UI languages: Deutsch, English, 简体中文

<table>
<tr>
<td><img src="assets/screenshots/stats-overlay.png" alt="Stats overlay corner" width="380"></td>
<td><img src="assets/screenshots/language-picker.png" alt="Language picker DE / EN / 简体中文" width="380"></td>
</tr>
</table>

## How vicash compares

| | vicash | OBS preview | Elgato 4K Capture | VLC / PotPlayer | TackleCast | VideoGameCapture |
|---|---|---|---|---|---|---|
| Single .exe, no install | yes | no | no | no | zip | Unity bundle |
| Audio passthrough with live sync slider | yes | partial | yes | no | yes (no slider) | JSON edit only |
| MS2109 / MS2130 mono-mix audio fix | yes | no | no | no | no | no |
| MJPEG HTTP relay | yes | no | no | no | no | no |
| Optimised for cheap MS2109 / MS2130 cards | yes | no | no | no | partial | partial |
| Works without vendor lock-in | yes | yes | Elgato only | yes | yes | yes |
| Live capture device / resolution / fps switch | yes | yes | partial | partial | partial | partial |
| Live colour adjustment with presets | yes | partial | partial | partial | no | no |
| Custom aspect lock (4:3 retro etc.) | yes | yes | no | partial | no | no |
| Screenshot / recording hotkey | F9 / F10 | scene-only | yes | no | no | screenshot only |
| Pipeline latency meter in UI | yes | no | no | no | no | no |

If you have an Elgato card and already run OBS, Elgato's own utility is fine. If you want one .exe that opens, shows your console, lets you tweak everything live, and tells you exactly how much latency the software is adding, vicash is the tighter fit.

## Run

Inside the preview window:

- `F1` opens the settings panel (Sprache, Monitor-Modus, Anzeige, Capture, Audio, Relay, Performance)
- `F9` saves a PNG screenshot to `<Pictures>/vicash`
- `F10` toggles MP4 video recording to the same folder (needs `ffmpeg.exe` reachable on PATH or next to `vicash.exe`)
- `F11` toggles borderless fullscreen, `Esc` always leaves fullscreen
- `Ctrl` + mouse-wheel zooms the picture; right-click anywhere resets zoom and pan

Command line, if you prefer:

```
vicash.exe                                    # interactive device picker
vicash.exe --list                             # list video devices and exit
vicash.exe --list-audio                       # list audio devices and exit
vicash.exe --device 0                         # open device 0 in a preview window
vicash.exe --device 0 --audio                 # also pass audio through to default output
vicash.exe --device 0 --serve 0.0.0.0:7777    # also serve MJPEG over HTTP
```

## Sharing the feed to a second PC (MJPEG relay)

If you have a two-PC setup (game PC running vicash, streaming PC running OBS) and you do not want to buy another capture card, vicash can publish the live feed over your local network and OBS picks it up like any other browser source.

### Start it from the F1 panel (recommended)

1. Open vicash, press **F1**, expand the **Relay** section.
2. Pick a **Port** (default 7777, change it if that port is taken).
3. Click **Starten** / **Start**. The panel updates to show:

   ```
   Relay läuft / Relay running
   active clients 0    total since start 0

   LAN (other PC / phone on the same network)
     http://192.168.1.42:7777/   [copy URL]
   Local (browser on this PC)
     http://127.0.0.1:7777/      [copy URL]

   Endpoints
     /              browser / OBS Browser Source
     /stream        raw MJPEG stream
     /snapshot.jpg  single JPEG frame
   ```

   The LAN IP is auto-detected. Tick **Beim nächsten Start automatisch starten** / **Start automatically on next launch** to have vicash bring the relay up on its own next time.

4. Drag the JPEG quality slider to taste. 70-80 is a good balance of bandwidth and visual quality; drop it if your LAN is congested or you want to save CPU on the game PC.

### Or pass `--serve` at launch

```
vicash.exe --device 0 --audio --serve 0.0.0.0:7777
```

`0.0.0.0:7777` listens on every network interface so any device on the LAN can connect. Use `127.0.0.1:7777` to keep the stream on this machine only.

### Streaming PC (the one running OBS)

1. In OBS, add a new **Browser Source**.
2. Untick "Local file" and paste the LAN URL from the vicash panel (e.g. `http://192.168.1.42:7777/`).
3. Set the width and height to match your capture resolution (default vicash setting is 1280x720).
4. Tick **Shutdown source when not visible** so the relay does not run when the scene is off-screen.
5. Done. The capture feed shows up in OBS as if it were a local source.

The Relay section in the F1 panel on the game PC shows **active clients 1** once OBS connects, which is handy for confirming the link is alive.

### Other consumers

- Any browser on any device on the LAN: open the LAN URL and you see the feed full-screen with a tiny help overlay. Works on a phone or tablet too.
- `ffplay http://192.168.1.42:7777/stream` if you want a separate window outside OBS.
- `curl -o frame.jpg http://192.168.1.42:7777/snapshot.jpg` to grab a single still.

### Port already in use? (Windows error 10013)

Windows reserves dynamic port ranges for Hyper-V, WSL, Docker and IIS. If the **Start** button shows a red error in the panel, pick a different port. 7777 is the new default for this reason; 8181, 9090 and 5500 are also usually free. Admin rights do **not** help, the OS-level reservation overrides everyone.

To see which ports are excluded on your machine:

```
netsh interface ipv4 show excludedportrange protocol=tcp
```

### Latency note

MJPEG over HTTP is simple and bulletproof but not the lowest-latency wire format. Expect roughly 100-200 ms more than the local vicash preview window. For tight gameplay you still play off the local vicash window; the relay is what your viewers / co-streamers / second-PC OBS see.

## A note on the cheap-card limit

A lot of capture cards sold as USB 3.0 are actually USB 2.0 chips inside a blue housing. They list 1920x1080 60fps as a supported mode but the USB pipe cannot sustain the bandwidth, so they silently fall back to 30 fps once you start streaming. vicash defaults to 1280x720 60 fps for this reason; that fits comfortably inside even fake-USB3 bandwidth and matches the native output of the most common console (Switch). If your card is genuinely fast enough for 1080p60, switch to 1080p in the F1 Capture section, hit Apply, and vicash will remember it.

The Performance section in F1 shows your actual capture interval (target 16.7 ms at 60 fps) so you can see whether the card is keeping up.

## Non goals

vicash does not replace OBS. It does not encode, composite, or stream to Twitch. No scenes, no transitions, no effects, no recording.

## Troubleshooting

### Desktop crashes / black wallpaper / DWM heap corruption

vicash uses wgpu for rendering. On Windows, wgpu picks DX12 first and falls back to Vulkan only if DX12 is unavailable. If you experienced a desktop crash before v0.1.2 with the Vulkan-backed build, that is exactly the DWM + Vulkan + old NVIDIA driver failure mode the DX12 preference avoids. Keep your GPU driver current and stick with the v0.1.2 or later build.

### Capture device stuck after a previous run

Cheap USB capture cards leave the Media Foundation source reader locked for a few seconds after a process exits. If vicash logs `Hardware MFT failed to start streaming due to lack of hardware resources` on launch, close any other app that touches the card (OBS, browser tab opened to /stream) and wait 5-10 seconds before trying again.

### Port 8080 / Windows error 10013

Windows reserves dynamic ranges for Hyper-V, WSL, Docker and IIS that often swallow 8080. vicash defaults to 7777 for that reason. If even 7777 is taken, try 8181, 9090 or 5500 in the F1 panel.

### Audio is local only

The MJPEG relay over LAN currently carries video, not audio. See [issue #1](https://github.com/caaatto/vicash/issues/1) for the state of the audio relay work.

## Build from source

Requires a Rust toolchain (stable). On Windows you also need either the MSVC build tools, or the GNU toolchain (`rustup default stable-x86_64-pc-windows-gnu`) with MinGW-w64 on PATH so `windres.exe` can compile the embedded icon resource.

```
cargo build --release
```

The binary lands at `target/release/vicash.exe`.

## License

MIT. The bundled JetBrains Mono font is under the SIL Open Font License 1.1 (see `assets/JetBrainsMono-OFL.txt`).

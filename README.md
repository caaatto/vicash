# vicash

Low overhead capture card preview and LAN relay for streamers on low end or workaround setups. VIdeo CApture SHare.

## What this is for

If you stream a console (Switch, PS, Xbox) through an HDMI splitter into a budget capture card, the standard tools work but cost you frames. OBS preview alone can eat real CPU, and adding a second PC usually means buying another capture card. vicash gives you the things that solve those pain points without extra hardware:

1. A direct, low latency preview window for the capture device. GPU rendered via wgpu, draws only when a new frame arrives so the GPU sits idle the rest of the time.
2. Audio passthrough from the capture card's audio input to your default output device, with a live sync delay slider in case the picture lags the sound.
3. An optional MJPEG over HTTP relay so a second PC or another OBS instance can pull the feed over your LAN as a browser source.

Built for the case where every CPU percent matters.

## Goals

- Single small native binary, no installer, no runtime.
- Idle when nothing is changing. GPU and audio only do work when there is work to do.
- Sane defaults that work the first time on a fresh Windows machine.
- Live settings: switch audio devices, adjust volume, sync delay, relay quality without restart.

## Non goals

- Replacing OBS. vicash does not encode, composite, or stream to Twitch.
- Fancy effects, overlays, scenes, or transitions.

## Status

Early development. Targeting Windows first because that is where capture cards live.

## Build

Requires a Rust toolchain (stable). On Windows you also need either the MSVC build tools or the GNU toolchain (rustup default stable-x86_64-pc-windows-gnu) with MinGW-w64 on PATH so `windres.exe` can compile the embedded icon resource.

```
cargo build --release
```

The binary lands at `target/release/vicash.exe`.

## Run

```
vicash.exe                                    # interactive device picker
vicash.exe --list                             # list video devices and exit
vicash.exe --list-audio                       # list audio devices and exit
vicash.exe --device 0                         # open device 0 in a preview window
vicash.exe --device 0 --audio                 # also pass audio through to default output
vicash.exe --device 0 --serve 0.0.0.0:8080    # also serve MJPEG over HTTP
```

Press `F1` in the preview window for the settings panel: fit mode, background color, JPEG quality for the relay, audio input/output device pickers, volume, mute, and the audio sync delay slider.

In OBS on the second PC, add a Browser source pointed at `http://<host>:8080/`.

## License

MIT. The bundled JetBrains Mono font is under the SIL Open Font License 1.1 (see `assets/JetBrainsMono-OFL.txt`).

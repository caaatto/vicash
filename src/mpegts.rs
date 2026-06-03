// Hand-rolled MPEG-TS muxer for H.264 video.
//
// Output is a continuous byte stream of 188-byte transport stream packets.
// Layout:
//   PID 0x0000 = PAT, sent every ~100ms
//   PID 0x1000 = PMT, sent every ~100ms
//   PID 0x0100 = H.264 video stream, stream type 0x1B
//
// PCR is carried in the adaptation field of the video PID. Timestamps are
// in the 90 kHz MPEG clock.
//
// This file is video-only on purpose; the AAC audio stream and its PID will
// be added in Phase 4 of the relay rewrite.

use std::time::Duration;

pub const PACKET_SIZE: usize = 188;
const SYNC_BYTE: u8 = 0x47;

const PAT_PID: u16 = 0x0000;
const PMT_PID: u16 = 0x1000;
pub const VIDEO_PID: u16 = 0x0100;

const STREAM_TYPE_H264: u8 = 0x1B;
const VIDEO_STREAM_ID: u8 = 0xE0; // Video stream 0
const PROGRAM_NUMBER: u16 = 1;
const TS_ID: u16 = 1;

/// MPEG-TS clock runs at 90 kHz.
pub const MPEGTS_CLOCK_HZ: u64 = 90_000;

pub struct MpegTsMuxer {
    video_cc: u8,
    pat_cc: u8,
    pmt_cc: u8,
    last_pat_pmt_pts: i64,
    /// Wall-clock-ish 90 kHz counter for the last PCR we emitted, so we can
    /// rate-limit PCR insertion if needed.
    started: bool,
}

impl MpegTsMuxer {
    pub fn new() -> Self {
        Self {
            video_cc: 0,
            pat_cc: 0,
            pmt_cc: 0,
            last_pat_pmt_pts: i64::MIN,
            started: false,
        }
    }

    /// Translate a Duration since stream start to a 90 kHz MPEG-TS timestamp.
    pub fn duration_to_90khz(d: Duration) -> u64 {
        (d.as_nanos() as u64 * MPEGTS_CLOCK_HZ) / 1_000_000_000
    }

    /// Returns a chunk of MPEG-TS bytes representing the supplied H.264
    /// access unit (one or more NAL units in Annex-B form) at the given PTS.
    /// If this is the first call or it has been at least 100 ms since the
    /// last PSI burst, the chunk also includes a fresh PAT + PMT.
    pub fn mux_video_au(
        &mut self,
        h264_au: &[u8],
        pts_90khz: u64,
        is_keyframe: bool,
    ) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(h264_au.len() + 4 * PACKET_SIZE);

        let pts_i64 = pts_90khz as i64;
        let need_psi = !self.started
            || pts_i64.saturating_sub(self.last_pat_pmt_pts)
                >= (MPEGTS_CLOCK_HZ as i64 / 10);
        if need_psi {
            out.extend_from_slice(&self.build_pat());
            out.extend_from_slice(&self.build_pmt());
            self.last_pat_pmt_pts = pts_i64;
            self.started = true;
        }

        let pes = build_pes_packet(VIDEO_STREAM_ID, pts_90khz, h264_au);
        self.packetize_pes(&pes, VIDEO_PID, pts_90khz, is_keyframe, &mut out);
        out
    }

    fn build_pat(&mut self) -> [u8; PACKET_SIZE] {
        // PAT table fields.
        // table_id=0x00, section_syntax_indicator=1, reserved=11,
        // section_length=13 (covers from after this byte to end of CRC).
        // After the standard PAT section header (8 bytes) we have one
        // program entry (4 bytes) and a 4-byte CRC.
        let mut section = Vec::with_capacity(17);
        section.push(0x00); // table_id
        // section_length = 9 (after this field) + 4 (program entry) + 4 (CRC) = 13
        section.extend_from_slice(&[0xB0, 0x0D]);
        section.extend_from_slice(&TS_ID.to_be_bytes());
        // reserved=11, version=0, current_next=1
        section.push(0xC1);
        section.push(0x00); // section_number
        section.push(0x00); // last_section_number
        // program_number, PMT PID with 3 reserved bits (=111)
        section.extend_from_slice(&PROGRAM_NUMBER.to_be_bytes());
        let pmt_pid = (0b1110_0000_0000_0000 | PMT_PID) & 0xFFFF;
        section.extend_from_slice(&pmt_pid.to_be_bytes());
        let crc = crc32_mpeg2(&section);
        section.extend_from_slice(&crc.to_be_bytes());

        let cc = self.pat_cc;
        self.pat_cc = (cc + 1) & 0x0F;
        psi_packet(PAT_PID, cc, &section)
    }

    fn build_pmt(&mut self) -> [u8; PACKET_SIZE] {
        // PMT section.
        let mut section = Vec::with_capacity(64);
        section.push(0x02); // table_id
        // We will patch section_length once we know the body length.
        section.extend_from_slice(&[0x00, 0x00]);
        section.extend_from_slice(&PROGRAM_NUMBER.to_be_bytes());
        section.push(0xC1); // reserved + version + current_next
        section.push(0x00); // section_number
        section.push(0x00); // last_section_number
        // PCR PID = VIDEO_PID, with 3 reserved bits (=111)
        let pcr_pid = (0b1110_0000_0000_0000 | VIDEO_PID) & 0xFFFF;
        section.extend_from_slice(&pcr_pid.to_be_bytes());
        // program_info_length (12 bits) - we have none, 4 reserved bits = 0xF0
        section.extend_from_slice(&[0xF0, 0x00]);

        // Elementary stream loop: one entry for H.264 video.
        section.push(STREAM_TYPE_H264);
        let es_pid = (0b1110_0000_0000_0000 | VIDEO_PID) & 0xFFFF;
        section.extend_from_slice(&es_pid.to_be_bytes());
        // ES_info_length = 0
        section.extend_from_slice(&[0xF0, 0x00]);

        // Patch section_length: covers everything after this 2-byte field
        // up to and including the CRC.
        let section_len = section.len() - 3 + 4; // +4 for CRC
        section[1] = 0xB0 | ((section_len >> 8) as u8 & 0x0F);
        section[2] = (section_len & 0xFF) as u8;

        let crc = crc32_mpeg2(&section);
        section.extend_from_slice(&crc.to_be_bytes());

        let cc = self.pmt_cc;
        self.pmt_cc = (cc + 1) & 0x0F;
        psi_packet(PMT_PID, cc, &section)
    }

    fn packetize_pes(
        &mut self,
        pes: &[u8],
        pid: u16,
        pcr_90khz: u64,
        is_keyframe: bool,
        out: &mut Vec<u8>,
    ) {
        let mut offset = 0usize;
        let mut first = true;
        while offset < pes.len() {
            let mut pkt = [0xFFu8; PACKET_SIZE];
            pkt[0] = SYNC_BYTE;
            let pusi: u16 = if first { 0x4000 } else { 0 };
            // transport_priority=0, PID = pid
            let pid_field = pusi | (pid & 0x1FFF);
            pkt[1] = (pid_field >> 8) as u8;
            pkt[2] = (pid_field & 0xFF) as u8;
            let cc = self.video_cc;
            self.video_cc = (cc + 1) & 0x0F;

            // Decide adaptation field. PCR / random_access on first packet
            // of a keyframe, otherwise just stuffing if we need it.
            let want_pcr = first;
            let want_rai = first && is_keyframe;

            // Header byte 3: transport_scrambling=00, adaptation field
            // control, continuity_counter.
            let mut adaptation_present = false;
            let mut adaptation: Vec<u8> = Vec::new();
            if want_pcr || want_rai {
                adaptation_present = true;
                // adaptation_field_length will be patched after we know.
                adaptation.push(0); // length placeholder
                let mut flags = 0u8;
                if want_rai {
                    flags |= 0x40;
                }
                if want_pcr {
                    flags |= 0x10;
                }
                adaptation.push(flags);
                if want_pcr {
                    // PCR = base (33 bits) + reserved (6 bits) + extension (9 bits)
                    let base = pcr_90khz;
                    let ext: u16 = 0;
                    adaptation.push((base >> 25) as u8);
                    adaptation.push((base >> 17) as u8);
                    adaptation.push((base >> 9) as u8);
                    adaptation.push((base >> 1) as u8);
                    adaptation.push(((base & 1) << 7) as u8 | 0x7E | ((ext >> 8) as u8 & 0x01));
                    adaptation.push((ext & 0xFF) as u8);
                }
                // Patch length = total bytes after this length byte.
                adaptation[0] = (adaptation.len() - 1) as u8;
            }

            // Payload room = 188 - 4 header - adaptation length (if any).
            let max_payload = if adaptation_present {
                PACKET_SIZE - 4 - adaptation.len()
            } else {
                PACKET_SIZE - 4
            };
            let chunk_len = (pes.len() - offset).min(max_payload);

            // If our chunk does not fill the packet, we need to pad via the
            // adaptation field; MPEG-TS does NOT allow short packets.
            let pad_needed = if !adaptation_present {
                max_payload - chunk_len
            } else {
                PACKET_SIZE - 4 - adaptation.len() - chunk_len
            };
            if pad_needed > 0 {
                if !adaptation_present {
                    adaptation_present = true;
                    adaptation.clear();
                    adaptation.push(0); // length placeholder
                    if pad_needed > 1 {
                        adaptation.push(0); // flags = 0
                        for _ in 0..(pad_needed - 2) {
                            adaptation.push(0xFF);
                        }
                    }
                    adaptation[0] = (adaptation.len() - 1) as u8;
                } else {
                    for _ in 0..pad_needed {
                        adaptation.push(0xFF);
                    }
                    adaptation[0] = (adaptation.len() - 1) as u8;
                }
            }

            let afc: u8 = if adaptation_present && chunk_len > 0 {
                0b11
            } else if adaptation_present {
                0b10
            } else {
                0b01
            };
            pkt[3] = (afc << 4) | cc;

            let mut write_at = 4;
            if adaptation_present {
                pkt[write_at..write_at + adaptation.len()].copy_from_slice(&adaptation);
                write_at += adaptation.len();
            }
            pkt[write_at..write_at + chunk_len].copy_from_slice(&pes[offset..offset + chunk_len]);
            offset += chunk_len;
            first = false;
            out.extend_from_slice(&pkt);
        }
    }
}

/// Wrap a PSI section in a single 188-byte TS packet (with pointer field 0).
fn psi_packet(pid: u16, cc: u8, section: &[u8]) -> [u8; PACKET_SIZE] {
    let mut pkt = [0xFFu8; PACKET_SIZE];
    pkt[0] = SYNC_BYTE;
    let pid_field: u16 = 0x4000 | (pid & 0x1FFF);
    pkt[1] = (pid_field >> 8) as u8;
    pkt[2] = (pid_field & 0xFF) as u8;
    pkt[3] = 0b0001_0000 | (cc & 0x0F);
    pkt[4] = 0; // pointer_field
    let max = PACKET_SIZE - 5;
    let n = section.len().min(max);
    pkt[5..5 + n].copy_from_slice(&section[..n]);
    pkt
}

/// Build a PES packet for one video access unit. The packet starts with the
/// 6-byte PES header, the 5-byte PTS-only optional header (PTS encoded into
/// 5 bytes per the spec), and then the H.264 Annex-B payload.
fn build_pes_packet(stream_id: u8, pts_90khz: u64, payload: &[u8]) -> Vec<u8> {
    let mut pes = Vec::with_capacity(14 + payload.len());
    pes.extend_from_slice(&[0x00, 0x00, 0x01]); // start code
    pes.push(stream_id);

    // PES_packet_length = 0 means "unbounded" which is required for video
    // streams whose access units exceed 65535 bytes. We use 0 always.
    pes.extend_from_slice(&[0x00, 0x00]);

    // Flags byte 1: '10' marker, scrambling=00, priority=0, alignment=0,
    // copyright=0, original=0.
    pes.push(0x80);
    // Flags byte 2: PTS_DTS_flags=10 (PTS only), all others off.
    pes.push(0x80);
    // PES_header_data_length = 5 (just PTS).
    pes.push(0x05);

    // PTS in 5-byte encoded form, prefix bits '0010' (PTS only).
    let pts = pts_90khz & ((1u64 << 33) - 1);
    let pts_bytes = encode_timestamp(pts, 0b0010);
    pes.extend_from_slice(&pts_bytes);

    pes.extend_from_slice(payload);
    pes
}

/// 33-bit timestamp encoded into 5 bytes with the given 4-bit prefix.
/// Byte 0: PPPP TTT 1
/// Byte 1: TTTTTTTT
/// Byte 2: TTTTTTT 1
/// Byte 3: TTTTTTTT
/// Byte 4: TTTTTTT 1
fn encode_timestamp(ts: u64, prefix: u8) -> [u8; 5] {
    let mut out = [0u8; 5];
    out[0] = (prefix << 4) | (((ts >> 30) & 0x07) as u8) << 1 | 1;
    out[1] = ((ts >> 22) & 0xFF) as u8;
    out[2] = (((ts >> 15) & 0x7F) as u8) << 1 | 1;
    out[3] = ((ts >> 7) & 0xFF) as u8;
    out[4] = ((ts & 0x7F) as u8) << 1 | 1;
    out
}

/// CRC-32/MPEG-2 (poly 0x04C11DB7, init 0xFFFFFFFF, no reflect, no xor out).
fn crc32_mpeg2(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &b in data {
        crc ^= (b as u32) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C11DB7
            } else {
                crc << 1
            };
        }
    }
    crc
}

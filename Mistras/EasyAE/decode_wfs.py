"""
AEwin64 .WFS File Decoder
=========================
Decodes binary .WFS (Waveform Stream) files produced by AEwin64 acoustic
emission software into numpy arrays.

File format source: AEwin64 User Manual / AppendixII_Data files (MISTRAS Group Inc.)
Empirically verified against real WFS captures (note: manual uses ID=173/0xAD
but real streaming files use ID=174/0xAE with a slightly different layout).

.WFS Binary Structure
---------------------
All multi-byte integers are little-endian (least significant byte first).

Each "message" in the file is laid out as:
    [LEN : 2 bytes] [MESSAGE BODY : LEN bytes]

The first byte of the body is the Message ID.

Hardware-setup message (ID=174, Sub-ID=42):
    Body layout:
        [0]     ID      = 174 (0xAE)
        [1]     Sub-ID  = 42  (0x2A)
        [2]     MVERS   = 110
        [3]     0x00    (padding)
        [4]     ADT     = 2   (16-bit signed samples)
        [5]     SETS    = number of channel setups
        [6:10]  SRATE   = sample rate in Hz, 4-byte LE uint  (e.g. 1000000 = 1 MSPS)
        ...rest of channel-setup block

Waveform message (ID=174, Sub-ID=1):
    Body layout (28-byte fixed header, then samples to end of message):
        [0]     ID      = 174 (0xAE)
        [1]     Sub-ID  = 1   (0x01)
        [2]     version / MVERS (0x64 = 100, constant)
        [3]     0x00    (padding)
        [4:8]   TOT low 4 bytes (0xAAAAAAAA when not timestamped)
        [8]     CID     = channel number (1-based)
        [9:28]  metadata / padding bytes
        [28:]   N × 2-byte signed int16 samples  (N = (LEN - 28) // 2)

NOTE: The documentation specifies ID=173 (0xAD) and a slightly different header,
but observed streaming files use ID=174 (0xAE) with a 28-byte header and no
explicit N field — samples implicitly fill to the end of the message.
"""

import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Message ID constants (empirical — streaming files use 174, not 173)
# ---------------------------------------------------------------------------
MSG_ID_STREAMING   = 174   # 0xAE — used in real WFS streaming files
MSG_ID_DOCUMENTED  = 173   # 0xAD — as written in the manual (legacy / non-streaming)

SUBID_WAVEFORM  = 1
SUBID_HW_SETUP  = 42

# Fixed header size in waveform messages (empirically determined)
# Samples begin at byte 28 and run to the end of the message body.
WAVEFORM_HEADER_BYTES = 28

# Offset of CID (channel ID) inside the 28-byte waveform header body
CID_OFFSET = 8


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class WaveformRecord:
    """One captured waveform burst."""
    channel: int                   # AE channel number (CID), 1-based
    samples: np.ndarray            # int16 numpy array of ADC counts
    sample_rate_hz: Optional[int] = None  # filled from hardware-setup message

    @property
    def time_axis_s(self) -> Optional[np.ndarray]:
        """Time axis for each sample in seconds, or None if rate unknown."""
        if self.sample_rate_hz is None:
            return None
        dt = 1.0 / self.sample_rate_hz
        return np.arange(len(self.samples)) * dt


@dataclass
class WFSFile:
    """Decoded contents of a .WFS file."""
    path: Path
    sample_rate_hz: Optional[int] = None
    waveforms: List[WaveformRecord] = field(default_factory=list)

    def to_array(self, channel: Optional[int] = None) -> np.ndarray:
        """
        Stack all waveforms (optionally filtered by channel) into a 2-D
        numpy array of shape (n_records, n_samples).

        All records have the same length in streaming-mode files, so no
        zero-padding is needed. Mixed-length files are zero-padded to the
        longest record.

        Parameters
        ----------
        channel : int, optional
            If given, only include records from this AE channel number.

        Returns
        -------
        np.ndarray  shape (n_records, max_samples), dtype int16
        """
        records = self.waveforms
        if channel is not None:
            records = [r for r in records if r.channel == channel]
        if not records:
            return np.empty((0, 0), dtype=np.int16)
        max_len = max(len(r.samples) for r in records)
        out = np.zeros((len(records), max_len), dtype=np.int16)
        for i, rec in enumerate(records):
            out[i, : len(rec.samples)] = rec.samples
        return out

    def channels(self) -> List[int]:
        """Unique channel numbers present in this file."""
        return sorted({r.channel for r in self.waveforms})


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_sample_rate(body: bytes) -> Optional[int]:
    """
    Extract sample rate in Hz from a hardware-setup message body
    (ID=174, Sub-ID=42).

    The sample rate is stored as a 4-byte little-endian uint at body[6:10],
    in units of Hz (e.g. 1000000 = 1 MSPS).
    """
    if len(body) < 10:
        return None
    srate = struct.unpack_from("<I", body, 6)[0]
    return srate if srate > 0 else None


def decode_wfs(path: str | Path,
               max_records: Optional[int] = None) -> WFSFile:
    """
    Parse a .WFS streaming file and return a :class:`WFSFile` with all
    waveform records decoded as int16 numpy arrays.

    Parameters
    ----------
    path : str or Path
        Path to the .wfs file.
    max_records : int, optional
        Stop after reading this many waveform records (useful for large files
        during development/testing).

    Returns
    -------
    WFSFile
        Decoded file object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    data = path.read_bytes()

    result = WFSFile(path=path)
    pos = 0
    total = len(data)
    n_waveforms = 0

    while pos + 2 <= total:
        if max_records is not None and n_waveforms >= max_records:
            break

        msg_len = struct.unpack_from("<H", data, pos)[0]
        pos += 2

        if msg_len == 0:
            continue  # zero-length padding message

        if pos + msg_len > total:
            break     # truncated file — stop gracefully

        body = data[pos : pos + msg_len]
        pos += msg_len

        if len(body) < 2:
            continue

        msg_id  = body[0]
        sub_id  = body[1]

        # ------------------------------------------------------------------ #
        #  Hardware setup → extract sample rate                               #
        # ------------------------------------------------------------------ #
        if sub_id == SUBID_HW_SETUP and msg_id in (MSG_ID_STREAMING, MSG_ID_DOCUMENTED):
            srate = _parse_sample_rate(body)
            if srate is not None:
                result.sample_rate_hz = srate
            continue

        # ------------------------------------------------------------------ #
        #  Waveform data                                                       #
        # ------------------------------------------------------------------ #
        if sub_id == SUBID_WAVEFORM and msg_id in (MSG_ID_STREAMING, MSG_ID_DOCUMENTED):
            if msg_len <= WAVEFORM_HEADER_BYTES:
                continue  # no samples in this message

            cid = body[CID_OFFSET] if len(body) > CID_OFFSET else 0

            n_samples = (msg_len - WAVEFORM_HEADER_BYTES) // 2
            sample_bytes = body[WAVEFORM_HEADER_BYTES : WAVEFORM_HEADER_BYTES + n_samples * 2]
            samples = np.frombuffer(sample_bytes, dtype="<i2").copy()

            result.waveforms.append(WaveformRecord(
                channel=cid,
                samples=samples,
                sample_rate_hz=result.sample_rate_hz,
            ))
            n_waveforms += 1

    # Back-fill sample rate on any records parsed before the hw-setup message
    if result.sample_rate_hz is not None:
        for rec in result.waveforms:
            if rec.sample_rate_hz is None:
                rec.sample_rate_hz = result.sample_rate_hz

    return result


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def load_continuous(path: str | Path,
                    channel: Optional[int] = None,
                    max_records: Optional[int] = None,
                    sample_rate_hz: Optional[int] = None):
    """
    Decode a .WFS file and concatenate all records into a single continuous
    time-series suitable for spectral analysis.

    Parameters
    ----------
    path : str or Path
        Path to the .wfs file.
    channel : int, optional
        If given, only include records from this AE channel number.
    max_records : int, optional
        Stop after reading this many waveform records.
    sample_rate_hz : int, optional
        Override the sample rate found in the file (Hz).

    Returns
    -------
    samples : np.ndarray  float64, shape (N,)
        Concatenated ADC counts as float64.
    time : np.ndarray  float64, shape (N,)
        Time axis in seconds from the start of the first sample.
    sample_rate_hz : int
        Sample rate used (either read from the file or the override value).

    Raises
    ------
    SystemExit
        If no waveform records are found, or the requested channel is absent.
    """
    import sys

    wfs = decode_wfs(path, max_records=max_records)

    if not wfs.waveforms:
        sys.exit(f"No waveform records found in {path}")

    sr = sample_rate_hz or wfs.sample_rate_hz
    if sr is None:
        print("WARNING: sample rate not found in hardware-setup message — "
              "defaulting to 1,000,000 Hz. Pass sample_rate_hz to override.")
        sr = 1_000_000

    records = wfs.waveforms
    if channel is not None:
        records = [r for r in records if r.channel == channel]
        if not records:
            sys.exit(f"No records found for channel {channel} in {path}")

    print(f"Loaded {len(records):,} records  |  sample rate: {sr:,} Hz"
          f"  |  samples per record: {len(records[0].samples):,}")

    raw = np.concatenate([r.samples for r in records]).astype(np.float64)
    t   = np.arange(len(raw)) / sr

    return raw, t, sr


def wfs_to_numpy(path: str | Path,
                 channel: Optional[int] = None,
                 max_records: Optional[int] = None) -> np.ndarray:
    """
    One-liner: decode a .WFS file and return a 2-D int16 numpy array.

    Shape: (n_waveforms, n_samples_per_waveform).

    Parameters
    ----------
    path : str or Path
        Path to the .wfs file.
    channel : int, optional
        If given, only return waveforms from that AE channel.
    max_records : int, optional
        Limit number of records read (useful for large files).

    Returns
    -------
    np.ndarray  shape (n_waveforms, n_samples), dtype int16
    """
    return decode_wfs(path, max_records=max_records).to_array(channel=channel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decode a .WFS streaming file and print summary info."
    )
    parser.add_argument("wfs_file", help="Path to the .wfs file")
    parser.add_argument(
        "--channel", type=int, default=None,
        help="Only show waveforms from this channel number"
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Stop after reading this many waveform records"
    )
    args = parser.parse_args()

    wfs = decode_wfs(args.wfs_file, max_records=args.max_records)

    print(f"File          : {wfs.path.name}")
    if wfs.sample_rate_hz:
        print(f"Sample rate   : {wfs.sample_rate_hz:,} Hz  ({wfs.sample_rate_hz/1e6:.3f} MSPS)")
    else:
        print("Sample rate   : unknown (no hardware-setup message found)")
    print(f"Total records : {len(wfs.waveforms):,}")
    print(f"Channels      : {wfs.channels()}")

    records = wfs.waveforms
    if args.channel is not None:
        records = [r for r in records if r.channel == args.channel]
        print(f"Records ch {args.channel} : {len(records):,}")
    print()

    for i, rec in enumerate(records[:5]):
        print(f"  [{i}] ch={rec.channel}  n={len(rec.samples):,} samples"
              f"  min={rec.samples.min()}  max={rec.samples.max()}"
              f"  std={rec.samples.std():.2f}")

    if len(records) > 5:
        print(f"  ... ({len(records) - 5:,} more records)")

    arr = wfs.to_array(channel=args.channel)
    print(f"\nnumpy array shape : {arr.shape}  dtype={arr.dtype}")
    print("Done.")

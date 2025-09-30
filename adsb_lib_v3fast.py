
"""
adsb_lib_v3fast.py — Same API as adsb_lib_v3, but with a much faster detector.

Changes vs v3:
- Replaced rolling median/MAD (O(n*win)) with a global/blocked MAD threshold (≈O(n)).
- Everything else (I/O, slicing, CRC/PI) is identical in spirit for compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any
import numpy as np

# -------------------- Data structures --------------------

@dataclass
class IQSamples:
    samples: np.ndarray
    fs: float
    center_freq: Optional[float] = None
    meta: Dict[str, Any] = None

@dataclass
class Detection:
    p0: int
    score: float
    fs: float
    up: int

@dataclass
class DecodeConfig:
    up: int = 2
    min_snr: float = 6.0
    offset_search: Tuple[int, int] = (-2, 3)
    try_lengths: Tuple[int, ...] = (112, 56)
    crc_poly: int = 0xFFF409
    check_df17_df18_pi: bool = True

@dataclass
class Frame:
    p0: int
    bits: np.ndarray
    length_bits: int
    df: Optional[int]
    ca: Optional[int]
    icao: Optional[int]
    me_bits: Optional[np.ndarray]
    parity: Optional[int]
    crc_ok: Optional[bool]
    hex: str

# -------------------- I/O --------------------

def load_cu8_iq(path: Union[str, Path], max_samples: Optional[int] = None) -> np.ndarray:
    b = np.fromfile(str(path), dtype=np.uint8)
    if max_samples is not None:
        b = b[:2*max_samples]
    if b.size % 2: b = b[:-1]
    i = b[0::2].astype(np.float32) - 127.5
    q = b[1::2].astype(np.float32) - 127.5
    return (i/127.5 + 1j*q/127.5).astype(np.complex64)

def load_cf32_iq(path: Union[str, Path], max_samples: Optional[int] = None) -> np.ndarray:
    b = np.fromfile(str(path), dtype=np.float32)
    if max_samples is not None:
        b = b[:2*max_samples]
    if b.size % 2: b = b[:-1]
    i = b[0::2]; q = b[1::2]
    return (i + 1j*q).astype(np.complex64)

def load_iq(path: Union[str, Path], fmt: str, fs: float, max_samples: Optional[int] = None,
            center_freq: Optional[float] = None, meta: Optional[Dict[str, Any]] = None) -> IQSamples:
    path = Path(path)
    if fmt.lower() in ("cu8","u8","cu","iq8"):
        x = load_cu8_iq(path, max_samples=max_samples)
    elif fmt.lower() in ("cf32","f32","float32"):
        x = load_cf32_iq(path, max_samples=max_samples)
    else:
        raise ValueError("fmt must be 'cu8' or 'cf32'")
    return IQSamples(samples=x, fs=float(fs), center_freq=center_freq, meta=meta or {})

# -------------------- DSP --------------------

def dc_remove(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)).astype(x.dtype, copy=False)

def to_magnitude(x: np.ndarray) -> np.ndarray:
    return np.abs(x).astype(np.float32)

def mag_oversample(mag: np.ndarray, fs: float, up: int = 2) -> Tuple[np.ndarray, float]:
    if up <= 1:
        return mag.astype(np.float32, copy=False), fs
    n = mag.size
    t_up = np.linspace(0, n-1, n*up, dtype=np.float32)
    mag_up = np.interp(t_up, np.arange(n, dtype=np.float32), mag).astype(np.float32)
    return mag_up, fs*up

# -------------------- Detection (fast) --------------------

def _preamble_template(fs: float) -> np.ndarray:
    us = lambda micro: int(round((micro*1e-6)*fs))
    segs = [(0.0,0.5,1.0),(0.5,1.0,0.0),(1.0,1.5,1.0),(1.5,3.5,0.0),
            (3.5,4.0,1.0),(4.0,5.5,0.0),(5.5,6.0,1.0),(6.0,8.0,0.0)]
    L = max(us(8.0), 8)
    tpl = np.zeros(L, dtype=np.float32)
    for a,b,val in segs: tpl[us(a):us(b)] = val
    if np.max(tpl)>0: tpl /= np.max(tpl)
    return tpl

def _blocked_median_mad(x: np.ndarray, block: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate rolling stats by blockwise med/MAD with nearest-neighbor expand."""
    n = x.size
    nb = (n + block - 1)//block
    med_b = np.empty(nb, dtype=np.float32)
    mad_b = np.empty(nb, dtype=np.float32)
    for b in range(nb):
        s = b*block; e = min(n, s+block)
        w = x[s:e]
        m = np.median(w)
        med_b[b] = m
        mad_b[b] = np.median(np.abs(w - m)) + 1e-6
    # expand to length n
    med = np.repeat(med_b, block)[:n]
    mad = np.repeat(mad_b, block)[:n]
    return med, mad

def detect_preambles(mag: np.ndarray, fs: float, min_snr: float = 6.0, up: int = 1,
                     use_block_stats: bool = True, block: int = 4096) -> List[Detection]:
    tpl = _preamble_template(fs)
    corr = np.convolve(mag, tpl[::-1], mode='same')

    if use_block_stats:
        med, mad = _blocked_median_mad(corr, block=block)
    else:
        # global stats (very fast)
        m = np.median(corr)
        med = np.full_like(corr, m)
        mad = np.full_like(corr, np.median(np.abs(corr - m)) + 1e-6)

    thr = med + (min_snr/2.0)*mad

    # local maxima above threshold
    c = corr
    peaks = (c[1:-1] > c[:-2]) & (c[1:-1] >= c[2:]) & (c[1:-1] > thr[1:-1])
    idx = np.nonzero(peaks)[0] + 1

    # NMS within ~4us
    guard = int(round(4e-6*fs))
    kept = []
    last = -10**9
    for p in idx:
        if not kept or p - last > guard:
            kept.append(p); last = p
        elif c[p] > c[last]:
            kept[-1] = p; last = p

    return [Detection(p0=int(p), score=float(c[p]), fs=fs, up=up) for p in kept]

# -------------------- Bit slicing & helpers --------------------

def slice_ppm_bits(mag: np.ndarray, fs: float, payload_bits: int, start_index: int) -> np.ndarray:
    start = start_index + int(round(8e-6*fs))
    spb = fs / 1e6
    h = int(round(0.5e-6*fs))
    bits = np.zeros(payload_bits, dtype=np.uint8)
    for k in range(payload_bits):
        s = int(round(start + k*spb))
        a = np.sum(mag[s:s+h])
        b = np.sum(mag[s+h:s+2*h])
        bits[k] = 1 if a > b else 0
    return bits

def bits_to_int(bits: np.ndarray, a: int = 0, b: Optional[int] = None) -> int:
    if b is None: b = len(bits)
    val = 0
    for bit in bits[a:b]:
        val = (val << 1) | int(bit)
    return val

def bits_to_hex(bits: np.ndarray) -> str:
    n = len(bits)
    if n % 4:
        pad = 4 - (n % 4)
        bits = np.concatenate([np.zeros(pad, dtype=np.uint8), bits])
    out = ''
    for i in range(0, len(bits), 4):
        nib = (bits[i]<<3) | (bits[i+1]<<2) | (bits[i+2]<<1) | (bits[i+3])
        out += f"{nib:X}"
    return out

# -------------------- CRC / Parity --------------------

def crc24_compute(bits: np.ndarray, poly: int) -> int:
    reg = 0
    top = 1 << 24
    mask24 = (1 << 24) - 1
    for b in bits:
        reg = ((reg << 1) | int(b)) & ((1 << 25) - 1)
        if reg & top:
            reg ^= (top | poly)
    return reg & mask24

def parse_modes_header(bits: np.ndarray):
    n = len(bits)
    if n not in (56, 112):
        return None, None, None, None, None
    df = bits_to_int(bits, 0, 5) if n >= 5 else None
    ca = bits_to_int(bits, 5, 8) if n >= 8 else None
    icao = bits_to_int(bits, 8, 32) if n >= 32 else None
    if n == 56:
        me_bits = bits[32:56-24] if 56-24 > 32 else np.array([], dtype=np.uint8)
        parity = bits_to_int(bits, 56-24, 56)
    else:
        me_bits = bits[32:112-24]
        parity = bits_to_int(bits, 112-24, 112)
    return df, ca, icao, me_bits, parity

def verify_crc_or_pi(bits: np.ndarray, cfg: DecodeConfig, icao: Optional[int], parity: Optional[int]) -> Optional[bool]:
    n = len(bits)
    if n not in (56, 112): return None
    msg_wo_parity = bits[:n-24]
    crc = crc24_compute(msg_wo_parity, cfg.crc_poly) if cfg.crc_poly is not None else None
    if crc is None or parity is None: return None
    df = bits_to_int(bits, 0, 5)
    if cfg.check_df17_df18_pi and df in (17,18) and icao is not None:
        return parity == (crc ^ (icao & ((1<<24)-1)))
    else:
        return parity == crc

# -------------------- Decoding --------------------

def decode_frames(mag: np.ndarray, fs: float, detections: Sequence[Detection], cfg: Optional[DecodeConfig] = None) -> List[Frame]:
    cfg = cfg or DecodeConfig()
    frames: List[Frame] = []
    for det in detections:
        for off in range(cfg.offset_search[0], cfg.offset_search[1]):
            p = det.p0 + off
            for nbits in cfg.try_lengths:
                try:
                    bits = slice_ppm_bits(mag, fs, nbits, p)
                except Exception:
                    continue
                df, ca, icao, me_bits, parity = parse_modes_header(bits)
                crc_ok = verify_crc_or_pi(bits, cfg, icao, parity)
                frames.append(Frame(
                    p0=p, bits=bits, length_bits=nbits,
                    df=df, ca=ca, icao=icao, me_bits=me_bits,
                    parity=parity, crc_ok=crc_ok, hex=bits_to_hex(bits)
                ))
    return frames

# -------------------- Synthesis & Pipeline --------------------

def synth_adsb_bits(n_payload_bits: int = 112, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0xAD5B)
    return (rng.random(n_payload_bits) > 0.5).astype(np.uint8)

def bits_to_ppm_wave(bits: np.ndarray, fs: float, amp: float = 1.0) -> np.ndarray:
    L_pre = int(round(8e-6*fs))
    L_bit = int(round(1e-6*fs))
    h = L_bit // 2
    tpl = _preamble_template(fs) * amp
    pre = np.zeros(L_pre, dtype=np.float32); pre[:len(tpl)] = tpl
    total = L_pre + L_bit*len(bits)
    w = np.zeros(total, dtype=np.float32); w[:L_pre] = pre
    pos = L_pre
    for b in bits:
        if b == 1: w[pos:pos+h] = amp
        else:      w[pos+h:pos+2*h] = amp
        pos += L_bit
    return w.astype(np.complex64)

def pad_and_add_noise(x: np.ndarray, snr_db: float = 12.0, pad_front: int = 200, pad_back: int = 800) -> np.ndarray:
    rng = np.random.default_rng(12345)
    x = x.astype(np.complex64)
    s_power = np.mean(np.abs(x)**2) + 1e-12
    n_power = s_power / (10.0**(snr_db/10.0))
    noise = np.sqrt(n_power/2.0) * (rng.standard_normal(x.size) + 1j*rng.standard_normal(x.size))
    y = x + noise.astype(np.complex64)
    return np.concatenate([np.zeros(pad_front, dtype=np.complex64), y, np.zeros(pad_back, dtype=np.complex64)])

def quick_pipeline(iq: IQSamples, up: int = 2, min_snr: float = 6.0, cfg: Optional[DecodeConfig] = None):
    x = dc_remove(iq.samples)
    mag = to_magnitude(x)
    mag_up, fs_up = mag_oversample(mag, iq.fs, up=up)
    dets = detect_preambles(mag_up, fs_up, min_snr=min_snr, up=up)
    frames = decode_frames(mag_up, fs_up, dets, cfg=cfg or DecodeConfig(up=up, min_snr=min_snr))
    return dets, frames, mag_up, fs_up

__all__ = [
    # data
    "IQSamples","Detection","DecodeConfig","Frame",
    # IO/DSP
    "load_iq","load_cu8_iq","load_cf32_iq","dc_remove","to_magnitude","mag_oversample",
    # detect/decode
    "detect_preambles","decode_frames","bits_to_hex","bits_to_int",
    # CRC
    "crc24_compute","parse_modes_header","verify_crc_or_pi",
    # synth/pipeline
    "synth_adsb_bits","bits_to_ppm_wave","pad_and_add_noise","quick_pipeline",
]

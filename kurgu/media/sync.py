"""Audio-correlation sync between camera files and a (long) screen recording.

Strategy (section 5.1):
1. Coarse: correlate onset envelopes (energy derivative at `envelope_rate` Hz) of each camera file against the
   whole screen recording. Peak-to-sidelobe ratio is the confidence; a camera with no confident match has no screen.
2. Fine: refine with raw audio at `refine_rate` Hz in a +-`refine_window_s` window.
3. Drift: probe several windows along the camera file, re-measure the local offset, and fit
   offset(t) = a + b*t (clock drift). Each screen region later re-measures locally with `local_offset`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np


def _fft_correlate_full(hay: np.ndarray, needle: np.ndarray) -> np.ndarray:
    """Equivalent of scipy.signal.correlate(hay, needle, 'full') via numpy FFT (no scipy dependency)."""
    n = len(hay) + len(needle) - 1
    nfft = 1 << (n - 1).bit_length()
    H = np.fft.rfft(hay.astype(np.float64), nfft)
    N = np.fft.rfft(needle[::-1].astype(np.float64), nfft)
    return np.fft.irfft(H * N, nfft)[:n]


def _decimate(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Integer-factor downsampling with a windowed-sinc low-pass (numpy only)."""
    if target >= sr:
        return x.astype(np.float32)
    k = int(round(sr / target))
    taps = 8 * k + 1
    t = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(t / k) / k * np.hamming(taps)
    y = np.convolve(x.astype(np.float64), h, mode="same")
    return y[::k].astype(np.float32)


def onset_envelope(audio: np.ndarray, sr: int, rate: int) -> np.ndarray:
    hop = max(1, sr // rate)
    n = len(audio) // hop * hop
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    fr = audio[:n].reshape(-1, hop)
    e = np.log10(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-10)
    d = np.diff(e, prepend=e[0])
    d = np.maximum(d, 0.0)
    d -= d.mean()
    sd = d.std()
    return (d / sd if sd > 0 else d).astype(np.float32)


def _xcorr_lag(needle: np.ndarray, hay: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Lag (in samples) where `needle` best matches inside `hay` (needle starts at hay[lag]). Returns (lag, psr, corr)."""
    needle = needle - needle.mean()
    hay = hay - hay.mean()
    c = _fft_correlate_full(hay, needle)
    # index k in 'full' corresponds to lag = k - (len(needle) - 1)
    k = int(np.argmax(c))
    lag = k - (len(needle) - 1)
    peak = c[k]
    excl = max(5, len(c) // 2000)
    mask = np.ones_like(c, dtype=bool)
    mask[max(0, k - excl):k + excl + 1] = False
    rest = c[mask]
    psr = float((peak - rest.mean()) / (rest.std() + 1e-12)) if rest.size else 0.0
    return lag, psr, c


@dataclass
class SyncResult:
    camera: str
    screen: str | None
    offset_s: float | None = None          # screen_time = camera_time + offset (at camera t=0)
    drift: float = 0.0                     # seconds of extra offset per second of camera time
    confidence: float = 0.0
    correlation: float = 0.0
    probes: list[dict] = field(default_factory=list)
    note: str = ""

    def screen_time(self, cam_t: float) -> float:
        return cam_t + self.offset_s + self.drift * cam_t

    def to_dict(self) -> dict:
        return asdict(self)


def _ncc_at(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Pearson correlation of a against b shifted so that a[0] aligns with b[lag] (overlap only)."""
    a0 = max(0, -lag)
    b0 = max(0, lag)
    n = min(len(a) - a0, len(b) - b0)
    if n < 10:
        return 0.0
    x = a[a0:a0 + n].astype(np.float64)
    y = b[b0:b0 + n].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(x, np.ones(n) / n, mode="same") if n > 1 else x


def coarse_offset(cam: np.ndarray, scr: np.ndarray, sr: int, rate: int) -> tuple[float, float, float]:
    """Returns (offset_s, peak-to-sidelobe ratio, normalized correlation at the peak).

    The correlation is measured on 100 ms-smoothed envelopes so small clock drift does not destroy it."""
    ec = onset_envelope(cam, sr, rate)
    es = onset_envelope(scr, sr, rate)
    k = max(1, rate // 10)
    if len(ec) > len(es):
        # camera longer than screen: find screen inside camera and negate
        lag, psr, _ = _xcorr_lag(es, ec)
        return -lag / rate, psr, _ncc_at(_smooth(es, k), _smooth(ec, k), lag)
    lag, psr, _ = _xcorr_lag(ec, es)
    return lag / rate, psr, _ncc_at(_smooth(ec, k), _smooth(es, k), lag)


def refine_offset(cam: np.ndarray, scr: np.ndarray, sr: int, coarse: float, cam_start: float, length: float,
                  window: float, refine_rate: int) -> tuple[float, float]:
    """Refine offset for camera segment [cam_start, cam_start+length) around `coarse`."""
    def rs(x):
        return _decimate(x, sr, refine_rate) if refine_rate != sr else x

    a0 = int(cam_start * sr)
    a1 = int(min(len(cam), (cam_start + length) * sr))
    needle = cam[a0:a1]
    s0 = int((cam_start + coarse - window) * sr)
    s1 = int((cam_start + coarse + length + window) * sr)
    if s0 < 0 or s1 > len(scr) or len(needle) < sr:
        s0c, s1c = max(0, s0), min(len(scr), s1)
        if s1c - s0c < len(needle):
            return coarse, 0.0
        s0, s1 = s0c, s1c
    hay = scr[s0:s1]
    n = rs(needle)
    h = rs(hay)
    if len(h) < len(n):
        return coarse, 0.0
    lag, psr, _ = _xcorr_lag(n, h)
    screen_at = s0 / sr + lag / refine_rate
    return screen_at - cam_start, psr


def sync_camera_to_screen(cam: np.ndarray, scr: np.ndarray, sr: int, cfg: dict, cam_name: str = "", scr_name: str = "") -> SyncResult:
    res = SyncResult(camera=cam_name, screen=scr_name)
    coarse, psr, ncc = coarse_offset(cam, scr, sr, cfg["envelope_rate"])
    res.confidence = psr
    res.correlation = ncc
    if psr < cfg["min_confidence"] or ncc < cfg["min_correlation"]:
        res.screen = None
        res.offset_s = None
        res.note = (f"Ekran kaydında eşleşme bulunamadı (tepe oranı {psr:.1f}, korelasyon {ncc:.2f}; "
                    f"eşik {cfg['min_confidence']}/{cfg['min_correlation']})")
        return res
    cam_len = len(cam) / sr
    every = cfg["drift_probe_every_s"]
    plen = min(cfg["drift_probe_len_s"], cam_len)
    starts = list(np.arange(0.0, max(0.0, cam_len - plen) + 1e-6, every)) or [0.0]
    if cam_len - plen - starts[-1] > every * 0.3:
        starts.append(cam_len - plen)
    for st in starts:
        off, c = refine_offset(cam, scr, sr, coarse, float(st), plen, cfg["refine_window_s"], cfg["refine_rate"])
        res.probes.append({"cam_t": float(st + plen / 2), "offset": float(off), "confidence": float(c)})
    good = [p for p in res.probes if p["confidence"] >= cfg["min_confidence"]]
    if not good:
        res.offset_s = coarse
        res.note = "İnce ayar başarısız, kaba senkron kullanıldı"
        return res
    if len(good) >= 3:
        t = np.array([p["cam_t"] for p in good])
        o = np.array([p["offset"] for p in good])
        b, a = np.polyfit(t, o, 1)
        resid = o - (a + b * t)
        keep = np.abs(resid) < max(0.02, 3 * np.median(np.abs(resid)) + 1e-4)
        if keep.sum() >= 3:
            b, a = np.polyfit(t[keep], o[keep], 1)
        res.offset_s, res.drift = float(a), float(b)
    else:
        res.offset_s = float(np.median([p["offset"] for p in good]))
        res.drift = 0.0
    res.note = f"{len(good)}/{len(res.probes)} ölçüm, kayma {res.drift * 3600:.3f} sn/saat"
    return res


def local_offset(cam: np.ndarray, scr: np.ndarray, sr: int, sync: SyncResult, cam_t0: float, cam_t1: float, cfg: dict) -> tuple[float, float]:
    """Re-measure the offset locally for a screen region (camera time range)."""
    length = max(2.0, min(cam_t1 - cam_t0, 30.0))
    mid = (cam_t0 + cam_t1) / 2
    start = max(0.0, mid - length / 2)
    predicted = sync.offset_s + sync.drift * mid
    return refine_offset(cam, scr, sr, predicted, start, length, 0.25, cfg["refine_rate"])

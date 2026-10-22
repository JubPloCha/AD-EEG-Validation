from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import h5py, numpy as np, scipy
from scipy import signal

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0),
         "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
CHANNELS = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
            "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8"]
PAIRS = [(i, j) for i in range(len(CHANNELS)) for j in range(i + 1, len(CHANNELS))]

@dataclass(frozen=True)
class FeatureConfig:
    fs_expected: float = 500.0
    source_window_sec: float = 4.0
    effective_window_sec: float = 1.0
    minimum_effective_sec: float = 60.0
    min_repeats_for_wpli: int = 8
    robust_clip_lower: float = 0.01
    robust_clip_upper: float = 0.99
    feature_version: str = "cpca_ad_nc_final_0.1"
    confirmed_rest_reference: bool = False


Config = FeatureConfig


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()

def matlab_char(f: h5py.File, key: str) -> str:
    try:
        a = np.asarray(f[key][...])
        if a.dtype == np.uint16:
            return "".join(chr(int(v)) for v in a.reshape(-1))
        return safe_text(a.squeeze())
    except Exception:
        return ""

def scalar(f: h5py.File, key: str, default=np.nan) -> float:
    try:
        return float(np.asarray(f[key][...]).squeeze())
    except Exception:
        return default

def deref_text(f: h5py.File, ref: Any) -> str:
    try:
        obj = f[ref]
        a = np.asarray(obj[...])
        if a.dtype == np.uint16:
            return "".join(chr(int(v)) for v in a.reshape(-1))
        return safe_text(a.squeeze())
    except Exception:
        return ""

def h5_chan_labels(f: h5py.File) -> list[str]:
    labels = []
    try:
        for ref in np.asarray(f["chanlocs/labels"][...]).reshape(-1):
            labels.append(deref_text(f, ref))
    except Exception:
        return []
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    normalized = [aliases.get(x.strip().upper(), x.strip().upper()) for x in labels]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Duplicate channel labels after alias mapping: {normalized}")
    return normalized

def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.98)
    lo = max(lo, 0.1)
    if hi <= lo or x.shape[-1] < 16:
        return x
    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    try:
        return signal.sosfiltfilt(sos, x, axis=-1)
    except ValueError:
        return signal.sosfilt(sos, x, axis=-1)

def hjorth_complexity(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4 or np.nanstd(x) == 0:
        return np.nan
    d1 = np.diff(x); d2 = np.diff(d1)
    v0, v1, v2 = np.var(x), np.var(d1), np.var(d2)
    if v0 <= 0 or v1 <= 0:
        return np.nan
    return float(np.sqrt(v2 / v1) / np.sqrt(v1 / v0))

def debiased_squared_wpli(im: np.ndarray) -> float:
    """Vinck-style debiased squared wPLI from repeated spectral estimates."""
    im = np.asarray(im, dtype=float)
    im = im[np.isfinite(im)]
    if im.size < 2:
        return np.nan
    s1 = np.sum(im)
    sabs = np.sum(np.abs(im))
    s2 = np.sum(im ** 2)
    den = sabs ** 2 - s2
    if den <= 0:
        return np.nan
    val = (s1 ** 2 - s2) / den
    return float(np.clip(val, -1.0, 1.0))

def _deref_array(f: h5py.File, ref: Any) -> np.ndarray:
    try:
        return np.asarray(f[ref][...]).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)

def _event_value(f: h5py.File, ref: Any) -> Any:
    a = _deref_array(f, ref)
    if a.size == 0:
        return ""
    if a.dtype.kind in "iu" and a.size > 1:
        try:
            return "".join(chr(int(v)) for v in a)
        except Exception:
            pass
    if a.size == 1:
        return float(a[0]) if np.issubdtype(a.dtype, np.number) else safe_text(a[0])
    return safe_text(a[0])

def _read_event_audit(f: h5py.File) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "event" not in f or not isinstance(f["event"], h5py.Group):
        return out
    eg = f["event"]
    lat = eg.get("latency")
    typ = eg.get("type")
    dur = eg.get("duration")
    if lat is None:
        return out
    refs = np.asarray(lat[...]).reshape(-1)
    for i, ref in enumerate(refs):
        try:
            latency = float(_event_value(f, ref))
        except Exception:
            continue
        event_type = ""
        if typ is not None and i < typ.size:
            event_type = safe_text(_event_value(f, np.asarray(typ[...]).reshape(-1)[i]))
        duration = np.nan
        if dur is not None and i < dur.size:
            try:
                duration = float(_event_value(f, np.asarray(dur[...]).reshape(-1)[i]))
            except Exception:
                duration = np.nan
        out.append({"index": i, "latency_matlab": latency,
                    "type": event_type, "duration_samples": duration})
    return out

def read_eeglab_hdf5(set_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read EEGLAB v7.3 data and event metadata without modifying the input."""
    with h5py.File(set_path, "r") as f:
        nbchan = int(round(scalar(f, "nbchan")))
        pnts = int(round(scalar(f, "pnts")))
        trials = int(round(scalar(f, "trials")))
        fs = scalar(f, "srate")
        labels = h5_chan_labels(f)
        data_ref = matlab_char(f, "data")
        datfile = matlab_char(f, "datfile")
        hist = matlab_char(f, "history")
        ref = matlab_char(f, "ref")
        comments = matlab_char(f, "comments")
        events = _read_event_audit(f)
    fdt_path = set_path.with_suffix(".fdt")
    if not fdt_path.exists():
        raise FileNotFoundError(f"Missing paired .fdt: {fdt_path.name}")
    expected = nbchan * pnts * trials
    nbytes = fdt_path.stat().st_size
    if nbytes != expected * 4:
        raise ValueError(f"fdt size mismatch: expected {expected * 4}, got {nbytes}")
    raw = np.fromfile(fdt_path, dtype="<f4")
    data = raw.reshape((nbchan, pnts, trials), order="F")
    meta = {"nbchan": nbchan, "pnts": pnts, "trials": trials, "srate": fs,
            "labels": labels, "data_ref": data_ref, "datfile": datfile,
            "history": hist, "ref": ref, "comments": comments,
            "events": events, "boundary_events": [e for e in events
                if "bound" in safe_text(e.get("type")).lower()],
            "set_sha256": sha256_file(set_path),
            "fdt_sha256": sha256_file(fdt_path)}
    return data.astype(np.float64, copy=False), meta

def _continuous_intervals(n_samples: int, events: list[dict[str, Any]]) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    """Split the current data at EEGLAB boundaries without deleting samples.

    EEGLAB ``boundary.duration`` describes data removed before the current
    dataset was saved.  It must therefore never be subtracted from the array
    being analysed.  MATLAB latency is converted to a zero-based cut with
    floor(L-0.5); half-sample and endpoint latencies are handled explicitly.
    """
    cuts: list[tuple[int, dict[str, Any]]] = []
    audit: list[dict[str, Any]] = []
    for e in events:
        if "bound" not in safe_text(e.get("type")).lower():
            continue
        raw = e.get("latency_matlab", np.nan)
        try:
            latency = float(raw)
        except Exception:
            latency = np.nan
        if not np.isfinite(latency):
            audit.append({**e, "valid": False, "invalid_reason": "nonfinite_latency"})
            continue
        cut = int(np.floor(latency - 0.5 + 1e-12))
        if cut < 0 or cut > n_samples:
            audit.append({**e, "valid": False, "invalid_reason": "latency_out_of_range",
                          "latency_zero_based": cut})
            continue
        row = {**e, "valid": True, "latency_zero_based": cut,
               "duration_ignored": e.get("duration_samples", np.nan)}
        audit.append(row); cuts.append((cut, row))
    cuts.sort(key=lambda z: z[0])
    unique: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for cut, row in cuts:
        if cut not in seen:
            unique.append((cut, row)); seen.add(cut)
        else:
            row["duplicate_cut"] = True
    intervals: list[tuple[int, int]] = []
    points = [0] + [c for c, _ in unique] + [n_samples]
    for left, right in zip(points[:-1], points[1:]):
        if right > left:
            intervals.append((left, right))
    for cut, row in unique:
        row["end_zero_based"] = cut
    return intervals, audit

def _make_segments(data: np.ndarray, meta: dict[str, Any], cfg: Config) -> tuple[np.ndarray, list[dict[str, Any]]]:
    fs = float(meta["srate"]); n_ep = data.shape[2]
    target_n = max(1, int(round(cfg.effective_window_sec * fs)))
    segments: list[np.ndarray] = []
    windows: list[dict[str, Any]] = []
    if n_ep > 1:
        # Original epochs are independent.  Boundary latencies in EEGLAB are
        # converted from concatenated epoch coordinates into epoch-local cuts.
        n = data.shape[1]
        all_events = meta.get("events", [])
        audit_all: list[dict[str, Any]] = []
        for ep in range(n_ep):
            offset = ep * n
            local_events = []
            for e in all_events:
                l = float(e.get("latency_matlab", np.nan)) if np.isfinite(e.get("latency_matlab", np.nan)) else np.nan
                if np.isfinite(l):
                    z = l - 0.5 - offset
                    if -0.5 <= z <= n + 0.5:
                        local_events.append({**e, "latency_matlab": l - offset})
            intervals, audit = _continuous_intervals(n, local_events)
            audit_all.extend([{**a, "epoch": ep + 1} for a in audit])
            if not intervals:
                intervals = [(0, n)]
            for left, right in intervals:
                for start in range(left, right - target_n + 1, target_n):
                    stop = start + target_n
                    segments.append(data[:, start:stop, ep])
                    windows.append({"epoch": ep + 1, "start": start, "stop": stop,
                                    "interval_left": left, "interval_right": right,
                                    "source": "original_epoch"})
        meta["_boundary_audit"] = audit_all
    else:
        intervals, audit = _continuous_intervals(data.shape[1], meta.get("events", []))
        meta["_boundary_audit"] = audit
        if not intervals:
            intervals = [(0, data.shape[1])]
        for left, right in intervals:
            for start in range(left, right - target_n + 1, target_n):
                stop = start + target_n
                segments.append(data[:, start:stop, 0])
                windows.append({"epoch": 1, "start": start, "stop": start + target_n,
                                "interval_left": left, "interval_right": right,
                                "source": "continuous_interval"})
    if not segments:
        raise ValueError("No usable within-boundary windows")
    return np.stack(segments, axis=0), windows

def _one_sided_psd(segs: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed one-sided PSD in V^2/Hz, matching the cross-spectrum window."""
    n = segs.shape[-1]
    x = segs - np.nanmean(segs, axis=-1, keepdims=True)
    win = signal.windows.hann(n, sym=False)
    scale = fs * np.sum(win ** 2)
    fft = np.fft.rfft(x * win[None, None, :], axis=-1)
    psd = (np.abs(fft) ** 2) / max(scale, np.finfo(float).tiny)
    if n > 1:
        if n % 2 == 0:
            psd[..., 1:-1] *= 2.0
        else:
            psd[..., 1:] *= 2.0
    return np.fft.rfftfreq(n, d=1.0 / fs), psd

def _integrate_band(freqs: np.ndarray, values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Integrate a discrete PSD on [lo, hi] with linearly interpolated edges."""
    lo = float(max(lo, freqs[0])); hi = float(min(hi, freqs[-1]))
    if hi <= lo:
        return np.full(values.shape[:-1], np.nan)
    inside = (freqs > lo) & (freqs < hi)
    fi = np.concatenate(([lo], freqs[inside], [hi]))
    left = np.take(values, [int(np.searchsorted(freqs, lo, side="left"))], axis=-1)
    right = np.take(values, [int(np.searchsorted(freqs, hi, side="left"))], axis=-1)
    # np.interp is applied by a small vectorized linear interpolation helper.
    def interp_edge(x: float) -> np.ndarray:
        if x <= freqs[0]: return values[..., 0]
        if x >= freqs[-1]: return values[..., -1]
        j = int(np.searchsorted(freqs, x)); f0, f1 = freqs[j - 1], freqs[j]
        w = (x - f0) / max(f1 - f0, np.finfo(float).tiny)
        return values[..., j - 1] * (1 - w) + values[..., j] * w
    vi = np.concatenate((interp_edge(lo)[..., None], values[..., inside], interp_edge(hi)[..., None]), axis=-1)
    return np.trapezoid(vi, fi, axis=-1)

def extract_subject_features(set_path: Path, cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    data, meta = read_eeglab_hdf5(set_path)
    fs = float(meta["srate"])
    if str(meta.get("ref", "")).strip().upper() != "REST" and not cfg.confirmed_rest_reference:
        raise ValueError("Expected already processed REST-referenced EEG; no rereferencing is performed")
    if not np.isfinite(fs) or abs(fs - cfg.fs_expected) > 1e-6:
        raise ValueError(f"Unexpected sampling rate {fs}")
    labels = meta["labels"]
    if labels != CHANNELS:
        missing = [c for c in CHANNELS if c not in labels]
        if missing:
            raise ValueError(f"Channel mismatch; missing required channels {missing}; received {labels}")
        idx = [labels.index(c) for c in CHANNELS]
        data = data[idx, :, :]
    segs, windows = _make_segments(data, meta, cfg)
    p99 = float(np.nanpercentile(np.abs(segs), 99))
    if not np.isfinite(p99) or p99 < 1e-3 or p99 > 1e6:
        raise ValueError(f"Signal units/amplitude incompatible with EEGLAB microvolts: abs_p99={p99}")
    duration = float(sum(w["stop"] - w["start"] for w in windows) / fs)
    if duration < cfg.minimum_effective_sec:
        raise ValueError(f"Effective duration below {cfg.minimum_effective_sec}s: {duration:.2f}s")
    freqs, spectra = _one_sided_psd(segs, fs)
    # Use exactly the same detrending/windowed FFT for power and connectivity.
    x = segs - np.nanmean(segs, axis=-1, keepdims=True)
    win = signal.windows.hann(segs.shape[-1], sym=False)
    complex_spectra = np.fft.rfft(x * win[None, None, :], axis=-1)
    all_lo, all_hi = 1.0, min(45.0, fs / 2.0)
    total_power = _integrate_band(freqs, spectra, all_lo, all_hi)
    feat: dict[str, Any] = {}
    connectivity: dict[str, np.ndarray] = {}
    freq_bins: dict[str, list[float]] = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        freq_bins[band] = freqs[mask].astype(float).tolist()
        if not mask.any():
            raise ValueError(f"No FFT bins for {band}")
        bp = _integrate_band(freqs, spectra, lo, hi)
        abs_log = np.log10(np.maximum(bp, 1e-30)).mean(axis=0)
        rel = bp / np.maximum(total_power, 1e-30)
        rel_log = np.log10(np.maximum(rel, 1e-30)).mean(axis=0)
        for ci, ch in enumerate(CHANNELS):
            feat[f"power_log10_{band}_{ch}"] = float(abs_log[ci])
            feat[f"relative_log10_{band}_{ch}"] = float(rel_log[ci])
        band_psd = spectra[:, :, mask]
        probs = band_psd / np.maximum(band_psd.sum(axis=-1, keepdims=True), 1e-30)
        ent = -(probs * np.log(np.maximum(probs, 1e-30))).sum(axis=-1) / np.log(max(2, mask.sum()))
        for ci, ch in enumerate(CHANNELS):
            feat[f"spectral_entropy_{band}_{ch}"] = float(np.nanmean(ent[:, ci]))
        conn = np.full(len(PAIRS), np.nan)
        for pi, (i, j) in enumerate(PAIRS):
            ims = np.imag(complex_spectra[:, i, mask] * np.conj(complex_spectra[:, j, mask]))
            if ims.shape[0] >= cfg.min_repeats_for_wpli:
                vals = np.asarray([debiased_squared_wpli(ims[:, fi]) for fi in range(ims.shape[1])])
                conn[pi] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        connectivity[band] = conn
    for band, conn in connectivity.items():
        for pi, (i, j) in enumerate(PAIRS):
            feat[f"dwpli2_{band}_{CHANNELS[i]}__{CHANNELS[j]}"] = float(conn[pi])
    for ci, ch in enumerate(CHANNELS):
        feat[f"hjorth_complexity_broad_{ch}"] = float(np.nanmean([hjorth_complexity(s[ci]) for s in segs]))
    for band, (lo, hi) in BANDS.items():
        filtered = bandpass(segs, fs, lo, hi)
        for ci, ch in enumerate(CHANNELS):
            feat[f"hjorth_complexity_{band}_{ch}"] = float(np.nanmean([hjorth_complexity(s[ci]) for s in filtered]))
    amp = np.abs(segs)
    fdt_path = set_path.with_suffix(".fdt")
    boundary_used = list(meta.get("_boundary_audit", [])) or [e for e in meta.get("events", []) if "bound" in safe_text(e.get("type")).lower()]
    valid_boundary = [e for e in boundary_used if e.get("valid", True)]
    interval_lengths = []
    if data.shape[2] == 1:
        interval_lengths = [b - a for a, b in _continuous_intervals(data.shape[1], meta.get("events", []))[0]]
    else:
        interval_lengths = [data.shape[1]] * data.shape[2]
    tail_samples = int(sum(n % int(round(cfg.effective_window_sec * fs)) for n in interval_lengths))
    windows_cross_boundary = any(
        (w["start"] < w.get("interval_left", w["start"])) or
        (w["stop"] > w.get("interval_right", w["stop"])) for w in windows)
    v2_wrong_extra_samples = int(sum(
        max(0, int(round(float(e.get("duration_samples")))))
        for e in valid_boundary
        if np.isfinite(e.get("duration_samples", np.nan))))
    feat_meta = {"effective_repeats": int(segs.shape[0]), "effective_duration_sec": duration,
                 "source_pnts": int(data.shape[1]), "source_trials": int(data.shape[2]),
                 "window_target_sec": cfg.source_window_sec,
                 "effective_window_sec": float(segs.shape[-1] / fs), "srate": fs,
                 "window_frequency_resolution_hz": float(fs / segs.shape[-1]),
                 "frequency_bins_json": json.dumps(freq_bins, ensure_ascii=False),
                 "window_rule": "non-overlapping within original epoch or EEGLAB continuous interval",
                 "boundary_event_count": len(boundary_used),
                 "valid_boundary_event_count": len(valid_boundary),
                 "invalid_boundary_event_count": int(sum(not e.get("valid", True) for e in boundary_used)),
                 "boundary_events_json": json.dumps(boundary_used, ensure_ascii=False),
                 "windows_cross_boundary": bool(windows_cross_boundary),
                 "tail_samples_after_windowing": tail_samples,
                 "v2_wrong_extra_boundary_samples": v2_wrong_extra_samples,
                 "v2_wrong_extra_boundary_sec": float(v2_wrong_extra_samples / fs),
                 "channel_labels": CHANNELS,
                 "ref": meta.get("ref", ""), "history": meta.get("history", ""),
                 "reference_evidence": "explicit user/source confirmation" if cfg.confirmed_rest_reference else "saved EEG.ref metadata",
                 "comments": meta.get("comments", ""),
                 "ica_history_present": bool(re.search(r"ica|icac|pop_subcomp", meta.get("history", ""), re.I)),
                 "bad_segment_rule": "none added by v2; source EEGLAB history retained",
                 "near_constant_channel_count": int(np.sum(np.nanstd(segs.reshape(-1, segs.shape[-1]), axis=1) < 1e-12)),
                 "amplitude_abs_p99": float(np.nanpercentile(amp, 99)),
                 "amplitude_abs_max": float(np.nanmax(amp)),
                 "window_variance_median": float(np.nanmedian(np.var(segs, axis=-1))),
                 "set_sha256": meta["set_sha256"], "fdt_sha256": meta["fdt_sha256"],
                 "feature_code_version": cfg.feature_version,
                 "dependency_versions": json.dumps({"numpy": np.__version__, "scipy": scipy.__version__}, ensure_ascii=False)}
    return feat, feat_meta

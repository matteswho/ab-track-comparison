#!/usr/bin/env python3
"""
Mastering Track Analyzer
Analysiert WAV, MP3, FLAC, AAC und gibt JSON aus.
Verwendung:
  python3 analyze_track.py mein_track.wav
  python3 analyze_track.py track1.wav track2.mp3   (mehrere Tracks)
  python3 analyze_track.py *.wav                    (Wildcard)
"""

import sys
import os
import json
import subprocess
import tempfile
import struct
import numpy as np
from scipy import signal as scipy_signal

# ─── Audio laden via ffmpeg ──────────────────────────────────────────────────

def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """Lädt beliebige Audiodatei via ffmpeg als float32 PCM (stereo)."""
    cmd = [
        "ffmpeg", "-i", filepath,
        "-f", "f32le",       # 32-bit float little-endian raw PCM
        "-ar", "48000",      # EBU R128 braucht 48 kHz
        "-ac", "2",          # immer stereo (mono wird dupliziert)
        "-v", "error",
        "pipe:1"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg Fehler: {result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if len(audio) == 0:
        raise RuntimeError("Keine Audiodaten gelesen – Datei leer oder Format unbekannt.")
    audio = audio.reshape(-1, 2)  # (samples, 2)
    return audio, 48000


def get_audio_metadata(filepath: str) -> dict:
    """Liest Metadaten via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


# ─── EBU R128 LUFS (ITU-R BS.1770-4) ───────────────────────────────────────

def design_k_weighting_filters(sr: int):
    """K-Weighting: Stage 1 (High-Shelf) + Stage 2 (High-Pass)."""
    # Stage 1: Pre-filter (High-shelf, +4 dB bei 1681 Hz)
    f0, Q, V0 = 1681.974450955533, 0.7071752369554196, 3.999843853973347
    Kf = np.tan(np.pi * f0 / sr)
    b0 = (V0 + np.sqrt(2 * V0) * Kf + Kf**2) / (1 + np.sqrt(2) * Kf + Kf**2)
    b1 = 2 * (Kf**2 - V0) / (1 + np.sqrt(2) * Kf + Kf**2)
    b2 = (V0 - np.sqrt(2 * V0) * Kf + Kf**2) / (1 + np.sqrt(2) * Kf + Kf**2)
    a1 = 2 * (Kf**2 - 1) / (1 + np.sqrt(2) * Kf + Kf**2)
    a2 = (1 - np.sqrt(2) * Kf + Kf**2) / (1 + np.sqrt(2) * Kf + Kf**2)
    hs_b, hs_a = [b0, b1, b2], [1.0, a1, a2]

    # Stage 2: High-pass 38 Hz (Butterworth 2. Ordnung)
    f_hp = 38.13547087602444
    Kf2 = np.tan(np.pi * f_hp / sr)
    b0 = 1 / (1 + np.sqrt(2) * Kf2 + Kf2**2)
    b1 = -2 * b0
    b2 = b0
    a1 = 2 * (Kf2**2 - 1) / (1 + np.sqrt(2) * Kf2 + Kf2**2)
    a2 = (1 - np.sqrt(2) * Kf2 + Kf2**2) / (1 + np.sqrt(2) * Kf2 + Kf2**2)
    hp_b, hp_a = [b0, b1, b2], [1.0, a1, a2]

    return (hs_b, hs_a), (hp_b, hp_a)


def k_weight(audio: np.ndarray, sr: int) -> np.ndarray:
    """Wendet K-Weighting auf alle Kanäle an."""
    (hs_b, hs_a), (hp_b, hp_a) = design_k_weighting_filters(sr)
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        x = scipy_signal.lfilter(hs_b, hs_a, audio[:, ch])
        x = scipy_signal.lfilter(hp_b, hp_a, x)
        out[:, ch] = x
    return out


def compute_lufs(audio: np.ndarray, sr: int) -> tuple[float, float, float]:
    """
    Berechnet Integrated LUFS, LRA und Short-Term LUFS (max).
    Gibt (integrated_lufs, lra, shortterm_max) zurück.
    """
    weighted = k_weight(audio, sr)

    block_size   = int(0.4 * sr)   # 400 ms Block
    hop_size     = int(0.1 * sr)   # 100 ms Hop (75% Overlap)
    min_block    = int(3.0 * sr)   # 3 s für Short-Term

    # Momentary blocks (400 ms)
    blocks = []
    for start in range(0, len(weighted) - block_size + 1, hop_size):
        block = weighted[start:start + block_size]
        mean_sq = np.mean(block**2, axis=0)
        # Channel-Summe (L + R, surround-Gewichtung irrelevant bei Stereo)
        loudness = -0.691 + 10 * np.log10(np.sum(mean_sq) + 1e-10)
        blocks.append(loudness)

    if not blocks:
        return -70.0, 0.0, -70.0

    blocks = np.array(blocks)

    # Absolute gate (-70 LUFS)
    gated_abs = blocks[blocks >= -70.0]
    if len(gated_abs) == 0:
        return -70.0, 0.0, -70.0

    # Relative gate (-10 LU unter ungegated Mittel)
    gamma_r = np.mean(10**((gated_abs + 0.691) / 10))
    gamma_r_lufs = -0.691 + 10 * np.log10(gamma_r + 1e-10) - 10.0
    gated_rel = gated_abs[gated_abs >= gamma_r_lufs]

    if len(gated_rel) == 0:
        gated_rel = gated_abs

    integrated = -0.691 + 10 * np.log10(np.mean(10**((gated_rel + 0.691) / 10)) + 1e-10)

    # LRA: Short-Term 3s blocks
    st_blocks = []
    for start in range(0, len(weighted) - min_block + 1, hop_size):
        block = weighted[start:start + min_block]
        mean_sq = np.mean(block**2, axis=0)
        loudness = -0.691 + 10 * np.log10(np.sum(mean_sq) + 1e-10)
        st_blocks.append(loudness)

    if st_blocks:
        st = np.array(st_blocks)
        st_gated = st[st >= -70.0]
        if len(st_gated) > 1:
            lra = float(np.percentile(st_gated, 95) - np.percentile(st_gated, 10))
            lra = max(0.0, lra)
        else:
            lra = 0.0
        st_max = float(np.max(st_gated)) if len(st_gated) > 0 else -70.0
    else:
        lra = 0.0
        st_max = -70.0

    return float(integrated), float(lra), float(st_max)


# ─── True Peak ──────────────────────────────────────────────────────────────

def compute_true_peak(audio: np.ndarray, sr: int) -> float:
    """True Peak mit 4x Oversampling."""
    oversampled = scipy_signal.resample_poly(audio, 4, 1)
    peak = np.max(np.abs(oversampled))
    return float(20 * np.log10(peak + 1e-10))


# ─── Spektrale Analyse ───────────────────────────────────────────────────────

def compute_spectral_balance(audio: np.ndarray, sr: int) -> dict:
    """
    FFT-basierte Spektralbalance, gemittelt über den ganzen Track.
    Gibt relative Energie in dB für mehrere Bänder zurück.
    """
    mono = audio.mean(axis=1)

    # Durchschnittliches Spektrum via Welch
    freqs, psd = scipy_signal.welch(mono, fs=sr, nperseg=4096, noverlap=2048)
    psd_db = 10 * np.log10(psd + 1e-20)

    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        if not np.any(mask):
            return -60.0
        return float(np.mean(psd_db[mask]))

    bands = {
        "sub_bass":   band_energy(20,   60),    # Sub (Kick-Fundament)
        "low_bass":   band_energy(60,   200),   # Low-End / Bass
        "low_mids":   band_energy(200,  800),   # Low-Mids (Wärme / Matsch)
        "mids":       band_energy(800,  2000),  # Mids (Körper)
        "high_mids":  band_energy(2000, 6000),  # Presence / Screams / Snare
        "presence":   band_energy(3000, 6000),  # Engerer Presence-Bereich
        "air":        band_energy(8000, 16000), # Air / Brillanz
        "ultra_air":  band_energy(16000, 22000),# Ultra-High
    }

    # Normalisiert auf Referenz: Low-Mids gelten als "0 dB" Referenz
    ref = bands["low_mids"]
    bands_norm = {k: round(v - ref, 2) for k, v in bands.items()}

    return {"absolute": {k: round(v, 2) for k, v in bands.items()},
            "relative": bands_norm}


# ─── Stereobreite ────────────────────────────────────────────────────────────

def compute_stereo_width(audio: np.ndarray) -> dict:
    """
    Stereokorrelation und Breite.
    correlation: +1 = mono, 0 = unkorrelliert, -1 = phasenproblem
    width: 0–100 Skala für UI
    """
    L, R = audio[:, 0], audio[:, 1]
    if np.std(L) < 1e-8 or np.std(R) < 1e-8:
        return {"correlation": 1.0, "width_score": 0, "mid_side_ratio": 0.0}

    corr = float(np.corrcoef(L, R)[0, 1])
    mid  = (L + R) / 2
    side = (L - R) / 2
    rms_mid  = float(np.sqrt(np.mean(mid**2)))
    rms_side = float(np.sqrt(np.mean(side**2)))
    ms_ratio = float(rms_side / (rms_mid + 1e-10))

    # Width-Score 0–100: bei corr=1 → 0, bei corr=0 → ~70, bei negativ → >85
    width_score = int(np.clip((1 - corr) * 70, 0, 100))

    return {
        "correlation": round(corr, 3),
        "width_score": width_score,
        "mid_side_ratio": round(ms_ratio, 3)
    }


# ─── Peak / RMS ─────────────────────────────────────────────────────────────

def compute_dynamics(audio: np.ndarray) -> dict:
    """Sample Peak, RMS Lautheit."""
    peak_linear = float(np.max(np.abs(audio)))
    peak_db = float(20 * np.log10(peak_linear + 1e-10))
    rms = float(np.sqrt(np.mean(audio**2)))
    rms_db = float(20 * np.log10(rms + 1e-10))
    crest = peak_db - rms_db
    return {
        "peak_db":    round(peak_db, 2),
        "rms_db":     round(rms_db, 2),
        "crest_factor": round(crest, 2)
    }


# ─── Hauptfunktion ───────────────────────────────────────────────────────────

def analyze_file(filepath: str) -> dict:
    print(f"  Lade:       {os.path.basename(filepath)}", flush=True)
    audio, sr = load_audio(filepath)
    duration = len(audio) / sr

    print(f"  Dauer:      {duration:.1f}s @ {sr} Hz | {audio.shape[1]}ch", flush=True)
    print(f"  LUFS ...", end="", flush=True)
    lufs, lra, st_max = compute_lufs(audio, sr)
    print(f" {lufs:.1f}", flush=True)

    print(f"  True Peak ..", end="", flush=True)
    true_peak = compute_true_peak(audio, sr)
    print(f" {true_peak:.1f} dBTP", flush=True)

    print(f"  Spektrum ..", flush=True)
    spectral = compute_spectral_balance(audio, sr)

    print(f"  Stereo ..", flush=True)
    stereo = compute_stereo_width(audio)

    dynamics = compute_dynamics(audio)

    # Metadaten
    meta = get_audio_metadata(filepath)
    fmt = meta.get("format", {})
    streams = meta.get("streams", [{}])
    astream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    result = {
        "file": os.path.basename(filepath),
        "path": os.path.abspath(filepath),
        "duration_s": round(duration, 2),
        "sample_rate": sr,
        "bit_depth": astream.get("bits_per_sample", astream.get("bits_per_raw_sample", "?")),
        "codec": astream.get("codec_name", "?"),
        "format": fmt.get("format_long_name", "?"),
        "loudness": {
            "lufs_integrated":  round(lufs, 2),
            "lra":              round(lra, 2),
            "shortterm_max":    round(st_max, 2),
            "true_peak_dbtp":   round(true_peak, 2),
            "rms_db":           dynamics["rms_db"],
            "peak_db":          dynamics["peak_db"],
            "crest_factor_db":  dynamics["crest_factor"]
        },
        "spectral": spectral,
        "stereo": stereo
    }

    return result


def main():
    files = sys.argv[1:]
    if not files:
        print("Verwendung: python3 analyze_track.py <datei1> [datei2] ...")
        print("Beispiel:   python3 analyze_track.py mein_master.wav referenz.mp3")
        sys.exit(1)

    results = []
    for f in files:
        if not os.path.exists(f):
            print(f"[!] Datei nicht gefunden: {f}")
            continue
        print(f"\n{'='*50}")
        print(f"Analysiere: {f}")
        print('='*50)
        try:
            r = analyze_file(f)
            results.append(r)
            print(f"\n  → LUFS: {r['loudness']['lufs_integrated']} | "
                  f"LRA: {r['loudness']['lra']} LU | "
                  f"True Peak: {r['loudness']['true_peak_dbtp']} dBTP")
        except Exception as e:
            print(f"[FEHLER] {e}")
            results.append({"file": os.path.basename(f), "error": str(e)})

    # JSON ausgeben
    output_path = "track_analysis.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"✓ Fertig! JSON gespeichert: {output_path}")
    print(f"  → Datei im Browser-Tool laden für die Visualisierung.")
    print('='*50)

    # Auch kurze Zusammenfassung in Terminal
    if len(results) >= 2 and all("error" not in r for r in results[:2]):
        a, b = results[0], results[1]
        print(f"\nSchnellvergleich: {a['file']}  vs.  {b['file']}")
        print(f"  LUFS:      {a['loudness']['lufs_integrated']:6.1f}  vs  {b['loudness']['lufs_integrated']:6.1f}  "
              f"(Δ {a['loudness']['lufs_integrated'] - b['loudness']['lufs_integrated']:+.1f})")
        print(f"  LRA:       {a['loudness']['lra']:6.1f}  vs  {b['loudness']['lra']:6.1f}  LU")
        print(f"  True Peak: {a['loudness']['true_peak_dbtp']:6.1f}  vs  {b['loudness']['true_peak_dbtp']:6.1f}  dBTP")
        print(f"  Stereo:    {a['stereo']['width_score']:6d}  vs  {b['stereo']['width_score']:6d}  (0–100)")


if __name__ == "__main__":
    main()

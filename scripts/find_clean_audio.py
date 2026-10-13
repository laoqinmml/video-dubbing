"""Find the cleanest stretch of speech in a clone-source recording.

Cloning quality follows the reference audio, so rather than handing MiniMax the
whole clip this profiles it段 by segment and reports, for each speech run:

  level        how loud the speech is (too quiet = weak clone)
  floor        the noise floor in the pauses around it
  snr          speech level minus that floor - the single most useful number
  flatness     spectral flatness of the pause: high = hiss, low = tonal hum
  hf_ratio     share of energy above 4 kHz: high can mean sibilance or hiss,
               very low means dull/muffled
  clip%        share of samples at/near full scale

Output: work/clone_segments.json
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import FFMPEG, WORK  # noqa: E402

SRC = Path(r"D:\0\0 其他人的站\中山市 三毛电气\FB专用文件（荣惠）2026.9.10"
           r"\2026年1月-7月拍摄视频\个人拜年视频（已发）.mp4")
WAV = WORK / "bn_full_44k.wav"
OUT = WORK / "clone_segments.json"

MIN_SEG = 1.0
MIN_PAUSE = 0.25
GATE_DB = -32.0


def extract() -> Path:
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(SRC),
                    "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(WAV)],
                   check=True)
    return WAV


def load() -> tuple[np.ndarray, int]:
    with wave.open(str(WAV), "rb") as w:
        sr = w.getframerate()
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return d, sr


def db(x: float) -> float:
    return 20 * float(np.log10(max(x, 1e-12)))


def metrics(x: np.ndarray, sr: int, a: float, b: float) -> dict:
    seg = x[int(a * sr):int(b * sr)]
    if seg.size < 512:
        return {}
    rms = float(np.sqrt(np.mean(seg ** 2)))
    peak = float(np.max(np.abs(seg)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    tot = spec.sum() + 1e-20
    p = spec / tot
    flat = float(np.exp(np.mean(np.log(np.maximum(p, 1e-12)))) / max(p.mean(), 1e-12))
    hf = float(spec[freqs > 4000].sum() / tot)
    clip = float(np.mean(np.abs(seg) > 0.985))
    return {"rms_db": db(rms), "peak_db": db(peak), "flatness": flat,
            "hf_ratio": hf, "clip_pct": 100 * clip}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    if not WAV.exists():
        extract()
    x, sr = load()
    print(f"source: {SRC.name}  {len(x)/sr:.2f}s @ {sr} Hz")

    hop = int(0.01 * sr)
    n = len(x) // hop
    env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    env_db = 20 * np.log10(env)

    active = env_db > GATE_DB
    runs, i = [], 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            runs.append([i * 0.01, j * 0.01])
            i = j
        else:
            i += 1
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < MIN_PAUSE:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = [(a, b) for a, b in merged if b - a >= MIN_SEG]

    print(f"\n{'#':>2} {'start':>7} {'end':>7} {'dur':>6} {'level':>7} "
          f"{'floor':>7} {'SNR':>6} {'flat':>6} {'hf%':>5} {'clip%':>6}")

    rows = []
    for k, (a, b) in enumerate(merged, 1):
        m = metrics(x, sr, a, b)
        # noise floor: the quietest 250 ms inside a 0.6 s skirt around the run
        lo = max(0.0, a - 0.6)
        hi = min(len(x) / sr, b + 0.6)
        skirt = x[int(lo * sr):int(hi * sr)]
        w = int(0.25 * sr)
        if skirt.size > w:
            fr = np.array([np.sqrt(np.mean(skirt[i:i + w] ** 2))
                           for i in range(0, skirt.size - w, w // 2)])
            floor = db(float(np.percentile(fr, 10)))
        else:
            floor = m["rms_db"]
        snr = m["rms_db"] - floor
        rows.append({**m, "id": k, "start": round(a, 2), "end": round(b, 2),
                     "dur": round(b - a, 2), "floor_db": round(floor, 1),
                     "snr_db": round(snr, 1)})
        print(f"{k:>2} {a:7.2f} {b:7.2f} {b-a:6.2f} {m['rms_db']:7.1f} "
              f"{floor:7.1f} {snr:6.1f} {m['flatness']:6.3f} "
              f"{100*m['hf_ratio']:5.1f} {m['clip_pct']:6.2f}")

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        best = max(rows, key=lambda r: r["snr_db"])
        print(f"\nhighest SNR segment: #{best['id']} "
              f"{best['start']}-{best['end']}s  SNR {best['snr_db']} dB")
        tot = sum(r["dur"] for r in rows)
        print(f"total speech across segments: {tot:.2f}s")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

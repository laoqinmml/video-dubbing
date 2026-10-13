"""Prepare the New Year greeting video's audio as a voice-clone reference.

The new source is a much better clone reference than the blender clip: 23.7 dB
dynamic range and a -29.4 dB noise floor versus 7.2 dB / -19.7 dB for the
compressed factory mix. It has no music bed (30-100 Hz = 0.1% of energy), so no
source separation is needed - just trim the silence and hand it to MiniMax.

MiniMax requires 10 s - 5 min, so the trimmed length is checked.

Output: work/bn_clone_ref.mp3, work/bn_prompt.mp3, work/bn_speech.json
"""
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import FFMPEG, WORK  # noqa: E402

SRC = WORK / "bn_16k.wav"
CLEAN_WAV = WORK / "bn_clean.wav"
REF_MP3 = WORK / "bn_clone_ref.mp3"
PROMPT_MP3 = WORK / "bn_prompt.mp3"
INFO = WORK / "bn_speech.json"

TRIM_DB = -38.0      # legacy fixed threshold (kept for reference)
PAD = 0.05
PROMPT_LEN = 7.0     # MiniMax wants the prompt clip under 8 s
FLOOR_MARGIN_DB = 8.0   # gate sits this far above the measured noise floor
MAX_TRIM_DB = 45.0      # but never more than this below the peak


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return d, sr


def trim(x: np.ndarray, sr: int):
    """Adaptive silence trim.

    A fixed threshold relative to the peak does not work here: this recording's
    noise floor sits only ~24 dB below its peak, so a -38 dB gate never opens.
    The gate is therefore placed relative to the measured floor instead.
    """
    win = int(0.01 * sr)
    n = len(x) // win
    env = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    env_db = 20 * np.log10(env + 1e-12)
    peak_db = float(np.percentile(env_db, 99))
    floor_db = float(np.percentile(env_db, 10))
    gate_db = max(floor_db + FLOOR_MARGIN_DB, peak_db - MAX_TRIM_DB)

    active = np.where(env_db > gate_db)[0]
    if active.size == 0:
        return 0.0, len(x) / sr, gate_db
    a = max(0, active[0] * win - int(PAD * sr))
    b = min(len(x), (active[-1] + 1) * win + int(PAD * sr))
    print(f"[prep] peak {peak_db:.1f} dB, floor {floor_db:.1f} dB -> gate {gate_db:.1f} dB")
    return a / sr, b / sr, gate_db


def main() -> None:
    x, sr = read_wav(SRC)
    a, b, gate = trim(x, sr)
    dur = b - a
    print(f"[prep] source {len(x)/sr:.2f}s -> speech {a:.2f}s..{b:.2f}s "
          f"({dur:.2f}s)")
    if dur < 10.0:
        raise SystemExit(f"trimmed audio is only {dur:.2f}s; MiniMax needs >= 10s")

    # WAV for local use
    seg = x[int(a * sr): int(b * sr)]
    with wave.open(str(CLEAN_WAV), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((seg * 32767).astype(np.int16).tobytes())

    # Mono 32 kHz mp3 for upload
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(CLEAN_WAV),
                    "-ac", "1", "-ar", "32000", "-b:a", "128k", str(REF_MP3)],
                   check=True)
    # First ~7 s as the cloning prompt clip
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(REF_MP3),
                    "-t", str(PROMPT_LEN), str(PROMPT_MP3)], check=True)

    info = {
        "source": str(SRC),
        "trim_start": round(a, 3),
        "trim_end": round(b, 3),
        "speech_sec": round(dur, 3),
        "ref_mp3": str(REF_MP3),
        "ref_kb": round(REF_MP3.stat().st_size / 1024, 1),
        "prompt_mp3": str(PROMPT_MP3),
        "prompt_sec": PROMPT_LEN,
        "prompt_start_in_source": round(a, 3),
    }
    INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prep] reference {REF_MP3.stat().st_size/1024:.0f} KB, "
          f"prompt {PROMPT_LEN}s -> {REF_MP3.name} / {PROMPT_MP3.name}")
    print(f"[prep] prompt covers source time {a:.2f}s .. {a + PROMPT_LEN:.2f}s")


if __name__ == "__main__":
    main()

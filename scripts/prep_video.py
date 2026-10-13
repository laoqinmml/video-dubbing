"""Stage 0 - prepare one video for the dubbing pipeline.

Produces, inside the per-video work directory:
    orig_stereo.wav      full-quality audio, for demucs and for reference
    orig_16k.wav         mono 16 kHz, what the ASR wants
    vocals_16k.wav       mono 16 kHz of the demucs vocal stem (ASR input)
    total_duration.txt   the video length the dub has to fill

Why demucs before ASR: these clips often carry background music or factory
noise, and the vocal stem measurably improves recognition. It costs about 20 s
per minute of audio on CPU, which is affordable for a batch.

Run:  python scripts/prep_video.py
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import FFMPEG, FFPROBE, SRC_VIDEO, WORK  # noqa: E402

STEREO = WORK / "orig_stereo.wav"
MONO16 = WORK / "orig_16k.wav"
VOCALS = WORK / "vocals_16k.wav"
TOTAL = WORK / "total_duration.txt"
INFO = WORK / "source_info.json"


def probe() -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "stream=codec_type,width,height,r_frame_rate,duration",
         "-show_entries", "format=duration", "-of", "json", str(SRC_VIDEO)],
        capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    j = json.loads(out)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    return {
        "file": SRC_VIDEO.name,
        "duration": round(float(j["format"]["duration"]), 3),
        "width": v.get("width"), "height": v.get("height"),
        "fps": v.get("r_frame_rate"),
        "has_audio": a is not None,
        "size_mb": round(SRC_VIDEO.stat().st_size / 1e6, 1),
    }


def wav_duration(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    if not SRC_VIDEO.exists():
        raise SystemExit(f"missing source: {SRC_VIDEO}")
    info = probe()
    if not info["has_audio"]:
        raise SystemExit(f"{SRC_VIDEO.name}: no audio track")
    print(f"[prep] {info['file']}  {info['duration']}s  "
          f"{info['width']}x{info['height']}  {info['size_mb']} MB")

    if not STEREO.exists():
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(SRC_VIDEO),
                        "-vn", "-ac", "2", "-ar", "44100",
                        "-c:a", "pcm_s16le", str(STEREO)], check=True)
    if not MONO16.exists():
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(STEREO),
                        "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(MONO16)], check=True)
    dur = wav_duration(MONO16)
    TOTAL.write_text(f"{dur:.3f}", encoding="utf-8")
    print(f"[prep] audio {dur:.2f}s -> {MONO16.name}, {TOTAL.name}")

    if not VOCALS.exists():
        outdir = WORK / "sep"
        print("[prep] demucs vocal separation (CPU)…")
        subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals",
                        "-n", "htdemucs", "-d", "cpu", "-o", str(outdir),
                        str(STEREO)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stem = outdir / "htdemucs" / STEREO.stem / "vocals.wav"
        if not stem.exists():
            raise SystemExit(f"demucs produced nothing at {stem}")
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(stem),
                        "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(VOCALS)], check=True)
    print(f"[prep] vocals {wav_duration(VOCALS):.2f}s -> {VOCALS.name}")

    INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("[prep] ready")


if __name__ == "__main__":
    main()

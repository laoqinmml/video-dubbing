"""Stage 7/8 - mux the English voice onto the video + emit SRT files.

Video is stream-copied (no re-encode).

Audio pipeline is deliberately two-stage:
  1. build the mixed bed+voice to a WAV (bed looped/trimmed/faded, ducked under
     the voice with sidechaincompress)
  2. measure THAT mix and only then apply loudnorm while muxing

Doing it in one pass is wrong: the loudnorm measurements would describe the
voice alone while the filter runs on the louder mix, so the target would be hit
only by luck.

BGM ducking matters because this dub is a single mono voice with no headroom -
a bed at a musical level simply swallows the speech.

Output: out/<stem>_EN_<tag>.mp4
"""
import argparse
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import FFMPEG, OUT, SRC_VIDEO, STEM, WORK  # noqa: E402

ALIGNED = WORK / "en_aligned.wav"
REPORT = WORK / "align_report.json"
ZH = WORK / "segments_lines.json"
ORIG_BED = WORK / "sep" / "htdemucs" / "orig_stereo" / "no_vocals.wav"
TOTAL = WORK / "total_duration.txt"


def ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, rows: list[tuple[float, float, str]]) -> None:
    lines = []
    for i, (a, b, text) in enumerate(rows, 1):
        lines.append(f"{i}\n{ts(a)} --> {ts(b)}\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[mux] srt -> {path}  ({len(rows)} cues)")


def measure_loudness(path: Path) -> dict:
    """ffmpeg writes the JSON report to stderr; force UTF-8 because a Chinese
    Windows locale would otherwise decode it as GBK and kill the reader thread."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path),
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    text = proc.stderr or ""
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        raise SystemExit(f"loudnorm measurement failed:\n{text[-800:]}")
    return json.loads(m.group(0))


def build_mix(bed: Path, total: float, gain_db: float, duck: float) -> Path:
    """Voice + ducked, faded, looped music bed -> work/bed_mixed.wav"""
    dst = WORK / "bed_mixed.wav"
    fade_out = max(0.0, total - 1.2)
    fc = (
        f"[1:a]aloop=loop=-1:size=2147483647:start=0,"
        f"atrim=0:{total:.3f},asetpts=N/SR/TB,"
        f"afade=t=in:st=0:d=0.35,afade=t=out:st={fade_out:.3f}:d=1.2,"
        f"volume={gain_db}dB[bed];"
        f"[bed][0:a]sidechaincompress="
        f"threshold={duck}:ratio=8:attack=15:release=350:makeup=1:level_sc=1[bedd];"
        f"[0:a][bedd]amix=inputs=2:duration=first:normalize=0[out]"
    )
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-i", str(ALIGNED), "-i", str(bed),
         "-filter_complex", fc, "-map", "[out]", "-ar", "44100", "-ac", "2",
         str(dst)],
        check=True,
    )
    return dst


def window_db(path: Path, a: float, b: float) -> float:
    """RMS level of a window, for checking the music actually sits under speech."""
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        w.setpos(int(a * sr))
        raw = w.readframes(int((b - a) * sr))
    import numpy as np
    d = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        d = d.reshape(-1, ch).mean(axis=1)
    return 20 * float(np.log10(max(float(np.sqrt(np.mean(d ** 2))), 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bgm", default=None,
                    help="music file, or 'original' for the demucs accompaniment "
                         "stem of the source video")
    ap.add_argument("--bgm-gain", type=float, default=-13.0)
    ap.add_argument("--duck", type=float, default=0.05,
                    help="sidechain threshold; lower = harder ducking")
    ap.add_argument("--lufs", type=float, default=None)
    ap.add_argument("--tp", type=float, default=-1.5)
    ap.add_argument("--tag", default="EN_dubbed")
    args = ap.parse_args()

    if not ALIGNED.exists():
        raise SystemExit(f"missing {ALIGNED} - run align.py first")

    total = float(TOTAL.read_text(encoding="utf-8").strip())
    # align_report.json comes from the per-line aligner. The single-call path
    # (place_single.py) does not produce it, because it lays one clip down and
    # already writes better subtitles from the synthesizer's real word timings -
    # so a missing report is normal now, not an error.
    report = None
    if REPORT.exists():
        report = json.loads(REPORT.read_text(encoding="utf-8"))
    else:
        print("[mux] no align_report.json - using the subtitles already written "
              "by place_single.py (real word timings)")
    if report:
        en_lines = json.loads((WORK / "segments_en.json").read_text(encoding="utf-8"))
        en_text = {s["id"]: s.get("en", "") for s in en_lines}
        write_srt(OUT / f"{STEM}_EN.srt",
                  [(r["start"], r["ends_at"], en_text.get(r["id"], "")) for r in report])
    if ZH.exists():
        zh = json.loads(ZH.read_text(encoding="utf-8"))
        write_srt(OUT / f"{STEM}_ZH.srt", [(z["start"], z["end"], z["zh"]) for z in zh])

    # --- stage 1: audio ------------------------------------------------------
    audio_src = ALIGNED
    if args.bgm:
        bed = ORIG_BED if args.bgm.lower() == "original" else Path(args.bgm)
        if not bed.exists():
            raise SystemExit(f"bgm not found: {bed}")
        print(f"[mux] bgm {bed.name}: gain {args.bgm_gain} dB, "
              f"duck threshold {args.duck}")
        audio_src = build_mix(bed, total, args.bgm_gain, args.duck)
        # Report the actual voice/music balance in the mix.
        gap = next((r for r in (report or []) if r.get("gap_after", 0) > 1.0), None)
        if gap:
            g0 = gap["ends_at"] + 0.15
            g1 = min(g0 + 1.0, total - 0.1)
            print(f"[mux] balance check: gap {g0:.1f}-{g1:.1f}s = "
                  f"{window_db(audio_src, g0, g1):.1f} dBFS, "
                  f"speech 18.5-19.5s = {window_db(audio_src, 18.5, 19.5):.1f} dBFS")

    meas = measure_loudness(audio_src)
    print(f"[mux] mix measures {meas['input_i']} LUFS / {meas['input_tp']} dBTP")

    # --- stage 2: mux --------------------------------------------------------
    cmd = [FFMPEG, "-y", "-v", "error", "-i", str(SRC_VIDEO), "-i", str(audio_src)]
    if args.lufs is not None:
        fc = (
            f"[1:a]loudnorm=measured_I={meas['input_i']}"
            f":measured_TP={meas['input_tp']}"
            f":measured_LRA={meas['input_lra']}"
            f":measured_thresh={meas['input_thresh']}"
            f":offset={meas['target_offset']}"
            f":I={args.lufs}:TP={args.tp}:LRA=11[aout]"
        )
        cmd += ["-filter_complex", fc, "-map", "0:v:0", "-map", "[aout]"]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]

    dst = OUT / f"{STEM}_{args.tag}.mp4"
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-shortest", str(dst)]
    subprocess.run(cmd, check=True)
    print(f"[mux] wrote {dst}  ({dst.stat().st_size / 1048576:.1f} MB)")

    after = measure_loudness(dst)
    print(f"[mux] output: {after['input_i']} LUFS / {after['input_tp']} dBTP")


if __name__ == "__main__":
    main()

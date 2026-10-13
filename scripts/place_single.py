"""Place the single-call voice track on the timeline, with subtitles.

Because the whole script is synthesized in one call, there are no per-line
placements to make any more: the clip is simply laid down and padded. Its word
timestamps come from MiniMax, so the SRT is derived from the actual audio rather
than from estimates.

Output: work/en_aligned.wav, out/<stem>_EN.srt
"""
import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import FFMPEG, OUT, STEM, WORK  # noqa: E402

WAV = WORK / "single_call.wav"
META = WORK / "single_call.json"
SUB = WORK / "single_call_subtitle.json"
ALIGNED = WORK / "en_aligned.wav"
TOTAL = WORK / "total_duration.txt"
LINES = WORK / "segments_en.json"

HEAD_PAD = 0.30    # the read starts here
TAIL_PAD = 0.40    # and the music carries the last stretch

# Extra breathing room inserted between sentences, in seconds.
#
# The client asked for "a little more pause between sentences". MiniMax's own
# `<#x#>` marker cannot deliver that: measured on a 9-sentence script, a 0.06 s
# marker added 322 ms per boundary, 0.10 s added 542 ms, 0.20 s added 537 ms -
# the model jumps straight to about half a second and does not scale with the
# value. Nine boundaries is ~5 s, 15% of a 33 s video, which would force heavy
# atempo.
#
# So the silence is inserted HERE, in post, where the amount is exact. The fitter
# is told to leave room for it (EXTRA_PAUSE in fit_script.py).
EXTRA_PAUSE = float(os.environ.get("EXTRA_PAUSE", "0.12"))


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        d = d.reshape(-1, ch).mean(axis=1)
    return d, sr


def ts(sec: float) -> str:
    sec = max(0.0, sec)
    h, m = int(sec // 3600), int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    total = float(TOTAL.read_text(encoding="utf-8").strip())
    x, sr = read_wav(WAV)
    info = json.loads(META.read_text(encoding="utf-8"))
    print(f"[place] voice clip {len(x)/sr:.2f}s, video {total:.2f}s")

    # --- sentence boundaries, from the token stream --------------------------
    #
    # A cue/token ending in . ! ? closes a sentence. Those points get the extra
    # pause inserted below. The list is built before the audio is assembled
    # because both the waveform and the subtitle times depend on it.
    bounds: list[float] = []
    if SUB.exists():
        for seg in json.loads(SUB.read_text(encoding="utf-8")):
            for w in seg.get("timestamped_words") or []:
                t = (w.get("word") or "").strip()
                if t[-1:] in (".", "!", "?") and w.get("time_end") is not None:
                    bounds.append(w["time_end"] / 1000.0)
    clip_dur = len(x) / sr
    bounds = sorted({round(b, 3) for b in bounds if 0.1 < b < clip_dur - 0.08})

    # The fitter reserved EXTRA_PAUSE for (script lines - 1) boundaries, but the
    # tokenizer does not always agree with the script - it split this one script
    # into 10 sentences for 9 lines - and that extra boundary pushed the clip
    # 0.24 s past the end of the video, where it was trimmed. So the pause is
    # clamped to whatever slack actually exists, and never causes an overrun.
    slack = total - HEAD_PAD - clip_dur
    # Keep a small margin: the per-sentence copy rounds to whole samples, and
    # without it the accumulated rounding reported a 0.11 s overrun and trimmed
    # the tail. 60 ms is inaudible as a shortened pause.
    slack -= 0.06
    extra = EXTRA_PAUSE if bounds else 0.0
    if bounds and extra * len(bounds) > slack:
        extra = max(0.0, slack / len(bounds))
        print(f"[place] only {slack:.2f}s of slack for {len(bounds)} pauses "
              f"-> reduced to {extra*1000:.0f} ms each")
    print(f"[place] {len(bounds)} sentence boundaries -> inserting "
          f"{extra*1000:.0f} ms each ({extra*len(bounds):.2f}s total)")

    def shifted(t: float) -> float:
        """Clip time -> timeline, accounting for the inserted silences."""
        return t + HEAD_PAD + sum(extra for b in bounds if t >= b)

    start = int(round(HEAD_PAD * sr))
    track = np.zeros(int(round(total * sr)), dtype=np.float32)
    # Splices are FADED, not butted together.
    #
    # The cut points come from MiniMax's word timings and land a few milliseconds
    # either side of the true end of a word, so the waveform is not at zero there.
    # Dropping the next segment in at full amplitude puts a step into the signal -
    # a click, and the "audio was not edited cleanly" artefact the client heard.
    # A 12 ms fade out and in at every boundary removes the step.
    #
    # The earlier test that appeared to clear the splicing was FLAWED: it compared
    # the file-wide maximum sample-to-sample jump, and the source already had a
    # bigger jump (0.2863) elsewhere, so a 0.2 click at a splice was invisible.
    FADE = max(1, int(round(0.012 * sr)))
    src_cursor, dst_cursor = 0.0, start
    for b in list(bounds) + [clip_dur]:
        seg = x[int(round(src_cursor * sr)):int(round(b * sr))].copy()
        if seg.size:
            n = min(FADE, seg.size // 2)
            if n > 0:
                seg[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
                seg[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
            take = min(seg.size, len(track) - dst_cursor)
            if take > 0:
                track[dst_cursor:dst_cursor + take] += seg[:take]
            dst_cursor += seg.size
        src_cursor = b
        dst_cursor += int(round(extra * sr))
    if dst_cursor > len(track):
        print(f"[place] WARNING: clip + pauses overrun by "
              f"{(dst_cursor - len(track))/sr:.2f}s and were trimmed")

    peak = float(np.max(np.abs(track))) or 1.0
    if peak > 0.99:
        track = track / peak * 0.99
    with wave.open(str(ALIGNED), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((track * 32767).astype(np.int16).tobytes())

    speech = len(x) / sr
    covered = HEAD_PAD + speech + extra * len(bounds)
    print(f"[place] wrote {ALIGNED.name}: speech {speech:.2f}s "
          f"+ pauses {extra*len(bounds):.2f}s from {HEAD_PAD}s "
          f"-> covers {covered:.2f}s of {total:.2f}s "
          f"({100*covered/total:.0f}%)")
    print(f"[place] music-only tail: {total - covered:.2f}s")

    # --- subtitles straight from the synthesizer's word timings ------------
    #
    # MiniMax returns only 2 coarse segments for the whole script, which is
    # useless as subtitles, and its token texts are SUB-WORD pieces ("He", "llo",
    # "bo", "ss"), so joining them with spaces produces gibberish like
    # "He llo bo ss,". What the tokens ARE good for is timing: their sentence
    # breaks line up one-to-one with the script lines, so each cue takes its
    # text from the script and only its times from the tokens.
    if SUB.exists():
        script = [r["en"].strip() for r in
                  json.loads(LINES.read_text(encoding="utf-8"))]
        segs = json.loads(SUB.read_text(encoding="utf-8"))
        tokens = []
        for seg in segs:
            for w in seg.get("timestamped_words") or []:
                if w.get("time_begin") is not None:
                    tokens.append({"text": (w.get("word") or "").strip(),
                                   "a": w["time_begin"] / 1000.0,
                                   "b": w["time_end"] / 1000.0})
        groups, cur = [], []
        for tk in tokens:
            cur.append(tk)
            if tk["text"][-1:] in (".", "!", "?"):
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)

        # Sentence splitting is not reliable (the tokenizer split this script
        # into 10 groups for 9 lines), so the cues are aligned by CHARACTER
        # COUNT instead: count the alphanumeric characters of the token stream
        # and of each script line, and cut the token stream wherever the running
        # totals meet. That is insensitive to how the tokenizer chunked things.
        def alnum(s: str) -> str:
            return "".join(c for c in s.lower() if c.isalnum())

        cum_tok, run = [], 0
        for tk in tokens:
            run += len(alnum(tk["text"]))
            cum_tok.append(run)

        cues, consumed, start_tok = [], 0, 0
        for line in script:
            consumed += len(alnum(line))
            end_tok = start_tok
            while end_tok < len(cum_tok) - 1 and cum_tok[end_tok] < consumed:
                end_tok += 1
            a = shifted(tokens[start_tok]["a"])
            b = shifted(tokens[end_tok]["b"])
            cues.append((a, b, line))
            start_tok = end_tok + 1
        print(f"[place] aligned {len(cues)} cues by character count "
              f"({len(tokens)} tokens, {len(groups)} sentence groups seen)")

        lines_out = []
        for i, (a, b, t) in enumerate(cues, 1):
            lines_out.append(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n")
        p = OUT / f"{STEM}_EN.srt"
        p.write_text("\n".join(lines_out), encoding="utf-8")
        print(f"[place] srt -> {p.name}  ({len(cues)} cues, script text + "
              f"measured times)")

    # --- zh subtitles, unchanged -------------------------------------------
    zh = WORK / "zh_cues.json"
    if zh.exists():
        rows = json.loads(zh.read_text(encoding="utf-8"))
        lines = [f"{i}\n{ts(r['start'])} --> {ts(r['end'])}\n{r['zh']}\n"
                 for i, r in enumerate(rows, 1)]
        (OUT / f"{STEM}_ZH.srt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

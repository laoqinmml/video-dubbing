"""Build the dubbing timeline from Volcano ASR word timestamps.

The old timeline was assembled indirectly:
    energy-envelope segmentation -> LLM transcription -> sentence regrouping ->
    text matching back onto segments
which needed manual merging, and the LLM's timestamps drifted (it once put the
last line at 38 s inside a 32.3 s file).

Volcano returns real per-WORD times, e.g.
    {"text": "boss", "start_time": 810, "end_time": 1090}
so clauses can be cut at punctuation and timed exactly. Utterance-level
segmentation alone is too coarse (it returned only 3 spans for 32 s), which is
why the words matter.

Run:  python scripts/build_segments_volcano.py [--audio work/vocals_16k.wav]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import FFMPEG, WORK  # noqa: E402

OUT = WORK / "segments_zh.json"
LINES = WORK / "segments_lines.json"
RAW = WORK / "volcano_raw.json"

CLAUSE_END = re.compile(r"[，。！？；、,.!?;]")
MIN_LINE = 0.60     # merge anything shorter into the previous line
MAX_LINE = 7.00     # split longer lines at a word boundary


def to_mp3(src: Path) -> Path:
    dst = Path(tempfile.gettempdir()) / "volc_asr_in.mp3"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src),
                    "-ac", "1", "-ar", "16000", "-b:a", "64k", str(dst)],
                   check=True)
    return dst


def words_of(utt: dict) -> list[dict]:
    """Real words only - the API pads with spaces carrying -1 timestamps,
    which the parser already dropped."""
    return [w for w in (utt.get("words") or []) if w.get("text", "").strip()]


def cut_at_punctuation(utts: list[dict]) -> list[dict]:
    """One segment per clause, timed from its first and last word.

    Punctuation is NOT a token in `words` (only space padding is, and it carries
    -1 times), so the punctuated utterance text is walked in parallel with the
    word list. Every punctuation mark marks a boundary at the current word
    index, which yields both exact times and the original wording.
    """
    segs: list[dict] = []
    for u in utts:
        ws = words_of(u)
        text = u.get("text", "")
        if not ws:
            continue
        ti = wi = seg_w = seg_t = 0
        while ti < len(text) and wi < len(ws):
            ch = text[ti]
            if ch.isspace():
                ti += 1
                continue
            if CLAUSE_END.match(ch):
                if wi > seg_w:
                    segs.append(_mk(ws[seg_w:wi], text[seg_t:ti + 1]))
                ti += 1
                seg_w, seg_t = wi, ti
                continue
            tok = ws[wi]["text"]
            if text.startswith(tok, ti):
                ti += len(tok)
                wi += 1
            else:                      # tolerate a stray mismatch
                ti += 1
        if seg_w < len(ws):
            segs.append(_mk(ws[seg_w:], text[seg_t:]))
    return [s for s in segs if s["text"]]


def _mk(ws: list[dict], text: str | None = None) -> dict:
    if not ws:
        return {"start": 0.0, "end": 0.0, "text": "", "words": []}
    return {
        "start": round(ws[0]["start"], 2),
        "end": round(ws[-1]["end"], 2),
        "text": (text if text is not None
                 else "".join(w["text"] for w in ws)).strip(),
        "words": ws,
    }


def merge_and_split(segs: list[dict]) -> list[dict]:
    # 1. merge runts forward
    merged: list[dict] = []
    for s in segs:
        dur = s["end"] - s["start"]
        if merged and dur < MIN_LINE:
            prev = merged[-1]
            prev["end"] = s["end"]
            prev["text"] += s["text"]
            prev["words"] = prev["words"] + s["words"]
        else:
            merged.append(dict(s))

    # 2. split runaways at a real word boundary (never duplicate text)
    final: list[dict] = []
    for s in merged:
        dur = s["end"] - s["start"]
        if dur <= MAX_LINE or len(s["words"]) < 4:
            final.append(s)
            continue
        ws = s["words"]
        target = s["start"] + dur / 2
        # nearest word boundary to the midpoint
        idx = min(range(1, len(ws)),
                  key=lambda i: abs(ws[i]["start"] - target))
        left, right = ws[:idx], ws[idx:]
        final.append(_mk(left))
        final.append(_mk(right))
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(WORK / "vocals_16k.wav"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.audio)
    if not src.exists():
        raise SystemExit(f"missing audio: {src}")

    if args.dry_run:
        import os
        has = bool(os.environ.get("VOLC_API_KEY") or os.environ.get("VOLC_APP_KEY"))
        print(f"audio  : {src}")
        print(f"api key: {'set' if has else 'MISSING'}")
        return

    import asr_volcano

    print(f"[volcano] transcribing {src.name}")
    res = asr_volcano.transcribe(to_mp3(src))
    RAW.write_text(json.dumps(res["raw"], ensure_ascii=False, indent=2),
                   encoding="utf-8")
    total = (res["duration_ms"] / 1000.0) or 32.32
    print(f"[volcano] {len(res['utterances'])} utterances, {total:.2f}s, "
          f"word timestamps present: "
          f"{bool(res['utterances'] and res['utterances'][0].get('words'))}")

    segs = merge_and_split(cut_at_punctuation(res["utterances"]))
    print(f"[volcano] {len(segs)} clauses after merge/split\n")

    out = []
    for i, s in enumerate(segs, 1):
        nxt = segs[i]["start"] if i < len(segs) else total
        out.append({
            "id": i,
            "start": s["start"],
            "end": s["end"],
            "dur": round(s["end"] - s["start"], 2),
            "slot": round(max(0.5, nxt - s["start"]), 2),
            "zh": s["text"],
        })

    if LINES.exists():
        backup = LINES.with_suffix(".json.bak-energy")
        if not backup.exists():
            shutil.copyfile(LINES, backup)
            print(f"[volcano] heuristic timeline kept at {backup.name}\n")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    LINES.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for s in out:
        print(f"  {s['id']:>2} {s['start']:6.2f}->{s['end']:6.2f} "
              f"(slot {s['slot']:5.2f}s)  {s['zh']}")
    print(f"\n[volcano] wrote {OUT.name} and {LINES.name}")


if __name__ == "__main__":
    main()

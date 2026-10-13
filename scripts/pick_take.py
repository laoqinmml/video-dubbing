"""Generate several takes of the script and keep the most evenly paced one.

MiniMax's pacing drifts within a single call: measuring this script's word
timestamps showed the tempo dropping from ~8 tokens/s at the start to ~4 around
13-18 s and back to ~7 by the end. The client hears that as "from 'We control'
onwards it slows down, which is weird".

`speed` cannot fix it - it scales the whole take uniformly, so it would speed up
the parts that are already fine. And splitting the script into pieces is what
caused the original problem.

What does work is selection: the model is not deterministic, so several takes are
generated, each is scored on how flat its ARTICULATION rate is (pause time
excluded, because pauses are supposed to vary), and the flattest one wins.

Output: work/single_call.wav (best take) + work/takes.json (scores)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import minimax  # noqa: E402
from synth_single import build_text  # noqa: E402
from config import EMOTION, WORK, voice_id  # noqa: E402

TAKES = int(os.environ.get("TAKES", "3"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "4"))
SD_BAR = float(os.environ.get("SD_BAR", "6.0"))
LINES = WORK / "segments_en.json"
WAV = WORK / "single_call.wav"
META = WORK / "single_call.json"
TAKE_DIR = WORK / "takes"
SCORES = WORK / "takes.json"

PAUSE_GAP = 0.20    # s - longer than this counts as a pause, not articulation
WIN = 3.0
STEP = 0.5


def tokens_from(sub_json) -> np.ndarray:
    ts = []
    for seg in sub_json:
        for w in seg.get("timestamped_words") or []:
            if w.get("time_begin") is not None:
                ts.append(w["time_begin"] / 1000.0)
    return np.sort(np.array(ts))


def evenness(ts: np.ndarray) -> tuple[float, list]:
    """Std-dev of the windowed articulation rate. Lower = steadier."""
    if ts.size < 20:
        return 1e9, []
    rate_rows = []
    for a in np.arange(0, max(ts[-1] - ts[0] - WIN + 0.1, 0.1), STEP):
        lo, hi = ts[0] + a, ts[0] + a + WIN
        idx = np.where((ts >= lo) & (ts < hi))[0]
        if idx.size < 8:
            continue
        g = np.diff(ts[idx[0]:idx[-1] + 1])
        art = g[g < PAUSE_GAP]
        if art.size < 6:
            continue
        rate_rows.append((round(lo, 2), round(1 / np.median(art), 2)))
    if len(rate_rows) < 4:
        return 1e9, rate_rows
    rates = np.array([r[1] for r in rate_rows])
    return float(rates.std()), rate_rows


def main() -> None:
    voice = voice_id()
    lines = json.loads(LINES.read_text(encoding="utf-8"))
    text = build_text(lines)
    words = sum(len(r["en"].split()) for r in lines)
    TAKE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[takes] voice={voice}  {words} words")

    # --reselect re-scores takes already on disk instead of paying for new ones.
    if "--reselect" in sys.argv and SCORES.exists():
        results = json.loads(SCORES.read_text(encoding="utf-8"))
        print(f"[takes] re-selecting from {len(results)} saved takes "
              f"(no new synthesis)")
        results = [r for r in results
                   if (TAKE_DIR / f"take{r['take']}.wav").exists()]
    else:
        # SYNTHESIZE-AND-CHECK, not "generate a batch and choose".
        #
        # MiniMax is non-deterministic: the same text came back anywhere between
        # 27.25 s and 31.83 s (17% spread) with pacing quality varying 30%. So a
        # single blind synthesis is a gamble - one run overran the video and had
        # its ending trimmed. But paying for 6-8 takes every time is a fixed
        # waste when ~71% of takes pass first time.
        #
        # So: synthesize one, check it, and only pay again if it fails.
        total = float((WORK / "total_duration.txt").read_text(
            encoding="utf-8").strip())
        head = float(os.environ.get("HEAD_PAD", "0.30"))
        print(f"[takes] synthesize-and-check: accept if it fits "
              f"({head}s head + clip <= {total:.2f}s) and sd <= {SD_BAR}\n")
        results = []
        for i in range(1, MAX_ATTEMPTS + 1):
            raw, ms, sub = minimax.t2a(text, voice, speed=1.0, pitch=0,
                                       emotion=EMOTION, subtitle_word=True)
            (TAKE_DIR / f"take{i}.wav").write_bytes(raw)
            sub_json = None
            if sub:
                try:
                    import requests
                    r = requests.get(sub, timeout=90)
                    sub_json = r.json()
                    (TAKE_DIR / f"take{i}_sub.json").write_text(
                        r.text, encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    print(f"[takes] take{i} subtitle failed: {str(exc)[:80]}")
            ts = tokens_from(sub_json) if sub_json else np.array([])
            sd, rows = evenness(ts)
            dur = round(ms / 1000, 2)
            fits = (head + dur) <= total
            ok = fits and sd <= SD_BAR
            results.append({"take": i, "duration_sec": dur,
                            "evenness_sd": round(sd, 3), "rates": rows,
                            "fits": fits, "accepted": ok})
            why = ("OK" if ok else
                   ("too long" if not fits else f"sd {sd:.2f} > {SD_BAR}"))
            print(f"[takes] attempt {i}: {dur:5.2f}s  sd={sd:5.3f}  "
                  f"{'ACCEPT' if ok else 'reject - ' + why}")
            if ok:
                break
        # re-read as plain dicts without the transient keys
        results = [{k: v for k, v in r.items()
                    if k not in ("fits", "accepted")} for r in results]

    # Selection has TWO criteria, and fit comes first. Scoring only on evenness
    # picked a 32.98 s take for a 32.32 s video, which then got trimmed and lost
    # the closing "call me!". A take that does not fit is not a candidate at all.
    total = float((WORK / "total_duration.txt").read_text(encoding="utf-8").strip())
    head = float(os.environ.get("HEAD_PAD", "0.30"))
    for r in results:
        r["fits"] = (head + r["duration_sec"]) <= total
    fitting = [r for r in results if r["fits"]]

    print()
    for r in results:
        print(f"[takes] take{r['take']}: {r['duration_sec']:5.2f}s  "
              f"sd={r['evenness_sd']:5.3f}  "
              f"{'fits' if r['fits'] else 'TOO LONG'}")

    if fitting:
        # Evenness alone is not the whole story: two takes can be equally
        # steady while one fills two more seconds of the video. So takes within
        # 5% of the best evenness are treated as tied, and the LONGEST of those
        # wins - steadier delivery AND fuller coverage.
        best_sd = min(r["evenness_sd"] for r in fitting)
        tied = [r for r in fitting if r["evenness_sd"] <= best_sd * 1.05]
        best = max(tied, key=lambda r: r["duration_sec"])
        print(f"\n[takes] most even fitting: sd={best_sd:.3f}; "
              f"{len(tied)} take(s) within 5% of it")
        print(f"[takes] best = take{best['take']} "
              f"(sd {best['evenness_sd']}, {best['duration_sec']}s, "
              f"head {head}s -> ends {head + best['duration_sec']:.2f}s "
              f"of {total:.2f}s = "
              f"{100*(head + best['duration_sec'])/total:.0f}%)")
    else:
        best = min(results, key=lambda r: r["duration_sec"])
        print(f"\n[takes] WARNING: no take fits ({total:.2f}s video, "
              f"{head}s head pad). Using the shortest, take{best['take']} "
              f"at {best['duration_sec']}s - the script needs fewer words.")
    for f in ("wav", "sub.json"):
        src = TAKE_DIR / f"take{best['take']}_{f}" if f != "wav" \
            else TAKE_DIR / f"take{best['take']}.wav"
        if src.exists():
            if f == "wav":
                WAV.write_bytes(src.read_bytes())
            else:
                (WORK / "single_call_subtitle.json").write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8")

    META.write_text(json.dumps({
        "voice_id": voice, "words": words, "chars": len(text),
        "duration_sec": best["duration_sec"],
        "words_per_sec": round(words / best["duration_sec"], 3),
        "take": best["take"], "evenness_sd": best["evenness_sd"],
        "takes": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    SCORES.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"[takes] wrote {WAV.name} from take{best['take']}")


if __name__ == "__main__":
    main()

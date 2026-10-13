"""Fit the English script to the video by MEASURING, not by guessing words.

Word targets do not work reliably here. Asked for 67 words on a 28.7 s video,
DeepSeek returned 110, then 84, then stalled at 84 across three repair rounds -
25% over, which can never fit. Guessing a "words per second" constant is what
produced that, and the constant moves anyway with pitch and text.

So this closes the loop on the thing that actually matters - the spoken
duration:

    synthesise one take -> measure it -> if it overruns, condense the script by
    the measured ratio -> synthesise again (max CONDENSE_ROUNDS)

Condensing drops the least important detail rather than mangling every sentence,
because "make it 25% shorter" applied uniformly produces clipped, unnatural
English. Each round also re-checks the pacing bar, so a take that fits but is
uneven is still rejected.

Output: work/segments_en.json (final wording), work/single_call.wav, work/takes.json
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import minimax  # noqa: E402
from config import (  # noqa: E402
    DEEPSEEK_BASE, DEEPSEEK_KEY, EMOTION, FFMPEG, WORK, voice_id,
)  # noqa: E402
from pick_take import evenness, tokens_from  # noqa: E402
from synth_single import build_text  # noqa: E402

EN = WORK / "segments_en.json"
ZH = WORK / "segments_lines.json"
WAV = WORK / "single_call.wav"
META = WORK / "single_call.json"
SUB = WORK / "single_call_subtitle.json"
TAKE_DIR = WORK / "takes"
SCORES = WORK / "takes.json"
TOTAL = WORK / "total_duration.txt"

CONDENSE_ROUNDS = int(os.environ.get("CONDENSE_ROUNDS", "3"))
SD_BAR = float(os.environ.get("SD_BAR", "6.0"))
HEAD = float(os.environ.get("HEAD_PAD", "0.30"))
# place_single.py inserts this much silence at every sentence boundary. The
# speech budget must leave room for it, or the pauses push the clip past the end
# of the video and get trimmed.
EXTRA_PAUSE = float(os.environ.get("EXTRA_PAUSE", "0.12"))
# Cap on the tempo compression applied to make a take fit.
#
# This was raised to 1.22 to maximise coverage and the client immediately heard
# it: "too fast, the original speed was good". Coverage is not worth speeding
# the voice up. At 1.06 the fitter instead condenses the script until it fits at
# (almost) its natural pace, which is the right trade.
# Now disabled entirely (1.0) at the client's request. The fitter must instead
# shorten the script until it fits at its natural pace.
ATEMPO_MAX = float(os.environ.get("ATEMPO_MAX", "1.0"))
# A take must also FILL the video, not merely fit inside it.
MIN_FILL = float(os.environ.get("MIN_FILL", "0.93"))

SYSTEM = """You are shortening the English voice-over of a short vertical \
marketing video from a Chinese kitchen-appliance factory (blenders, export).

The current script is too LONG for the video once spoken aloud. Shorten it.

AUDIENCE, in every rewrite: the listener is the OVERSEAS buyer, never a \
Chinese exporter. Never address "exporters" or "those of you in export", and \
never write "your export market" - write "your market" or "your brand". Preserve \
that framing through any shortening or lengthening.

RULES
1. Hit the target word count closely. The target is computed from the measured
   overrun, so treat it as a hard budget.
2. CUT BY DROPPING WHOLE LINES if that is what it takes. The previous attempt
   only reworded sentences and stalled 15% over budget - renaming words cannot
   buy that much. You are explicitly allowed to output FEWER lines than you were
   given: merge two related lines, or delete a line whose content is already
   implied elsewhere. Losing a minor example is fine; losing the whole video's
   message is not.
3. KEEP, in priority order: the closing call to action and the speaker's name,
   the years of experience, the product term (blender), the factory scale or
   location. Drop from the bottom of that list first.
4. Never leave a sentence that is grammatically broken or sounds abrupt, and do
   not clip every sentence a little - that reads as unnatural.
5. Natural spoken English, proud and upbeat.
6. NUMBERS STAY AS WORDS. Never introduce a digit. "20,000" counts as one word but is spoken as two, so digits corrupt the word budget and the synthesizer may read them digit by digit. If you rewrite a phrase containing a number, keep it spelled out ("twenty thousand", "fifteen years").
7. Output ONLY a JSON array of {"id": <int>, "en": "<english>"}, with the ids you
   kept. Fewer items than the input is allowed and expected.
"""


def total_duration() -> float:
    return float(TOTAL.read_text(encoding="utf-8").strip())


def drop_lines(rows: list[dict], target_words: int) -> list[dict] | None:
    """Drop whole lines until the script is short enough.

    Deterministic fallback for when the rewriting model will not cut: asked for
    32 words it returned 42, three rounds running. Lines are removed from the
    MIDDLE first, working outwards, so the opening hook and the closing call to
    action survive - those are the two lines that carry the sale. Returns None
    when there is nothing left worth dropping (keeps at least three lines).
    """
    out = [dict(r) for r in rows]
    while sum(len(r["en"].split()) for r in out) > target_words and len(out) > 3:
        mid = len(out) // 2
        # alternate outwards from the middle: mid, mid-1, mid+1, mid-2 ...
        order = sorted(range(1, len(out) - 1),
                       key=lambda i: (abs(i - mid), i))
        out.pop(order[0]) if order else out.pop(-2)
    if sum(len(r["en"].split()) for r in out) > target_words:
        return None
    return out


def condense(rows: list[dict], target_words: int, measured: float,
             budget: float) -> list[dict]:
    return _rewrite(rows, target_words, measured, budget, "condense")


def expand(rows: list[dict], target_words: int, measured: float,
           budget: float) -> list[dict]:
    return _rewrite(rows, target_words, measured, budget, "expand")


def _rewrite(rows: list[dict], target_words: int, measured: float,
             budget: float, mode: str) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(base_url=DEEPSEEK_BASE, api_key=DEEPSEEK_KEY, timeout=180.0)
    current = sum(len(r["en"].split()) for r in rows)
    payload = {
        "mode": mode,
        "target_total_words": target_words,
        "current_total_words": current,
        "spoken_seconds_measured": round(measured, 2),
        "spoken_seconds_allowed": round(budget, 2),
        "lines": [{"id": r["id"], "en": r["en"]} for r in rows],
    }
    print(f"[fit] {mode} {current} -> ~{target_words} words "
          f"(measured {measured:.2f}s vs {budget:.2f}s allowed)")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": SYSTEM if mode == "condense"
                   else SYSTEM_EXPAND},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)}],
        temperature=0.3,
    )
    txt = resp.choices[0].message.content or ""
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        raise ValueError(f"no JSON array in {mode} reply")
    fixed = json.loads(m.group(0))
    by_id = {int(d["id"]): str(d["en"]).strip() for d in fixed}
    out = [{"id": r["id"], "zh": r.get("zh", ""),
            "en": by_id.get(r["id"], r["en"])} for r in rows]
    # expansion may legitimately add lines back
    for d in fixed:
        i = int(d["id"])
        if i not in {r["id"] for r in out} and str(d["en"]).strip():
            out.append({"id": i, "zh": "", "en": str(d["en"]).strip()})
    out.sort(key=lambda r: r["id"])
    return [r for r in out if r["en"]]


SYSTEM_EXPAND = """You are lengthening the English voice-over of a short vertical \
marketing video from a Chinese kitchen-appliance factory (blenders, export).

The current script FITS the video but leaves too much silence at the end - the \
spoken audio finishes well before the video does, so the track has a long \
music-only stretch. It needs to be LONGER.

AUDIENCE, in every rewrite: the listener is the OVERSEAS buyer, never a \
Chinese exporter. Never address "exporters" or "those of you in export", and \
never write "your export market" - write "your market" or "your brand". Preserve \
that framing through any shortening or lengthening.

RULES
1. Reach the target word count closely. It is computed from how much of the \
video is currently unfilled.
2. Add substance, never padding: bring back a concrete detail that a shortening \
pass dropped, spell a benefit out fully, or describe a process more completely. \
Do NOT add empty filler, do not repeat an idea you already made, and do not \
invent facts, numbers or specifications.
3. Keep every existing fact intact - the speaker's name, the product term \
(blender), the years of experience, the factory scale or location, and any \
closing call to action.
4. Natural spoken English, proud and upbeat, flowing continuously from line to \
line.
5. NUMBERS STAY AS WORDS. Never introduce a digit: "20,000" counts as one word but is spoken as two, which corrupts the length estimate, and the synthesizer may read it digit by digit.
6. Output ONLY a JSON array of {"id": <int>, "en": "<english>"}. You may add \
lines if that reads better.
"""


def synth_once(text: str, voice: str, speed: float = 1.0
               ) -> tuple[float, float, bytes, dict | None]:
    raw, ms, sub = minimax.t2a(text, voice, speed=speed, pitch=0,
                               emotion=EMOTION, subtitle_word=True)
    sub_json = None
    if sub:
        try:
            import requests
            r = requests.get(sub, timeout=90)
            sub_json = r.json()
        except Exception:  # noqa: BLE001
            pass
    ts = tokens_from(sub_json) if sub_json else np.array([])
    sd, _ = evenness(ts)
    return ms / 1000.0, sd, raw, sub_json


def atempo(raw: bytes, factor: float) -> tuple[bytes, float]:
    """Speed the clip up by an exact factor with ffmpeg, returning new bytes.

    `speed` on the API is a performance hint and measured badly non-linear (asked
    for x1.104, got 3% shorter). atempo is an exact time-stretch, which is what a
    fitting mechanism needs.
    """
    import subprocess
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="atempo_"))
    src, dst = tmp / "in.wav", tmp / "out.wav"
    src.write_bytes(raw)
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src),
                    "-filter:a", f"atempo={factor:.6f}",
                    "-c:a", "pcm_s16le", str(dst)], check=True)
    import wave as _w
    with _w.open(str(dst), "rb") as f:
        dur = f.getnframes() / f.getframerate()
    return dst.read_bytes(), dur


def main() -> None:
    voice = voice_id()
    rows = json.loads(EN.read_text(encoding="utf-8"))
    total = total_duration()
    n_bounds = max(0, len(rows) - 1)
    budget = total - HEAD - EXTRA_PAUSE * n_bounds
    print(f"[fit] reserving {EXTRA_PAUSE*1000:.0f} ms x {n_bounds} sentence "
          f"pauses = {EXTRA_PAUSE*n_bounds:.2f}s")
    TAKE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fit] voice={voice}  video={total:.2f}s  speech budget={budget:.2f}s")

    # Candidates are COLLECTED, then the best is chosen at the end.
    #
    # Deciding round by round does not survive this model's variance: the same
    # text came back at 26.2 s, then 25.0 s, then 33.8 s, so "accept the first
    # one that passes" oscillated between 92%, 88% and 119% and ended up
    # dumping to the worst option. Scoring every take on how full it can
    # actually make the video, and picking the winner afterwards, is stable.
    cands: list[dict] = []
    # A while loop, not a for: dropping lines is a real edit that deserves a
    # synthesis of its own. With a fixed range the drop landed on the LAST round
    # and the shortened script was never spoken - 外贸破壁机4 shipped at 113%
    # with 2.47 s trimmed off its ending.
    rnd, max_rounds = 1, CONDENSE_ROUNDS
    while rnd <= max_rounds:
        text = build_text(rows)
        words = sum(len(r["en"].split()) for r in rows)
        dur, sd, raw, sub_json = synth_once(text, voice)
        fill = dur / budget if budget else 0.0
        cands.append({"round": rnd, "words": words, "chars": len(text),
                      "dur": dur, "sd": sd, "raw": raw, "sub": sub_json,
                      "rows": [dict(r) for r in rows]})
        print(f"[fit] round {rnd}: {words} words -> {dur:5.2f}s "
              f"(fill {fill:.0%})  sd={sd:5.3f}")
        (TAKE_DIR / f"fit{rnd}.wav").write_bytes(raw)
        if sub_json:
            (TAKE_DIR / f"fit{rnd}_sub.json").write_text(
                json.dumps(sub_json, ensure_ascii=False), encoding="utf-8")

        # Good enough: fills the video AND paces acceptably. Stop paying.
        if 0.97 <= fill <= 1.0 and sd <= SD_BAR:
            print("[fit]   fills 97-100% with acceptable pacing -> done")
            break

        if fill > ATEMPO_MAX:
            need = int(round(words * min(0.95, budget / max(dur, 0.1))))
            print(f"[fit]   {fill:.0%} is past the atempo cap; condensing to "
                  f"~{need} words")
            before = words
            try:
                rows = condense(rows, max(12, need), dur, budget)
            except Exception as exc:  # noqa: BLE001
                print(f"[fit] condense failed: {str(exc)[:120]}")
                break
            after = sum(len(r["en"].split()) for r in rows)
            if rnd >= 2 and after >= before:
                # The model keeps refusing to cut - it returned 42 words three
                # times when asked for 32. Rewriting has stalled, so whole lines
                # are dropped here instead. That is the only deterministic way to
                # make a too-long script fit without speeding the voice up, and
                # it beats the alternative of an overrunning clip whose ending
                # gets trimmed away by place_single.py.
                target = max(10, int(round(after * budget / max(dur, 0.1))))
                dropped = drop_lines(rows, target)
                if dropped is None:
                    print("[fit] cannot drop any more lines; giving up on "
                          "shrinking and letting the last take stand")
                    break
                rows = dropped
                max_rounds += 1        # the shortened script needs its own take
                print(f"[fit]   model stalled at {after} words -> dropped whole "
                      f"lines down to {sum(len(r['en'].split()) for r in rows)}; "
                      f"extra round granted")
        elif fill < MIN_FILL:
            need = int(round(words * (budget * 1.02) / max(dur, 0.1)))
            print(f"[fit]   fills only {fill:.0%}; expanding to ~{need} words")
            try:
                rows = expand(rows, max(words + 4, need), dur, budget)
            except Exception as exc:  # noqa: BLE001
                print(f"[fit] expand failed: {str(exc)[:120]}")
                break
        else:
            print(f"[fit]   {fill:.0%} is workable; trying another take of the "
                  f"same script")
        rnd += 1

    # FINAL GUARD: with atempo disabled the chosen take may still overrun, and an
    # overrunning clip has its ending cut off by place_single.py - the closing
    # line simply disappears. The in-loop fallbacks only fire when the rewriting
    # model visibly stalls, which is not every case (马来 and 先试单 both shipped
    # at 102%). So if the winner still does not fit, drop whole lines and spend
    # one more synthesis; keep whichever result actually fits.
    def _worst(c: dict) -> float:
        f = c["dur"] / budget if budget else 0.0
        return max(f, c["dur"] / (budget * ATEMPO_MAX)) if ATEMPO_MAX else f

    if cands and _worst(max(cands, key=lambda c: -_worst(c))) > 1.0:
        pick = min(cands, key=_worst)
        if _worst(pick) > 1.0:
            words_now = pick["words"]
            target = max(10, int(round(words_now * budget / pick["dur"] * 0.97)))
            trimmed = drop_lines(pick["rows"], target)
            if trimmed:
                print(f"[fit] final guard: best take still overruns "
                      f"({pick['dur']:.2f}s vs {budget:.2f}s); dropping "
                      f"{words_now} -> "
                      f"{sum(len(r['en'].split()) for r in trimmed)} words and "
                      f"re-synthesising once")
                text = build_text(trimmed)
                dur2, sd2, raw2, sub2 = synth_once(text, voice)
                fits2 = dur2 <= budget
                print(f"[fit]   guard take: {dur2:.2f}s  "
                      f"{'FITS' if fits2 else 'STILL OVER'}")
                rnd += 1
                (TAKE_DIR / f"fit{rnd}.wav").write_bytes(raw2)
                if sub2:
                    (TAKE_DIR / f"fit{rnd}_sub.json").write_text(
                        json.dumps(sub2, ensure_ascii=False), encoding="utf-8")
                if fits2 or dur2 < pick["dur"]:
                    cands.append({"round": rnd, "words":
                                  sum(len(r["en"].split()) for r in trimmed),
                                  "chars": len(text), "dur": dur2, "sd": sd2,
                                  "raw": raw2, "sub": sub2, "rows": trimmed})

    def achievable(c: dict) -> tuple[float, float]:
        """(best fill reachable, atempo factor to apply). 1.0 factor = no change.

        When the take is longer than even the atempo cap can fix, it CANNOT be
        made to fit - it will overrun and place_single.py will trim the tail,
        losing the closing line. That case used to be scored as if it reached
        only the cap when in fact it overshot, and 外贸破壁机33 shipped at 103%.
        Such a take is now scored by how far it overshoots, so a fitting take
        always wins.
        """
        f = c["dur"] / budget if budget else 0.0
        if f <= 1.0:
            return f, 1.0
        if f <= ATEMPO_MAX:
            return 1.0, f          # atempo lands it exactly on the budget
        capped = c["dur"] / ATEMPO_MAX          # length after the cap
        return budget / capped, ATEMPO_MAX      # <1.0: overruns, penalised

    best = max(cands, key=lambda c: (round(achievable(c)[0], 3),
                                     -abs(c["sd"] - 5.5)))
    got, factor = achievable(best)

    # FINAL GUARD, keyed off the ACTUAL winner.
    #
    # An earlier attempt at this sat before the selection and tested
    # `_worst(max(cands))`, which measured the wrong thing and silently did
    # nothing, so 马来 still shipped at 102% with 0.71 s of its ending trimmed
    # away. Testing the chosen take directly cannot miss.
    if best["dur"] > budget:
        words_now = best["words"]
        target = max(10, int(round(words_now * budget / best["dur"] * 0.96)))
        trimmed = drop_lines(best["rows"], target)
        if trimmed:
            print(f"\n[fit] GUARD: the winner still overruns "
                  f"({best['dur']:.2f}s vs {budget:.2f}s budget) -> dropping "
                  f"{words_now} -> "
                  f"{sum(len(r['en'].split()) for r in trimmed)} words and "
                  f"synthesising once more")
            text = build_text(trimmed)
            dur2, sd2, raw2, sub2 = synth_once(text, voice)
            print(f"[fit]   guard take: {dur2:.2f}s  "
                  f"{'FITS' if dur2 <= budget else 'STILL OVER'}")
            rnd += 1
            (TAKE_DIR / f"fit{rnd}.wav").write_bytes(raw2)
            if sub2:
                (TAKE_DIR / f"fit{rnd}_sub.json").write_text(
                    json.dumps(sub2, ensure_ascii=False), encoding="utf-8")
            if dur2 <= best["dur"]:
                best = {"round": rnd, "words": sum(len(r["en"].split())
                                                   for r in trimmed),
                        "chars": len(text), "dur": dur2, "sd": sd2,
                        "raw": raw2, "sub": sub2, "rows": trimmed}
                cands.append(best)
                got, factor = achievable(best)

    print(f"\n[fit] best of {len(cands)} takes: round {best['round']}, "
          f"{best['words']} words, {best['dur']:.2f}s, sd={best['sd']:.3f}, "
          f"reachable fill {got:.0%}")

    raw, sub_json, dur, sd = best["raw"], best["sub"], best["dur"], best["sd"]
    if factor > 1.0:
        print(f"[fit] atempo x{factor:.4f} to place it exactly on the budget")
        raw, dur = atempo(raw, factor)
        if sub_json:
            for seg in sub_json:
                for w in seg.get("timestamped_words") or []:
                    for k in ("time_begin", "time_end", "word_begin",
                              "word_end", "pronounce_word_begin",
                              "pronounce_word_end"):
                        if isinstance(w.get(k), (int, float)):
                            w[k] = w[k] / factor
    if got < 0.95:
        print(f"[fit] WARNING: best reachable fill is only {got:.0%}. "
              f"This video's Chinese carries more than a natural English read "
              f"fits; consider cutting a line by hand.")

    attempts = [{k: (round(v, 3) if isinstance(v, float) else v)
                 for k, v in c.items() if k not in ("raw", "sub", "rows")}
                for c in cands]
    attempts[-1] = dict(attempts[-1], chosen=True, final_sec=round(dur, 2),
                        atempo=round(factor, 4))
    rows, words, chars = best["rows"], best["words"], best["chars"]

    EN.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    WAV.write_bytes(raw)
    if sub_json:
        SUB.write_text(json.dumps(sub_json, ensure_ascii=False), encoding="utf-8")
    META.write_text(json.dumps({
        "voice_id": voice, "words": words, "chars": chars,
        "duration_sec": round(dur, 2),
        "words_per_sec": round(words / dur, 3) if dur else 0,
        "evenness_sd": round(sd, 3), "attempts": attempts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    SCORES.write_text(json.dumps(attempts, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\n[fit] final: {words} words, {dur:.2f}s "
          f"({100*(HEAD+dur)/total:.0f}% of video), sd={sd:.3f}")


if __name__ == "__main__":
    main()

"""Stage 3 - write the English dub to fill the video's audio track.

Two very different objectives have lived in this file:

  * strict mode (old): translate line by line, each confined to the time slot of
    the original Chinese sentence. Faithful to the original timeline, but the
    English came out shorter than the video and the leftover silence made it
    sound disjointed.

  * fill mode (current): the client asked to stop chasing the original timings
    and let the English run at a normal video pace. So the script is written to
    a target WORD COUNT derived from the video length, split into a sensible
    number of lines, and the lines are later laid end to end with even pauses.
    The goal is that speech covers the whole track with no long holes.

Output: work/segments_en.json
"""
import json
import math
import os
import re
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    DEEPSEEK_BASE, DEEPSEEK_KEY, FILL_MODE, FILL_TARGET_RATIO, HEAD_PAD,
    TAIL_PAD, WORK, ZENMUX_BASE, ZENMUX_KEY,
)

SRC = WORK / "segments_lines.json"     # sentence-level Chinese content
OUT = WORK / "segments_en.json"
AUDIO = WORK / "orig_16k.wav"

WORDS_PER_SECOND = 2.55  # MEASURED on this voice with the lifted pitch and no
                         # slow-downs. The earlier 2.0 estimate was calibrated on
                         # a slower, partly-compressed render and under-filled
                         # the track by ~15%.
TARGET_LINES = 9         # lines may be SPLIT to stay sayable in one breath

SYSTEM_STRICT = """You are a professional Chinese-to-English dubbing translator for \
short vertical marketing videos from a Chinese kitchen-appliance factory \
(blenders, export customers).

You will receive a JSON array of subtitle lines. Each item has:
  id, dur (seconds of the original Chinese line), slot (seconds available \
before the next line starts), max_words, zh (the Chinese text).

For every item, produce natural SPOKEN English for a voice-over.

HARD RULES
1. LENGTH: the English must be speakable at a natural pace within max_words \
words. If a literal translation is longer, tighten it: use contractions, cut \
redundancy, prefer short common words. Never pad to fill space.
2. NATURAL PUNCTUATION: write it the way a person would say it. Commas, \
question marks and full stops are welcome - a line that reads naturally matters \
more than shaving a word. BUT keep short lines (max_words 5 or below) free of \
internal punctuation: in a short line every comma or colon costs about a third \
of a second of pause and blows the budget. Prefer a plain phrase there.
3. MEANING: keep the selling point, any number, unit, model code and brand name \
exactly accurate. Never invent features or specifications. Drop a decorative \
adjective rather than overrun.
4. TONE - UPBEAT AND POSITIVE: this is a promotional video, so sound proud, \
warm and upbeat. Lead with the strength, use active verbs, keep the energy up, \
and make the closing line a warm invitation. Positive framing only - no \
"no worries", no negative phrasing, no apologetic hedging. It must still sound \
like a real person talking, never corporate, never shouty.
5. OUTPUT: a JSON array only, same length and same ids, each item \
{"id": <int>, "en": "<english>"}. No markdown fence, no commentary.
"""

SYSTEM_FILL = """You are writing the English voice-over for a short vertical \
marketing video from a Chinese kitchen-appliance factory (blenders). \
The speaker is Eva, introducing her factory to OVERSEAS buyers.

You will receive the Chinese script, the video length, and a HARD word budget.

AUDIENCE - READ THIS BEFORE ANYTHING ELSE
The English version is shown to OVERSEAS customers: importers, wholesalers and \
brand owners in Southeast Asia and elsewhere. They are the BUYER.

Many of these videos were originally shot for a DOMESTIC Chinese audience of \
exporters who were looking for a supplier, and the Chinese speaks to them - \
"做外贸的老板们", "想找货源的", "做出口的". THAT FRAMING MUST NOT SURVIVE INTO \
ENGLISH. Never address "exporters", "foreign trade bosses" or "those of you in \
export", and never explain how to export or how to pick a supplier. The viewer is \
not someone hunting for goods to resell - the viewer IS the overseas buyer, and \
this factory is the supplier.

Rewrite the ADDRESS, not just the words. "做外贸的老板们，想找稳定的破壁机货源" \
becomes "Looking for a reliable blender supplier?" - said straight to the buyer. \
Never write "if you're in export" or "for your export market"; write "for your \
market", "for your brand", "for your customers". The purpose of the copy is to \
make the overseas buyer want to contact this factory.

THE BUDGET IS THE JOB. Read this first:
The Chinese is information-dense and almost always contains MORE material than \
fits in a natural English read of that video. You are NOT asked to translate all \
of it. You are asked to choose the strongest selling points and write ONLY those, \
so that the English lands within about 5% of the word budget.

This is a selection task, not a compression task. If you translate everything \
and then tighten the wording, you will end up 20-50% over budget and the video \
will not work. Decide what to leave OUT before you start writing.

HOW TO CHOOSE
- Must keep: the speaker's name and role, the product term (blender), the years \
of experience, the factory scale or location if stated, and the closing call to \
action.
- Keep the most concrete proof of capability (a machine type, an in-house \
process, a part they make).
- KEEP THE SPECIFICS. Capacities (two litre, seven litre), model counts \
("over twenty styles"), machine types and year counts are the proof that sells - \
worth more than any adjective. Cut an adjective before you cut a number.
- Drop any line whose only purpose was to speak to Chinese exporters.
- Drop repetition, scene-setting flourishes, and any second example that makes \
the same point as the first.

STYLE
- REWRITE, DO NOT TRANSLATE CLOSELY. You are not bound to the Chinese sentence \
by sentence. Keep the message and the selling points; write the English that \
works best as a spoken ad. If a Chinese sentence does not earn its place in \
English, replace it with something better or drop it.
- ONE BREATH PER LINE. Every line must be sayable in a single breath: about \
fourteen words maximum. If an idea needs more, SPLIT IT INTO TWO LINES. Long \
lines are the main cause of rushed, uneven delivery once synthesized.
- CLEAN GRAMMAR. These are read aloud to buyers, so a sentence that is \
ungrammatical or awkward is unacceptable. Read each line back to yourself: if a \
native speaker would not say it that way, rewrite it. Avoid dangling references \
like "come and see ours".
- Plain, confident, spoken. No corporate phrasing, no stacked adjectives.

RULES
1. Land within about 5% of the word budget. Count the words before answering.
2. NUMBERS ARE WRITTEN AS WORDS, ALWAYS. This is not cosmetic and not optional:
   the word budget and the spoken length are both measured on the script, and
   "20,000" counts as ONE word while it is spoken as TWO ("twenty thousand").
   Leaving digits in makes every length calculation wrong, and the synthesizer
   may read them digit by digit ("two zero zero zero zero"). So write
   "twenty thousand square meters", "fifteen years", "two hundred units".
   Chinese 万 = ten thousand: 2万 -> "twenty thousand". Never output a digit.
3. Natural spoken English, proud, warm, upbeat, active verbs. A real person \
talking to camera - never corporate, never shouty.
4. Continuous flow: the lines are read back to back, so each should lead into \
the next. A comma may continue across two lines; do not force a full stop.
5. Never invent facts. Keep every unit and product term accurate.
6. Vary sentence openings; never repeat the same connective twice in one line.
7. LINE COUNT IS YOURS TO SET. You may return MORE lines than the Chinese had, \
because splitting a long idea is required by the one-breath rule above. Do not \
force the original line count.
8. OUTPUT: a JSON array only, one item per line, each \
{"id": <int>, "en": "<english>"}. No commentary.
"""


def total_duration() -> float:
    with wave.open(str(AUDIO), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def build_payload_strict() -> list[dict]:
    segs = json.loads(SRC.read_text(encoding="utf-8"))
    total = total_duration()
    out = []
    for i, s in enumerate(segs):
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else total
        slot = max(0.5, round(nxt - s["start"], 3))
        allowed = min(s["dur"] * 1.15, slot)
        out.append({
            "id": s["id"], "dur": s["dur"], "slot": slot,
            "max_words": max(2, int(allowed * 1.8)), "zh": s["zh"],
        })
    return out


def build_payload_fill(target_words_override: int | None = None) -> dict:
    segs = json.loads(SRC.read_text(encoding="utf-8"))
    total = total_duration()
    speech_budget = (total - HEAD_PAD - TAIL_PAD) * FILL_TARGET_RATIO
    target_words = (target_words_override
                    if target_words_override
                    else max(20, int(round(speech_budget * WORDS_PER_SECOND))))
    n_lines = min(TARGET_LINES, max(4, len(segs)))
    return {
        "video_seconds": round(total, 2),
        "target_total_words": target_words,
        "target_lines": n_lines,
        "words_per_second_reference": WORDS_PER_SECOND,
        "full_chinese_script": "".join(s["zh"] for s in segs),
        "chinese_segments": [s["zh"] for s in segs],
    }


def _ask(client, model: str, system: str, user: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.35,
    )
    txt = resp.choices[0].message.content or ""
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        raise ValueError(f"no JSON array in reply: {txt[:200]}")
    return json.loads(m.group(0))


REPAIR_SYSTEM = """You adjust English dubbing lines to a word budget.

You receive the current lines plus the total word count they must reach. Rewrite
them so the TOTAL is within about 5% of the target:
- If the total is too LOW, make sentences fuller and warmer - add natural spoken
  connective phrasing and spell out benefits, without inventing facts and
  without empty filler.
- If the total is too HIGH, tighten wording.
Keep the same number of items and ids, keep every fact and product term exact,
and keep the upbeat spoken tone.
Return ONLY a JSON array of {"id": <int>, "en": "<english>"}.
"""


def adjust_total(by_id: dict[int, str], target_words: int,
                 max_rounds: int = 3) -> dict[int, str]:
    """Nudge the whole script toward the target length.

    Writing to a word count is a whole-script property, so the repair pass works
    on the total rather than line by line.
    """
    from openai import OpenAI

    client = OpenAI(base_url=DEEPSEEK_BASE, api_key=DEEPSEEK_KEY,
                    timeout=180.0, max_retries=2)
    for rnd in range(1, max_rounds + 1):
        current = sum(len(v.split()) for v in by_id.values())
        drift = (current - target_words) / target_words
        print(f"[translate] round {rnd}: {current} words vs target "
              f"{target_words} ({drift:+.1%})")
        if abs(drift) <= 0.05:
            print("[translate] within 5% of target")
            return by_id
        payload = {
            "target_total_words": target_words,
            "current_total_words": current,
            "direction": "expand" if drift < 0 else "tighten",
            "lines": [{"id": i, "en": by_id[i]} for i in sorted(by_id)],
        }
        try:
            fixed = _ask(client, "deepseek-chat", REPAIR_SYSTEM,
                         json.dumps(payload, ensure_ascii=False, indent=1))
            for d in fixed:
                i = int(d["id"])
                if i in by_id and str(d["en"]).strip():
                    by_id[i] = str(d["en"]).strip()
        except Exception as exc:  # noqa: BLE001
            print(f"[translate] adjust failed: {str(exc)[:140]}")
            break
    return by_id


def call_llm(system: str, user_payload) -> list[dict]:
    """DeepSeek official first (direct connection); ZenMux opt-in only."""
    from openai import OpenAI

    user = json.dumps(user_payload, ensure_ascii=False, indent=1)
    attempts = [("DeepSeek", DEEPSEEK_BASE, DEEPSEEK_KEY, "deepseek-chat")]
    if os.environ.get("USE_ZENMUX") == "1":
        attempts.append(("ZenMux", ZENMUX_BASE, ZENMUX_KEY,
                         "deepseek/deepseek-v4-flash"))

    last = None
    for name, base, key, model in attempts:
        if not key:
            continue
        for try_i in range(1, 4):
            try:
                print(f"[translate] {name} / {model}"
                      + (f" (attempt {try_i})" if try_i > 1 else ""))
                client = OpenAI(base_url=base, api_key=key, timeout=180.0,
                                max_retries=2)
                return _ask(client, model, system, user)
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"[translate] {name} failed: {str(exc)[:160]}")
                time.sleep(2 * try_i)
    raise SystemExit(f"all translation backends failed: {last}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--target-words", type=int, default=None,
                    help="override the computed word target")
    args = ap.parse_args()

    segs = json.loads(SRC.read_text(encoding="utf-8"))

    if FILL_MODE:
        payload = build_payload_fill(args.target_words)
        print(f"[translate] FILL mode: {payload['target_lines']} lines, "
              f"target {payload['target_total_words']} words "
              f"for a {payload['video_seconds']}s video")
        data = call_llm(SYSTEM_FILL, payload)
        by_id = {int(d["id"]): str(d["en"]).strip() for d in data}
        by_id = adjust_total(by_id, payload["target_total_words"])
        rows = [{"id": i, "zh": "", "en": by_id[i]} for i in sorted(by_id)]
    else:
        payload = build_payload_strict()
        budget = {p["id"]: p for p in payload}
        print(f"[translate] STRICT mode: {len(payload)} lines, budget "
              f"{payload[0]['max_words']}..{max(p['max_words'] for p in payload)} words")
        data = call_llm(SYSTEM_STRICT, payload)
        by_id = {int(d["id"]): str(d["en"]).strip() for d in data}
        rows = []
        for s in segs:
            rows.append({"id": s["id"], "zh": s["zh"], "en": by_id.get(s["id"], ""),
                         "start": s["start"], "dur": s["dur"], "slot": s["slot"],
                         "max_words": budget[s["id"]]["max_words"]})

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    total_words = sum(len(r["en"].split()) for r in rows)
    print(f"\n[translate] wrote {OUT}: {len(rows)} lines, {total_words} words")
    for r in rows:
        print(f"  {r['id']:>2} ({len(r['en'].split()):>2}w)  {r['en']}")


if __name__ == "__main__":
    main()

"""Synthesize the whole script in ONE call, with explicit pause markers.

Why one call
------------
Synthesizing line by line gave every line its own tempo: measured 1.77 - 3.19
words/second across the script, an 80% spread, which the client heard as "some
sentences slow, some rushed". `speed` cannot fix that - it scales a line
uniformly, it does not make the model pace differently.

Why explicit pauses
-------------------
A single call helped but not enough: measuring the word timestamps showed a U
shape (8.0 tokens/s at the start, 4.0 in the middle, 7.0 at the end), because
the model decides its own pauses and lingers on the long comma-heavy sentence in
the middle. So the pauses are now written into the text as MiniMax `<#x#>`
markers at every sentence boundary - uniform, mine, not the model's whim.

Output: work/single_call.wav + work/single_call.json
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import minimax  # noqa: E402
from config import EMOTION, WORK, voice_id  # noqa: E402

LINES = WORK / "segments_en.json"
WAV = WORK / "single_call.wav"
META = WORK / "single_call.json"

SENTENCE_PAUSE = 0.0    # 0 = no markers at all; see build_text() for why
CLAUSE_PAUSE = 0.0      # 0 = leave internal commas to the model
MAX_CHARS = 9500        # MiniMax sync limit is 10,000

# TTS-only respellings. The client wants "Eva" said as EE-vuh (/ˈiːvə/), but this
# Chinese voice reads the name inconsistently. Measured by synthesising known
# vowels on the SAME voice and matching, the options scored:
#
#     Eeva            0.772 against /iː/     <- best
#     Eeva + IPA      0.571
#     Eva (bare)      0.428
#     Eva + (ˈiːvə)  -0.279  -> came out /ɛ/, i.e. the IPA hint made it WORSE
#
# So the name is respelled for the synthesizer only. Subtitles keep "Eva",
# because the SRT is built from segments_en.json, not from this string.
TTS_RESPELL = {"Eva": "Eeva"}

# Numbers carrying a thousands separator break this voice: "20,000 square meter"
# came back as "20 gel gel gel square meter" (the ASR's transcription of it),
# because the embedded comma is read as punctuation in the middle of a number.
# They are written out as words instead. Subtitles keep the digits - only the TTS
# string is touched.
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _under_1000(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")
    return (_ONES[n // 100] + " hundred"
            + (" " + _under_1000(n % 100) if n % 100 else ""))


def number_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    parts = []
    for div, name in ((1_000_000, "million"), (1_000, "thousand")):
        if n >= div:
            parts.append(_under_1000(n // div) + " " + name)
            n %= div
    if n:
        parts.append(_under_1000(n))
    return " ".join(parts)


def spell_numbers(t: str) -> str:
    """Write numbers out in words so the synthesizer reads them properly.

    Observed failures on this voice:
      * "20,000"  - the thousands separator got mangled into "20, 000" by the
                    comma normaliser and then read digit by digit
                    ("two zero zero zero zero")
      * "20000"   - no separator, still read digit by digit
      * "15"      - came back as a bare digit token, so even short numbers are
                    not guaranteed to be spoken as words

    Everything is therefore written out: 15 -> "fifteen", 20000 -> "twenty
    thousand". Only the TTS string is touched; subtitles keep the digits.

    A digit inside a word-like token (model codes such as "SM-800W", "24/7") is
    left alone by the \\b anchors, which is the desired behaviour.
    """
    def sub(m: "re.Match") -> str:
        try:
            return number_to_words(int(m.group(0).replace(",", "")))
        except Exception:  # noqa: BLE001
            return m.group(0)

    t = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", sub, t)   # 20,000 / 2,000,000
    t = re.sub(r"\b\d+\b", sub, t)                   # 20000 / 15 / 7
    return t


def build_text(lines: list[dict]) -> str:
    """Join the script into one clean paragraph for a single synthesis call.

    Two punctuation traps were hit here, both of which change how the model
    breathes:

    1. The lines end with a mix of '.', ',' and '!'. Stripping only a trailing
       full stop produced ",." and "!." - malformed punctuation the synthesizer
       hesitates on.
    2. The opposite mistake is to strip everything and join with ". ", which
       turns a CONTINUING clause into a sentence break. Line 8 ends in a comma
       and runs straight into line 9 ("...blender supplier, and you like our
       factory, call me!"), so forcing a full stop there invents a pause that the
       sentence does not have.

    So the original punctuation decides the join: a comma continues, a full stop
    or bang ends. This keeps the model's pauses where the writing puts them.

    Pause markers (`<#0.25#>`) were tried as well and removed: they made the clip
    36.2 s against 29.0 s plain, because the model adds its own pause on top of
    the marker.
    """
    parts = []
    for r in lines:
        t = re.sub(r"\s+", " ", r["en"].strip())
        # Numbers FIRST, before any comma handling. Doing it the other way round
        # was a real bug: the comma normaliser below turns "20,000" into
        # "20, 000", after which spell_numbers() no longer matches (its pattern
        # needs the digits tight against the comma) and the synthesizer reads the
        # digits one by one - "two zero zero zero zero".
        t = spell_numbers(t)
        t = re.sub(r"\s*,\s*", ", ", t)
        for word, respell in TTS_RESPELL.items():
            t = re.sub(rf"\b{re.escape(word)}\b", respell, t)
        m = re.search(r"([,.!?;])\s*$", t)
        punct = m.group(1) if m else "."
        if m:
            t = t[:m.start()].strip()
        parts.append((t, punct))

    out = ""
    for i, (t, punct) in enumerate(parts):
        out += t
        if i < len(parts) - 1:
            out += punct + " "
        else:
            out += "." if punct == "," else punct
    return out


def main() -> None:
    if not os.environ.get("MINIMAX_API_KEY"):
        print("[single] WARNING: MINIMAX_API_KEY not set")
    voice = voice_id()
    lines = json.loads(LINES.read_text(encoding="utf-8"))
    text = build_text(lines)
    if len(text) > MAX_CHARS:
        raise SystemExit(f"text too long: {len(text)} chars")
    words = sum(len(r["en"].split()) for r in lines)
    print(f"[single] voice={voice}  {words} words -> {len(text)} chars "
          f"(with pause markers)")
    print(f"[single] text: {text[:160]}...")

    raw, ms, sub = minimax.t2a(text, voice, speed=1.0, pitch=0,
                               emotion=EMOTION, subtitle_word=True)
    WAV.write_bytes(raw)
    info = {"voice_id": voice, "words": words, "chars": len(text),
            "duration_ms": ms, "duration_sec": round(ms / 1000, 3),
            "words_per_sec": round(words / (ms / 1000), 3),
            "sentence_pause": SENTENCE_PAUSE, "clause_pause": CLAUSE_PAUSE,
            "subtitle_url": sub}
    META.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[single] duration {ms/1000:.2f}s  ({words/(ms/1000):.2f} words/s)")
    print(f"[single] -> {WAV.name}")

    if sub:
        try:
            import requests
            r = requests.get(sub, timeout=90)
            (WORK / "single_call_subtitle.json").write_text(r.text, encoding="utf-8")
            print("[single] word timestamps saved")
        except Exception as exc:  # noqa: BLE001
            print(f"[single] subtitle download failed: {str(exc)[:100]}")


if __name__ == "__main__":
    main()

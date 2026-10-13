"""Shared paths / config for the CN->EN dubbing pipeline.

Batch mode: set DUB_SLUG (and DUB_SRC) and every path below follows, so the same
scripts run per video without editing anything. With no DUB_SLUG the original
single-video layout is used, which keeps the 外贸破壁机8 state working.
"""
import os
from pathlib import Path

# Where work/ and out/ live. Set DUB_ROOT to run the pipeline against any
# folder; otherwise it defaults to this file's project root, which keeps the
# skill self-contained when checked out on its own.
ROOT = Path(os.environ.get("DUB_ROOT")
            or Path(__file__).resolve().parent.parent)
OUT = ROOT / "out"

SRC_DIR = Path(
    r"D:\0\0 其他人的站\中山市 三毛电气\FB专用文件（荣惠）2026.9.10"
    r"\2026年1月-7月拍摄视频"
)

_SLUG = os.environ.get("DUB_SLUG", "").strip()
if _SLUG:
    STEM = _SLUG
    SRC_VIDEO = Path(os.environ.get("DUB_SRC", "")) if os.environ.get("DUB_SRC") \
        else SRC_DIR / f"{_SLUG}.mp4"
    WORK = ROOT / "work" / "batch" / _SLUG
else:
    SRC_VIDEO = SRC_DIR / "外贸破壁机8.m4v"
    STEM = "外贸破壁机8"
    WORK = ROOT / "work"

TTS = WORK / "tts"
for d in (WORK, OUT, TTS):
    d.mkdir(parents=True, exist_ok=True)

# State shared by every video in a batch: the cloned voice and the music. In
# batch mode each video gets its own WORK, so these must be looked up centrally
# rather than assumed to sit beside the current video.
SHARED = ROOT / "work"


def voice_id() -> str:
    """The cloned voice to synthesize with, from wherever it was created."""
    import json as _json
    for p in (WORK / "voice.json", SHARED / "voice.json"):
        if p.exists():
            return _json.loads(p.read_text(encoding="utf-8"))["voice_id"]
    raise SystemExit("no voice.json found - run clone_voice.py first")


def bgm_track() -> Path | None:
    """Background music, if a推介曲 has been chosen."""
    import json as _json
    p = SHARED / "bgm.json"
    if p.exists():
        f = _json.loads(p.read_text(encoding="utf-8")).get("file")
        if f and Path(f).exists():
            return Path(f)
    return None

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE = r"C:\ffmpeg\bin\ffprobe.exe"

# --- network -----------------------------------------------------------------
# External services (ZenMux, Gemini, HuggingFace) need the local proxy, because
# curl/WinINET direct access to them is blocked here. But a stale proxy setting
# would also break the DOMESTIC services that work fine over a direct
# connection (api.minimaxi.com, api.deepseek.com) - that happened once when the
# local proxy process died. So the port is probed and the proxy only used when
# something is actually listening.
PROXY = "http://127.0.0.1:10808"


def _proxy_alive(host: str = "127.0.0.1", port: int = 10808,
                 timeout: float = 0.4) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


PROXY_AVAILABLE = _proxy_alive()
if PROXY_AVAILABLE:
    os.environ.setdefault("HTTP_PROXY", PROXY)
    os.environ.setdefault("HTTPS_PROXY", PROXY)
else:
    # Direct connections only: drop any inherited proxy so requests/httpx do
    # not keep trying a dead port.
    for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
               "ALL_PROXY", "all_proxy"):
        os.environ.pop(_v, None)
    os.environ["NO_PROXY"] = "*"
    print("[config] WARNING: local proxy 127.0.0.1:10808 is not listening -> "
          "using DIRECT connections. Domestic hosts (MiniMax/DeepSeek) work; "
          "ZenMux/Gemini steps will fail until the proxy is back.")

# --- model / service endpoints ----------------------------------------------
# Domestic MiniMax site (platform.minimaxi.com). Requirements for voice cloning:
#   * a key created ON the domestic site (overseas keys are rejected: code 1004)
#   * personal or enterprise real-name verification (else code 2038)
#   * a topped-up RMB balance (else code 1008)
# Domestic API host is reachable directly - no proxy needed for MiniMax.
MINIMAX_BASE = os.environ.get("MINIMAX_BASE", "https://api.minimaxi.com")
MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY", "")

ZENMUX_BASE = "https://zenmux.ai/api/v1"
ZENMUX_KEY = os.environ.get(
    "ZENMUX_API_KEY",
    "sk-ai-v1-a3a73a38a1597aabcb5757b9b2bddc70b90a7d0f8524fc1b0c2ad45f838a4d47",
)
DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_KEY = os.environ.get(
    "DEEPSEEK_API_KEY", os.environ.get("DEEPSEEK_API_KEY", "")
)

# --- dubbing parameters ------------------------------------------------------
# Alignment mode C (hybrid): start of every line snaps to the original start.
# Overflow <= this ratio eats following silence; beyond it we compress.
OVERFLOW_TOLERANCE = 0.15
# MiniMax natural-sounding speed window (compression happens on the model side).
SPEED_MIN, SPEED_MAX = 0.80, 1.20
# English delivery rate used for the translation length budget (words/second).
# MEASURED from real speech-2.8-hd output, not assumed. Natural sentences with
# punctuation land at 1.7-2.2 w/s (commas and question marks cost real pause
# time), so budgeting at 1.8 is what keeps the whole dub inside the video while
# still letting every line be read at its natural pace.
WORDS_PER_SECOND = 1.8

# --- pacing mode -------------------------------------------------------------
# RELAXED_SPEECH = False -> STRICT alignment: every line is nailed to the
#   original Chinese start time, and any line that does not fit is compressed
#   (MiniMax re-performs it up to SPEED_MAX first, then align.py applies a small
#   ffmpeg atempo for whatever is left). This is "尽量对齐原有中文时间线".
RELAXED_SPEECH = False
MAX_RELAX_SPEED = 1.10  # only used when RELAXED_SPEECH is True
MIN_LINE_GAP = 0.12     # silence kept between two dub lines (relaxed mode only)
DRIFT_WARN = 0.30       # warn if a line starts this much later than the original

# --- natural-fill mode -------------------------------------------------------
# The client asked to stop chasing the original speech timings and instead lay
# the English out at a normal video pace, writing ENOUGH copy to fill the whole
# audio track so there are no long silences between sentences.
#   * FILL_MODE replaces "line N starts at time T" with "lines flow continuously"
#   * the script is written to a target word count derived from the video length
#   * a global pitch lift compensates for the clone sitting below the speaker
#   * speed is never allowed below 1.0, which is what caused the drawn-out parts
FILL_MODE = True
FILL_TARGET_RATIO = 0.94   # share of the available audio to fill with speech
HEAD_PAD = 0.35            # silence before the first line
TAIL_PAD = 0.50            # silence after the last line
GAP_MIN, GAP_MAX = 0.15, 0.70   # bounds for the inter-line pause
# Global pitch lift + per-line jitter.
#
# IMPORTANT - measured, do not raise blindly: MiniMax's `pitch` moves the
# fundamental WITHOUT moving the formants, so past a point the voice just gets
# thin and piercing. Measured on this clone with one fixed sentence:
#
#     pitch   F0       spectral centroid   vs pitch 0
#       0    246.4 Hz      818 Hz              -
#       1    259.4 Hz      796 Hz           -2.7%
#       2    290.1 Hz      841 Hz           +2.8%
#       3    299.0 Hz     1063 Hz          +30.0%   <-- timbre breaks here
#       4    321.9 Hz     1000 Hz          +22.2%
#
# So the usable window is 0..2. A lift of 3 or more "matches the speaker's F0"
# on paper (that was the mistake: median F0 is NOT a timbre metric) while making
# the voice sound sharp and wrong. Keep LIFT + JITTER <= 2.
PITCH_LIFT = 0             # client asked for the clone's own pitch, untouched
PITCH_JITTER = 0           # no per-line pitch variation either
MIN_SPEED = 1.00           # never slow the voice down

# Pacing uniformity.
# The per-line speed used to follow the ORIGINAL speaker's syllable rate, which
# produced four different tempos across the script (1.00 / 1.026 / 1.048 /
# 1.091). The client heard that as "some sentences slow, some rushed", so pace
# is now held flat for the whole read. Pitch and level variation are kept - those
# are what make it sound alive without making the tempo wander.
UNIFORM_PACE = True
# Silence above this gap resets the timeline (breath / cut between takes).
RESYNC_GAP = 0.60

# Proper nouns the TTS otherwise mispronounces. Value is an IPA hint.
# "Eva" was rendered as "Yiwa" without this.
PRONUNCIATION = {
    "Eva": "(ˈiːvə)",
}

# Delivery emotion for MiniMax TTS. On speech-2.8-hd the supported values are
# happy / sad / angry / fearful / disgusted / surprised / calm.
# "happy" is what gives the upbeat, positive read; set to None to let the model
# pick automatically (which is flatter).
EMOTION = "happy"

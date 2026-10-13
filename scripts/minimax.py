"""Thin MiniMax speech client: file upload, voice cloning, T2A synthesis.

Deliberately dependency-light (requests only) because the T2A schema is
MiniMax-specific, not OpenAI-compatible. Every response is checked against
base_resp.status_code - HTTP 200 does NOT mean success on this API.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from config import MINIMAX_BASE, MINIMAX_KEY

RETRIES = 4
BACKOFF = 2.0


def _request(method: str, url: str, **kwargs) -> requests.Response:
    """HTTP with retries.

    Calls go out through a local proxy and dropped TLS connections
    (SSLEOFError / ConnectionError) show up regularly, so transient transport
    failures are retried instead of failing the whole stage.
    """
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            return requests.request(method, url, **kwargs)
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            last = exc
            if attempt == RETRIES:
                break
            wait = BACKOFF * attempt
            print(f"[minimax] transport error ({type(exc).__name__}), "
                  f"retry {attempt}/{RETRIES - 1} in {wait:.0f}s")
            time.sleep(wait)
    raise MiniMaxError(0, f"transport failed after {RETRIES} attempts: {last}")

# Error codes worth naming explicitly (from MiniMax docs).
ERR_NOT_VERIFIED = 2038      # account not cleared for voice cloning
ERR_AUTH = 1004
ERR_BAD_KEY = 2049
ERR_BALANCE = 1008
ERR_SIMILARITY = 1043


class MiniMaxError(RuntimeError):
    def __init__(self, code: int, msg: str, raw: dict | None = None):
        self.code = code
        self.msg = msg
        self.raw = raw or {}
        hint = ""
        if code == ERR_NOT_VERIFIED:
            hint = "  <- 该账号未完成实名认证/未开通音色复刻权限"
        elif code in (ERR_AUTH, ERR_BAD_KEY):
            hint = "  <- API Key 无效或站点选错（国内 minimaxi.com / 海外 minimax.io）"
        elif code == ERR_BALANCE:
            hint = "  <- 账户余额不足"
        super().__init__(f"MiniMax error {code}: {msg}{hint}")


def _key() -> str:
    key = os.environ.get("MINIMAX_API_KEY") or MINIMAX_KEY
    if not key:
        raise MiniMaxError(0, "MINIMAX_API_KEY not set")
    return key


def _base() -> str:
    return (os.environ.get("MINIMAX_BASE") or MINIMAX_BASE).rstrip("/")


def _check(payload: dict) -> dict:
    br = payload.get("base_resp") or {}
    code = br.get("status_code", -1)
    if code != 0:
        raise MiniMaxError(code, br.get("status_msg", "unknown"), payload)
    return payload


def _post(path: str, body: dict, timeout: int = 300) -> dict:
    url = f"{_base()}{path}"
    r = _request(
        "POST",
        url,
        headers={
            "Authorization": f"Bearer {_key()}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )
    try:
        return _check(r.json())
    except ValueError:
        raise MiniMaxError(0, f"HTTP {r.status_code}: {r.text[:300]}")


def upload(path: str | Path, purpose: str = "voice_clone") -> int:
    """purpose: 'voice_clone' (10s-5min) or 'prompt_audio' (<8s)."""
    path = Path(path)
    with path.open("rb") as fh:
        r = _request(
            "POST",
            f"{_base()}/v1/files/upload",
            headers={"Authorization": f"Bearer {_key()}"},
            data={"purpose": purpose},
            files={"file": (path.name, fh)},
            timeout=600,
        )
    payload = _check(r.json())
    file_id = payload["file"]["file_id"]
    print(f"[minimax] uploaded {path.name} ({path.stat().st_size/1024:.0f} KB) "
          f"purpose={purpose} -> file_id={file_id}")
    return int(file_id)


def clone(
    file_id: int,
    voice_id: str,
    prompt_file_id: int | None = None,
    prompt_text: str | None = None,
    model: str = "speech-2.8-hd",
    noise_reduction: bool = True,
    volume_normalization: bool = True,
    preview_text: str | None = None,
) -> dict:
    body: dict = {
        "file_id": file_id,
        "voice_id": voice_id,
        "need_noise_reduction": noise_reduction,
        "need_volume_normalization": volume_normalization,
    }
    if prompt_file_id:
        body["clone_prompt"] = {
            "prompt_audio": prompt_file_id,
            "prompt_text": prompt_text or "",
        }
    if preview_text:
        body["text"] = preview_text
        body["model"] = model
    payload = _post("/v1/voice_clone", body, timeout=600)
    print(f"[minimax] clone ok voice_id={voice_id}")
    return payload


def t2a(
    text: str,
    voice_id: str,
    model: str = "speech-2.8-hd",
    speed: float = 1.0,
    pitch: int = 0,
    vol: float = 1.0,
    audio_format: str = "wav",
    sample_rate: int = 44100,
    language_boost: str = "English",
    emotion: str | None = None,
    subtitle_word: bool = False,
    pronunciation: dict[str, str] | None = None,
) -> tuple[bytes, int, str | None]:
    """Synthesize one line. Returns (audio_bytes, audio_length_ms, subtitle_url).

    `pitch` is [-12, 12] and `speed` is [0.5, 2]; both are used to give the read
    the same rises and falls the original speaker had, instead of one flat
    delivery. `pronunciation` maps a literal word to an IPA/pinyin replacement,
    e.g. {"Eva": "(ˈiːvə)"}.
    """
    vs: dict = {
        "voice_id": voice_id,
        "speed": round(speed, 2),
        "vol": round(vol, 3),
        "pitch": int(pitch),
    }
    if emotion:
        vs["emotion"] = emotion
    body: dict = {
        "model": model,
        "text": text,
        "stream": False,
        "language_boost": language_boost,
        "output_format": "hex",
        "voice_setting": vs,
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate": 256000,
            "format": audio_format,
            "channel": 1,
        },
    }
    if subtitle_word:
        body["subtitle_enable"] = True
        body["subtitle_type"] = "word"
    if pronunciation:
        body["pronunciation_dict"] = {
            "tone": [f"{k}/{v}" for k, v in pronunciation.items()]
        }

    payload = _post("/v1/t2a_v2", body, timeout=600)
    data = payload.get("data") or {}
    audio_hex = data.get("audio")
    if not audio_hex:
        raise MiniMaxError(0, f"empty audio in response: {str(payload)[:200]}")
    audio = bytes.fromhex(audio_hex)
    length_ms = int((payload.get("extra_info") or {}).get("audio_length", 0))
    return audio, length_ms, data.get("subtitle_file")


def get_voice(voice_type: str = "all") -> dict:
    return _post("/v1/get_voice", {"voice_type": voice_type}, timeout=120)


if __name__ == "__main__":
    import sys

    if "--auth" in sys.argv:
        # Cheapest possible auth + permission probe: one short preset-voice line.
        audio, ms, _ = t2a(
            "Authentication probe.",
            voice_id="English_Trustworth_Man",
            speed=1.0,
        )
        print(f"[minimax] AUTH OK - synthesized {len(audio)} bytes, {ms} ms")
    elif "--voices" in sys.argv:
        v = get_voice()
        for group in ("system_voice", "voice_cloning", "voice_generation"):
            items = v.get(group) or []
            print(f"{group}: {len(items)}")
            for it in items[:8]:
                print("   ", it.get("voice_id"), "-", it.get("voice_name"))

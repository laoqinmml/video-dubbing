"""Volcano Engine (火山引擎/豆包语音) 录音文件识别 backend.

Auth scheme (verified against the live API):
    x-api-key: <key>                      single key, NOT app-id + access-token
    X-Api-Resource-Id: volc.seedasr.auc
    X-Api-Request-Id: <uuid>

Proof it is the right scheme, from a controlled test:
    real key  -> X-Api-Status-Code 45000092 "task not found"  (auth OK)
    wrong key -> HTTP 401, 45000010 "Invalid X-Api-Key"        (auth rejected)

Flow: POST /submit, then poll POST /query with the same request id.

Why this backend matters: Volcano is a DOMESTIC service answering over a direct
connection, while the Gemini/ZenMux path needs the local proxy. When that proxy
died, transcription was the first thing to break - this removes that dependency.
Its utterances also carry millisecond timestamps, which can replace the
energy-envelope segmentation with real ASR timing.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

HOST = "https://openspeech.bytedance.com"
SUBMIT_PATH = "/api/v3/auc/bigmodel/submit"
QUERY_PATH = "/api/v3/auc/bigmodel/query"
RESOURCE_ID = "volc.seedasr.auc"

# Status codes that mean "still working" while polling.
PENDING = {"20000001", "20000002"}
DONE = "20000000"

DEFAULT_KEY = os.environ.get("VOLC_API_KEY", "")


class VolcanoError(RuntimeError):
    def __init__(self, code, msg, raw=None):
        self.code = code
        self.msg = msg
        self.raw = raw or {}
        hint = ""
        if str(code) in ("45000010", "401"):
            hint = "  <- x-api-key 无效"
        elif str(code) in ("45000092",):
            hint = "  <- 任务不存在（正常：submit 之后才 query 同一个 request-id）"
        elif str(code).startswith("450000") and str(code) != DONE:
            hint = "  <- 检查该 App 是否开通「录音文件识别」以及是否还有额度"
        super().__init__(f"Volcano ASR {code}: {msg}{hint}")


def _key() -> str:
    key = os.environ.get("VOLC_API_KEY") or os.environ.get("VOLC_APP_KEY") or DEFAULT_KEY
    return key.strip()


def _headers(request_id: str, sequence: int = -1) -> dict:
    return {
        "x-api-key": _key(),
        "X-Api-Resource-Id": RESOURCE_ID,
        "X-Api-Request-Id": request_id,
        "X-Api-Sequence": str(sequence),
        "Content-Type": "application/json",
    }


def _body(audio_b64: str, fmt: str) -> dict:
    return {
        "user": {"uid": "dub-pipeline"},
        "audio": {"format": fmt, "data": audio_b64},
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
        },
    }


def submit(path: Path, request_id: str | None = None) -> str:
    path = Path(path)
    fmt = path.suffix.lstrip(".").lower()
    rid = request_id or str(uuid.uuid4())
    audio_b64 = base64.b64encode(path.read_bytes()).decode()
    r = requests.post(
        HOST + SUBMIT_PATH,
        headers=_headers(rid),
        data=json.dumps(_body(audio_b64, fmt)).encode(),
        timeout=600,
    )
    status = r.headers.get("X-Api-Status-Code", "")
    msg = r.headers.get("X-Api-Message", "")
    print(f"[volcano] submit HTTP {r.status_code} status={status} {msg}")
    if r.status_code != 200 or status not in ("", DONE, "20000001", "20000002"):
        raise VolcanoError(status or r.status_code, msg or r.text[:300])
    return rid


def query(request_id: str, timeout: int = 600) -> dict:
    r = requests.post(
        HOST + QUERY_PATH, headers=_headers(request_id),
        data=b"{}", timeout=timeout,
    )
    status = r.headers.get("X-Api-Status-Code", "")
    msg = r.headers.get("X-Api-Message", "")
    return {"status": status, "message": msg, "http": r.status_code,
            "text": r.text}


def transcribe(path: Path, *, poll_interval: float = 2.0,
               max_polls: int = 300) -> dict:
    """Submit and poll until done. Returns text + utterance timestamps."""
    rid = submit(path)
    print(f"[volcano] request-id {rid}")
    for i in range(1, max_polls + 1):
        time.sleep(poll_interval)
        q = query(rid)
        if q["status"] in PENDING:
            if i % 5 == 0:
                print(f"[volcano] still working ({i} polls)")
            continue
        if q["status"] != DONE:
            raise VolcanoError(q["status"], q["message"] or q["text"][:300])
        print(f"[volcano] done after {i} polls")
        return _parse(q)
    raise VolcanoError(0, f"gave up after {max_polls} polls")


def _parse(q: dict) -> dict:
    payload = json.loads(q["text"] or "{}")
    result = payload.get("result") or {}
    utts = []
    for u in result.get("utterances") or []:
        # Keep the per-word times: they are what makes clause-level timing
        # possible. Utterance spans alone are far too coarse (3 spans / 32 s).
        words = []
        for w in u.get("words") or []:
            st, en = w.get("start_time", -1), w.get("end_time", -1)
            if st is None or en is None or st < 0 or en < 0:
                continue
            words.append({"text": w.get("text", ""), "start": st / 1000.0,
                          "end": en / 1000.0})
        utts.append({
            "start": round(u.get("start_time", 0) / 1000.0, 2),
            "end": round(u.get("end_time", 0) / 1000.0, 2),
            "text": u.get("text", ""),
            "words": words,
        })
    return {
        "text": result.get("text", ""),
        "utterances": utts,
        "duration_ms": (payload.get("audio_info") or {}).get("duration", 0),
        "request_id": q.get("request_id", ""),
        "raw": payload,
    }


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).parent.parent / "work" / "bn_clone_ref.mp3"
    print(f"[volcano] key set: {bool(_key())}  resource: {RESOURCE_ID}")
    print(f"[volcano] file: {src.name} ({src.stat().st_size/1024:.0f} KB)")
    try:
        out = transcribe(src)
        print(f"\ntext: {out['text']}")
        print(f"duration {out['duration_ms']} ms, "
              f"utterances {len(out['utterances'])}")
        for u in out["utterances"][:25]:
            print(f"  {u['start']:6.2f} -> {u['end']:6.2f}  {u['text']}")
    except VolcanoError as exc:
        print(f"\n{exc}")

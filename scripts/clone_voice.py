"""Stage 4 - clone the speaker's timbre on MiniMax.

Default reference is the New Year greeting recording (work/bn_clone_ref.mp3),
prepared by prep_clone_source.py: it is a clean dry capture with 23.7 dB of
dynamic range, versus the 7.2 dB compressed factory mix used for the first
clone. A reference clip plus its transcript ("clone_prompt") is also uploaded
because MiniMax uses it to sharpen similarity and stability.

Note: voice_id must be unique per account, so a new id is used for a re-clone.

Common failures: code 2038 = account not real-name verified.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import minimax  # noqa: E402
from config import FFMPEG, WORK  # noqa: E402

# The New Year greeting, first ~7 s, matching work/bn_prompt.mp3.
BN_PROMPT_TEXT = "千好万好身体好，我的客户没烦恼。小财大财意外财，我的客户全发财。心中有梦，眼里有光。"

REF = WORK / "bn_clone_ref.mp3"
PROMPT = WORK / "bn_prompt.mp3"
OLD_STEM = WORK / "sep" / "htdemucs" / "orig_stereo" / "vocals.wav"
OUT = WORK / "voice.json"
STATE = WORK / "mm_state.json"


def build_from_stem(start: float, end: float, dst: Path) -> Path:
    """Fallback path: cut a clip straight out of the old demucs vocal stem."""
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-ss", str(start), "-to", str(end),
         "-i", str(OLD_STEM), "-ac", "1", "-ar", "32000", "-b:a", "128k", str(dst)],
        check=True,
    )
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(REF) if REF.exists() else None,
                    help="audio to clone from (10s-5min, mp3/m4a/wav)")
    ap.add_argument("--prompt", default=str(PROMPT) if PROMPT.exists() else None,
                    help="optional <8s reference clip that sharpens the clone")
    ap.add_argument("--prompt-text", default=BN_PROMPT_TEXT if PROMPT.exists() else None)
    ap.add_argument("--no-prompt", action="store_true",
                    help="skip clone_prompt entirely. MiniMax can leak the tail of "
                         "the prompt text into every synthesized line, so this is "
                         "the safe setting when a leak is detected.")
    ap.add_argument("--voice-id", default="SmacoEvaV2")
    ap.add_argument("--noise-reduction", action="store_true", default=True)
    args = ap.parse_args()

    if args.no_prompt:
        args.prompt = None
        args.prompt_text = None

    if not args.ref:
        raise SystemExit("no reference audio: run prep_clone_source.py first")

    ref = Path(args.ref)
    print(f"[clone] reference   : {ref.name} ({ref.stat().st_size/1024:.0f} KB)")
    file_id = minimax.upload(ref, purpose="voice_clone")

    prompt_id = None
    if args.prompt and Path(args.prompt).exists():
        try:
            prompt_id = minimax.upload(Path(args.prompt), purpose="prompt_audio")
            print(f"[clone] prompt clip : {Path(args.prompt).name} "
                  f"({Path(args.prompt).stat().st_size/1024:.0f} KB)")
        except minimax.MiniMaxError as exc:
            print(f"[clone] prompt audio rejected ({exc}); cloning without it")
            prompt_id = None

    try:
        minimax.clone(
            file_id,
            voice_id=args.voice_id,
            prompt_file_id=prompt_id,
            prompt_text=args.prompt_text if prompt_id else None,
            noise_reduction=args.noise_reduction,
        )
    except minimax.MiniMaxError as exc:
        if exc.code == minimax.ERR_NOT_VERIFIED:
            raise SystemExit(
                "\n阻断：账号未开通音色复刻权限（2038），请先完成实名认证。\n")
        raise

    OUT.write_text(json.dumps({
        "voice_id": args.voice_id,
        "site": "https://api.minimaxi.com",
        "clone_file_id": file_id,
        "prompt_file_id": prompt_id,
        "prompt_text": args.prompt_text if prompt_id else None,
        "reference": str(ref),
        "reference_kb": round(ref.stat().st_size / 1024, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[clone] voice ready: {args.voice_id} -> {OUT}")

    if STATE.exists():
        st = json.loads(STATE.read_text(encoding="utf-8"))
        st["cloned_voice_id"] = args.voice_id
        st["clone_reference"] = str(ref)
        st["current_blocker"] = None
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

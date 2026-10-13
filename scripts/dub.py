"""One command to dub a video: Chinese -> English, in the cloned voice.

    python scripts/dub.py --src "<path to video>" [--slug <short name>] [--force]

Stages, each resumable (a stage is skipped when its output already exists, so a
failed run can simply be re-run):

    1 prep       extract audio, separate vocals (demucs), record the video length
    2 asr        Volcano word-level transcription -> Chinese clause timeline
    3 translate  DeepSeek -> English script in fill mode
    4 fit        synthesize, MEASURE, condense if it overruns, re-synthesize
    5 place      lay the take on the timeline and write subtitles
    6 mux        background music + loudness normalisation -> final MP4

Everything is driven through the existing stage scripts so there is exactly one
implementation of each step. This file only sequences them and reports.

Environment:
    MINIMAX_API_KEY  required for stage 4
    VOLC_API_KEY     required for stage 2 (defaults to the configured key)
    DUB_BGM          optional override for the music file
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

SLUGS_DONE: list[str] = []


def run_stage(name: str, script: str, args: list[str], env: dict,
              log: list) -> bool:
    print(f"\n{'=' * 70}\n=== {name}\n{'=' * 70}")
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(HERE / script), *args],
                          env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = [ln for ln in out.splitlines()
            if ln.strip() and "not listening -> using DIRECT" not in ln]
    for ln in tail[-25:]:
        print("   " + ln)
    ok = proc.returncode == 0
    log.append({"stage": name, "ok": ok, "seconds": round(time.time() - t0, 1),
                "returncode": proc.returncode})
    if not ok:
        print(f"!!! {name} FAILED (exit {proc.returncode})")
    return ok


def stage_done(stage: str, files: list, work: Path, root: Path) -> bool:
    """Is this stage genuinely finished, so it can be skipped?

    Existence of the outputs is NOT enough. Two failures were caught only because
    shipped output happened to be inspected:

    1. `fit` - if the script never fitted the video, single_call.wav exists but is
       too long, and skipping pushes an overrunning take into place_single.py,
       which trims it and quietly loses the closing line.
    2. `mux` - the MP4 only changes when it is REBUILT. 外贸破壁机14's MP4 sat 44
       minutes OLDER than the audio it was meant to contain: an early full run
       produced it, and the later re-fit stopped at `place`. The delivered video
       was voiced from the discarded first draft.

    So every stage also checks that its output is NEWER than its input.
    """
    if not all(p.exists() for p in files):
        return False

    inputs = {
        "place": [work / "single_call.wav", work / "segments_en.json"],
        "mux": [work / "en_aligned.wav", work / "segments_en.json"],
        "fit": [work / "segments_en.json"],
    }.get(stage, [])
    for out in files:
        for src in inputs:
            if src.exists() and out.stat().st_mtime < src.stat().st_mtime:
                print(f"[dub] {out.name} is OLDER than {src.name} "
                      f"-> re-running {stage}")
                return False

    if stage == "fit":
        try:
            m = json.loads((work / "single_call.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return False
        atts = m.get("attempts") or []
        if not atts:
            return False
        if not atts[-1].get("fits", False):
            print("[dub] the saved take does not fit the video -> re-fitting")
            return False
    return True


# The client asked for more than one bed: "don't only use this one music track".
# Every track in the folder is used in turn. DUB_BGM still overrides everything.
BGM_DIR = Path(r"D:\DeepSeek Harness\活力工厂推介曲")


def pick_bgm(slug: str) -> str:
    """Rotate through the music folder, one track per video.

    A stable hash of the slug was tried first and distributed badly - 5 of the 7
    Southeast Asia videos landed on the same track. A persisted counter cycles
    1,2,3,1,2,3... so a batch is genuinely spread across the tracks.
    """
    if os.environ.get("DUB_BGM"):
        return os.environ["DUB_BGM"]
    tracks = sorted(BGM_DIR.glob("*.wav")) if BGM_DIR.exists() else []
    if not tracks:
        return ""
    counter = HERE.parent / "work" / "bgm_counter.json"
    idx = 0
    try:
        if counter.exists():
            idx = int(json.loads(counter.read_text(encoding="utf-8"))["next"])
    except Exception:  # noqa: BLE001
        idx = 0
    chosen = tracks[idx % len(tracks)]
    try:
        counter.parent.mkdir(parents=True, exist_ok=True)
        counter.write_text(json.dumps({"next": (idx + 1) % len(tracks)}),
                           encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(f"[dub] bgm rotation: {(idx % len(tracks)) + 1}/{len(tracks)} "
          f"-> {chosen.name}")
    return str(chosen)


def mux_args(src: Path, slug: str, work: Path) -> list[str]:
    bgm = pick_bgm(slug)
    if not bgm:
        b = Path(r"D:\DeepSeek Harness\活力工厂推介曲\DM_20261010175129_001.wav")
        bgm = str(b) if b.exists() else ""
    if bgm and Path(bgm).exists():
        return ["--bgm", bgm, "--bgm-gain", "-23", "--lufs", "-14",
                "--tag", "EN_dubbed"]
    print("[dub] no background music found - rendering voice only")
    return ["--lufs", "-14", "--tag", "EN_dubbed"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source video path")
    ap.add_argument("--slug", default=None, help="short name; defaults to stem")
    ap.add_argument("--force", action="store_true",
                    help="re-run every stage, ignoring saved state")
    ap.add_argument("--from", dest="start_at", default=None,
                    help="start at this stage (prep/asr/translate/fit/place/mux)")
    ap.add_argument("--until", dest="stop_at", default=None,
                    help="stop AFTER this stage, e.g. --until translate writes "
                         "the script but does not synthesize")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    if not src.exists():
        raise SystemExit(f"missing source video: {src}")
    slug = args.slug or src.stem
    root = HERE.parent

    env = dict(os.environ)
    env["DUB_SLUG"] = slug
    env["DUB_SRC"] = str(src)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUTF8", "1")
    # Every service this pipeline uses is domestic and must be reached directly
    # (Volcano, DeepSeek, MiniMax). An inherited HTTP(S)_PROXY pointing at the
    # local 10808 proxy - which is usually down - makes the SDK raise
    # "Unable to connect to proxy / WinError 10061" instead of failing over, so
    # the proxy is stripped here rather than relied on being detected later.
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy"):
        env.pop(v, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    if not env.get("MINIMAX_API_KEY"):
        raise SystemExit("MINIMAX_API_KEY is not set")
    if not env.get("VOLC_API_KEY"):
        env["VOLC_API_KEY"] = os.environ.get("VOLC_API_KEY", "")

    work = root / "work" / "batch" / slug
    out = root / "out"
    print(f"dub: {src.name}\n  slug : {slug}\n  work : {work}")

    stages = ["prep", "asr", "translate", "fit", "place", "mux"]
    if args.start_at:
        if args.start_at not in stages:
            raise SystemExit(f"unknown stage {args.start_at}")
        stages = stages[stages.index(args.start_at):]
    if args.stop_at:
        if args.stop_at not in stages:
            raise SystemExit(f"unknown --until stage {args.stop_at}")
        stages = stages[: stages.index(args.stop_at) + 1]

    markers = {
        "prep": [work / "vocals_16k.wav", work / "total_duration.txt"],
        "asr": [work / "segments_lines.json"],
        "translate": [work / "segments_en.json"],
        "fit": [work / "single_call.wav", work / "single_call.json"],
        "place": [work / "en_aligned.wav"],
        "mux": [out / f"{slug}_EN_dubbed.mp4"],
    }

    log: list = []
    t_start = time.time()
    # The stages form a dependency chain, so once one stage actually RUNS, every
    # later stage has to run too: its inputs changed. Without this, re-fitting
    # left place_single.py skipped and it re-used an alignment built from the
    # previous take.
    #
    # Naming a stage with --from is also a statement that the work from there is
    # stale, so that stage runs even if its outputs exist - otherwise
    # "--from place" after a re-synthesis silently keeps the OLD alignment.
    dirty = bool(args.force) or bool(args.start_at)
    for stage in stages:
        if not dirty and stage_done(stage, markers[stage], work, root):
            print(f"\n--- {stage}: already done, skipping "
                  f"({', '.join(p.name for p in markers[stage])})")
            log.append({"stage": stage, "ok": True, "skipped": True})
            continue
        dirty = True
        if stage == "prep":
            ok = run_stage("prep", "prep_video.py", [], env, log)
        elif stage == "asr":
            ok = run_stage("asr", "build_segments_volcano.py",
                           ["--audio", str(work / "vocals_16k.wav")], env, log)
        elif stage == "translate":
            ok = run_stage("translate", "translate.py", [], env, log)
        elif stage == "fit":
            ok = run_stage("fit", "fit_script.py", [], env, log)
        elif stage == "place":
            ok = run_stage("place", "place_single.py", [], env, log)
        else:
            ok = run_stage("mux", "mux.py", mux_args(src, slug, work), env, log)
        if not ok:
            print(f"\n[dub] stopped at stage '{stage}'. Fix and re-run the same "
                  f"command - finished stages are skipped.")
            report(slug, work, out, log, time.time() - t_start, ok=False)
            raise SystemExit(1)

    report(slug, work, out, log, time.time() - t_start, ok=True)


def report(slug: str, work: Path, out: Path, log: list, elapsed: float,
           ok: bool) -> None:
    """Write a per-video summary, including what the synthesis cost."""
    cost = {"rounds": 0, "chars": 0, "yuan": 0.0}
    meta = work / "single_call.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            atts = m.get("attempts") or []
            cost["rounds"] = len(atts)
            cost["chars"] = sum(int(a.get("chars", 0)) for a in atts)
            cost["yuan"] = round(cost["chars"] / 1000 * 0.35, 3)
            cost["duration_sec"] = m.get("duration_sec")
            cost["words"] = m.get("words")
        except Exception:  # noqa: BLE001
            pass

    final = out / f"{slug}_EN_dubbed.mp4"
    lines = [f"# {slug}", "", f"- 状态: {'完成' if ok else '未完成'}",
             f"- 耗时: {elapsed:.0f}s"]
    if final.exists():
        lines.append(f"- 成片: {final.name} "
                     f"({final.stat().st_size / 1e6:.1f} MB)")
    if cost["rounds"]:
        lines += [f"- 合成轮次: {cost['rounds']}",
                  f"- 合成字符: {cost['chars']}",
                  f"- 语音成本: ¥{cost['yuan']:.3f}"]
        if cost.get("words"):
            lines.append(f"- 文稿: {cost['words']} 词, "
                         f"{cost['duration_sec']}s 语音")
    lines += ["", "## 阶段", ""]
    for e in log:
        mark = "跳过" if e.get("skipped") else ("OK" if e["ok"] else "失败")
        lines.append(f"- {e['stage']}: {mark}"
                     + (f" ({e['seconds']}s)" if e.get("seconds") else ""))
    (work / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[dub] {'DONE' if ok else 'INCOMPLETE'} in {elapsed:.0f}s  "
          f"(report: {work / 'REPORT.md'})")


if __name__ == "__main__":
    main()

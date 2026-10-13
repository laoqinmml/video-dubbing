"""Build the client review sheet: English script + Chinese back-translation.

The client approves the COPY before any synthesis happens, so the review has to
show what the English actually says - not the Chinese it came from. Each English
line is therefore back-translated literally, and the original Chinese is printed
above for comparison, because the whole point of these Southeast Asia videos is
that the audience was CHANGED: they were shot for Chinese exporters and must now
speak to the overseas buyer.

Output: 待审文案.md in the project root.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DEEPSEEK_BASE, DEEPSEEK_KEY, ROOT  # noqa: E402

OUT_MD = ROOT / "待审文案.md"

JOBS = [
    ("SEA_14_新加坡", "外贸破壁机14 新加坡.m4v", "新加坡"),
    ("SEA_15_越南", "外贸破壁机15 越南.m4v", "越南"),
    ("SEA_16_马来", "外贸破壁机16 马来.m4v", "马来西亚"),
    ("SEA_17_菲律宾", "外贸破壁机17 菲律宾.m4v", "菲律宾"),
    ("SEA_33", "外贸破壁机33（蓝微） 东南亚.mp4", "东南亚（蓝微）"),
    ("SEA_07", "外贸破壁机7（改电机素材2版）.m4v", "通用"),
    ("SEA_04_先试单", "4：先试单，满意再来 重要.mp4", "通用（先试单）"),
]

SYSTEM = """把下面的英文文案逐句回译成中文，用于给客户核对英文的准确含义。
要求：忠实直译含义，不润色、不增删信息、不合并句子，逐句对应。
只输出 JSON 数组 [{"id":<int>,"zh":"<中文>"}]。"""


def back_translate(rows: list[dict]) -> dict[int, str]:
    from openai import OpenAI
    client = OpenAI(base_url=DEEPSEEK_BASE, api_key=DEEPSEEK_KEY, timeout=180.0)
    payload = {"lines": [{"id": r["id"], "en": r["en"]} for r in rows]}
    resp = client.chat.completions.create(
        model="deepseek-chat", temperature=0.2,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)}])
    txt = resp.choices[0].message.content or ""
    m = re.search(r"\[.*\]", txt, re.S)
    return {int(d["id"]): d["zh"] for d in json.loads(m.group(0))} if m else {}


def main() -> None:
    from config import SHARED
    total_cost = 0.0
    blocks = []
    for slug, filename, market in JOBS:
        work = ROOT / "work" / "batch" / slug
        en_p, zh_p, tot_p = (work / "segments_en.json",
                             work / "segments_lines.json",
                             work / "total_duration.txt")
        if not en_p.exists():
            blocks.append(f"## {market} — {filename}\n\n_(尚未生成)_\n")
            continue
        en = json.loads(en_p.read_text(encoding="utf-8"))
        zh_lines = json.loads(zh_p.read_text(encoding="utf-8")) if zh_p.exists() else []
        total = float(tot_p.read_text(encoding="utf-8").strip()) if tot_p.exists() else 0
        words = sum(len(r["en"].split()) for r in en)
        back = back_translate(en)
        info = json.loads((work / "source_info.json").read_text(encoding="utf-8")) \
            if (work / "source_info.json").exists() else {}

        b = [f"## {market}　`{filename}`", ""]
        b.append(f"- 时长 **{total:.1f}s**　|　英文 **{len(en)} 句 / {words} 词**"
                 f"　|　分辨率 {info.get('width','?')}×{info.get('height','?')}")
        b.append("")
        b.append("**中文原稿（原受众）**")
        b.append("")
        b.append("> " + "".join(r["zh"] for r in zh_lines))
        b.append("")
        b.append("**英文文案（面向海外买家）+ 中文回译**")
        b.append("")
        b.append("| # | English | 中文含义 |")
        b.append("|---|---|---|")
        for r in en:
            e = r["en"].replace("|", "\\|")
            z = back.get(r["id"], "").replace("|", "\\|")
            b.append(f"| {r['id']} | {e} | {z} |")
        b.append("")
        blocks.append("\n".join(b))

    doc = ["# 东南亚素材 · 英文文案待审", "",
           "以下 7 条**只出文案、尚未合成**。请逐条核对英文含义。", "",
           "**重点看**：这些视频原本是对**国内出口商**说的，英文已改为**直接对海外买家**说话"
           " —— 不该再出现「做外贸的朋友」「如果你做出口」「你的出口市场」这类表述。", "",
           "---", ""]
    doc.extend(blocks)
    OUT_MD.write_text("\n".join(doc), encoding="utf-8")
    print(f"-> {OUT_MD}  ({len(''.join(doc))} 字符)")
    print("\n".join(doc))


if __name__ == "__main__":
    main()

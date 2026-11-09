#!/usr/bin/env python3
"""把 design-review 母版 HTML 生成为「去模拟手机外壳」版本，供服务端下发。

为什么需要它
------------
`design-review/cassette-learning-interaction-review.html` 是唯一界面母版。它在桌面
浏览器里把应用装进一个**模拟手机外壳**（深色边框、34px 圆角、投影、456x972 固定
比例）—— 那是为了在浏览器里"更形象"。真机本身就有物理外壳，不能再套一层假壳。

APK 构建时由 `android-experience/app/build.gradle` 的 `stageMasterHtml` 任务做 8 处
严格替换来产出"去外壳变体"。改造后网页改由**服务端下发**（这样前端改动只需 push，
用户重开 APP 即最新版），所以服务端必须产出同一份去外壳版，否则真机上会显示成
"手机里套手机"。

与 build.gradle 的关系
----------------------
下面的 REPLACEMENTS 与 `stageMasterHtml` 的 8 组 from/to **逐字一致**（含第 7 处的
`updateSelection()` 修正，见该条注释）。两边必须同步改：改了这里就要改 build.gradle
（或反过来）。

失败即拒绝产出
--------------
任何一条 from 在母版里找不到，脚本**直接报错退出、不写文件**。这是刻意的：
宁可不上线，也不能把模拟手机外壳悄悄发给用户。母版变了就应当先在这里暴露出来。

用法
----
    python3 build_web.py                                 # 输出 deploy/web-dist/index.html
    python3 build_web.py --out /opt/lang/web/index.html   # 指定输出
    python3 build_web.py --check                          # 只校验 8 处锚点，不写文件
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 开源包布局：frontend/build_web.py 与 frontend/<母版>.html 同级，
# 产物写到 frontend/web-dist/index.html。
MASTER_HTML = os.path.join(
    HERE, "cassette-learning-interaction-review.html"
)
DEFAULT_OUT = os.path.join(HERE, "web-dist", "index.html")

REPLACEMENTS = [
    # 1. 页面背景：桌面预览底色 → 机身壳色，等比缩放留下的极窄 letterbox 缝隙与机身同色、不可见。
    (
        "background:var(--skin-body-bg)",
        "background:var(--shell)",
        "body 底色改为机身壳色",
    ),
    # 2. 模拟手机外壳 → 铺满全屏：去掉边框、圆角、投影、宽高比锁定和外边距。
    #    挖孔由下面第 6/8 处在画布内部处理，所以机身可以一直铺到屏幕最顶端。
    (
        "width:min(calc(100vw - 24px),46.9136dvh,456px);aspect-ratio:456/972;padding:7px;"
        "border:1px solid #48524d;border-radius:34px;background:linear-gradient(112deg,#5f6962 0,"
        "#8d9690 9%,#4f5953 20%,#353d38 51%,#6e7871 82%,#aeb6ae 100%);box-shadow:0 28px 68px "
        "rgba(29,34,31,.34),inset 1px 1px rgba(255,255,255,.2),inset -2px -2px rgba(28,36,31,.32)",
        "width:100%;height:100%;margin:0;padding:0;border:0;border-radius:0;"
        "background:var(--shell);box-shadow:none",
        "外壳铺满全屏（去边框/圆角/投影/比例锁）",
    ),
    # 3. 外壳内圈高光线。
    (
        '.repeater-preview::before{content:"";position:absolute;inset:5px;'
        "border:1px solid rgba(236,239,228,.19);border-radius:28px;pointer-events:none}"
        ".repeater-preview::after{content:none}",
        ".repeater-preview::before,.repeater-preview::after{content:none;display:none}",
        "去掉外壳内圈高光线",
    ),
    # 4. 屏幕内嵌：去掉 27px 圆角。
    (
        ".fit{position:relative;width:100%;height:100%;overflow:hidden;"
        "border-radius:27px;background:var(--shell)}",
        ".fit{position:relative;width:100%;height:100%;overflow:hidden;"
        "border-radius:0;background:var(--shell)}",
        "屏幕内嵌去圆角",
    ),
    # 5. 应用舞台：去掉圆角与屏幕边缘内阴影 —— 这些由真机屏幕本身提供。
    (
        "border:0;border-radius:27px;background:var(--shell);box-shadow:inset 1px 1px "
        "rgba(255,255,255,.82),inset -1px -2px rgba(76,84,78,.18);isolation:isolate",
        "border:0;border-radius:0;background:var(--shell);box-shadow:none;isolation:isolate",
        "舞台去圆角与屏幕内阴影",
    ),
    # 6. 设计画布按安全区"长高"：机身色铺到屏幕最顶端（含挖孔后面），内容避开挖孔。
    (
        "left:50%;top:50%;width:440px;height:956px;overflow:hidden",
        "left:50%;top:50%;width:calc(440px + var(--safe-left,env(safe-area-inset-left)) + "
        "var(--safe-right,env(safe-area-inset-right)));height:calc(956px + "
        "var(--safe-top,env(safe-area-inset-top)) + var(--safe-bottom,env(safe-area-inset-bottom)))"
        ";overflow:hidden",
        "设计画布按安全区长高",
    ),
    # 7. fit() 必须按真实（已长高的）画布缩放，不能写死 440/956。
    #    ⚠️ 2026-09-14 修正：母版后来在 setProperty(...) 之后插入了 updateSelection()，
    #    原锚点 `956))}fit();` 已不匹配 → APK 构建（stageMasterHtml 第 7 处）会直接报
    #    "pattern not found" 而失败。这里把 updateSelection() 一并纳入 替换，
    #    即保住该调用又达成"按真实画布缩放"的原意。build.gradle 需同步修正。
    (
        "function fit(){const f=$('#fit');$('#stage').style.setProperty('--fit',"
        "Math.min(f.clientWidth/440,f.clientHeight/956));updateSelection();}fit();",
        "function fit(){const f=$('#fit'),s=$('#stage');s.style.setProperty('--fit',"
        "Math.min(f.clientWidth/s.offsetWidth,f.clientHeight/s.offsetHeight));"
        "updateSelection();}fit();",
        "fit() 按真实画布缩放（保留 updateSelection）",
    ),
    # 8. 画布内避让挖孔，作为最后一段 CSS 追加（放在 </style> 前所以优先级最高）。
    #    主机顶栏用 margin 下移（标题是顶对齐的，加高不会移动它）；按键靠 main 的
    #    下内边距避开手势条；磁带柜网格吸收多出的高度；横屏时两侧安全区成为视图内边距。
    #    没有安全区时所有 calc 归零，预览与母版完全一致。
    (
        "</style>",
        ".main .topbar{margin-top:calc(3px + var(--safe-top,env(safe-area-inset-top)))}\n"
        ".main{padding-top:10px;padding-bottom:calc(10px + var(--safe-bottom,env(safe-area-inset-bottom)))}\n"
        ".library .library-grid{height:calc(830px + var(--safe-top,env(safe-area-inset-top)))}\n"
        ".view{padding-left:calc(14px + var(--safe-left,env(safe-area-inset-left)));"
        "padding-right:calc(14px + var(--safe-right,env(safe-area-inset-right)))}\n"
        "</style>",
        "追加挖孔/手势条安全区 CSS",
    ),
]


def build(master_path: str) -> str:
    """读母版、依次做 8 处替换，返回去外壳版文本。任一处找不到即抛错。"""
    if not os.path.isfile(master_path):
        raise SystemExit(f"[build_web] 找不到母版 HTML：{master_path}")

    with open(master_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    for idx, (frm, to, label) in enumerate(REPLACEMENTS, start=1):
        pos = text.find(frm)
        if pos < 0:
            raise SystemExit(
                f"[build_web] 第 {idx} 处替换失败（{label}）：母版里找不到目标片段。\n"
                f"         母版可能已改动，请同步更新 build_web.py 与 "
                f"android-experience/app/build.gradle 的 stageMasterHtml。\n"
                f"         片段（前 90 字符）：{frm[:90]}"
            )
        if text.find(frm, pos + 1) >= 0:
            raise SystemExit(
                f"[build_web] 第 {idx} 处替换失败（{label}）：目标片段在母版里出现多次，"
                f"无法确定应改哪一处。\n"
                f"         片段（前 90 字符）：{frm[:90]}"
            )
        text = text[:pos] + to + text[pos + len(frm):]
        print(f"[build_web] {idx}/8 {label} ✓")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="生成服务端下发的『去外壳版』前端页面")
    ap.add_argument("--master", default=MASTER_HTML, help="母版 HTML 路径")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出文件路径")
    ap.add_argument("--check", action="store_true", help="只校验 8 处锚点，不写文件")
    args = ap.parse_args()

    text = build(args.master)

    if args.check:
        print(f"[build_web] 8 处锚点全部匹配（母版 {args.master}）")
        return 0

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"[build_web] 已写出 {args.out}")
    print(f"[build_web] 字节数 {len(text.encode('utf-8'))}  sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

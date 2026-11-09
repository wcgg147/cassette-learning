# -*- coding: utf-8 -*-
"""从母版 HTML 抽出所有 <script> 块，用 node --check 做语法检查。

用法：py -3.12 tools/check_frontend_js.py
依赖：PATH 里有 node（没有的话会明确报出来，不静默跳过）。
拆包后前端是单文件、零外部资源，这个检查是「母版改完没改坏」的第一道闸。
"""
import io
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MASTER = os.path.join(ROOT, "frontend", "cassette-learning-interaction-review.html")

html = io.open(MASTER, encoding="utf-8").read()
blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
print("master:", MASTER)
print("script blocks:", len(blocks))
if not blocks:
    print("FAILED: no <script> blocks found")
    sys.exit(1)

ok = True
for i, block in enumerate(blocks):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        f.write(block)
        path = f.name
    try:
        r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        if r.returncode != 0:
            ok = False
            print(f"block {i}: SYNTAX ERROR\n{r.stderr[:1500]}")
        else:
            print(f"block {i}: OK ({len(block)} chars)")
    except FileNotFoundError:
        print("FAILED: node not found in PATH")
        sys.exit(2)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)

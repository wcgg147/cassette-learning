"""Dump the HTTP route surface of a backend package as JSON.

Usage:  py -3.12 route_dump.py <api_dir> <out.json>

Run with cwd = api_dir so the package's own `sys.path` bootstrap applies.
"""
import json
import os
import sys

api_dir = os.path.abspath(sys.argv[1])
out_path = os.path.abspath(sys.argv[2])
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)
os.chdir(api_dir)

import main  # noqa: E402

routes = []
for r in main.app.routes:
    if not hasattr(r, "path"):
        continue
    methods = sorted(r.methods or [])
    routes.append([" ".join(methods), r.path])

routes.sort()
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(routes, fh, ensure_ascii=False, indent=1)
print("ROUTES_DUMPED", len(routes), out_path)

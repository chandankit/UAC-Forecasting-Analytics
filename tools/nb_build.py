"""Build a notebook from a percent-format script and execute it top-to-bottom in a clean kernel.
Usage: python tools/nb_build.py <source.py> <out.ipynb> [--no-exec] [--timing-json path]"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import nbformat
from nbclient import NotebookClient

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
execute = "--no-exec" not in sys.argv
timing_json = sys.argv[sys.argv.index("--timing-json") + 1] if "--timing-json" in sys.argv else None

cells, kind, buf = [], None, []


def flush():
    text = "\n".join(buf).strip("\n")
    if kind is None or not text:
        return
    if kind == "md":
        lines = [l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l) for l in text.splitlines()]
        cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
    else:
        cells.append(nbformat.v4.new_code_cell(text))


for line in src.read_text(encoding="utf-8-sig").splitlines():
    if line.startswith("# %%"):
        flush()
        kind, buf = ("md" if "[markdown]" in line else "code"), []
    else:
        buf.append(line)
flush()

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
status = "not executed"
if execute:
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=3600, kernel_name="python3", record_timing=True,
                       resources={"metadata": {"path": str(dst.parent)}}).execute()
        status = "ok"
    except Exception as e:  # keep partial outputs for diagnosis
        status = f"error: {type(e).__name__}: {str(e)[:400]}"
    print(f"executed in {time.time() - t0:.0f}s -> {status}")
    if timing_json:
        rows = []
        for i, c in enumerate(nb.cells):
            if c.cell_type != "code":
                continue
            ex = c.metadata.get("execution", {})
            a, b = ex.get("iopub.execute_input"), ex.get("shell.execute_reply")
            sec = (datetime.fromisoformat(b.replace("Z", "+00:00")) - datetime.fromisoformat(a.replace("Z", "+00:00"))).total_seconds() if a and b else None
            rows.append({"cell": i, "header": c.source.splitlines()[0][:90] if c.source else "", "seconds": sec,
                         "errors": [o.get("ename") for o in c.outputs if o.output_type == "error"]})
        Path(timing_json).write_text(json.dumps({"status": status, "total_seconds": time.time() - t0, "cells": rows}, indent=1))
nbformat.write(nb, dst)
print(f"wrote {dst} ({len(cells)} cells)")
if status.startswith("error"):
    sys.exit(1)

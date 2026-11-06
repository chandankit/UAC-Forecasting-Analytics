"""Export a notebook to an editable percent-format script (# %% code cells, # %% [markdown] cells).
Usage: python tools/nb_export.py <notebook.ipynb> <out.py>"""
import sys
from pathlib import Path
import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
out = []
for c in nb.cells:
    if c.cell_type == "markdown":
        out.append("# %% [markdown]")
        out += [("# " + l) if l.strip() else "#" for l in c.source.splitlines()]
    elif c.cell_type == "code":
        out.append("# %%")
        out += c.source.splitlines()
    out.append("")
Path(sys.argv[2]).write_text("\n".join(out), encoding="utf-8")
print(f"exported {len(nb.cells)} cells -> {sys.argv[2]}")

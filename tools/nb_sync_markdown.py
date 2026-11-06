"""Copy markdown edits from the percent-format source into an already executed notebook, without re-execution.
Refuses to run unless every code cell in the source is identical to the executed notebook's code cells (so outputs stay valid).
Usage: python tools/nb_sync_markdown.py <source.py> <executed.ipynb> <out.ipynb>"""
import sys
from pathlib import Path
import nbformat

src, executed, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cells, kind, buf = [], None, []


def flush():
    text = "\n".join(buf).strip("\n")
    if kind and text:
        if kind == "md":
            text = "\n".join(l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l) for l in text.splitlines())
        cells.append((kind, text))


for line in src.read_text(encoding="utf-8-sig").splitlines():
    if line.startswith("# %%"):
        flush()
        kind, buf = ("md" if "[markdown]" in line else "code"), []
    else:
        buf.append(line)
flush()

nb = nbformat.read(executed, as_version=4)
if len(cells) != len(nb.cells):
    sys.exit(f"cell count differs: source {len(cells)} vs executed {len(nb.cells)}")
changed = 0
for i, ((k, text), c) in enumerate(zip(cells, nb.cells)):
    if (k == "code") != (c.cell_type == "code"):
        sys.exit(f"cell {i}: type differs")
    if k == "code":
        if text != c.source:
            sys.exit(f"cell {i}: code differs from the executed notebook -> re-execute instead")
    elif text != c.source:
        c.source = text
        changed += 1
nbformat.validate(nb)
nbformat.write(nb, dst)
print(f"markdown cells updated: {changed}; code cells verified identical: {sum(k == 'code' for k, _ in cells)} -> {dst}")

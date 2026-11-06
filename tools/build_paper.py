"""Build the research paper, its Appendix volume and the executive summary, export them to PDF with Word, and run the quality checks.

    python tools/build_paper.py                  # full build, then tools/check_paper.py (exit code = number of failed checks)
    python tools/build_paper.py --no-appendix    # paper and summary only (faster while editing)

Word (via PowerShell COM) gives the true page count; close the .docx files in Word first.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from md2docx import Builder  # noqa: E402

PAPER = "Research_Paper_UAC_Forecasting"
APPENDIX = "Research_Paper_Appendix"
SUMMARY = "Executive_Summary_UAC_Forecasting"


def _word(script):
    out = subprocess.run(["powershell", "-NoProfile", "-Command", "$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0; " + script],
                         capture_output=True, text=True, timeout=1500)
    if out.returncode:
        raise RuntimeError(f"Word failed: {out.stderr[-500:]}")
    return out.stdout


def clear_word_identity(docx):
    """When Word saves a file it records the Windows user as 'last modified by'. Put the document's own author back, so no personal name is published."""
    doc = Document(docx)
    doc.core_properties.last_modified_by = doc.core_properties.author
    doc.save(docx)


def to_pdf(docx, update_fields=False):
    """Export a .docx to a .pdf next to it using Word and return the page count Word reports.

    With update_fields, Word first fills in the table of contents and saves; the saved file's identity stamp is then cleared,
    and the PDF is exported in a second, read-only pass, so the PDF is always newer than the Word file it came from."""
    docx, pdf = Path(docx).resolve(), Path(docx).with_suffix(".pdf").resolve()
    if update_fields:
        _word(f"$d = $w.Documents.Open('{docx}', $false, $false); $d.Repaginate(); $null = $d.Fields.Update(); "
              f"foreach ($t in $d.TablesOfContents) {{ $t.Update() }}; $d.Save(); $d.Close(0); $w.Quit()")
        clear_word_identity(docx)
    out = _word(f"$d = $w.Documents.Open('{docx}', $false, $true); $d.Repaginate(); $n = $d.ComputeStatistics(2); "
                f"$d.ExportAsFixedFormat('{pdf}', 17); $d.Close(0); $w.Quit(); Write-Output $n")
    if not out.strip():
        raise RuntimeError(f"Word export failed for {docx}")
    return int(out.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-appendix", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    Builder(ROOT / f"{PAPER}.md", ROOT / f"{PAPER}.docx").build()
    print(f"{PAPER}: {to_pdf(ROOT / f'{PAPER}.docx')} pages")
    Builder(ROOT / f"{SUMMARY}.md", ROOT / f"{SUMMARY}.docx").build()
    print(f"{SUMMARY}: {to_pdf(ROOT / f'{SUMMARY}.docx')} pages")
    if not a.no_appendix:
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_paper_appendix.py")], check=True)
        Builder(ROOT / f"{APPENDIX}.md", ROOT / f"{APPENDIX}.docx", table_pt=7.5).build()
        print(f"{APPENDIX}: {to_pdf(ROOT / f'{APPENDIX}.docx', update_fields=True)} pages")
    if not a.no_check:
        sys.exit(subprocess.run([sys.executable, str(ROOT / "tools" / "check_paper.py")]).returncode)


if __name__ == "__main__":
    main()

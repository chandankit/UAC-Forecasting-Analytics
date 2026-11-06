"""Checks that keep the public repository deployable on Streamlit Community Cloud and its README honest."""
import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from conftest import ROOT

README = (ROOT / "README.md").read_text(encoding="utf-8")
NOT_PUBLISHED = {"intern.pdf", "unified_details.png", "Project_Details.txt", "launch.json"}     # personal or course documents, local settings


def published_files():
    """Files git tracks or has staged, or None outside a git checkout."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    files = set(out.splitlines())
    return files or None


def test_readme_links_point_to_published_files():
    links = [link for link in re.findall(r"\]\(([^)\s]+)\)", README) if not re.match(r"[a-z]+://|#|mailto:", link)]
    assert links
    files = published_files()
    for link in links:
        path = link.split("#")[0]
        assert (ROOT / path).exists(), f"README links to a missing file: {link}"
        if files is not None:
            assert path in files or any(f.startswith(path.rstrip("/") + "/") for f in files), f"{link} exists here but is not published (add it to .gitignore)"


def test_readme_run_commands_match_the_entry_point():
    assert "python -m streamlit run uac_forecast_dashboard.py" in README
    assert (ROOT / "uac_forecast_dashboard.py").is_file()
    assert "pip install -r requirements.txt" in README and "git clone https://github.com/chandankit/UAC-Forecasting-Analytics.git" in README


def test_streamlit_config_does_not_pin_the_server():
    """Community Cloud must be able to serve the app: a pinned address or port can stop it. Visitors must never see tracebacks."""
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert not {"address", "port"} & set(cfg.get("server", {}))
    assert cfg["client"]["showErrorDetails"] == "type"


def test_requirements_parse_cleanly_and_cover_every_app_import():
    raw = (ROOT / "requirements.txt").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "a byte-order mark can break uv, which Community Cloud uses to install requirements"
    names = {re.split(r"[=<>!~\[ ;]", line.strip())[0].lower() for line in raw.decode("utf-8").splitlines() if line.strip() and not line.startswith("#")}
    alias = {"sklearn": "scikit-learn", "docx": "python-docx"}
    for path in [ROOT / "uac_forecast_dashboard.py", *sorted((ROOT / "src").glob("*.py"))]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):      # Python accepts a leading byte-order mark in source files
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top in sys.stdlib_module_names or top == "src":
                    continue
                assert alias.get(top, top).lower() in names, f"{path.name} imports {top}, which requirements.txt does not install"


def test_app_reads_only_relative_paths():
    source = (ROOT / "uac_forecast_dashboard.py").read_text(encoding="utf-8")
    assert not re.search(r"(?i)\b[A-Z]:[\\/]", source), "absolute Windows path in the app"
    assert 'Path(__file__).resolve().parent / "artifacts"' in source


def test_nothing_personal_or_local_is_published():
    files = published_files()
    if files is None:
        pytest.skip("not a git checkout")
    assert not NOT_PUBLISHED & {Path(f).name for f in files}
    local_path = re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]")
    for f in sorted(files):
        if Path(f).suffix in {".py", ".md", ".csv", ".toml", ".ini", ".txt", ".json", ".ipynb"}:
            assert not local_path.search((ROOT / f).read_text(encoding="utf-8", errors="replace")), f"local path in {f}"

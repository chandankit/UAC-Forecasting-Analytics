"""ASGI entrypoint for hosting the Streamlit dashboard on Vercel.

Vercel's Python runtime serves ASGI/WSGI apps and loads the top-level `app` from this file. Streamlit >= 1.67 exposes
the dashboard as an ASGI app through `st.App`. The dashboard itself stays in uac_forecast_dashboard.py, so local runs
(`python -m streamlit run uac_forecast_dashboard.py`) and Streamlit Community Cloud are unchanged.
"""
import os
from pathlib import Path

# Serverless instances have a read-only filesystem and are never edited in place: no file watcher, no browser launch.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

import streamlit as st  # noqa: E402  (environment must be set before Streamlit reads its config)

app = st.App(str(Path(__file__).resolve().parent / "uac_forecast_dashboard.py"))

"""
Karpa Live — minimal monitoring dashboard.

Reads from the local chain state + run outputs and displays:
  - Current king + king history
  - Submission feed with scores and tier
  - Noise floor reference
  - Training loss curves from completed runs
  - Calibration reference timings

Usage:
    pip install 'karpa-subnet[dashboard]'
    streamlit run dashboard/app.py -- --karpa-root /path/to/karpa

Phase 0.5: reads local JSON files. Phase 1+: reads from Bittensor chain +
HuggingFace Hub.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def load_events(karpa_root: Path) -> list[dict]:
    path = karpa_root / "chain" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_king(karpa_root: Path) -> dict | None:
    path = karpa_root / "chain" / "king.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_noise_floor(karpa_root: Path) -> dict | None:
    for d in ["runs/h100_noise_floor", "runs/noise_floor"]:
        path = karpa_root / d / "noise_floor_summary.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def load_calibration(karpa_root: Path) -> dict | None:
    for d in ["runs/h100_calibration", "runs"]:
        path = karpa_root / d / "calibration.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def load_training_log(log_path: Path) -> pd.DataFrame | None:
    if not log_path.exists():
        return None
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    return pd.DataFrame(lines)


def find_training_runs(karpa_root: Path) -> list[tuple[str, Path]]:
    runs_dir = karpa_root / "runs"
    if not runs_dir.exists():
        return []
    results = []
    for run_dir in sorted(runs_dir.iterdir()):
        log = run_dir / "training_log.jsonl"
        if not log.exists():
            log = run_dir / "training" / "training_log.jsonl"
        if log.exists():
            results.append((run_dir.name, log))
    return results


def main():
    st.set_page_config(page_title="Karpa Live", page_icon="⛰️", layout="wide")
    st.title("⛰️ Karpa Live")
    st.caption("Phase 0.5 monitoring dashboard — canonical baseline trajectory, submissions, and network health")

    # Auto-refresh selector (applied at the bottom of the page after all content renders).
    refresh = st.sidebar.selectbox("Auto-refresh", ["Off", "10s", "30s", "60s"], index=1)

    karpa_root = Path(sys.argv[-1]) if len(sys.argv) > 1 and Path(sys.argv[-1]).exists() else Path(".")

    # --- Current King ---
    king = load_king(karpa_root)
    col1, col2, col3 = st.columns(3)
    if king:
        col1.metric("👑 Current King", king.get("miner_hotkey", "?")[:20])
        col2.metric("val_bpb", f"{king.get('val_bpb', 0):.4f}")
        col3.metric("Bundle", king.get("bundle_hash", "?")[:12] + "…")
    else:
        st.info("No king crowned yet. Run the smoke test or submit a baseline.")

    st.divider()

    # --- Two columns: events + noise floor ---
    left, right = st.columns([2, 1])

    with left:
        st.subheader("📋 Submission Feed")
        events = load_events(karpa_root)
        scored = [e for e in events if e.get("type") == "submission_scored"]
        if scored:
            rows = []
            for e in reversed(scored):
                rows.append({
                    "miner": e.get("miner_hotkey", "?")[:20],
                    "val_bpb": round(e.get("val_bpb", 0), 4),
                    "quality_gain": round(e.get("quality_gain", 0), 4),
                    "score": round(e.get("score", 0), 4),
                    "accepted": "✅" if e.get("accepted_as_king") else "❌",
                    "decisive": "✅" if e.get("decisively_beats_king") else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No submissions yet.")

        st.subheader("📜 Chain Events")
        if events:
            for e in reversed(events[-20:]):
                icon = {"initial_king": "👑", "king_changed": "🔄", "submission_scored": "📊",
                        "submission_rejected": "🚫"}.get(e.get("type", ""), "•")
                hotkey = e.get("miner_hotkey") or e.get("new_king", {}).get("miner_hotkey", "")
                st.text(f"{icon} {e['type']:25s}  {hotkey[:16]}")
        else:
            st.caption("No events yet.")

    with right:
        st.subheader("📊 Noise Floor")
        nf = load_noise_floor(karpa_root)
        if nf:
            st.metric("val_bpb mean", f"{nf['val_bpb']['mean']:.4f}")
            st.metric("val_bpb std (σ)", f"{nf['val_bpb']['std']:.4f}")
            st.metric("Margin (2σ)", f"{nf['suggested_noise_floor_margin']:.4f}")
            st.metric("Runs", nf["runs"])
            values = nf["val_bpb"]["values"]
            fig = px.strip(y=values, labels={"y": "val_bpb"}, title="Per-seed val_bpb")
            fig.update_layout(height=200, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Run noise_floor.py first.")

        st.subheader("🖥️ Calibration")
        cal = load_calibration(karpa_root)
        if cal:
            st.metric("GPU", cal.get("gpu_name", "CPU"))
            st.metric("Matmul", f"{cal['matmul_ms']:.3f} ms")
            st.metric("Attention", f"{cal['attention_ms']:.3f} ms")
            st.metric("Total", f"{cal['total_ms']:.3f} ms")
        else:
            st.caption("No calibration data.")

    st.divider()

    # --- Live Training Progress ---
    runs = find_training_runs(karpa_root)
    if runs:
        latest_name, latest_log = runs[-1]
        df_latest = load_training_log(latest_log)
        if df_latest is not None and len(df_latest) > 0:
            last = df_latest.iloc[-1]
            has_total = "total_steps" in df_latest.columns
            total_steps = int(last.get("total_steps", 0)) if has_total else None

            st.subheader(f"🔴 Live: {latest_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Step", f"{int(last['step'])}" + (f" / {total_steps}" if total_steps else ""))
            c2.metric("Loss", f"{last['loss']:.4f}")
            c3.metric("Tok/s", f"{last.get('tokens_per_sec', 0):,.0f}")
            c4.metric("Elapsed", f"{last.get('elapsed_s', 0) / 60:.1f} min")

            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(x=df_latest["step"], y=df_latest["loss"],
                                          mode="lines", name="loss", line=dict(color="#ff6b6b")))
            fig_live.update_layout(xaxis_title="Step", yaxis_title="Loss", height=300,
                                   margin=dict(t=10, b=30))
            st.plotly_chart(fig_live, use_container_width=True)

    st.divider()

    # --- All Training Loss Curves ---
    st.subheader("📈 Training Loss Curves")
    if runs:
        selected = st.multiselect("Select runs", [name for name, _ in runs],
                                  default=[runs[-1][0]] if runs else [])
        if selected:
            fig = go.Figure()
            for name, log_path in runs:
                if name in selected:
                    df = load_training_log(log_path)
                    if df is not None and "step" in df.columns and "loss" in df.columns:
                        fig.add_trace(go.Scatter(x=df["step"], y=df["loss"], mode="lines", name=name))
            fig.update_layout(xaxis_title="Step", yaxis_title="Loss", height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No training runs found yet.")

    # Auto-refresh: sleep AFTER all content has rendered, then rerun.
    if refresh != "Off":
        import time as _time
        secs = {"10s": 10, "30s": 30, "60s": 60}[refresh]
        _time.sleep(secs)
        st.rerun()


if __name__ == "__main__":
    main()

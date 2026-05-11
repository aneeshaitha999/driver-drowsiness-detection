

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


LOGS_DIR = Path("logs")


def latest_log_file() -> Path | None:
    files = sorted(LOGS_DIR.glob("drowsiness_log_*.csv"))
    return files[-1] if files else None


def main() -> None:
    st.set_page_config(page_title="Driver Drowsiness Dashboard", layout="wide")
    st.title("AI-Powered Driver Drowsiness Detection Dashboard")

    log_file = latest_log_file()
    if not log_file:
        st.warning("No logs found yet. Run app.py first to generate CSV logs.")
        return

    st.caption(f"Using log file: {log_file}")
    df = pd.read_csv(log_file)
    if df.empty:
        st.warning("Log file exists but contains no rows yet.")
        return

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Risk Score", f"{float(df['risk_score'].iloc[-1]):.1f}")
    col2.metric("Blink Count", f"{int(df['blink_count'].max())}")
    col3.metric("Yawn Count", f"{int(df['yawn_count'].max())}")
    col4.metric("Alert Frames", f"{int((df['status'] == 'ALERT').sum())}")

    st.subheader("Signal Trends")
    st.line_chart(df.set_index("timestamp")[["ear", "mar", "risk_score"]], use_container_width=True)

    st.subheader("Status Distribution")
    st.bar_chart(df["status"].value_counts())

    st.subheader("Recent Events")
    event_rows = df[df["event"].astype(str).str.len() > 0].tail(30)
    if event_rows.empty:
        st.info("No explicit event tags yet.")
    else:
        st.dataframe(event_rows[["timestamp", "status", "event", "ear", "mar", "risk_score"]], use_container_width=True)

    st.subheader("Raw Log Data")
    st.dataframe(df.tail(200), use_container_width=True)


if __name__ == "__main__":
    main()


import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_FILE = PROJECT_ROOT / "monitoring" / "reference_stats.json"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "digit_db"),
    "user": os.getenv("DB_USER", "digit_user"),
    "password": os.getenv("DB_PASSWORD", "digit_password"),
}

FEATURES = [
    "input_mean",
    "input_std",
    "input_min",
    "input_max",
    "nonzero_ratio",
]

st.set_page_config(
    page_title="Digit Recognizer Monitoring",
    
    layout="wide",
)

st.title("Digit Recognizer Monitoring Dashboard")

if st.button("Refresh dashboard"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data(ttl=5)
def load_data():
    query = """
        SELECT
            id, request_time, prediction, confidence,
            cached, latency_ms, error,
            input_mean, input_std, input_min,
            input_max, nonzero_ratio
        FROM predictions
        ORDER BY request_time DESC
        LIMIT 10000
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)

    if not df.empty:
        df["request_time"] = pd.to_datetime(df["request_time"])
        df = df.sort_values("request_time")

    return df


@st.cache_data
def load_reference():
    with REFERENCE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def calculate_psi(reference, live, bin_count=10):
    """Compare two distributions using reference-based quantile bins."""
    reference = np.asarray(reference, dtype=float)
    live = np.asarray(live, dtype=float)

    reference = reference[np.isfinite(reference)]
    live = live[np.isfinite(live)]

    if len(reference) == 0 or len(live) == 0:
        return np.nan

    # Choose bin boundaries using reference data only.
    quantiles = np.linspace(0, 1, bin_count + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    inner_edges = edges[1:-1]

    # Search-based binning also handles live values outside
    # the reference range by placing them in the outer bins.
    ref_bins = np.searchsorted(inner_edges, reference, side="right")
    live_bins = np.searchsorted(inner_edges, live, side="right")

    bins = len(inner_edges) + 1
    ref_counts = np.bincount(ref_bins, minlength=bins).astype(float)
    live_counts = np.bincount(live_bins, minlength=bins).astype(float)

    # Smoothing avoids division by zero for empty bins.
    epsilon = 1e-6
    ref_pct = (ref_counts + epsilon) / (
        ref_counts.sum() + epsilon * bins
    )
    live_pct = (live_counts + epsilon) / (
        live_counts.sum() + epsilon * bins
    )

    return float(
        np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct))
    )


def drift_status(psi):
    if not np.isfinite(psi):
        return "Insufficient data"
    if psi < 0.10:
        return "Low"
    if psi < 0.25:
        return "Moderate"
    return "High"


try:
    df = load_data()
except psycopg.Error as exc:
    st.error(f"Could not connect to PostgreSQL: {exc}")
    st.stop()

if df.empty:
    st.warning("No prediction records found. Submit some images first.")
    st.stop()


# API performance
st.header("1. API Performance")

total = len(df)
failed = df["error"].notna().sum()
success = total - failed
latencies = df["latency_ms"].dropna()
confidence = df["confidence"].dropna()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Requests", f"{total:,}")
c2.metric("Successful", f"{success:,}")
c3.metric("Error rate", f"{failed / total * 100:.2f}%")
c4.metric(
    "Average latency",
    f"{latencies.mean():.2f} ms" if not latencies.empty else "N/A",
)
c5.metric(
    "P95 latency",
    f"{latencies.quantile(0.95):.2f} ms"
    if not latencies.empty else "N/A",
)

latency_df = df.dropna(subset=["latency_ms"]).set_index("request_time")
st.subheader("Latency over time")
if not latency_df.empty:
    st.line_chart(latency_df["latency_ms"])

st.subheader("Requests over time")
st.bar_chart(
    df.set_index("request_time")
    .resample("1min")
    .size()
    .rename("requests")
)


# Prediction monitoring
st.header("2. Prediction Monitoring")

left, right = st.columns(2)

with left:
    st.subheader("Predicted digit distribution")
    digit_counts = (
        df["prediction"].dropna().astype(int)
        .value_counts()
        .reindex(range(10), fill_value=0)
        .sort_index()
    )
    st.bar_chart(digit_counts)

with right:
    st.subheader("Confidence distribution")
    if not confidence.empty:
        st.bar_chart(
            confidence.clip(0, 1).value_counts(
                bins=10, sort=False
            ).sort_index()
        )

cache_rate = df["cached"].mean() * 100
left, right = st.columns(2)
left.metric("Cache hit rate", f"{cache_rate:.2f}%")
right.metric(
    "Average confidence",
    f"{confidence.mean():.3f}" if not confidence.empty else "N/A",
)


# Data drift
st.header("3. Data Drift Detection")
st.caption(
    "Reference: benchmark_images. Live data: recent API input "
    "statistics stored in PostgreSQL."
)

try:
    reference = load_reference()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Could not load reference statistics: {exc}")
    st.info("Run: python monitoring/build_reference.py")
    st.stop()

st.metric(
    "Reference benchmark images",
    reference.get("image_count", "Unknown"),
)

window_size = st.selectbox(
    "Number of recent requests to compare",
    [50, 100, 250, 500, 1000],
    index=2,
)

# df is sorted oldest to newest; select the most recent records.
live_df = df.tail(window_size)

if len(live_df) < 30:
    st.warning(
        f"Only {len(live_df)} requests are available. "
        "Collect at least 30 requests for a more useful comparison."
    )

reference_features = reference.get("features", {})
rows = []

for feature in FEATURES:
    ref_values = reference_features.get(feature, [])
    live_values = live_df[feature].dropna().to_numpy()

    if len(ref_values) == 0 or len(live_values) == 0:
        psi = np.nan
        ref_mean = np.nan
        live_mean = np.nan
        live_count = len(live_values)
    else:
        psi = calculate_psi(ref_values, live_values)
        ref_mean = float(np.mean(ref_values))
        live_mean = float(np.mean(live_values))
        live_count = len(live_values)

    rows.append({
        "Feature": feature,
        "Reference mean": ref_mean,
        "Live mean": live_mean,
        "Live samples": live_count,
        "PSI": psi,
        "Status": drift_status(psi),
    })

drift_df = pd.DataFrame(rows)

valid_psi = drift_df["PSI"].dropna()
if valid_psi.empty:
    st.warning(
        "Not enough reference or live feature data to calculate drift. "
        "Check that server.py stores input statistics."
    )
else:
    overall_psi = valid_psi.mean()
    st.metric("Mean feature PSI", f"{overall_psi:.4f}")
    st.caption(
        "This is the mean of the available feature PSI scores. "
    )

st.subheader("Feature drift results")
st.dataframe(
    drift_df.style.format({
        "Reference mean": "{:.4f}",
        "Live mean": "{:.4f}",
        "PSI": "{:.4f}",
    }),
    use_container_width=True,
)

st.subheader("PSI by feature")
chart_df = drift_df.dropna(subset=["PSI"]).set_index("Feature")
if not chart_df.empty:
    st.bar_chart(chart_df["PSI"])

st.caption(
    "interpretation: PSI < 0.10 = low, "
    "0.10–0.25 = moderate, and >= 0.25 = high drift. "
)

st.subheader("Reference vs live feature means")
means_df = drift_df.set_index("Feature")[
    ["Reference mean", "Live mean"]
]
st.bar_chart(means_df)

# Recent request records
st.header("4. Recent Requests")
st.dataframe(
    df.sort_values("request_time", ascending=False).head(100),
    use_container_width=True,
)
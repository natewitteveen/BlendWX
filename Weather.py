# Weather.py
# BlendWX — Weather Model Comparison (Open-Meteo)
#
# Run (PowerShell):
#   cd "$HOME\OneDrive - Altas Corporation\Desktop\weather-app"
#   .\.venv\Scripts\Activate.ps1
#   python -m streamlit run Weather.py

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -----------------------------
# App constants
# -----------------------------
APP_NAME = "BlendWX"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Models supported via /v1/forecast?models=...
MODEL_CANDIDATES = {
    "ECMWF IFS": ["ecmwf_ifs", "ecmwf_ifs04", "ecmwf_ifs025"],
    "GFS Seamless": ["gfs_seamless"],
    "GEM Seamless": ["gem_seamless"],
}
DEFAULT_MODELS = ["ECMWF IFS", "GFS Seamless", "GEM Seamless"]

HORIZON_OPTIONS = [3, 7, 10, 14, 16]
DEFAULT_HORIZON = 10

PRECIP_BAR_INTERVALS = ["1H", "3H", "6H", "12H", "24H"]
DEFAULT_BAR_INTERVAL = "3H"

# Hourly variables supported in /v1/forecast (keep conservative)
HOURLY_REQUEST = {
    "temp_c": "temperature_2m",
    "feels_like_c": "apparent_temperature",
    "rh_pct": "relative_humidity_2m",
    "cloud_pct": "cloud_cover",
    "precip_mm": "precipitation",
    "precip_prob_pct": "precipitation_probability",
    "wind_kmh": "wind_speed_10m",
    "wind_gust_kmh": "wind_gusts_10m",
    "pressure_hpa": "surface_pressure",
    "shortwave_wm2": "shortwave_radiation",
    "sunshine_duration_s": "sunshine_duration",
    "cape_jkg": "cape",
    "weather_code": "weather_code",
}

SUNSHINE_THRESHOLD_WM2 = 120.0  # threshold to approximate sunny hour if sunshine_duration missing

# X-axis formatting: short ticks, detailed hover (North American, AM/PM)
X_TICKFORMAT = "%I %p"  # e.g., 01 PM
X_HOVERFORMAT = "%a %b %d, %I:%M %p"


# -----------------------------
# HTTP session (retries help with intermittent TLS / proxy flakiness)
# -----------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "BlendWX/1.0"})
    return s


SESSION = make_session()


# -----------------------------
# Data utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def geocode_city(name: str, count: int = 10) -> list[dict]:
    name = (name or "").strip()
    if not name:
        return []
    params = {"name": name, "count": count, "language": "en", "format": "json"}
    r = SESSION.get(GEOCODE_URL, params=params, timeout=25)
    r.raise_for_status()
    return (r.json().get("results") or [])[:count]


def build_params(lat: float, lon: float, forecast_days: int, model_slug: str) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_REQUEST.values()),
        "forecast_days": int(forecast_days),
        "timezone": "auto",
        "timeformat": "iso8601",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "wind_speed_unit": "kmh",
        "models": model_slug,
    }


def parse_time_index(times, tz_name: str | None) -> tuple[pd.DatetimeIndex, str]:
    s = pd.Series(pd.to_datetime(times, errors="coerce"), name="timestamp").dropna()

    tz_label = tz_name or "UTC"
    if tz_name:
        try:
            tz = ZoneInfo(tz_name)
            s = s.dt.tz_localize(tz)
        except Exception:
            tz_label = "UTC"
            s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_localize("UTC")

    return pd.DatetimeIndex(s), tz_label


@st.cache_data(show_spinner=False)
def fetch_one_model(display_name: str, model_slug: str, lat: float, lon: float, forecast_days: int) -> pd.DataFrame:
    params = build_params(lat, lon, forecast_days, model_slug)
    r = SESSION.get(FORECAST_URL, params=params, timeout=35)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.reason}: {r.text[:300]}")
    payload = r.json() or {}

    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    tz_name = payload.get("timezone")

    ts, tz_label = parse_time_index(times, tz_name)
    n = len(ts)

    def get_arr(hourly_key: str) -> pd.Series:
        arr = hourly.get(hourly_key)
        if arr is None:
            return pd.Series([pd.NA] * n)
        arr = list(arr)
        if len(arr) >= n:
            arr = arr[:n]
        else:
            arr = arr + [pd.NA] * (n - len(arr))
        return pd.Series(arr)

    df = pd.DataFrame({"timestamp": ts, "model": display_name, "model_slug": model_slug})
    for canon, api_key in HOURLY_REQUEST.items():
        df[canon] = pd.to_numeric(get_arr(api_key), errors="coerce")

    # clip percent variables
    for c in ["rh_pct", "cloud_pct", "precip_prob_pct"]:
        df[c] = df[c].clip(lower=0, upper=100)

    # defensive: precipitation should not be negative
    df["precip_mm"] = df["precip_mm"].where(df["precip_mm"].isna() | (df["precip_mm"] >= 0), np.nan)

    df.attrs["timezone"] = tz_label
    df.attrs["model_slug"] = model_slug
    return df


@st.cache_data(show_spinner=False)
def fetch_model_with_candidates(display_name: str, lat: float, lon: float, forecast_days: int) -> pd.DataFrame:
    candidates = MODEL_CANDIDATES.get(display_name, [])
    if not candidates:
        raise ValueError(f"Model not configured: {display_name}")

    last_err = None
    for slug in candidates:
        try:
            return fetch_one_model(display_name, slug, lat, lon, forecast_days)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All candidate slugs failed. Last error: {last_err}")


def add_blend(df: pd.DataFrame, label: str = "Blend") -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [c for c in df.columns if c not in ("timestamp", "model", "model_slug")]
    b = df.groupby("timestamp", as_index=False)[numeric_cols].mean(numeric_only=True)
    b["model"] = label
    b["model_slug"] = "blend"
    return pd.concat([df, b], ignore_index=True, sort=False)


def compute_daily_sunshine(df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_hourly.empty:
        return pd.DataFrame(columns=["date", "model", "sunshine_hours", "method"])

    out = df_hourly.copy()
    out["date"] = out["timestamp"].dt.floor("D")

    if out["sunshine_duration_s"].notna().any():
        daily = (
            out.groupby(["date", "model"], as_index=False)["sunshine_duration_s"]
            .sum(min_count=1)
            .rename(columns={"sunshine_duration_s": "sunshine_seconds"})
        )
        daily["sunshine_hours"] = daily["sunshine_seconds"] / 3600.0
        daily["method"] = "Native sunshine duration"
        return daily.drop(columns=["sunshine_seconds"])

    if out["shortwave_wm2"].notna().any():
        out["sunny_hour"] = out["shortwave_wm2"] > SUNSHINE_THRESHOLD_WM2
        daily = (
            out.groupby(["date", "model"], as_index=False)["sunny_hour"]
            .sum(min_count=1)
            .rename(columns={"sunny_hour": "sunshine_hours"})
        )
        daily["sunshine_hours"] = pd.to_numeric(daily["sunshine_hours"], errors="coerce")
        daily["method"] = f"Estimated (shortwave > {SUNSHINE_THRESHOLD_WM2:g} W/m²)"
        return daily

    daily = out.groupby(["date", "model"], as_index=False).size()[["date", "model"]]
    daily["sunshine_hours"] = pd.NA
    daily["method"] = "Not available"
    return daily


def severe_index(df: pd.DataFrame) -> pd.Series:
    """
    Simple 0-100 proxy index (not an official meteorological index):
      - 50% precipitation probability
      - 30% wind gusts (cap 100 km/h)
      - 20% CAPE (cap 2000 J/kg)
      +10 for thunder-like weather codes (95/96/99)
    """
    pop = df["precip_prob_pct"].fillna(0).clip(0, 100)
    gust = df["wind_gust_kmh"].fillna(0).clip(0, 100)
    cape = df["cape_jkg"].fillna(0).clip(0, 2000) / 2000 * 100
    wc = df["weather_code"]
    bump = wc.isin([95, 96, 99]).fillna(False).astype(float) * 10.0
    idx = (0.50 * pop + 0.30 * gust + 0.20 * cape + bump).clip(0, 100)

    if (df["precip_prob_pct"].notna().any() is False) and (df["wind_gust_kmh"].notna().any() is False) and (df["cape_jkg"].notna().any() is False):
        return pd.Series([pd.NA] * len(df), index=df.index)
    return idx


def daily_summary(df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_hourly.empty:
        return pd.DataFrame()

    d = df_hourly.copy()
    d["date"] = d["timestamp"].dt.floor("D")
    agg = d.groupby(["date", "model"], as_index=False).agg(
        temp_min_c=("temp_c", "min"),
        temp_max_c=("temp_c", "max"),
        precip_total_mm=("precip_mm", "sum"),
    )
    sun = compute_daily_sunshine(df_hourly)
    out = agg.merge(sun[["date", "model", "sunshine_hours", "method"]], on=["date", "model"], how="left")
    for c in ["temp_min_c", "temp_max_c", "precip_total_mm", "sunshine_hours"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["temp_min_c"] = out["temp_min_c"].round(1)
    out["temp_max_c"] = out["temp_max_c"].round(1)
    out["precip_total_mm"] = out["precip_total_mm"].round(1)
    out["sunshine_hours"] = out["sunshine_hours"].round(1)
    return out.sort_values(["date", "model"])


# -----------------------------
# Plot helpers (dark theme, hover, day separators)
# -----------------------------
def _midnights_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if start.tzinfo is None or end.tzinfo is None:
        return []
    d0 = start.floor("D") + pd.Timedelta(days=1)
    mids = []
    cur = d0
    while cur < end:
        mids.append(cur)
        cur = cur + pd.Timedelta(days=1)
    return mids


def _apply_chart_theme(fig: go.Figure, start: pd.Timestamp, end: pd.Timestamp):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(18,20,26,0.96)",
            bordercolor="rgba(255,255,255,0.10)",
            font=dict(size=14, color="rgba(255,255,255,0.92)"),
        ),
        margin=dict(l=12, r=12, t=62, b=18),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=12),
        ),
        title=dict(y=0.95),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        tickformat=X_TICKFORMAT,
        hoverformat=X_HOVERFORMAT,
        title_text="Local time",
        zeroline=False,
        nticks=8,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
    )

    # Day separators at midnight
    for m in _midnights_between(start, end):
        fig.add_vline(x=m, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.18)")

    # Centered day labels at midday
    if start.tzinfo is not None:
        day_start = start.floor("D")
        day_end = end.floor("D")
        cur = day_start
        while cur <= day_end:
            mid = cur + pd.Timedelta(hours=12)
            if start <= mid <= end:
                fig.add_annotation(
                    x=mid,
                    y=1.02,
                    xref="x",
                    yref="paper",
                    text=mid.strftime("%a %b %d"),
                    showarrow=False,
                    font=dict(size=12, color="rgba(255,255,255,0.70)"),
                )
            cur += pd.Timedelta(days=1)

    return fig


def _hovertemplate(name: str, unit: str | None):
    u = f" {unit}" if unit else ""
    return f"<b>%{{fullData.name}}</b><br>{name}: <b>%{{y:.1f}}{u}</b><extra></extra>"


def plot_timeseries(
    df: pd.DataFrame,
    y: str,
    title: str,
    y_title: str,
    unit: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    display_mode: str,
    y_range: list[float | None] | None = None,
    to_zero: bool = False,
):
    plot_df = df[["timestamp", "model", y]].copy().dropna(subset=["timestamp"])
    if plot_df.empty:
        st.info("No data available for this metric over the selected time window.")
        return

    if display_mode == "Models only":
        series_order = [m for m in plot_df["model"].unique() if m != "Blend"]
    elif display_mode == "Blend only":
        series_order = ["Blend"] if "Blend" in plot_df["model"].unique() else []
    else:
        series_order = (["Blend"] if "Blend" in plot_df["model"].unique() else []) + [
            m for m in plot_df["model"].unique() if m != "Blend"
        ]

    fig = go.Figure()
    for model_name in series_order:
        g = plot_df[plot_df["model"] == model_name].sort_values("timestamp")
        if g.empty:
            continue

        # Keep model colors, but de-emphasize in Models + Blend
        if display_mode == "Models + Blend":
            if model_name == "Blend":
                line = dict(width=4)
                opacity = 1.0
            else:
                line = dict(width=1)
                opacity = 0.35
        else:
            line = dict(width=3 if model_name == "Blend" else 2)
            opacity = 1.0

        fig.add_trace(
            go.Scatter(
                x=g["timestamp"],
                y=g[y],
                mode="lines",
                name=model_name,
                line=line,
                opacity=opacity,
                hovertemplate=_hovertemplate(y_title, unit),
            )
        )

    fig.update_layout(title=title)
    fig.update_yaxes(title_text=y_title)

    if to_zero:
        fig.update_yaxes(rangemode="tozero")
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    _apply_chart_theme(fig, start, end)
    st.plotly_chart(fig, use_container_width=True)


def plot_precip_bars(
    df: pd.DataFrame,
    title: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    display_mode: str,
    bar_interval: str,
):
    plot_df = df[["timestamp", "model", "precip_mm"]].copy()
    plot_df["precip_mm"] = pd.to_numeric(plot_df["precip_mm"], errors="coerce").fillna(0.0)

    if display_mode == "Blend only":
        plot_df = plot_df[plot_df["model"] == "Blend"]
    elif display_mode == "Models only":
        plot_df = plot_df[plot_df["model"] != "Blend"]

    if plot_df.empty:
        st.info("No precipitation data available for the selected time window.")
        return

    plot_df["bucket"] = plot_df["timestamp"].dt.floor(bar_interval)
    agg = plot_df.groupby(["bucket", "model"], as_index=False)["precip_mm"].sum()

    model_order = (["Blend"] if "Blend" in agg["model"].unique() and display_mode == "Models + Blend" else []) + [
        m for m in agg["model"].unique() if m != "Blend"
    ]

    fig = go.Figure()
    for m in model_order:
        g = agg[agg["model"] == m].sort_values("bucket")
        if g.empty:
            continue

        if display_mode == "Models + Blend":
            marker = dict(opacity=0.85 if m == "Blend" else 0.35)
        else:
            marker = dict(opacity=0.85)

        fig.add_trace(
            go.Bar(
                x=g["bucket"],
                y=g["precip_mm"],
                name=m,
                marker=marker,
                hovertemplate="<b>%{fullData.name}</b><br>Precipitation: <b>%{y:.1f} mm</b><extra></extra>",
            )
        )

    fig.update_layout(title=title, barmode="group")
    fig.update_yaxes(title_text="Precipitation (mm)", rangemode="tozero")
    _apply_chart_theme(fig, start, end)
    st.plotly_chart(fig, use_container_width=True)


def plot_precip_cumulative(
    df: pd.DataFrame,
    title: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    display_mode: str,
):
    x = df[["timestamp", "model", "precip_mm"]].copy()
    x["precip_mm"] = pd.to_numeric(x["precip_mm"], errors="coerce").fillna(0.0)
    x = x.sort_values(["model", "timestamp"])
    x["cum_precip_mm"] = x.groupby("model")["precip_mm"].cumsum()

    if display_mode == "Blend only":
        x = x[x["model"] == "Blend"]
    elif display_mode == "Models only":
        x = x[x["model"] != "Blend"]

    if x.empty:
        st.info("No precipitation data available for the selected time window.")
        return

    tmp = x.rename(columns={"cum_precip_mm": "_cum"})
    plot_timeseries(
        df=tmp,
        y="_cum",
        title=title,
        y_title="Cumulative precipitation (mm)",
        unit="mm",
        start=start,
        end=end,
        display_mode=display_mode,
        y_range=None,
        to_zero=True,
    )


# -----------------------------
# Streamlit page theming (dark, modern) + header/instructions
# -----------------------------
st.set_page_config(page_title=APP_NAME, layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
      .stApp {
        background: radial-gradient(1200px 600px at 15% 0%, rgba(60,95,255,0.18), rgba(0,0,0,0) 55%),
                    radial-gradient(900px 500px at 85% 10%, rgba(0,200,255,0.14), rgba(0,0,0,0) 60%),
                    linear-gradient(180deg, rgba(14,16,22,1) 0%, rgba(10,12,16,1) 100%);
        color: rgba(255,255,255,0.92);
      }
      h1, h2, h3 { letter-spacing: -0.02em; color: rgba(255,255,255,0.92); }
      .blendwx-hero {
        padding: 14px 16px;
        border-radius: 16px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 12px 30px rgba(0,0,0,0.35);
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
      }
      .blendwx-subtle { color: rgba(255,255,255,0.68); font-size: 0.95rem; }
      .blendwx-small { color: rgba(255,255,255,0.62); font-size: 0.88rem; line-height: 1.35; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="blendwx-hero">
      <div style="display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;">
        <h1 style="margin:0;">{APP_NAME}</h1>
        <div class="blendwx-subtle">Compare leading global forecast models — with an optional blended signal.</div>
      </div>
      <div class="blendwx-small" style="margin-top:8px;">
        <b>How to use:</b> Pick a location, choose models, set a forecast horizon and time window.
        Use <b>Series display</b> to show Models only, Blend only, or Models + Blend.
        In Models + Blend, the blend is emphasized for readability.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Location")

    loc_mode = st.radio("Input method", ["City search", "Latitude / Longitude"], horizontal=True)

    lat = lon = None
    location_label = None

    if loc_mode == "City search":
        q = st.text_input("City", placeholder="e.g., Toronto")
        results = []
        if q.strip():
            try:
                results = geocode_city(q, count=10)
            except Exception as e:
                st.error(f"Geocoding failed: {e}")

        if results:
            options = []
            for r in results:
                name = r.get("name")
                admin1 = r.get("admin1")
                country = r.get("country")
                tz = r.get("timezone")
                label = f"{name}, {admin1 or ''} {country or ''}".replace("  ", " ").strip()
                if tz:
                    label += f" (TZ: {tz})"
                options.append(label)

            sel = st.selectbox("Select a match", options, index=0)
            chosen = results[options.index(sel)]
            lat = float(chosen["latitude"])
            lon = float(chosen["longitude"])
            location_label = sel
        else:
            st.caption("Type a city name to search.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", value=43.6532, format="%.6f")
        with c2:
            lon = st.number_input("Longitude", value=-79.3832, format="%.6f")
        location_label = f"Lat {lat:.4f}, Lon {lon:.4f}"

    st.divider()
    st.header("Forecast settings")

    horizon = st.selectbox("Forecast horizon", HORIZON_OPTIONS, index=HORIZON_OPTIONS.index(DEFAULT_HORIZON))

    display_mode = st.radio(
        "Series display",
        ["Models only", "Blend only", "Models + Blend"],
        index=2,
        help="Blend is the mean across the selected models at each timestamp.",
    )

    bar_interval = st.selectbox("Precipitation bar interval", PRECIP_BAR_INTERVALS, index=PRECIP_BAR_INTERVALS.index(DEFAULT_BAR_INTERVAL))

    st.divider()
    st.header("Models")
    models = st.multiselect("Select forecast models", list(MODEL_CANDIDATES.keys()), default=DEFAULT_MODELS)

# Guardrails
if lat is None or lon is None:
    st.stop()
if not models:
    st.warning("Select at least one model.")
    st.stop()

# -----------------------------
# Fetch
# -----------------------------
with st.spinner("Fetching forecasts…"):
    frames = []
    tz_labels = set()
    resolved = []

    for m in models:
        try:
            df_m = fetch_model_with_candidates(m, lat, lon, forecast_days=horizon)
            tz_labels.add(df_m.attrs.get("timezone", "UTC"))
            resolved.append({"Model": m, "Resolved models= slug": df_m.attrs.get("model_slug", "")})
            frames.append(df_m)
        except Exception as e:
            st.error(f"Failed to fetch {m}: {e}")

df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if df_all.empty:
    st.error("No data returned for the selected inputs.")
    st.stop()

# Add blend depending on mode
if display_mode in ["Blend only", "Models + Blend"]:
    df_all = add_blend(df_all, label="Blend")

# Severe index
df_all["severe_index"] = severe_index(df_all)

# Time bounds
min_ts = df_all["timestamp"].min()
max_ts = df_all["timestamp"].max()

st.subheader("Overview")
st.write(f"**Location:** {location_label}")
st.caption(
    f"**Timezone:** {', '.join(sorted(tz_labels))} • "
    f"**Available window:** {min_ts.strftime('%a %b %d, %Y %I:%M %p')} → {max_ts.strftime('%a %b %d, %Y %I:%M %p')}"
)

with st.expander("Model resolution details"):
    st.dataframe(pd.DataFrame(resolved), use_container_width=True)

# Time window slider
default_start = min_ts
default_end = min(max_ts, min_ts + pd.Timedelta(days=horizon))

start_dt, end_dt = st.slider(
    "Time window",
    min_value=min_ts.to_pydatetime(),
    max_value=max_ts.to_pydatetime(),
    value=(default_start.to_pydatetime(), default_end.to_pydatetime()),
    format="MMM D, YYYY h:mm A",
)

start_ts = pd.Timestamp(start_dt)
end_ts = pd.Timestamp(end_dt)
if start_ts.tzinfo is None:
    start_ts = start_ts.tz_localize(min_ts.tz)
if end_ts.tzinfo is None:
    end_ts = end_ts.tz_localize(min_ts.tz)

df = df_all[(df_all["timestamp"] >= start_ts) & (df_all["timestamp"] <= end_ts)].copy()

st.divider()
st.subheader("Forecast charts")

# Requested order:
plot_timeseries(df, "temp_c", "Air temperature", "Temperature (°C)", "°C", start_ts, end_ts, display_mode)
plot_timeseries(df, "feels_like_c", "Feels-like temperature", "Feels-like (°C)", "°C", start_ts, end_ts, display_mode)
plot_timeseries(df, "rh_pct", "Relative humidity", "Humidity (%)", "%", start_ts, end_ts, display_mode, y_range=[0, 100])
plot_timeseries(df, "cloud_pct", "Cloud cover", "Cloud cover (%)", "%", start_ts, end_ts, display_mode, y_range=[0, 100])

st.markdown("### Sunshine (daily)")
daily_sun = compute_daily_sunshine(df)
if daily_sun.empty or daily_sun["sunshine_hours"].isna().all():
    st.info("Sunshine hours are not available for the selected time window/models.")
else:
    if display_mode == "Blend only":
        daily_sun = daily_sun[daily_sun["model"] == "Blend"]
    elif display_mode == "Models only":
        daily_sun = daily_sun[daily_sun["model"] != "Blend"]

    models_order = (["Blend"] if "Blend" in daily_sun["model"].unique() else []) + [
        m for m in daily_sun["model"].unique() if m != "Blend"
    ]

    fig = go.Figure()
    for m in models_order:
        g = daily_sun[daily_sun["model"] == m].sort_values("date")
        if g.empty:
            continue

        if display_mode == "Models + Blend":
            if m == "Blend":
                line = dict(width=4)
                opacity = 1.0
            else:
                line = dict(width=1)
                opacity = 0.35
        else:
            line = dict(width=3 if m == "Blend" else 2)
            opacity = 1.0

        fig.add_trace(
            go.Scatter(
                x=g["date"],
                y=g["sunshine_hours"],
                mode="lines+markers",
                name=m,
                line=line,
                opacity=opacity,
                hovertemplate="<b>%{fullData.name}</b><br>Sunshine: <b>%{y:.1f} h</b><extra></extra>",
            )
        )

    fig.update_layout(title="Sunshine hours (daily)")
    fig.update_yaxes(title_text="Hours", rangemode="tozero")
    fig.update_xaxes(
        title_text="Date",
        tickformat="%a %b %d",
        hoverformat="%a %b %d",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
    )
    _apply_chart_theme(fig, start_ts, end_ts)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Sunshine calculation notes"):
        st.write(
            "BlendWX uses Open-Meteo sunshine duration when available. "
            "If unavailable, it estimates sunshine hours by counting hours where shortwave radiation exceeds a threshold."
        )
        st.dataframe(daily_sun.groupby("model", as_index=False)["method"].first(), use_container_width=True)

st.markdown("### Precipitation")
plot_precip_bars(df, f"Precipitation (bar totals per {bar_interval})", start_ts, end_ts, display_mode, bar_interval)
plot_precip_cumulative(df, "Cumulative precipitation (selected window)", start_ts, end_ts, display_mode)
plot_timeseries(df, "precip_prob_pct", "Precipitation probability", "Probability (%)", "%", start_ts, end_ts, display_mode, y_range=[0, 100])

st.markdown("### Wind")
plot_timeseries(df, "wind_kmh", "Wind speed (10 m)", "Wind speed (km/h)", "km/h", start_ts, end_ts, display_mode, to_zero=True)
plot_timeseries(df, "wind_gust_kmh", "Wind gusts (10 m)", "Wind gusts (km/h)", "km/h", start_ts, end_ts, display_mode, to_zero=True)

st.markdown("### Pressure")
plot_timeseries(df, "pressure_hpa", "Surface pressure", "Pressure (hPa / mbar)", "hPa", start_ts, end_ts, display_mode)

st.markdown("### Severe weather (proxy)")
plot_timeseries(df, "severe_index", "Severe weather index (proxy)", "Index (0–100)", None, start_ts, end_ts, display_mode, y_range=[0, 100])

with st.expander("About the severe weather index"):
    st.write(
        "This is a simple heuristic intended for relative comparison across models. "
        "It is not an official meteorological index."
    )
    st.write(
        "- 50% precipitation probability\n"
        "- 30% wind gusts (capped)\n"
        "- 20% CAPE (capped)\n"
        "- +10 points for thunder-like weather codes (95/96/99)\n"
    )

st.divider()
st.subheader("Daily summary")
st.dataframe(daily_summary(df), use_container_width=True)

st.divider()
with st.expander("Export"):
    cols = [
        "timestamp",
        "model",
        "temp_c",
        "feels_like_c",
        "rh_pct",
        "cloud_pct",
        "precip_mm",
        "precip_prob_pct",
        "wind_kmh",
        "wind_gust_kmh",
        "pressure_hpa",
        "severe_index",
    ]
    export_df = df[[c for c in cols if c in df.columns]].copy()

    export_df["timestamp"] = pd.to_datetime(export_df["timestamp"]).dt.strftime("%a %b %d, %Y %I:%M %p %Z")

    for c in export_df.columns:
        if c not in ("timestamp", "model"):
            export_df[c] = pd.to_numeric(export_df[c], errors="coerce").round(1)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="blendwx_hourly.csv", mime="text/csv")

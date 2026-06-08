"""
Flight Turnaround Time Prediction — Streamlit App
Two Tabs: (1) Prediction  (2) EDA
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Turnaround Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#F44336"]
SEED = 42

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    """Load saved model artefacts if they exist."""
    base = "model_artefacts"
    artefacts = {}
    files = {
        "model":    "lgbm_turnaround_model.pkl",
        "encoders": "label_encoders.pkl",
        "features": "feature_names.pkl",
        "scaler":   "standard_scaler.pkl",
    }
    for key, fname in files.items():
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                artefacts[key] = pickle.load(f)
    return artefacts


@st.cache_data
def load_dataset():
    """Load the CSV dataset for EDA if present."""
    path = "airline_ground_ops_dataset.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        dt_cols = [
            "scheduled_arrival_time", "scheduled_departure_time",
            "actual_on_block_time", "actual_off_block_time",
        ]
        for c in dt_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        return df
    return None


def compute_engineered(f: dict, max_sim_flights: float = 25.0) -> dict:
    """Replicate notebook feature engineering for a single flight dict."""
    f["total_service_duration"] = (
        f["unloading_duration_minutes"]
        + f["loading_duration_minutes"]
        + f["fueling_duration_minutes"]
        + f["cleaning_duration_minutes"]
        + f["catering_duration_minutes"]
        + f["security_check_duration_minutes"]
    )
    f["total_delay_minutes"] = (
        f["unloading_start_delay_minutes"] + f["maintenance_delay_minutes"]
    )
    f["cargo_density"] = f["cargo_weight_kg"] / (f["cargo_volume_cbm"] + 1)
    f["congestion_score"] = (
        f["airport_congestion_index"] * 0.4
        + f["gate_occupancy_percent"] / 10 * 0.3
        + f["simultaneous_flights_at_terminal"] / max_sim_flights * 10 * 0.3
    )
    f["weather_severity_composite"] = (
        f["weather_severity_score"] * 0.5
        + 1 / (f["visibility_km"] + 0.1) * 5
        + f["wind_speed_kmh"] / 100 * 5
    )
    f["equipment_availability_score"] = f["equipment_availability_percent"] / 100
    f["crew_efficiency_score"] = f["average_crew_experience_years"] * (
        f["available_ground_crew_count"] / (f["assigned_ground_crew_count"] + 0.1)
    )
    f["terminal_activity_ratio"] = f["simultaneous_flights_at_terminal"] / (
        100 - f["gate_occupancy_percent"] + 1
    )
    f["cargo_complexity_flag"] = int(
        f["special_cargo_flag"] or f["hazardous_goods_flag"]
    )
    f["scheduled_turnaround_minutes"] = f["planned_turnaround_minutes"]  # proxy
    return f


def run_prediction(inputs: dict, artefacts: dict) -> float | None:
    if not artefacts.get("model") or not artefacts.get("features"):
        return None

    f = compute_engineered(dict(inputs))
    nf_df = pd.DataFrame([f])

    # Label encode
    le_map = artefacts.get("encoders", {})
    for col, le in le_map.items():
        if col in nf_df.columns:
            try:
                nf_df[col] = le.transform(nf_df[col].astype(str))
            except Exception:
                nf_df[col] = 0

    # OHE
    ohe_cols = [c for c in ["flight_type", "gate_type", "weather_condition"] if c in nf_df.columns]
    nf_df = pd.get_dummies(nf_df, columns=ohe_cols, drop_first=False)

    # Align to training features
    feat = artefacts["features"]
    for col in feat:
        if col not in nf_df.columns:
            nf_df[col] = 0
    nf_df = nf_df[feat]

    return float(artefacts["model"].predict(nf_df)[0])


# ── Dropdown options ──────────────────────────────────────────────────────────
AIRLINES = [
    "Emirates", "IndiGo", "Air India", "Lufthansa", "British Airways",
    "Singapore Airlines", "Qatar Airways", "Ryanair", "EasyJet", "Delta",
    "United Airlines", "American Airlines", "Cathay Pacific", "KLM", "Air France",
]
FLIGHT_TYPES  = ["International", "Domestic"]
AIRCRAFT_TYPES = [
    "A320", "A321", "A330", "A350", "A380",
    "B737", "B747", "B767", "B777", "B787",
    "ATR72", "E190", "CRJ900",
]
GATE_TYPES   = ["Contact", "Remote", "Bus Gate"]
WEATHER_COND = ["Clear", "Cloudy", "Rain", "Fog", "Storm", "Snow"]
AIRPORTS = [
    "DEL", "BOM", "BLR", "HYD", "MAA", "CCU",
    "DXB", "LHR", "CDG", "FRA", "SIN", "JFK",
    "ORD", "LAX", "AMS", "SYD", "HKG", "ICN",
]
CONGESTION_LEVELS = list(range(1, 11))   # 1–10


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <h1 style='text-align:center; color:#2196F3;'>✈️ Flight Turnaround Time Prediction</h1>
    <p style='text-align:center; color:#666; font-size:1.05rem;'>
        Machine Learning Pipeline · Airport Ground Operations
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

artefacts = load_artefacts()
df_data   = load_dataset()

tab_pred, tab_eda = st.tabs(["Prediction", "EDA"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:

    if not artefacts.get("model"):
        st.warning(
            "No trained model found. Please run the Jupyter notebook first to "
            "generate `model_artefacts/lgbm_turnaround_model.pkl`, then restart the app.",
            icon="",
        )

    st.markdown("### Enter Flight Details")
    st.caption("Fill in all fields and click **Predict Turnaround Time** at the bottom.")

    # ── Section 1: Flight Information ─────────────────────────────────────────
    with st.expander("Flight Information", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            airline_code  = st.selectbox("Airline", AIRLINES, index=0)
        with c2:
            flight_type   = st.selectbox("Flight Type", FLIGHT_TYPES, index=0)
        with c3:
            aircraft_type = st.selectbox("Aircraft Type", AIRCRAFT_TYPES, index=9)
        with c4:
            gate_type     = st.selectbox("Gate Type", GATE_TYPES, index=0)

        c5, c6 = st.columns(2)
        with c5:
            arrival_airport   = st.selectbox("Arrival Airport (IATA)", AIRPORTS, index=0)
        with c6:
            departure_airport = st.selectbox("Departure Airport (IATA)", AIRPORTS, index=10)

    # ── Section 2: Schedule ────────────────────────────────────────────────────
    with st.expander("Schedule", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            sched_arr_hour  = st.selectbox(
                "Scheduled Arrival Hour", list(range(0, 24)),
                index=14, format_func=lambda h: f"{h:02d}:00",
            )
        with c2:
            sched_dep_hour  = st.selectbox(
                "Scheduled Departure Hour", list(range(0, 24)),
                index=16, format_func=lambda h: f"{h:02d}:00",
            )
        with c3:
            planned_trnrnd  = st.number_input(
                "Planned Turnaround (min)", min_value=20, max_value=300, value=160, step=5,
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            day_of_week = st.selectbox(
                "Day of Week", list(range(7)),
                format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d],
                index=2,
            )
        with c5:
            month = st.selectbox(
                "Month", list(range(1, 13)),
                format_func=lambda m: [
                    "Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"
                ][m - 1],
                index=5,
            )
        with c6:
            is_weekend  = int(day_of_week in [5, 6])
            is_peak_hour = int(sched_arr_hour in list(range(7, 11)) + list(range(17, 21)))
            st.metric("Weekend?",   "Yes" if is_weekend   else "No")
        st.caption(f"Peak Hour (07–10 or 17–20): {'Yes ✅' if is_peak_hour else 'No'}")

    # ── Section 3: Ground Operations ───────────────────────────────────────────
    with st.expander("Ground Operations Durations (minutes)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            unloading_dur   = st.number_input("Unloading Duration",  10, 120, 35, 1)
            loading_dur     = st.number_input("Loading Duration",     10, 120, 40, 1)
        with c2:
            fueling_dur     = st.number_input("Fueling Duration",     10, 120, 38, 1)
            cleaning_dur    = st.number_input("Cleaning Duration",    5,  90,  28, 1)
        with c3:
            catering_dur    = st.number_input("Catering Duration",    5,  90,  30, 1)
            security_dur    = st.number_input("Security Check Dur.",  5,  60,  12, 1)

        c4, c5 = st.columns(2)
        with c4:
            unload_start_delay = st.number_input("Unloading Start Delay (min)", 0, 60, 3, 1)
        with c5:
            pass  # spacer

    # ── Section 4: Cargo ──────────────────────────────────────────────────────
    with st.expander("Cargo", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            cargo_weight    = st.number_input("Cargo Weight (kg)",   500, 60000, 15000, 500)
        with c2:
            cargo_volume    = st.number_input("Cargo Volume (cbm)",  1.0, 150.0, 22.0, 0.5)
        with c3:
            num_uld         = st.number_input("No. of ULD Containers", 0, 30, 4, 1)

        c4, c5 = st.columns(2)
        with c4:
            special_cargo   = st.selectbox("Special Cargo?",   ["No", "Yes"], index=0)
        with c5:
            hazardous_goods = st.selectbox("Hazardous Goods?", ["No", "Yes"], index=0)

    # ── Section 5: Crew ───────────────────────────────────────────────────────
    with st.expander("Crew", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            assigned_crew   = st.number_input("Assigned Crew Count",    4, 50, 14, 1)
        with c2:
            available_crew  = st.number_input("Available Crew Count",   4, 50, 13, 1)
        with c3:
            crew_exp        = st.slider("Avg Crew Experience (yrs)", 0.5, 20.0, 6.5, 0.5)
        with c4:
            crew_util       = st.slider("Crew Utilisation (%)", 30.0, 100.0, 88.0, 0.5)

    # ── Section 6: Airport Conditions ────────────────────────────────────────
    with st.expander("Airport Conditions", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            congestion_idx  = st.selectbox(
                "Congestion Index (1=Low, 10=High)",
                CONGESTION_LEVELS, index=5,
            )
        with c2:
            gate_occupancy  = st.slider("Gate Occupancy (%)",          10.0, 100.0, 72.0, 0.5)
        with c3:
            equip_avail     = st.slider("Equipment Availability (%)",  40.0, 100.0, 91.0, 0.5)
        with c4:
            sim_flights     = st.number_input("Simultaneous Flights at Terminal", 1, 50, 14, 1)

    # ── Section 7: Weather ────────────────────────────────────────────────────
    with st.expander("Weather", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            weather_cond    = st.selectbox("Weather Condition", WEATHER_COND, index=1)
        with c2:
            visibility_km   = st.slider("Visibility (km)",  0.5, 20.0, 10.0, 0.5)
        with c3:
            wind_speed      = st.slider("Wind Speed (km/h)", 0.0, 120.0, 18.0, 1.0)
        with c4:
            weather_sev     = st.slider("Weather Severity Score", 0.0, 10.0, 1.5, 0.1)

    # ── Section 8: Maintenance ────────────────────────────────────────────────
    with st.expander("Maintenance", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            maint_required  = st.selectbox("Maintenance Required?", ["No", "Yes"], index=0)
        with c2:
            maint_delay     = st.number_input("Maintenance Delay (min)", 0, 180, 0, 5)

    # ── Section 9: Historical Averages ────────────────────────────────────────
    with st.expander(" Historical Averages (contextual, not leakage)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            ac_avg  = st.number_input("Aircraft Avg Turnaround (min)", 30, 200, 87, 1)
        with c2:
            gate_avg = st.number_input("Gate Avg Turnaround (min)",    30, 200, 78, 1)
        with c3:
            al_avg  = st.number_input("Airline Avg Turnaround (min)", 30, 200, 92, 1)

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("---")
    col_btn, col_res = st.columns([1, 2])

    with col_btn:
        predict_clicked = st.button(
            "Predict Turnaround Time",
            type="primary",
            use_container_width=True,
        )

    if predict_clicked:
        inputs = {
            # Flight info
            "airline_code":              airline_code,
            "flight_type":               flight_type,
            "aircraft_type":             aircraft_type,
            "arrival_airport":           arrival_airport,
            "departure_airport":         departure_airport,
            "gate_type":                 gate_type,
            # Schedule
            "scheduled_arrival_hour":    sched_arr_hour,
            "scheduled_departure_hour":  sched_dep_hour,
            "planned_turnaround_minutes": planned_trnrnd,
            "day_of_week":               day_of_week,
            "month":                     month,
            "is_weekend":                is_weekend,
            "is_peak_hour":              is_peak_hour,
            # Ops
            "unloading_start_delay_minutes": unload_start_delay,
            "unloading_duration_minutes":    unloading_dur,
            "loading_duration_minutes":      loading_dur,
            "fueling_duration_minutes":      fueling_dur,
            "cleaning_duration_minutes":     cleaning_dur,
            "catering_duration_minutes":     catering_dur,
            "security_check_duration_minutes": security_dur,
            # Cargo
            "cargo_weight_kg":           cargo_weight,
            "cargo_volume_cbm":          cargo_volume,
            "number_of_uld_containers":  num_uld,
            "special_cargo_flag":        1 if special_cargo == "Yes" else 0,
            "hazardous_goods_flag":      1 if hazardous_goods == "Yes" else 0,
            # Crew
            "assigned_ground_crew_count":   assigned_crew,
            "available_ground_crew_count":  available_crew,
            "average_crew_experience_years": crew_exp,
            "crew_utilization_percent":     crew_util,
            # Airport
            "airport_congestion_index":     congestion_idx,
            "gate_occupancy_percent":       gate_occupancy,
            "equipment_availability_percent": equip_avail,
            "simultaneous_flights_at_terminal": sim_flights,
            # Weather
            "weather_condition":        weather_cond,
            "visibility_km":            visibility_km,
            "wind_speed_kmh":           wind_speed,
            "weather_severity_score":   weather_sev,
            # Maintenance
            "maintenance_required_flag": 1 if maint_required == "Yes" else 0,
            "maintenance_delay_minutes": maint_delay,
            # Historical
            "aircraft_avg_turnaround_minutes": ac_avg,
            "gate_avg_turnaround_minutes":     gate_avg,
            "airline_avg_turnaround_minutes":  al_avg,
        }

        predicted = run_prediction(inputs, artefacts)

        with col_res:
            if predicted is None:
                # ── Heuristic fallback ────────────────────────────────────────
                total_svc = (
                    unloading_dur + loading_dur + fueling_dur
                    + cleaning_dur + catering_dur + security_dur
                )
                base = max(total_svc * 0.85, planned_trnrnd * 0.9)
                weather_mult = {
                    "Clear": 1.0, "Cloudy": 1.02, "Rain": 1.07,
                    "Fog": 1.10, "Storm": 1.20, "Snow": 1.18,
                }.get(weather_cond, 1.0)
                cong_mult = 1 + (congestion_idx - 5) * 0.015
                maint_add = maint_delay * 0.6
                predicted = round(base * weather_mult * cong_mult + maint_add + unload_start_delay * 0.4, 1)
                st.info("Model artefacts not found — showing heuristic estimate.", icon="ℹ️")

            delta = predicted - planned_trnrnd
            status_color = "#F44336" if delta > 10 else "#FF9800" if delta > 0 else "#4CAF50"
            status_label = (
                f"🔴 +{delta:.0f} min DELAY"  if delta > 10  else
                f"🟡 +{delta:.0f} min SLIGHT DELAY" if delta > 0 else
                f"🟢 {delta:.0f} min AHEAD OF PLAN"
            )

            st.markdown(
                f"""
                <div style="background:#1a1a2e;border-radius:12px;padding:24px;border-left:6px solid {status_color};">
                    <h2 style="color:#fff;margin:0 0 4px;">Predicted Turnaround</h2>
                    <h1 style="color:{status_color};font-size:3.5rem;margin:0;">{predicted:.0f} <span style="font-size:1.5rem">min</span></h1>
                    <p style="color:#aaa;margin:8px 0 0;">{status_label}</p>
                    <hr style="border-color:#333;margin:16px 0;">
                    <table style="color:#ddd;width:100%;font-size:0.95rem;">
                        <tr><td>Airline</td><td><b>{airline_code}</b></td></tr>
                        <tr><td>Aircraft</td><td><b>{aircraft_type}</b></td></tr>
                        <tr><td>Route</td><td><b>{arrival_airport} → {departure_airport}</b></td></tr>
                        <tr><td>Flight Type</td><td><b>{flight_type}</b></td></tr>
                        <tr><td>Weather</td><td><b>{weather_cond}</b></td></tr>
                        <tr><td>Congestion</td><td><b>{congestion_idx}/10</b></td></tr>
                        <tr><td>Planned</td><td><b>{planned_trnrnd} min</b></td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Gauge chart ──────────────────────────────────────────────────────
        st.markdown("#### Prediction Breakdown")
        total_svc = (
            unloading_dur + loading_dur + fueling_dur
            + cleaning_dur + catering_dur + security_dur
        )
        breakdown = {
            "Total Service Duration": total_svc,
            "Planned Buffer":         max(0, planned_trnrnd - total_svc),
            "Congestion Overhead":    round((congestion_idx - 1) * 1.5, 1),
            "Weather Impact":         round({"Clear":0,"Cloudy":2,"Rain":5,"Fog":8,"Storm":18,"Snow":15}.get(weather_cond, 0), 1),
            "Maintenance Delay":      maint_delay,
            "Other / Residual":       max(0, round(predicted - total_svc - maint_delay, 1)),
        }
        bk_df = pd.DataFrame(list(breakdown.items()), columns=["Component", "Minutes"])
        bk_df = bk_df[bk_df["Minutes"] > 0]

        fig, ax = plt.subplots(figsize=(9, 3))
        left = 0
        colors_bk = ["#2196F3","#90CAF9","#FF9800","#4FC3F7","#F44336","#B0BEC5"]
        for i, (_, row) in enumerate(bk_df.iterrows()):
            ax.barh(0, row["Minutes"], left=left, color=colors_bk[i % len(colors_bk)],
                    edgecolor="white", linewidth=1.5, label=f"{row['Component']} ({row['Minutes']:.0f} min)")
            left += row["Minutes"]
        ax.axvline(predicted, color="red", lw=2.5, ls="--", label=f"Predicted: {predicted:.0f} min")
        ax.axvline(planned_trnrnd, color="orange", lw=2, ls=":", label=f"Planned: {planned_trnrnd} min")
        ax.set_yticks([])
        ax.set_xlabel("Minutes")
        ax.set_title("Turnaround Breakdown", fontweight="bold")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:

    st.markdown("### Exploratory Data Analysis")

    if df_data is None:
        st.warning(
            " Dataset `airline_ground_ops_dataset.csv` not found. "
            "Place it in the same directory as `app.py` to enable EDA.",
            icon="",
        )
        st.stop()

    df = df_data.copy()
    TARGET = "actual_turnaround_minutes"
    PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#F44336"]
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

    # ── Dataset Overview ───────────────────────────────────────────────────────
    st.markdown("#### 📋 Dataset Overview")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Records",    f"{len(df):,}")
    mc2.metric("Features",         f"{df.shape[1]}")
    mc3.metric("Avg Turnaround",   f"{df[TARGET].mean():.1f} min")
    mc4.metric("Max Turnaround",   f"{df[TARGET].max():.0f} min")

    with st.expander("Show raw data sample (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)

    st.markdown("---")

    # ── 5.1 Target Variable ────────────────────────────────────────────────────
    st.markdown("#### 5.1 Target Variable — `actual_turnaround_minutes`")

    fig = plt.figure(figsize=(18, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.hist(df[TARGET], bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
    ax1.axvline(df[TARGET].mean(),   color="red",    lw=2, ls="--", label=f"Mean: {df[TARGET].mean():.1f}")
    ax1.axvline(df[TARGET].median(), color="orange", lw=2, ls="--", label=f"Median: {df[TARGET].median():.1f}")
    ax1.set_title("Distribution", fontweight="bold")
    ax1.set_xlabel("Turnaround Time (min)")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    df[TARGET].plot.kde(ax=ax2, color="#2196F3", lw=2.5)
    ax2.fill_between(ax2.lines[0].get_xdata(), ax2.lines[0].get_ydata(), alpha=0.2, color="#2196F3")
    ax2.set_title("KDE", fontweight="bold")
    ax2.set_xlabel("Turnaround Time (min)")

    ax3 = fig.add_subplot(gs[2])
    ax3.boxplot(df[TARGET], patch_artist=True, widths=0.5,
                medianprops=dict(color="red", lw=2),
                boxprops=dict(facecolor="#90CAF9"))
    ax3.set_title("Boxplot", fontweight="bold")
    ax3.set_ylabel("Turnaround Time (min)")
    ax3.set_xticks([])

    plt.suptitle("Target Variable Analysis", fontsize=14, fontweight="bold")
    st.pyplot(fig)
    plt.close()

    skew = df[TARGET].skew()
    kurt = df[TARGET].kurtosis()
    st.caption(
        f"Skewness: **{skew:.3f}** "
        f"({'right-skewed' if skew > 0.5 else 'left-skewed' if skew < -0.5 else 'approx. symmetric'})  |  "
        f"Kurtosis: **{kurt:.3f}**"
    )

    st.markdown("---")

    # ── 5.2 Numerical Feature Distributions ───────────────────────────────────
    st.markdown("#### 5.2 Numerical Feature Distributions")

    num_features = [
        "planned_turnaround_minutes", "cargo_weight_kg", "cargo_volume_cbm",
        "assigned_ground_crew_count", "available_ground_crew_count",
        "average_crew_experience_years", "crew_utilization_percent",
        "airport_congestion_index", "gate_occupancy_percent",
        "equipment_availability_percent", "simultaneous_flights_at_terminal",
        "visibility_km", "wind_speed_kmh", "weather_severity_score",
        "maintenance_delay_minutes", "unloading_duration_minutes",
        "loading_duration_minutes", "fueling_duration_minutes",
        "cleaning_duration_minutes", "catering_duration_minutes",
    ]
    num_features = [c for c in num_features if c in df.columns]

    fig, axes = plt.subplots(4, 5, figsize=(22, 14))
    axes = axes.flatten()
    for i, col in enumerate(num_features):
        axes[i].hist(df[col], bins=35, color=PALETTE[i % len(PALETTE)],
                     edgecolor="white", alpha=0.8)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=9, fontweight="bold")
        axes[i].tick_params(labelsize=8)
    for j in range(len(num_features), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── 5.3 Categorical Feature Distributions ────────────────────────────────
    st.markdown("#### 5.3 Categorical Feature Distributions")

    cat_features = [c for c in ["flight_type", "aircraft_type", "gate_type", "weather_condition", "airline_code"] if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    for i, col in enumerate(cat_features):
        vc   = df[col].value_counts()
        bars = axes[i].bar(vc.index, vc.values, color=PALETTE[:len(vc)], edgecolor="white", alpha=0.88)
        axes[i].set_title(col.replace("_", " ").title(), fontweight="bold")
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, vc.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                         f"{val:,}", ha="center", va="bottom", fontsize=8)
    for j in range(len(cat_features), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── 5.4 Operational Insights ──────────────────────────────────────────────
    st.markdown("#### 5.4 Operational Insights")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    if "aircraft_type" in df.columns:
        order_ac = df.groupby("aircraft_type")[TARGET].median().sort_values().index
        sns.boxplot(data=df, x="aircraft_type", y=TARGET, order=order_ac, palette="Set2", ax=axes[0, 0])
        axes[0, 0].set_title("Aircraft Type vs Turnaround", fontweight="bold")
        axes[0, 0].tick_params(axis="x", rotation=30)

    if "flight_type" in df.columns:
        sns.violinplot(data=df, x="flight_type", y=TARGET, palette=["#2196F3", "#FF5722"],
                       inner="box", ax=axes[0, 1])
        axes[0, 1].set_title("Flight Type vs Turnaround", fontweight="bold")

    if "weather_condition" in df.columns:
        order_wc = df.groupby("weather_condition")[TARGET].median().sort_values().index
        sns.boxplot(data=df, x="weather_condition", y=TARGET, order=order_wc,
                    palette="coolwarm", ax=axes[1, 0])
        axes[1, 0].set_title("Weather Condition vs Turnaround", fontweight="bold")
        axes[1, 0].tick_params(axis="x", rotation=30)

    if "airport_congestion_index" in df.columns:
        sns.boxplot(data=df, x="airport_congestion_index", y=TARGET, palette="YlOrRd", ax=axes[1, 1])
        axes[1, 1].set_title("Congestion Index vs Turnaround", fontweight="bold")

    plt.suptitle("Operational Insights", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── 5.5 Correlation Analysis ──────────────────────────────────────────────
    st.markdown("#### 5.5 Correlation Heatmap")

    corr_cols = [
        TARGET, "planned_turnaround_minutes",
        "cargo_weight_kg", "cargo_volume_cbm",
        "assigned_ground_crew_count", "available_ground_crew_count",
        "average_crew_experience_years", "crew_utilization_percent",
        "airport_congestion_index", "gate_occupancy_percent",
        "equipment_availability_percent", "simultaneous_flights_at_terminal",
        "weather_severity_score", "maintenance_delay_minutes",
        "unloading_duration_minutes", "loading_duration_minutes",
        "fueling_duration_minutes", "cleaning_duration_minutes",
        "catering_duration_minutes",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, linewidths=0.5,
                annot_kws={"size": 7.5}, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Top correlations bar chart
    st.markdown("#### Top Feature Correlations with Turnaround Time")
    target_corr = df[corr_cols].corr()[[TARGET]].drop(TARGET).sort_values(TARGET)
    colors_bar = ["#F44336" if v < 0 else "#2196F3" for v in target_corr[TARGET]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(target_corr.index, target_corr[TARGET], color=colors_bar, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Feature Correlation with Turnaround Time", fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")
    for bar, val in zip(bars, target_corr[TARGET]):
        ax.text(
            val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9,
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── 5.6 Crew Experience Analysis ─────────────────────────────────────────
    if "average_crew_experience_years" in df.columns:
        st.markdown("#### 5.6 Crew Experience vs Turnaround Time")

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        axes[0].scatter(df["average_crew_experience_years"], df[TARGET],
                        alpha=0.15, s=12, color="#2196F3")
        z = np.polyfit(df["average_crew_experience_years"], df[TARGET], 1)
        p = np.poly1d(z)
        xs = np.linspace(df["average_crew_experience_years"].min(),
                         df["average_crew_experience_years"].max(), 100)
        axes[0].plot(xs, p(xs), color="red", lw=2, ls="--", label="Trend")
        axes[0].set_title("Crew Experience vs Turnaround", fontweight="bold")
        axes[0].set_xlabel("Avg Crew Experience (years)")
        axes[0].set_ylabel("Turnaround Time (min)")
        axes[0].legend()

        df_tmp = df.copy()
        df_tmp["exp_bin"] = pd.cut(df_tmp["average_crew_experience_years"], bins=5)
        exp_grouped = df_tmp.groupby("exp_bin")[TARGET].mean().reset_index()
        axes[1].bar([str(x) for x in exp_grouped["exp_bin"]], exp_grouped[TARGET],
                    color="#4CAF50", edgecolor="white", alpha=0.85)
        axes[1].set_title("Avg Turnaround by Experience Band", fontweight="bold")
        axes[1].set_xlabel("Experience Band (years)")
        axes[1].set_ylabel("Mean Turnaround Time (min)")
        axes[1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Business Insights Summary ─────────────────────────────────────────────
    # st.markdown("#### 💡 Key Business Insights")

    # insights = {
    #     "Total Service Duration":        "Ground handling time is the primary driver — optimise parallel ops.",
    #     "Planned Turnaround Accuracy":   "Better planning ≈ less deviation. Scheduling accuracy is critical.",
    #     "Fueling Duration":              "Fueling is often on the critical path — pre-position tankers early.",
    #     "Cargo Weight":                  "Heavy loads extend loading/unloading. Consider weight caps per stand.",
    #     "Congestion Score":              "Terminal congestion cascades into delays — slot management is key.",
    #     "Maintenance Delay":             "Unplanned maintenance is high-impact — invest in predictive maintenance.",
    #     "Crew Efficiency":               "Experienced crews significantly reduce turnaround time.",
    #     "Weather Severity":              "Weather planning and flexible scheduling mitigates impact.",
    #     "Equipment Availability":        "Equipment downtime is avoidable with proactive maintenance schedules.",
    #     "Gate Occupancy":                "Gate saturation limits buffer times — optimise gate allocation.",
    # }

    # for feat, insight in insights.items():
    #     st.markdown(f"- **{feat}**: {insight}")

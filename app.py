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
TEST_INPUT_DATA_PATH = "flight_input_test_dataset.csv"


@st.cache_data
def generate_test_flight_dataset(num_records: int = 80, seed: int = SEED) -> pd.DataFrame:
    """Generate realistic test records keyed by Flight_ID for UI auto-population."""
    rng = np.random.default_rng(seed)
    records = []
    delayed_flags = np.array([1] * (num_records // 2) + [0] * (num_records - num_records // 2))
    rng.shuffle(delayed_flags)

    for i in range(1, num_records + 1):
        is_delayed_flight = int(delayed_flags[i - 1])
        airline = str(rng.choice(AIRLINES))
        flight_type = str(rng.choice(FLIGHT_TYPES, p=[0.62, 0.38]))
        aircraft = str(rng.choice(AIRCRAFT_TYPES))
        weather = str(rng.choice(WEATHER_COND, p=[0.42, 0.24, 0.16, 0.07, 0.07, 0.04]))

        arr_airport = str(rng.choice(AIRPORTS))
        dep_airport = str(rng.choice([a for a in AIRPORTS if a != arr_airport]))
        arr_terminal = f"T{int(rng.integers(1, 5))}"
        dep_terminal = f"T{int(rng.integers(1, 5))}"
        gate_type = str(rng.choice(GATE_TYPES, p=[0.56, 0.28, 0.16]))

        passengers = int(rng.integers(60, 340))
        baggage_count = int(max(20, passengers * rng.uniform(0.7, 1.6)))
        transit_passengers = int(rng.integers(0, max(2, passengers // 4)))
        cargo_load = float(rng.integers(1200, 48000))
        cargo_volume = float(np.round(cargo_load / rng.uniform(550, 900), 1))

        special_cargo_count = int(rng.integers(0, 4))
        dangerous_goods_flag = int(rng.choice([0, 1], p=[0.9, 0.1]))

        assigned_crew = int(rng.integers(6, 28))
        available_crew = int(max(4, assigned_crew + rng.integers(-3, 4)))
        crew_experience = float(np.round(rng.uniform(1.0, 18.0), 1))
        crew_util = float(np.round(np.clip(72 + rng.normal(10, 8), 35, 100), 1))

        fueling_required = int(rng.choice([0, 1], p=[0.12, 0.88]))
        fueling_duration = int(rng.integers(12, 65) if fueling_required else 0)
        catering_required = int(rng.choice([0, 1], p=[0.08, 0.92]))
        catering_duration = int(rng.integers(10, 50) if catering_required else 0)
        cleaning_required = int(rng.choice([0, 1], p=[0.03, 0.97]))
        cleaning_duration = int(rng.integers(8, 45) if cleaning_required else 0)
        unloading_duration = int(rng.integers(18, 70))
        loading_duration = int(rng.integers(20, 75))
        security_duration = int(rng.integers(8, 28))

        loader_available = int(rng.choice([0, 1], p=[0.08, 0.92]))
        tug_available = int(rng.choice([0, 1], p=[0.06, 0.94]))
        gpu_available = int(rng.choice([0, 1], p=[0.1, 0.9]))
        belt_loader_count = int(rng.integers(1, 7))
        equipment_availability_percent = float(np.round((loader_available + tug_available + gpu_available) / 3 * 100, 1))

        active_flights_terminal = int(rng.integers(2, 45))
        gate_occupancy = float(np.round(rng.uniform(42, 98), 1))
        apron_congestion = int(rng.integers(1, 11))
        runway_queue_length = int(rng.integers(0, 22))

        temperature_c = float(np.round(rng.uniform(8, 42), 1))
        wind_speed = float(np.round(np.clip(rng.normal(20, 12), 0, 100), 1))
        visibility = float(np.round(np.clip(rng.normal(11, 4.5), 0.5, 20), 1))
        rainfall = float(np.round(max(0.0, rng.gamma(1.6, 2.2) - 1), 1))

        weather_severity = float(np.round(
            np.clip(
                (2.5 if weather in ["Rain", "Fog"] else 5.5 if weather in ["Storm", "Snow"] else 0.8)
                + (100 - visibility * 5) / 20
                + wind_speed / 35
                + rainfall / 8,
                0,
                10,
            ),
            1,
        ))

        if is_delayed_flight == 1:
            arrival_delay = int(np.clip(rng.normal(14 + runway_queue_length * 0.7, 8), 6, 60))
            fueling_delay = int(np.clip(rng.normal(6 if fueling_required else 2, 3.5), 1, 35))
            catering_delay = int(np.clip(rng.normal(5 if catering_required else 1.5, 3), 1, 30))
            crew_delay = int(np.clip(rng.normal(5, 3.5), 1, 25))
            maintenance_delay = int(np.clip(rng.normal(12, 12), 0, 90))
        else:
            arrival_delay = int(np.clip(rng.normal(0.6, 1.0), 0, 3))
            fueling_delay = int(np.clip(rng.normal(0.3 if fueling_required else 0.0, 0.7), 0, 2))
            catering_delay = int(np.clip(rng.normal(0.2 if catering_required else 0.0, 0.6), 0, 2))
            crew_delay = int(np.clip(rng.normal(0.3, 0.8), 0, 2))
            maintenance_delay = int(np.clip(rng.normal(0.5, 1.2), 0, 5))
        maintenance_required = int(maintenance_delay > 8)

        sched_arr_hour = int(rng.integers(0, 24))
        sched_dep_hour = int(np.clip(sched_arr_hour + rng.integers(1, 5), 0, 23))
        day_of_week = int(rng.integers(0, 7))
        month = int(rng.integers(1, 13))
        planned_turnaround = int(
            np.clip(
                unloading_duration + loading_duration + fueling_duration + cleaning_duration + catering_duration + security_duration
                + arrival_delay * 0.35 + crew_delay * 0.2 + 20,
                45,
                260,
            )
        )

        records.append(
            {
                # Required columns from the user request
                "Flight_ID": f"FL-{i:04d}",
                "Flight_Number": f"{airline.split()[0][:2].upper()}{int(rng.integers(100, 9999))}",
                "Airline_Code": airline,
                "Aircraft_Type": aircraft,
                "Flight_Type": flight_type,
                "Arrival_Terminal": arr_terminal,
                "Departure_Terminal": dep_terminal,
                "Cargo_Load_KG": cargo_load,
                "Baggage_Count": baggage_count,
                "Passenger_Count": passengers,
                "Special_Cargo_Count": special_cargo_count,
                "Dangerous_Goods_Flag": dangerous_goods_flag,
                "Transit_Passenger_Count": transit_passengers,
                "Assigned_Ground_Crew": assigned_crew,
                "Crew_Experience_Years": crew_experience,
                "Fueling_Required": fueling_required,
                "Fueling_Duration_Minutes": fueling_duration,
                "Catering_Required": catering_required,
                "Catering_Duration_Minutes": catering_duration,
                "Cleaning_Required": cleaning_required,
                "Cleaning_Duration_Minutes": cleaning_duration,
                "Loader_Available": loader_available,
                "Tug_Available": tug_available,
                "GPU_Available": gpu_available,
                "Belt_Loader_Count": belt_loader_count,
                "Active_Flights_At_Terminal": active_flights_terminal,
                "Gate_Occupancy_Percentage": gate_occupancy,
                "Apron_Congestion_Index": apron_congestion,
                "Runway_Queue_Length": runway_queue_length,
                "Weather_Condition": weather,
                "Temperature_Celsius": temperature_c,
                "Wind_Speed_KMPH": wind_speed,
                "Visibility_KM": visibility,
                "Rainfall_MM": rainfall,
                "Arrival_Delay_Minutes": arrival_delay,
                "Fueling_Delay_Minutes": fueling_delay,
                "Catering_Delay_Minutes": catering_delay,
                "Crew_Delay_Minutes": crew_delay,
                "Maintenance_Delay_Minutes": maintenance_delay,
                "Delay_Flight": is_delayed_flight,
                # Extra helper columns for direct widget prefill
                "Arrival_Airport": arr_airport,
                "Departure_Airport": dep_airport,
                "Gate_Type": gate_type,
                "Scheduled_Arrival_Hour": sched_arr_hour,
                "Scheduled_Departure_Hour": sched_dep_hour,
                "Day_Of_Week": day_of_week,
                "Month": month,
                "Planned_Turnaround_Minutes": planned_turnaround,
                "Unloading_Duration_Minutes": unloading_duration,
                "Loading_Duration_Minutes": loading_duration,
                "Security_Check_Duration_Minutes": security_duration,
                "Cargo_Volume_CBM": cargo_volume,
                "Available_Ground_Crew": available_crew,
                "Crew_Utilization_Percent": crew_util,
                "Equipment_Availability_Percent": equipment_availability_percent,
                "Weather_Severity_Score": weather_severity,
                "Maintenance_Required_Flag": maintenance_required,
                "Aircraft_Avg_Turnaround_Minutes": int(np.clip(planned_turnaround + rng.integers(-15, 18), 35, 220)),
                "Gate_Avg_Turnaround_Minutes": int(np.clip(planned_turnaround + rng.integers(-20, 20), 30, 210)),
                "Airline_Avg_Turnaround_Minutes": int(np.clip(planned_turnaround + rng.integers(-16, 22), 35, 230)),
            }
        )

    return pd.DataFrame(records)


@st.cache_data
def load_or_create_test_flight_data(path: str = TEST_INPUT_DATA_PATH) -> pd.DataFrame:
    """Load existing test dataset, else generate and persist it."""
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        if "Delay_Flight" not in df_existing.columns or df_existing.empty:
            df = generate_test_flight_dataset(num_records=80, seed=SEED)
            df.to_csv(path, index=False)
            return df
        delay_share = float(df_existing["Delay_Flight"].mean())
        if not (0.49 <= delay_share <= 0.51):
            df = generate_test_flight_dataset(num_records=len(df_existing), seed=SEED)
            df.to_csv(path, index=False)
            return df
        return df_existing
    df = generate_test_flight_dataset(num_records=80, seed=SEED)
    df.to_csv(path, index=False)
    return df


def _pick(options, value, fallback):
    """Safely choose selectbox index from options."""
    if value in options:
        return value
    return fallback


def _safe_float(value, fallback):
    return fallback if pd.isna(value) else float(value)


def _safe_int(value, fallback):
    return fallback if pd.isna(value) else int(value)


def build_prefill(selected_record: pd.Series | None) -> dict:
    """Build default widget values from selected Flight_ID record."""
    defaults = {
        "airline_code": "Emirates",
        "flight_type": "International",
        "aircraft_type": "B787",
        "gate_type": "Contact",
        "arrival_airport": "DEL",
        "departure_airport": "SIN",
        "sched_arr_hour": 14,
        "sched_dep_hour": 16,
        "planned_trnrnd": 160,
        "day_of_week": 2,
        "month": 6,
        "unloading_dur": 35,
        "loading_dur": 40,
        "fueling_dur": 38,
        "cleaning_dur": 28,
        "catering_dur": 30,
        "security_dur": 12,
        "unload_start_delay": 3,
        "cargo_weight": 15000,
        "cargo_volume": 22.0,
        "num_uld": 4,
        "special_cargo": "No",
        "hazardous_goods": "No",
        "assigned_crew": 14,
        "available_crew": 13,
        "crew_exp": 6.5,
        "crew_util": 88.0,
        "congestion_idx": 6,
        "gate_occupancy": 72.0,
        "equip_avail": 91.0,
        "sim_flights": 14,
        "weather_cond": "Cloudy",
        "visibility_km": 10.0,
        "wind_speed": 18.0,
        "weather_sev": 1.5,
        "maint_required": "No",
        "maint_delay": 0,
        "ac_avg": 87,
        "gate_avg": 78,
        "al_avg": 92,
    }

    if selected_record is None:
        return defaults

    mapped = {
        "airline_code": _pick(AIRLINES, selected_record.get("Airline_Code"), defaults["airline_code"]),
        "flight_type": _pick(FLIGHT_TYPES, selected_record.get("Flight_Type"), defaults["flight_type"]),
        "aircraft_type": _pick(AIRCRAFT_TYPES, selected_record.get("Aircraft_Type"), defaults["aircraft_type"]),
        "gate_type": _pick(GATE_TYPES, selected_record.get("Gate_Type"), defaults["gate_type"]),
        "arrival_airport": _pick(AIRPORTS, selected_record.get("Arrival_Airport"), defaults["arrival_airport"]),
        "departure_airport": _pick(AIRPORTS, selected_record.get("Departure_Airport"), defaults["departure_airport"]),
        "sched_arr_hour": _safe_int(selected_record.get("Scheduled_Arrival_Hour"), defaults["sched_arr_hour"]),
        "sched_dep_hour": _safe_int(selected_record.get("Scheduled_Departure_Hour"), defaults["sched_dep_hour"]),
        "planned_trnrnd": _safe_int(selected_record.get("Planned_Turnaround_Minutes"), defaults["planned_trnrnd"]),
        "day_of_week": _safe_int(selected_record.get("Day_Of_Week"), defaults["day_of_week"]),
        "month": _safe_int(selected_record.get("Month"), defaults["month"]),
        "unloading_dur": _safe_int(selected_record.get("Unloading_Duration_Minutes"), defaults["unloading_dur"]),
        "loading_dur": _safe_int(selected_record.get("Loading_Duration_Minutes"), defaults["loading_dur"]),
        "fueling_dur": _safe_int(selected_record.get("Fueling_Duration_Minutes"), defaults["fueling_dur"]),
        "cleaning_dur": _safe_int(selected_record.get("Cleaning_Duration_Minutes"), defaults["cleaning_dur"]),
        "catering_dur": _safe_int(selected_record.get("Catering_Duration_Minutes"), defaults["catering_dur"]),
        "security_dur": _safe_int(selected_record.get("Security_Check_Duration_Minutes"), defaults["security_dur"]),
        "unload_start_delay": _safe_int(selected_record.get("Arrival_Delay_Minutes"), defaults["unload_start_delay"]),
        "cargo_weight": _safe_int(selected_record.get("Cargo_Load_KG"), defaults["cargo_weight"]),
        "cargo_volume": _safe_float(selected_record.get("Cargo_Volume_CBM"), defaults["cargo_volume"]),
        "num_uld": _safe_int(selected_record.get("Belt_Loader_Count"), defaults["num_uld"]),
        "special_cargo": "Yes" if _safe_int(selected_record.get("Special_Cargo_Count"), 0) > 0 else "No",
        "hazardous_goods": "Yes" if _safe_int(selected_record.get("Dangerous_Goods_Flag"), 0) == 1 else "No",
        "assigned_crew": _safe_int(selected_record.get("Assigned_Ground_Crew"), defaults["assigned_crew"]),
        "available_crew": _safe_int(selected_record.get("Available_Ground_Crew"), defaults["available_crew"]),
        "crew_exp": _safe_float(selected_record.get("Crew_Experience_Years"), defaults["crew_exp"]),
        "crew_util": _safe_float(selected_record.get("Crew_Utilization_Percent"), defaults["crew_util"]),
        "congestion_idx": _safe_int(selected_record.get("Apron_Congestion_Index"), defaults["congestion_idx"]),
        "gate_occupancy": _safe_float(selected_record.get("Gate_Occupancy_Percentage"), defaults["gate_occupancy"]),
        "equip_avail": _safe_float(selected_record.get("Equipment_Availability_Percent"), defaults["equip_avail"]),
        "sim_flights": _safe_int(selected_record.get("Active_Flights_At_Terminal"), defaults["sim_flights"]),
        "weather_cond": _pick(WEATHER_COND, selected_record.get("Weather_Condition"), defaults["weather_cond"]),
        "visibility_km": _safe_float(selected_record.get("Visibility_KM"), defaults["visibility_km"]),
        "wind_speed": _safe_float(selected_record.get("Wind_Speed_KMPH"), defaults["wind_speed"]),
        "weather_sev": _safe_float(selected_record.get("Weather_Severity_Score"), defaults["weather_sev"]),
        "maint_required": "Yes" if _safe_int(selected_record.get("Maintenance_Required_Flag"), 0) == 1 else "No",
        "maint_delay": _safe_int(selected_record.get("Maintenance_Delay_Minutes"), defaults["maint_delay"]),
        "ac_avg": _safe_int(selected_record.get("Aircraft_Avg_Turnaround_Minutes"), defaults["ac_avg"]),
        "gate_avg": _safe_int(selected_record.get("Gate_Avg_Turnaround_Minutes"), defaults["gate_avg"]),
        "al_avg": _safe_int(selected_record.get("Airline_Avg_Turnaround_Minutes"), defaults["al_avg"]),
    }

    # Keep values inside widget bounds to avoid Streamlit range errors.
    mapped["sched_arr_hour"] = int(np.clip(mapped["sched_arr_hour"], 0, 23))
    mapped["sched_dep_hour"] = int(np.clip(mapped["sched_dep_hour"], 0, 23))
    mapped["planned_trnrnd"] = int(np.clip(mapped["planned_trnrnd"], 20, 300))
    mapped["day_of_week"] = int(np.clip(mapped["day_of_week"], 0, 6))
    mapped["month"] = int(np.clip(mapped["month"], 1, 12))
    mapped["unloading_dur"] = int(np.clip(mapped["unloading_dur"], 10, 120))
    mapped["loading_dur"] = int(np.clip(mapped["loading_dur"], 10, 120))
    mapped["fueling_dur"] = int(np.clip(mapped["fueling_dur"], 10, 120))
    mapped["cleaning_dur"] = int(np.clip(mapped["cleaning_dur"], 5, 90))
    mapped["catering_dur"] = int(np.clip(mapped["catering_dur"], 5, 90))
    mapped["security_dur"] = int(np.clip(mapped["security_dur"], 5, 60))
    mapped["unload_start_delay"] = int(np.clip(mapped["unload_start_delay"], 0, 60))
    mapped["cargo_weight"] = int(np.clip(mapped["cargo_weight"], 500, 60000))
    mapped["cargo_volume"] = float(np.clip(mapped["cargo_volume"], 1.0, 150.0))
    mapped["num_uld"] = int(np.clip(mapped["num_uld"], 0, 30))
    mapped["assigned_crew"] = int(np.clip(mapped["assigned_crew"], 4, 50))
    mapped["available_crew"] = int(np.clip(mapped["available_crew"], 4, 50))
    mapped["crew_exp"] = float(np.clip(mapped["crew_exp"], 0.5, 20.0))
    mapped["crew_util"] = float(np.clip(mapped["crew_util"], 30.0, 100.0))
    mapped["congestion_idx"] = int(np.clip(mapped["congestion_idx"], 1, 10))
    mapped["gate_occupancy"] = float(np.clip(mapped["gate_occupancy"], 10.0, 100.0))
    mapped["equip_avail"] = float(np.clip(mapped["equip_avail"], 40.0, 100.0))
    mapped["sim_flights"] = int(np.clip(mapped["sim_flights"], 1, 50))
    mapped["visibility_km"] = float(np.clip(mapped["visibility_km"], 0.5, 20.0))
    mapped["wind_speed"] = float(np.clip(mapped["wind_speed"], 0.0, 120.0))
    mapped["weather_sev"] = float(np.clip(mapped["weather_sev"], 0.0, 10.0))
    mapped["maint_delay"] = int(np.clip(mapped["maint_delay"], 0, 180))
    mapped["ac_avg"] = int(np.clip(mapped["ac_avg"], 30, 200))
    mapped["gate_avg"] = int(np.clip(mapped["gate_avg"], 30, 200))
    mapped["al_avg"] = int(np.clip(mapped["al_avg"], 30, 200))
    return mapped


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

    # --- INSERTION: Flight ID test dataset loading + selector ---
    flight_data = load_or_create_test_flight_data()
    selected_record = None

    if flight_data is None or flight_data.empty:
        st.warning("Flight test dataset is empty. Using manual default values.", icon="⚠️")
    elif "Flight_ID" not in flight_data.columns:
        st.warning("`Flight_ID` column missing in the test dataset. Using manual default values.", icon="⚠️")
    else:
        flight_ids = flight_data["Flight_ID"].dropna().astype(str).unique().tolist()
        if not flight_ids:
            st.warning("No valid Flight_ID values found in dataset. Using manual default values.", icon="⚠️")
        else:
            selected_flight_id = st.selectbox("Select Flight ID", flight_ids)
            selected_rows = flight_data[flight_data["Flight_ID"].astype(str) == str(selected_flight_id)]
            if selected_rows.empty:
                st.warning("Selected Flight_ID was not found. Using manual default values.", icon="⚠️")
            else:
                selected_record = selected_rows.iloc[0]
                null_cols = selected_record[selected_record.isna()].index.tolist()
                if null_cols:
                    st.warning(
                        "Some fields for the selected Flight_ID contain null values. "
                        "Defaults are used for missing fields.",
                        icon="⚠️",
                    )

    prefill = build_prefill(selected_record)

    # ── Section 1: Flight Information ─────────────────────────────────────────
    with st.expander("Flight Information", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            airline_code  = st.selectbox("Airline", AIRLINES, index=AIRLINES.index(prefill["airline_code"]))
        with c2:
            flight_type   = st.selectbox("Flight Type", FLIGHT_TYPES, index=FLIGHT_TYPES.index(prefill["flight_type"]))
        with c3:
            aircraft_type = st.selectbox("Aircraft Type", AIRCRAFT_TYPES, index=AIRCRAFT_TYPES.index(prefill["aircraft_type"]))
        with c4:
            gate_type     = st.selectbox("Gate Type", GATE_TYPES, index=GATE_TYPES.index(prefill["gate_type"]))

        c5, c6 = st.columns(2)
        with c5:
            arrival_airport   = st.selectbox("Arrival Airport (IATA)", AIRPORTS, index=AIRPORTS.index(prefill["arrival_airport"]))
        with c6:
            departure_airport = st.selectbox("Departure Airport (IATA)", AIRPORTS, index=AIRPORTS.index(prefill["departure_airport"]))

    # ── Section 2: Schedule ────────────────────────────────────────────────────
    with st.expander("Schedule", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            sched_arr_hour  = st.selectbox(
                "Scheduled Arrival Hour", list(range(0, 24)),
                index=prefill["sched_arr_hour"], format_func=lambda h: f"{h:02d}:00",
            )
        with c2:
            sched_dep_hour  = st.selectbox(
                "Scheduled Departure Hour", list(range(0, 24)),
                index=prefill["sched_dep_hour"], format_func=lambda h: f"{h:02d}:00",
            )
        with c3:
            planned_trnrnd  = st.number_input(
                "Planned Turnaround (min)", min_value=20, max_value=300, value=prefill["planned_trnrnd"], step=5,
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            day_of_week = st.selectbox(
                "Day of Week", list(range(7)),
                format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d],
                index=prefill["day_of_week"],
            )
        with c5:
            month = st.selectbox(
                "Month", list(range(1, 13)),
                format_func=lambda m: [
                    "Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"
                ][m - 1],
                index=prefill["month"] - 1,
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
            unloading_dur   = st.number_input("Unloading Duration",  10, 120, prefill["unloading_dur"], 1)
            loading_dur     = st.number_input("Loading Duration",     10, 120, prefill["loading_dur"], 1)
        with c2:
            fueling_dur     = st.number_input("Fueling Duration",     10, 120, prefill["fueling_dur"], 1)
            cleaning_dur    = st.number_input("Cleaning Duration",    5,  90,  prefill["cleaning_dur"], 1)
        with c3:
            catering_dur    = st.number_input("Catering Duration",    5,  90,  prefill["catering_dur"], 1)
            security_dur    = st.number_input("Security Check Dur.",  5,  60,  prefill["security_dur"], 1)

        c4, c5 = st.columns(2)
        with c4:
            unload_start_delay = st.number_input("Unloading Start Delay (min)", 0, 60, prefill["unload_start_delay"], 1)
        with c5:
            pass  # spacer

    # ── Section 4: Cargo ──────────────────────────────────────────────────────
    with st.expander("Cargo", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            cargo_weight    = st.number_input("Cargo Weight (kg)",   500, 60000, prefill["cargo_weight"], 500)
        with c2:
            cargo_volume    = st.number_input("Cargo Volume (cbm)",  1.0, 150.0, prefill["cargo_volume"], 0.5)
        with c3:
            num_uld         = st.number_input("No. of ULD Containers", 0, 30, prefill["num_uld"], 1)

        c4, c5 = st.columns(2)
        with c4:
            special_cargo   = st.selectbox("Special Cargo?",   ["No", "Yes"], index=["No", "Yes"].index(prefill["special_cargo"]))
        with c5:
            hazardous_goods = st.selectbox("Hazardous Goods?", ["No", "Yes"], index=["No", "Yes"].index(prefill["hazardous_goods"]))

    # ── Section 5: Crew ───────────────────────────────────────────────────────
    with st.expander("Crew", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            assigned_crew   = st.number_input("Assigned Crew Count",    4, 50, prefill["assigned_crew"], 1)
        with c2:
            available_crew  = st.number_input("Available Crew Count",   4, 50, prefill["available_crew"], 1)
        with c3:
            crew_exp        = st.slider("Avg Crew Experience (yrs)", 0.5, 20.0, prefill["crew_exp"], 0.5)
        with c4:
            crew_util       = st.slider("Crew Utilisation (%)", 30.0, 100.0, prefill["crew_util"], 0.5)

    # ── Section 6: Airport Conditions ────────────────────────────────────────
    with st.expander("Airport Conditions", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            congestion_idx  = st.selectbox(
                "Congestion Index (1=Low, 10=High)",
                CONGESTION_LEVELS, index=CONGESTION_LEVELS.index(prefill["congestion_idx"]),
            )
        with c2:
            gate_occupancy  = st.slider("Gate Occupancy (%)",          10.0, 100.0, prefill["gate_occupancy"], 0.5)
        with c3:
            equip_avail     = st.slider("Equipment Availability (%)",  40.0, 100.0, prefill["equip_avail"], 0.5)
        with c4:
            sim_flights     = st.number_input("Simultaneous Flights at Terminal", 1, 50, prefill["sim_flights"], 1)

    # ── Section 7: Weather ────────────────────────────────────────────────────
    with st.expander("Weather", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            weather_cond    = st.selectbox("Weather Condition", WEATHER_COND, index=WEATHER_COND.index(prefill["weather_cond"]))
        with c2:
            visibility_km   = st.slider("Visibility (km)",  0.5, 20.0, prefill["visibility_km"], 0.5)
        with c3:
            wind_speed      = st.slider("Wind Speed (km/h)", 0.0, 120.0, prefill["wind_speed"], 1.0)
        with c4:
            weather_sev     = st.slider("Weather Severity Score", 0.0, 10.0, prefill["weather_sev"], 0.1)

    # ── Section 8: Maintenance ────────────────────────────────────────────────
    with st.expander("Maintenance", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            maint_required  = st.selectbox("Maintenance Required?", ["No", "Yes"], index=["No", "Yes"].index(prefill["maint_required"]))
        with c2:
            maint_delay     = st.number_input("Maintenance Delay (min)", 0, 180, prefill["maint_delay"], 5)

    # ── Section 9: Historical Averages ────────────────────────────────────────
    with st.expander(" Historical Averages (contextual, not leakage)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            ac_avg  = st.number_input("Aircraft Avg Turnaround (min)", 30, 200, prefill["ac_avg"], 1)
        with c2:
            gate_avg = st.number_input("Gate Avg Turnaround (min)",    30, 200, prefill["gate_avg"], 1)
        with c3:
            al_avg  = st.number_input("Airline Avg Turnaround (min)", 30, 200, prefill["al_avg"], 1)

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
        estimate_mode = False

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
                estimate_mode = True
                st.info("Model artefacts not found. Displaying heuristic estimate.", icon="ℹ️")

            delta = predicted - planned_trnrnd
            status = "Delayed" if delta > 0 else "On Time / Early"
            status_color = "#2E7D32" if delta <= 0 else "#D32F2F"
            delta_text = f"{delta:+.0f} min vs plan"

            k1, k2, k3 = st.columns(3)
            k1.metric("Predicted Turnaround", f"{predicted:.0f} min")
            k1.markdown(
                f"<p style='margin:4px 0 0;font-weight:600;color:{status_color};'>{delta_text}</p>",
                unsafe_allow_html=True,
            )
            k2.metric("Operational Status", status)
            k3.metric("Planning Confidence", "Estimated" if estimate_mode else "Model-based")

            st.markdown(
                f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-weight:600;color:#0f172a;">Flight Summary</span>
                        <span style="font-weight:700;color:{status_color};">{status}</span>
                    </div>
                    <div style="margin-top:8px;color:#334155;font-size:0.95rem;line-height:1.5;">
                        <b>{airline_code}</b> · {flight_type} · {aircraft_type}<br>
                        Route: <b>{arrival_airport} → {departure_airport}</b> · Weather: <b>{weather_cond}</b> · Congestion: <b>{congestion_idx}/10</b><br>
                        Planned Turnaround: <b>{planned_trnrnd} min</b> · Predicted: <b>{predicted:.0f} min</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Professional charts from prediction output ───────────────────────
        st.markdown("#### Prediction Insights")
        total_svc = (
            unloading_dur + loading_dur + fueling_dur
            + cleaning_dur + catering_dur + security_dur
        )

        weather_impact = round({"Clear": 0, "Cloudy": 2, "Rain": 5, "Fog": 8, "Storm": 18, "Snow": 15}.get(weather_cond, 0), 1)
        congestion_overhead = round((congestion_idx - 1) * 1.5, 1)
        planned_buffer = max(0, planned_trnrnd - total_svc)

        breakdown = {
            "Total Service Duration": total_svc,
            "Planned Buffer": planned_buffer,
            "Congestion Overhead": congestion_overhead,
            "Weather Impact": weather_impact,
            "Maintenance Delay":      maint_delay,
            "Other / Residual":       max(0, round(predicted - total_svc - maint_delay, 1)),
        }
        bk_df = pd.DataFrame(list(breakdown.items()), columns=["Component", "Minutes"])
        bk_df = bk_df[bk_df["Minutes"] > 0]

        c_chart1, c_chart2 = st.columns([1, 1])

        with c_chart1:
            fig, ax = plt.subplots(figsize=(6.2, 4.1))
            labels = ["Planned", "Predicted"]
            vals = [planned_trnrnd, predicted]
            bar_colors = ["#64748b", status_color]
            bars = ax.bar(labels, vals, color=bar_colors, width=0.55)
            ax.set_title("Plan vs Predicted Turnaround", fontweight="bold")
            ax.set_ylabel("Minutes")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c_chart2:
            fig, ax = plt.subplots(figsize=(7.4, 4.1))
            colors_bk = ["#2563eb", "#7dd3fc", "#f59e0b", "#06b6d4", "#ef4444", "#94a3b8"]
            bars = ax.barh(bk_df["Component"], bk_df["Minutes"], color=colors_bk[:len(bk_df)], edgecolor="white")
            ax.set_title("Predicted Time Composition", fontweight="bold")
            ax.set_xlabel("Minutes")
            ax.grid(axis="x", linestyle="--", alpha=0.25)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
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

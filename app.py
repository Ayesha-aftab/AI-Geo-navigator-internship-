
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn

# ============================================================
# Pakistan Heatwave Prediction Dashboard
# Based on the uploaded Colab notebook
# ============================================================

INPUT_SEQ_LEN = 30
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3

HEATWAVE_THRESHOLD = 40.0
HIGH_RISK_THRESHOLD = 43.0
EXTREME_RISK_THRESHOLD = 45.0

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "pakistan_climate_data_2005_2025.csv"
SCALER_PATH = BASE_DIR / "climate_scaler.pkl"
MODEL_PATH = BASE_DIR / "custom_climate_llm.pth"

device = torch.device("cpu")


# ============================================================
# Model architecture — same as the Colab notebook
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CustomClimateLLM(nn.Module):
    def __init__(
        self,
        input_dim=4,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        output_dim=2,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim, max_len=100
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        return self.head(x[:, -1, :])


# ============================================================
# Load model/data
# ============================================================

@st.cache_resource
def load_model_and_scaler():
    missing = [
        p.name
        for p in [DATA_PATH, SCALER_PATH, MODEL_PATH]
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing deployment file(s): " + ", ".join(missing)
        )

    scaler = joblib.load(SCALER_PATH)

    model = CustomClimateLLM().to(device)
    state = torch.load(
        MODEL_PATH,
        map_location=device,
    )
    model.load_state_dict(state)
    model.eval()

    return model, scaler


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}"
        )

    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])

    data = (
        data.sort_values(["city", "date"])
        .reset_index(drop=True)
    )

    _, scaler = load_model_and_scaler()

    data[["temp_scaled", "humidity_scaled"]] = scaler.transform(
        data[["temperature_celsius", "relative_humidity_pct"]]
    )

    return data


# ============================================================
# Helper functions — same logic as notebook
# ============================================================

def get_category(temp):
    if temp < 12.0:
        return "COLD ❄️"
    elif temp <= 37.0:
        return "MILD 🌤️"
    return "HOT ☀️"


def get_cyclical_features(dt_obj):
    doy = dt_obj.dayofyear
    return (
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    )


def calculate_heatwave_risk(temperature, humidity):
    # Kept exactly aligned with the uploaded notebook:
    # humidity is displayed but the risk rule is temperature-based.

    if temperature >= EXTREME_RISK_THRESHOLD:
        return (
            "HEATWAVE",
            "EXTREME",
            95,
            "Extreme heatwave conditions detected.",
        )

    if temperature >= HIGH_RISK_THRESHOLD:
        return (
            "HEATWAVE",
            "HIGH",
            82,
            "High heatwave risk detected.",
        )

    if temperature >= HEATWAVE_THRESHOLD:
        return (
            "HEATWAVE",
            "MODERATE",
            68,
            "Heatwave conditions detected.",
        )

    return (
        "NO HEATWAVE",
        "LOW",
        25,
        "Temperature is below the heatwave threshold.",
    )


def predict_heatwave(city, target_date, df, model, scaler):
    city_df = (
        df[df["city"] == city]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(city_df) < INPUT_SEQ_LEN:
        raise ValueError(
            "Not enough historical data for this city."
        )

    target_date = pd.Timestamp(target_date)
    last_known_date = city_df["date"].max()

    if target_date <= last_known_date:
        matches = city_df[city_df["date"] == target_date]

        if len(matches) == 0:
            raise ValueError(
                f"{target_date.strftime('%Y-%m-%d')} is not available "
                "as a historical date in the dataset."
            )

        idx = matches.index[0]

        if idx < INPUT_SEQ_LEN:
            raise ValueError(
                "Not enough history before the selected date."
            )

        history_df = city_df.iloc[
            idx - INPUT_SEQ_LEN : idx
        ].copy()

        doy = history_df["date"].dt.dayofyear

        history_df["day_sin"] = np.sin(
            2 * np.pi * doy / 365.25
        )
        history_df["day_cos"] = np.cos(
            2 * np.pi * doy / 365.25
        )

        features = history_df[
            [
                "temp_scaled",
                "humidity_scaled",
                "day_sin",
                "day_cos",
            ]
        ].values

        input_tensor = torch.tensor(
            features,
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = (
                model(input_tensor)
                .cpu()
                .numpy()
            )

        prediction = scaler.inverse_transform(
            pred_scaled
        )[0]

        predicted_temperature = float(prediction[0])
        predicted_humidity = float(prediction[1])

        actual_temperature = float(
            city_df.loc[idx, "temperature_celsius"]
        )
        actual_humidity = float(
            city_df.loc[idx, "relative_humidity_pct"]
        )

        forecast_type = "Historical Prediction"

    else:
        last_30_df = city_df.tail(INPUT_SEQ_LEN).copy()

        sequence = []

        for _, row in last_30_df.iterrows():
            sin_value, cos_value = get_cyclical_features(
                row["date"]
            )

            sequence.append(
                [
                    row["temp_scaled"],
                    row["humidity_scaled"],
                    sin_value,
                    cos_value,
                ]
            )

        current_date = last_known_date
        predicted_temperature = None
        predicted_humidity = None

        while current_date < target_date:
            current_date += pd.Timedelta(days=1)

            input_tensor = torch.tensor(
                sequence[-INPUT_SEQ_LEN:],
                dtype=torch.float32,
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction_scaled = (
                    model(input_tensor)
                    .cpu()
                    .numpy()[0]
                )

            prediction = scaler.inverse_transform(
                [prediction_scaled]
            )[0]

            predicted_temperature = float(prediction[0])
            predicted_humidity = float(prediction[1])

            sin_value, cos_value = get_cyclical_features(
                current_date
            )

            sequence.append(
                [
                    prediction_scaled[0],
                    prediction_scaled[1],
                    sin_value,
                    cos_value,
                ]
            )

        actual_temperature = None
        actual_humidity = None
        forecast_type = "AI Future Forecast"

    (
        heatwave_status,
        risk_level,
        risk_score,
        risk_message,
    ) = calculate_heatwave_risk(
        predicted_temperature,
        predicted_humidity,
    )

    return {
        "temperature": predicted_temperature,
        "humidity": predicted_humidity,
        "actual_temperature": actual_temperature,
        "actual_humidity": actual_humidity,
        "forecast_type": forecast_type,
        "heatwave_status": heatwave_status,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_message": risk_message,
        "last_known_date": last_known_date,
        "city_df": city_df,
    }


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Pakistan Heatwave Prediction",
    page_icon="☀️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, #172554, #070b14 45%, #030712);
    }

    .hero {
        text-align: center;
        padding: 30px 20px;
        margin-bottom: 20px;
        border-radius: 24px;
        background: linear-gradient(135deg, #172554, #0f172a);
        border: 1px solid rgba(148,163,184,.18);
        box-shadow: 0 20px 50px rgba(0,0,0,.35);
    }

    .hero-icon { font-size: 48px; }

    .hero-title {
        font-size: 36px;
        font-weight: 800;
        margin-top: 6px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 8px;
    }

    .badge {
        display: inline-block;
        margin-top: 14px;
        padding: 7px 15px;
        border-radius: 999px;
        background: rgba(249,115,22,.12);
        border: 1px solid rgba(249,115,22,.3);
        color: #fb923c;
        font-size: 12px;
        font-weight: 700;
    }

    .metric {
        padding: 18px;
        min-height: 125px;
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(30,41,59,.95), rgba(15,23,42,.95));
        border: 1px solid rgba(148,163,184,.14);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .7px;
    }

    .metric-value {
        margin-top: 7px;
        font-size: 28px;
        font-weight: 800;
    }

    .status {
        padding: 18px 22px;
        margin: 16px 0;
        border-radius: 18px;
    }

    .heatwave {
        background: linear-gradient(135deg, rgba(127,29,29,.55), rgba(69,10,10,.30));
        border: 1px solid rgba(248,113,113,.35);
    }

    .normal {
        background: linear-gradient(135deg, rgba(20,83,45,.40), rgba(6,78,59,.25));
        border: 1px solid rgba(74,222,128,.25);
    }

    .status-title {
        font-size: 24px;
        font-weight: 800;
    }

    .status-message {
        color: #94a3b8;
        margin-top: 4px;
    }

    .small-note {
        color: #94a3b8;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load resources
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">☀️</div>
        <div class="hero-title">Pakistan Heatwave Prediction</div>
        <div class="hero-subtitle">
            AI-Powered Temperature Forecasting & Heatwave Risk Assessment
        </div>
        <div class="badge">🌡️ CLIMATE INTELLIGENCE SYSTEM</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, scaler = load_model_and_scaler()
    df = load_data()
except Exception as e:
    st.error("Dashboard setup is incomplete.")
    st.code(str(e))
    st.stop()


cities = sorted(
    df["city"].dropna().unique().tolist()
)


# ============================================================
# Inputs
# ============================================================

st.subheader("🎯 Generate Heatwave Prediction")

col1, col2, col3 = st.columns([1.1, 1.1, 1])

with col1:
    city = st.selectbox(
        "🏙️ Select City",
        cities,
    )

with col2:
    selected_date = st.date_input(
        "📅 Prediction Date",
        value=pd.Timestamp("2026-06-15").date(),
        min_value=df["date"].min().date(),
        max_value=pd.Timestamp("2030-12-31").date(),
    )

with col3:
    st.markdown(
        """
        <div class="small-note">
        <b>Model:</b> Custom Climate Transformer<br>
        <b>Input sequence:</b> 30 days<br>
        <b>Historical data:</b> 2005–2025<br>
        <b>Forecast:</b> Autoregressive
        </div>
        """,
        unsafe_allow_html=True,
    )

predict = st.button(
    "🔥 Predict Heatwave Risk",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Prediction
# ============================================================

if predict:
    try:
        result = predict_heatwave(
            city,
            pd.Timestamp(selected_date),
            df,
            model,
            scaler,
        )

        temp = result["temperature"]
        humidity = result["humidity"]
        risk = result["risk_level"]
        score = result["risk_score"]

        if result["heatwave_status"] == "HEATWAVE":
            st.markdown(
                f"""
                <div class="status heatwave">
                    <div style="font-size:36px">🔥</div>
                    <div class="status-title">HEATWAVE DETECTED — {risk} RISK</div>
                    <div class="status-message">{result["risk_message"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status normal">
                    <div style="font-size:36px">🌤️</div>
                    <div class="status-title">NO HEATWAVE — LOW RISK</div>
                    <div class="status-message">{result["risk_message"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-title">🌡️ PREDICTED TEMPERATURE</div>
                    <div class="metric-value">{temp:.1f} °C</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-title">💧 RELATIVE HUMIDITY</div>
                    <div class="metric-value">{humidity:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-title">⚠️ HEATWAVE RISK</div>
                    <div class="metric-value">{risk}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c4:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-title">📊 RISK SCORE</div>
                    <div class="metric-value">{score}/100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("📈 Temperature & Heatwave Trend")

        city_df = result["city_df"]
        graph_data = city_df.tail(30).copy()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            graph_data["date"],
            graph_data["temperature_celsius"],
            marker="o",
            linewidth=2.2,
            markersize=4,
            label="Historical Temperature",
        )

        ax.axhline(
            HEATWAVE_THRESHOLD,
            linestyle="--",
            linewidth=2,
            label=f"Heatwave Threshold ({HEATWAVE_THRESHOLD}°C)",
        )

        ax.scatter(
            [pd.Timestamp(selected_date)],
            [temp],
            s=220,
            marker="*",
            zorder=10,
            label="AI Prediction",
        )

        ax.set_title(
            f"{city} — Temperature & Heatwave Trend",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, linestyle="--", alpha=.25)
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("📋 Prediction Summary")

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Location",
                    "Prediction Date",
                    "Forecast Type",
                    "Predicted Temperature",
                    "Predicted Humidity",
                    "Heatwave Status",
                    "Risk Level",
                    "Risk Score",
                    "Heatwave Threshold",
                ],
                "Value": [
                    city,
                    pd.Timestamp(selected_date).strftime("%d %B %Y"),
                    result["forecast_type"],
                    f"{temp:.2f} °C",
                    f"{humidity:.2f} %",
                    result["heatwave_status"],
                    risk,
                    f"{score}/100",
                    f"{HEATWAVE_THRESHOLD:.1f} °C",
                ],
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.error("Prediction Error")
        st.exception(e)

else:
    st.info(
        "Select a city and prediction date, then click "
        "'Predict Heatwave Risk'."
    )

st.markdown(
    """
    <div class="small-note" style="text-align:center;margin-top:25px">
        Custom Climate Transformer • 30-day input sequence •
        Pakistan Climate Data 2005–2025 • Autoregressive forecasting
    </div>
    """,
    unsafe_allow_html=True,
)

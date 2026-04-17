import streamlit as st
import numpy as np
import joblib
import json
from pathlib import Path
import pandas as pd


BASE_FEATURES = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    height_m = df["height"] / 100.0
    df["bmi"] = df["weight"] / np.where(height_m > 0, height_m**2, np.nan)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["map"] = df["ap_lo"] + (df["pulse_pressure"] / 3.0)
    df["age_bp_interaction"] = df["age"] * df["ap_hi"]
    df["lifestyle_risk"] = df[["smoke", "alco"]].sum(axis=1) + (1 - df["active"])
    return df


@st.cache_resource
def load_artifacts():
    project_root = Path(__file__).resolve().parent
    model_obj = joblib.load(project_root / "heart_model.pkl")
    scaler_obj = None

    scaler_path = project_root / "scaler.pkl"
    if scaler_path.exists():
        scaler_obj = joblib.load(scaler_path)

    report_path = project_root / "model_report.json"
    report_obj = {}
    if report_path.exists():
        report_obj = json.loads(report_path.read_text(encoding="utf-8"))
    return model_obj, scaler_obj, report_obj


def predict_probability(model_obj, legacy_scaler_obj, report_obj, input_df: pd.DataFrame) -> float:
    expected_features = report_obj.get("features", [])

    # Preferred path: modern pipeline artifact with preprocessing and model bundled.
    if hasattr(model_obj, "named_steps"):
        model_input = add_engineered_features(input_df)
        if expected_features:
            model_input = model_input[expected_features]
        return float(model_obj.predict_proba(model_input)[0][1])

    # Legacy path: separate scaler + plain estimator.
    if legacy_scaler_obj is None:
        raise ValueError("Missing scaler for legacy model artifacts.")

    legacy_input = legacy_scaler_obj.transform(input_df[BASE_FEATURES].values)
    return float(model_obj.predict_proba(legacy_input)[0][1])


def validate_inputs(ap_hi_value, ap_lo_value, height_value, weight_value):
    errors = []
    if ap_hi_value <= ap_lo_value:
        errors.append("Systolic blood pressure must be higher than diastolic blood pressure.")
    if not (120 <= height_value <= 230):
        errors.append("Height must be between 120 and 230 cm.")
    if not (30 <= weight_value <= 250):
        errors.append("Weight must be between 30 and 250 kg.")
    return errors

st.set_page_config(
    page_title="Cardiovascular Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

model, legacy_scaler, report = load_artifacts()
threshold = float(report.get("threshold", 0.5))
best_model = report.get("best_model", "Unknown")

st.title("❤️ Cardiovascular Disease Risk Prediction")
st.write("Educational cardiovascular risk screening powered by machine learning.")

if best_model:
    st.caption(f"Model: {best_model} | Decision Threshold: {threshold:.2f}")

with st.expander("How to use this tool"):
    st.markdown(
        "- Enter patient demographics, vitals, and lifestyle indicators.\n"
        "- Click **Predict Risk** to get probability and category.\n"
        "- This app supports screening, not diagnosis."
    )

st.divider()

st.subheader("Patient Inputs")

with st.container(border=True):
    st.markdown("**Demographics**")
    col_demo_1, col_demo_2 = st.columns(2)
    with col_demo_1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
    with col_demo_2:
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

with st.container(border=True):
    st.markdown("**Vitals**")
    col_vital_1, col_vital_2 = st.columns(2)
    with col_vital_1:
        height = st.number_input("Height (cm)", min_value=120, max_value=230, value=170)
        ap_hi = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=260, value=120)
    with col_vital_2:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0)
        ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=80)

with st.container(border=True):
    st.markdown("**Lab and Lifestyle**")
    col_life_1, col_life_2 = st.columns(2)
    with col_life_1:
        cholesterol = st.selectbox(
            "Cholesterol",
            [1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x],
        )
        gluc = st.selectbox(
            "Glucose",
            [1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x],
        )
    with col_life_2:
        smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        alco = st.selectbox("Alcohol Intake", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        active = st.selectbox("Physically Active", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

st.divider()

if st.button("Predict Risk"):
    validation_errors = validate_inputs(ap_hi, ap_lo, height, weight)
    if validation_errors:
        for err in validation_errors:
            st.error(err)
        st.stop()

    input_df = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active,
            }
        ]
    )

    try:
        probability = predict_probability(model, legacy_scaler, report, input_df)
    except Exception as exc:
        st.error("Prediction failed. Re-train model artifacts and retry.")
        st.caption(f"Debug detail: {exc}")
        st.stop()

    st.subheader("🩺 Risk Assessment Result")
    st.metric("Predicted Disease Probability", f"{probability * 100:.2f}%")

    if probability < 0.30:
        st.success("🟢 **LOW RISK**")
        st.markdown(
            "- Continue healthy lifestyle habits\n"
            "- Maintain routine screening and exercise"
        )
    elif probability < max(0.60, threshold):
        st.warning("🟡 **MEDIUM RISK**")
        st.markdown(
            "- Improve diet quality and physical activity\n"
            "- Monitor blood pressure, glucose, and cholesterol"
        )
    else:
        st.error("🔴 **HIGH RISK**")
        st.markdown(
            "- Consult a physician or cardiologist\n"
            "- Follow clinical advice for lifestyle and treatment planning"
        )

    with st.expander("Derived Indicators"):
        bmi = weight / ((height / 100) ** 2)
        pulse_pressure = ap_hi - ap_lo
        map_value = ap_lo + ((ap_hi - ap_lo) / 3)
        st.write(f"BMI: {bmi:.2f}")
        st.write(f"Pulse Pressure: {pulse_pressure:.2f} mmHg")
        st.write(f"Mean Arterial Pressure: {map_value:.2f} mmHg")


st.divider()
st.caption("⚠️ Educational use only. This tool is not a medical diagnosis system.")

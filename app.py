import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ---------------------------
# Load model
# ---------------------------
model = joblib.load("xgb_optuna.pkl")

st.set_page_config(page_title="FNOL Claim Prediction", layout="wide")
st.title("🚗 FNOL Claim Prediction App")
st.markdown("Fill in the details below. The form is compact and organized into sections.")

# ===========================
# 1. Date & Time Inputs
# ===========================
with st.expander("📅 Date & Time Features", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        date_of_loss = st.date_input("Date of Loss", value=datetime(2023, 6, 15))
    with c2:
        notification_period = st.number_input("Notification Period (days)", min_value=0, step=1, value=45)
    with c3:
        inception_to_loss = st.number_input("Inception to Loss (days)", min_value=0, step=1, value=120)
    with c4:
        time_hour = st.slider("Hour of Loss", 0, 23, 14)

# ===========================
# 2. Flags
# ===========================
with st.expander("🚦 Binary Flags", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        vehicle_registration_present = st.checkbox("Vehicle Reg. Present", value=True)
    with c2:
        incident_details_present = st.checkbox("Incident Details Present", value=True)
    with c3:
        injury_details_present = st.checkbox("Injury Details Present", value=False)

# ===========================
# 3. TP Type (counts)
# ===========================
tp_type_max = {
    "insd_pass_back": 4, "insd_pass_front": 0, "driver": 5, "pass_back": 6,
    "pass_front": 2, "bike": 2, "cyclist": 1, "pass_multi": 0,
    "pedestrian": 1, "other": 6, "nk": 6
}
tp_type_features = {}
with st.expander("👥 Third Party Types (Counts)", expanded=False):
    cols = st.columns(4)
    for i, (x, max_val) in enumerate(tp_type_max.items()):
        with cols[i % 4]:
            tp_type_features[f"tp_type_{x}"] = st.number_input(
                f"{x}", min_value=0, max_value=max_val, value=0, step=1,
                key=f"tp_type_{x}"   # <-- unique key
            )

# ===========================
# 4. TP Injury (counts)
# ===========================
tp_injury_max = {"whiplash": 7, "traumatic": 4, "fatality": 2, "unclear": 7, "nk": 6}
tp_injury_features = {}
with st.expander("🤕 Third Party Injuries (Counts)", expanded=False):
    cols = st.columns(5)
    for i, (x, max_val) in enumerate(tp_injury_max.items()):
        with cols[i % 5]:
            tp_injury_features[f"tp_injury_{x}"] = st.number_input(
                f"{x}", min_value=0, max_value=max_val, value=0, step=1,
                key=f"tp_injury_{x}"   # <-- unique key
            )

# ===========================
# 5. TP Region (counts)
# ===========================
tp_region_max = {
    "eastang": 5, "eastmid": 6, "london": 9, "north": 4, "northw": 7,
    "outerldn": 5, "scotland": 6, "southe": 9, "southw": 5,
    "wales": 7, "westmid": 6, "yorkshire": 5
}
tp_region_features = {}
with st.expander("🌍 Third Party Regions (Counts)", expanded=False):
    cols = st.columns(4)
    for i, (x, max_val) in enumerate(tp_region_max.items()):
        with cols[i % 4]:
            tp_region_features[f"tp_region_{x}"] = st.number_input(
                f"{x}", min_value=0, max_value=max_val, value=0, step=1,
                key=f"tp_region_{x}"   # <-- unique key
            )

# ===========================
# 6. Other categorical features
# ===========================
with st.expander("⚙️ Other Features", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        ph_fault = st.selectbox("PH considered TP at fault", ["N", "Unknown", "Y"], index=0)
        vehicle_mobile = st.selectbox("Vehicle Mobile", ["N", "Unknown", "Y"], index=0)
    with c2:
        location = st.selectbox(
            "Location of Incident",
            ["Car Park", "Home Address", "Main Road", "Minor Road", "Motorway", "Other"],
            index=2
        )
        weather = st.selectbox("Weather Conditions", ["NORMAL", "SNOW_ICE_FOG", "Unknown", "WET"], index=0)
    with c3:
        notifier = st.selectbox("Notifier", ["CNF", "NamedDriver", "Other", "PH", "TP"], index=3)
        main_driver = st.selectbox("Main Driver", ["N", "Other", "Y"], index=2)

# ===========================
# Feature Engineering
# ===========================
df = pd.DataFrame([{
    "notification_period": notification_period,
    "inception_to_loss": inception_to_loss,
    "time_hour": time_hour,
    "vechile_registration_present": int(vehicle_registration_present),
    "incident_details_present": int(incident_details_present),
    "injury_details_present": int(injury_details_present),
    **tp_type_features,
    **tp_injury_features,
    **tp_region_features,
    "ph_tp_fault_n": 1 if ph_fault=="N" else 0,
    "ph_tp_fault_unknown": 1 if ph_fault=="Unknown" else 0,
    "ph_tp_fault_y": 1 if ph_fault=="Y" else 0,
    "vehiclemobile_n": 1 if vehicle_mobile=="N" else 0,
    "vehiclemobile_unknown": 1 if vehicle_mobile=="Unknown" else 0,
    "vehiclemobile_y": 1 if vehicle_mobile=="Y" else 0,
    **{f"location_{x.lower().replace(' ','_')}": 1 if location==x else 0
       for x in ["Car Park","Home Address","Main Road","Minor Road","Motorway","Other"]},
    **{f"weather_{x.lower()}": 1 if weather==x else 0
       for x in ["NORMAL","SNOW_ICE_FOG","Unknown","WET"]},
    **{f"notifier_{x.lower()}": 1 if notifier==x else 0
       for x in ["CNF","NamedDriver","Other","PH","TP"]},
    **{f"maindriver_{x.lower()}": 1 if main_driver==x else 0
       for x in ["N","Other","Y"]}
}])

# --- Date engineered features ---
date_of_loss = pd.to_datetime(date_of_loss)
df["year"] = date_of_loss.year
df["is_wknd"] = int(date_of_loss.weekday() >= 5)
df["is_month_start"] = int(date_of_loss.is_month_start)
df["is_month_end"] = int(date_of_loss.is_month_end)
df["month_to_loss"] = (pd.to_datetime("today") - date_of_loss).days / 30

# Cyclical encodings
df["month_sin"] = np.sin(2 * np.pi * date_of_loss.month / 12)
df["month_cos"] = np.cos(2 * np.pi * date_of_loss.month / 12)
df["day_of_week_sin"] = np.sin(2 * np.pi * (date_of_loss.dayofweek+1) / 7)
df["day_of_week_cos"] = np.cos(2 * np.pi * (date_of_loss.dayofweek+1) / 7)
df["day_of_year_sin"] = np.sin(2 * np.pi * date_of_loss.dayofyear / 365)
df["day_of_year_cos"] = np.cos(2 * np.pi * date_of_loss.dayofyear / 365)
df["week_of_year_sin"] = np.sin(2 * np.pi * date_of_loss.isocalendar().week / 52)
df["week_of_year_cos"] = np.cos(2 * np.pi * date_of_loss.isocalendar().week / 52)
df["day_of_month_sin"] = np.sin(2 * np.pi * date_of_loss.day / date_of_loss.days_in_month)
df["day_of_month_cos"] = np.cos(2 * np.pi * date_of_loss.day / date_of_loss.days_in_month)

season_num = 1 if date_of_loss.month in [12,1,2] else 2 if date_of_loss.month in [3,4,5] else 3 if date_of_loss.month in [6,7,8] else 4
df["season_sin"] = np.sin(2 * np.pi * season_num / 4)
df["season_cos"] = np.cos(2 * np.pi * season_num / 4)

# --- Interaction Features ---
df["tp_injury_whiplash_x_inception_to_loss"] = df["tp_injury_whiplash"] * df["inception_to_loss"]
df["tp_injury_whiplash_x_notification_period"] = df["tp_injury_whiplash"] * df["notification_period"]
df["tp_injury_whiplash_x_time_hour"] = df["tp_injury_whiplash"] * df["time_hour"]
df["tp_injury_whiplash_x_tp_injury_nk"] = df["tp_injury_whiplash"] * df["tp_injury_nk"]

df["tp_injury_traumatic_x_inception_to_loss"] = df["tp_injury_traumatic"] * df["inception_to_loss"]
df["tp_injury_traumatic_x_notification_period"] = df["tp_injury_traumatic"] * df["notification_period"]
df["tp_injury_traumatic_x_time_hour"] = df["tp_injury_traumatic"] * df["time_hour"]
df["tp_injury_traumatic_x_tp_injury_nk"] = df["tp_injury_traumatic"] * df["tp_injury_nk"]
df["tp_injury_traumatic_x_tp_injury_whiplash"] = df["tp_injury_traumatic"] * df["tp_injury_whiplash"]

df["tp_type_pass_front_x_inception_to_loss"] = df["tp_type_pass_front"] * df["inception_to_loss"]
df["tp_type_pass_front_x_notification_period"] = df["tp_type_pass_front"] * df["notification_period"]
df["tp_type_pass_front_x_time_hour"] = df["tp_type_pass_front"] * df["time_hour"]
df["tp_type_pass_front_x_tp_injury_nk"] = df["tp_type_pass_front"] * df["tp_injury_nk"]
df["tp_type_pass_front_x_tp_injury_whiplash"] = df["tp_type_pass_front"] * df["tp_injury_whiplash"]

df["tp_injury_nk_x_inception_to_loss"] = df["tp_injury_nk"] * df["inception_to_loss"]
df["tp_injury_nk_x_notification_period"] = df["tp_injury_nk"] * df["notification_period"]
df["tp_injury_nk_x_time_hour"] = df["tp_injury_nk"] * df["time_hour"]
df["tp_injury_nk_x_tp_injury_whiplash"] = df["tp_injury_nk"] * df["tp_injury_whiplash"]

# ===========================
# Preview + Prediction
# ===========================
st.subheader("📊 Generated Features")
st.dataframe(df)
st.write(f"Number of features: **{df.shape[1]}**")

if st.button("🔮 Predict Capped Incurred"):
    prediction = model.predict(df)[0]
    st.success(f"💰 Predicted Capped Incurred: **£{prediction:,.2f}**")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ---------------------------
# Load model
# ---------------------------
model = joblib.load("xgb_optuna.pkl")

st.title("🚗 FNOL Claim Prediction App")

st.write("Fill in the details to generate model-ready features.")

# ---------------------------
# 1. Date & Time Inputs
# ---------------------------
date_of_loss = st.date_input("Date of Loss", value=datetime(2023, 6, 15))
notification_period = st.number_input("Notification Period (days)", min_value=0, step=1, value=45)
inception_to_loss = st.number_input("Inception to Loss (days)", min_value=0, step=1, value=120)
time_hour = st.slider("Time of Loss (Hour)", 0, 23, 14)

# ---------------------------
# 2. Binary Flags
# ---------------------------
vehicle_registration_present = st.checkbox("Vehicle Registration Present", value=True)
incident_details_present = st.checkbox("Incident Details Present", value=True)
injury_details_present = st.checkbox("Injury Details Present", value=False)

# ---------------------------
# 3. TP Type
# ---------------------------
tp_type = st.multiselect(
    "Third Party Types",
    ["insd_pass_back","insd_pass_front","driver","pass_back","pass_front",
     "bike","cyclist","pass_multi","pedestrian","other","nk"],
    default=["driver"]
)

# ---------------------------
# 4. TP Injury
# ---------------------------
tp_injury = st.multiselect(
    "Third Party Injuries",
    ["whiplash","traumatic","fatality","unclear","nk"],
    default=["whiplash"]
)

# ---------------------------
# 5. TP Region
# ---------------------------
tp_region = st.selectbox(
    "Third Party Region",
    ["eastang","eastmid","london","north","northw","outerldn",
     "scotland","southe","southw","wales","westmid","yorkshire"],
    index=2
)

# ---------------------------
# 7. PH Fault
# ---------------------------
ph_fault = st.selectbox("PH considered TP at fault", ["N","Unknown","Y"], index=0)

# ---------------------------
# 8. Vehicle Mobile
# ---------------------------
vehicle_mobile = st.selectbox("Vehicle Mobile", ["N","Unknown","Y"], index=0)

# ---------------------------
# 9. Location
# ---------------------------
location = st.selectbox(
    "Location of Incident",
    ["Car Park","Home Address","Main Road","Minor Road","Motorway","Other"],
    index=2
)

# ---------------------------
# 10. Weather
# ---------------------------
weather = st.selectbox("Weather Conditions", ["NORMAL","SNOW_ICE_FOG","Unknown","WET"], index=0)

# ---------------------------
# 11. Notifier
# ---------------------------
notifier = st.selectbox("Notifier", ["CNF","NamedDriver","Other","PH","TP"], index=3)

# ---------------------------
# 12. Main Driver
# ---------------------------
main_driver = st.selectbox("Main Driver", ["N","Other","Y"], index=2)

# ---------------------------
# Feature Engineering
# ---------------------------
df = pd.DataFrame([{
    # Continuous
    "notification_period": notification_period,
    "inception_to_loss": inception_to_loss,
    "time_hour": time_hour,

    # Flags
    "vechile_registration_present": int(vehicle_registration_present),
    "incident_details_present": int(incident_details_present),
    "injury_details_present": int(injury_details_present),

    # TP Types
    **{f"tp_type_{x}": int(x in tp_type) for x in ["insd_pass_back","insd_pass_front","driver","pass_back","pass_front","bike","cyclist","pass_multi","pedestrian","other","nk"]},

    # TP Injuries
    **{f"tp_injury_{x}": int(x in tp_injury) for x in ["whiplash","traumatic","fatality","unclear","nk"]},

    # TP Regions
    **{f"tp_region_{x}": int(x == tp_region) for x in ["eastang","eastmid","london","north","northw","outerldn","scotland","southe","southw","wales","westmid","yorkshire"]},

    # PH Fault
    "ph_tp_fault_n": 1 if ph_fault=="N" else 0,
    "ph_tp_fault_unknown": 1 if ph_fault=="Unknown" else 0,
    "ph_tp_fault_y": 1 if ph_fault=="Y" else 0,

    # Vehicle Mobile
    "vehiclemobile_n": 1 if vehicle_mobile=="N" else 0,
    "vehiclemobile_unknown": 1 if vehicle_mobile=="Unknown" else 0,
    "vehiclemobile_y": 1 if vehicle_mobile=="Y" else 0,

    # Location
    **{f"location_{x.lower().replace(' ','_')}": 1 if location==x else 0 for x in ["Car Park","Home Address","Main Road","Minor Road","Motorway","Other"]},

    # Weather
    **{f"weather_{x.lower()}": 1 if weather==x else 0 for x in ["NORMAL","SNOW_ICE_FOG","Unknown","WET"]},

    # Notifier
    **{f"notifier_{x.lower()}": 1 if notifier==x else 0 for x in ["CNF","NamedDriver","Other","PH","TP"]},

    # Main Driver
    **{f"maindriver_{x.lower()}": 1 if main_driver==x else 0 for x in ["N","Other","Y"]}
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

# Season cyclical
season_map = {1:"Winter",2:"Spring",3:"Summer",4:"Autumn"}
season_num = 1 if date_of_loss.month in [12,1,2] else 2 if date_of_loss.month in [3,4,5] else 3 if date_of_loss.month in [6,7,8] else 4
df["season_sin"] = np.sin(2 * np.pi * season_num / 4)
df["season_cos"] = np.cos(2 * np.pi * season_num / 4)

# ---------------------------
# Interaction Features
# ---------------------------
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

# ---------------------------
# Prediction
# ---------------------------
st.subheader("Generated Features")
st.dataframe(df)
st.write(f"📊 Number of features: {df.shape[1]}")


# --- Prediction ---
if st.button("Predict Capped Incurred"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Capped Incurred: **£{prediction:,.2f}**")

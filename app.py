import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgb_optuna.pkl")   # Replace with your saved model file

st.title("🚗 FNOL Claim Prediction App")

st.write("Enter claim details below to predict **Capped Incurred**")

# --- Inputs ---
date_of_loss = st.date_input("Date of Loss")
notification_period = st.number_input("Notification Period (days)", min_value=0, step=1)
inception_to_loss = st.number_input("Inception to Loss (days)", min_value=0, step=1)
time_hour = st.slider("Time of Loss (Hour)", 0, 23, 12)

vehicle_mobile = st.selectbox("Vehicle Mobile", ["Y", "N", "n/k"])
notifier = st.selectbox("Notifier", ["PH", "CNF", "Other", "TP", "NamedDriver"])
location = st.selectbox("Location of Incident", ["Main Road", "Minor Road", "Motorway", "Car Park", "Home Address", "Other"])
weather = st.selectbox("Weather Conditions", ["NORMAL", "WET", "SNOW/ICE/FOG", "Unknown"])
ph_fault = st.selectbox("PH Considered TP at Fault", ["Y", "N", "Unknown"])
main_driver = st.selectbox("Main Driver", ["Y", "N", "Other"])

# --- Transform inputs into one-hot encoding ---
input_dict = {
    "Notification_period": notification_period,
    "Inception_to_loss": inception_to_loss,
    "Time_hour": time_hour,
    "VehicleMobile_Y": 1 if vehicle_mobile == "Y" else 0,
    "VehicleMobile_N": 1 if vehicle_mobile == "N" else 0,
    "VehicleMobile_Unknown": 1 if vehicle_mobile == "n/k" else 0,
    "Notifier_PH": 1 if notifier == "PH" else 0,
    "Notifier_CNF": 1 if notifier == "CNF" else 0,
    "Notifier_Other": 1 if notifier == "Other" else 0,
    "Notifier_TP": 1 if notifier == "TP" else 0,
    "Notifier_NamedDriver": 1 if notifier == "NamedDriver" else 0,
    "Location_Main_Road": 1 if location == "Main Road" else 0,
    "Location_Minor_Road": 1 if location == "Minor Road" else 0,
    "Location_Motorway": 1 if location == "Motorway" else 0,
    "Location_Car_Park": 1 if location == "Car Park" else 0,
    "Location_Home_Address": 1 if location == "Home Address" else 0,
    "Location_Other": 1 if location == "Other" else 0,
    "Weather_NORMAL": 1 if weather == "NORMAL" else 0,
    "Weather_WET": 1 if weather == "WET" else 0,
    "Weather_SNOW_ICE_FOG": 1 if weather == "SNOW/ICE/FOG" else 0,
    "Weather_Unknown": 1 if weather == "Unknown" else 0,
    "PH_TP_Fault_Y": 1 if ph_fault == "Y" else 0,
    "PH_TP_Fault_N": 1 if ph_fault == "N" else 0,
    "PH_TP_Fault_Unknown": 1 if ph_fault == "Unknown" else 0,
    "MainDriver_Y": 1 if main_driver == "Y" else 0,
    "MainDriver_N": 1 if main_driver == "N" else 0,
    "MainDriver_Other": 1 if main_driver == "Other" else 0,
}

input_df = pd.DataFrame([input_dict])

# --- Prediction ---
if st.button("Predict Capped Incurred"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Capped Incurred: **£{prediction:,.2f}**")

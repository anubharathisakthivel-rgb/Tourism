import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="praneeth232/machine_failure_model", filename="best_machine_failure_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a people selecting the Tourism Package Prediction based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=1, max_value=120, value=19, step=1)
Type_Of_Contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", ["1", "2", "3"])
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=1, max_value=400, value=10, step=1)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Large Business", "Small Business"])
Gender = st.selectbox("Gender", ["Female", "Male"])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=5, value=2, step=1)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=6, value=2, step=1)
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1, max_value=6, value=2, step=1)
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Divorced", "Married", "Unmarried"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=1, max_value=100, value=10)
Passport = st.selectbox("Passport", ["1", "0"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=1)
OwnCar = st.selectbox("OwnCar", ["1", "0"])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=3)
Designation = st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=300000, value=1000)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeOfContact': Type_Of_Contact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Wellness Tourism Package Selection"):
    prediction = model.predict(input_data)[0]
    result = "Customer will select the Wellness Toursim Package" if prediction == 1 else "Customer won't select the Wellness Toursim Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

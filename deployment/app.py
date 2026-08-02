
# Create the Streamlit application

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Set the browser title and page layout
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide",
)

# Resolve the model relative to the repository root
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_model.joblib"

# Keep the inputs in the same order used during training
FEATURE_COLUMNS = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

# Match the data types expected by the trained pipeline
FEATURE_DTYPES = {
    "Age": "float64",
    "TypeofContact": "object",
    "CityTier": "int64",
    "DurationOfPitch": "float64",
    "Occupation": "object",
    "Gender": "object",
    "NumberOfPersonVisiting": "int64",
    "NumberOfFollowups": "float64",
    "ProductPitched": "object",
    "PreferredPropertyStar": "float64",
    "MaritalStatus": "object",
    "NumberOfTrips": "float64",
    "Passport": "int64",
    "PitchSatisfactionScore": "int64",
    "OwnCar": "int64",
    "NumberOfChildrenVisiting": "float64",
    "Designation": "object",
    "MonthlyIncome": "float64",
}

@st.cache_resource
def load_prediction_model():
    """Load the trained model once and reuse it across app reruns."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)

# Stop the application early if the model cannot be loaded
try:
    model = load_prediction_model()
except Exception as error:
    st.error(
        f"Unable to load the prediction model: {error}"
    )
    st.stop()

# Display the application heading and usage guidance
st.title("Tourism Package Purchase Prediction")

st.subheader("Turning Traveller Profiles into Purchase Possibilities.") # v2 - commit!

st.write(
    "Enter the customer and sales-pitch details to estimate "
    "whether the customer is likely to purchase the tourism package."
)

st.caption(
    "This prediction can support sales prioritization. "
    "It should not be used as an automated customer eligibility decision."
)

# Collect customer and sales-pitch details in one form
with st.form("tourism_prediction_form"):

    customer_column, sales_column = st.columns(2)

    with customer_column:

        st.subheader("Customer information")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=61,
            value=36,
            step=1,
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Freelancer",
                "Large Business",
                "Salaried",
                "Small Business",
            ],
        )

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male",
            ],
        )

        marital_status = st.selectbox(
            "Marital status",
            [
                "Divorced",
                "Married",
                "Single",
                "Unmarried",
            ],
        )

        designation = st.selectbox(
            "Designation",
            [
                "AVP",
                "Executive",
                "Manager",
                "Senior Manager",
                "VP",
            ],
        )

        monthly_income = st.number_input(
            "Monthly income",
            min_value=1000.0,
            max_value=100000.0,
            value=22400.0,
            step=500.0,
        )

        city_tier = st.selectbox(
            "City tier",
            [1, 2, 3],
        )

        passport_option = st.selectbox(
            "Has passport?",
            ["No", "Yes"],
        )

        own_car_option = st.selectbox(
            "Owns a car?",
            ["No", "Yes"],
        )

    with sales_column:

        st.subheader("Visit and sales-pitch information")

        contact_type = st.selectbox(
            "Type of contact",
            [
                "Company Invited",
                "Self Enquiry",
            ],
        )

        product_pitched = st.selectbox(
            "Product pitched",
            [
                "Basic",
                "Deluxe",
                "King",
                "Standard",
                "Super Deluxe",
            ],
        )

        duration_of_pitch = st.number_input(
            "Duration of pitch in minutes",
            min_value=5.0,
            max_value=127.0,
            value=14.0,
            step=1.0,
        )

        number_of_followups = st.number_input(
            "Number of follow-ups",
            min_value=1,
            max_value=6,
            value=4,
            step=1,
        )

        pitch_satisfaction_score = st.selectbox(
            "Pitch satisfaction score",
            [1, 2, 3, 4, 5],
            index=2,
        )

        preferred_property_star = st.selectbox(
            "Preferred property rating",
            [3, 4, 5],
        )

        number_of_trips = st.number_input(
            "Number of trips taken annually",
            min_value=1,
            max_value=22,
            value=3,
            step=1,
        )

        number_of_people_visiting = st.number_input(
            "Number of people visiting",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )

        number_of_children_visiting = st.number_input(
            "Number of children visiting",
            min_value=0,
            max_value=3,
            value=1,
            step=1,
        )

    submitted = st.form_submit_button(
        "Predict purchase likelihood",
        type="primary",
        use_container_width=True,
    )

# Generate a prediction only after the form is submitted
if submitted:

    # Store the submitted values in the model's expected feature order
    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": contact_type,
                "CityTier": city_tier,
                "DurationOfPitch": duration_of_pitch,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": (
                    number_of_people_visiting
                ),
                "NumberOfFollowups": number_of_followups,
                "ProductPitched": product_pitched,
                "PreferredPropertyStar": (
                    preferred_property_star
                ),
                "MaritalStatus": marital_status,
                "NumberOfTrips": number_of_trips,
                "Passport": (
                    1 if passport_option == "Yes" else 0
                ),
                "PitchSatisfactionScore": (
                    pitch_satisfaction_score
                ),
                "OwnCar": (
                    1 if own_car_option == "Yes" else 0
                ),
                "NumberOfChildrenVisiting": (
                    number_of_children_visiting
                ),
                "Designation": designation,
                "MonthlyIncome": monthly_income,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    # Apply the same input types used during model training
    input_data = input_data.astype(
        FEATURE_DTYPES
    )

    # Predict the class and its purchase probability
    try:
        prediction = int(
            model.predict(input_data)[0]
        )

        purchase_probability = float(
            model.predict_proba(input_data)[0, 1]
        )

    except Exception as error:
        st.error(
            f"Prediction could not be completed: {error}"
        )
        st.stop()

    # Present the prediction in a simple business-friendly format
    st.subheader("Prediction result")

    if prediction == 1:
        st.success(
            "The customer is likely to purchase "
            "the tourism package."
        )
    else:
        st.info(
            "The customer is not currently likely to purchase "
            "the tourism package."
        )

    st.metric(
        "Estimated purchase probability",
        f"{purchase_probability:.1%}",
    )

    # Allow the submitted model input to be reviewed
    with st.expander("View submitted model input"):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from tensorflow.keras.models import load_model


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)


# ==========================================
# PROJECT DIRECTORY
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "laptop_price_model.keras"
DROPDOWN_PATH = BASE_DIR / "dropdowns.pkl"
COLUMNS_PATH = BASE_DIR / "model_columns.pkl"


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #101827;
    color: #FFFFFF;
}

.main-title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    color: #60A5FA;
}

.subtitle {
    text-align: center;
    color: #CBD5E1;
    font-size: 16px;
    margin-bottom: 25px;
}

h1, h2, h3, p, label {
    color: #FFFFFF !important;
}

div[data-testid="stMetric"] {
    background-color: #1E293B;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
}

div.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    height: 48px;
    width: 100%;
    font-size: 17px;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #172337;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL AND PICKLE FILES
# ==========================================

@st.cache_resource
def load_prediction_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    return load_model(MODEL_PATH)


@st.cache_data
def load_dropdown_data():

    if not DROPDOWN_PATH.exists():
        raise FileNotFoundError(
            f"Dropdown file not found: {DROPDOWN_PATH.name}"
        )

    with open(DROPDOWN_PATH, "rb") as file:
        data = pickle.load(file)

    return data


@st.cache_data
def load_model_columns():

    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Columns file not found: {COLUMNS_PATH.name}"
        )

    with open(COLUMNS_PATH, "rb") as file:
        columns = pickle.load(file)

    return columns


# ==========================================
# INITIALIZE PROJECT
# ==========================================

try:

    model = load_prediction_model()

    dropdown_data = load_dropdown_data()

    model_columns = load_model_columns()

except Exception as e:

    st.error(f"Unable to load project files: {e}")

    st.info(
        "Make sure all required files are present "
        "in the same folder as app.py."
    )

    st.stop()


# ==========================================
# GET PREPROCESSING INFORMATION
# ==========================================

try:

    dropdowns = dropdown_data["dropdowns"]

    category = dropdown_data["category"]

    numerical_columns = dropdown_data["numerical_columns"]

    numeric_info = dropdown_data["numeric_info"]

    scaler = dropdown_data["scaler"]

except KeyError as e:

    st.error(f"Missing preprocessing information: {e}")

    st.stop()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">💻 Laptop Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict laptop prices using Artificial Neural Networks'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# LAPTOP SPECIFICATIONS
# ==========================================

st.subheader("🖥️ Enter Laptop Specifications")

st.write(
    "Select the laptop specifications below to estimate its price."
)

user_input = {}

col1, col2 = st.columns(2)


# ==========================================
# CATEGORICAL INPUTS
# ==========================================

with col1:

    st.markdown("### Brand & Configuration")

    for feature in category[:4]:

        options = dropdowns[feature]

        user_input[feature] = st.selectbox(
            feature.replace("_", " ").title(),
            options=options,
            key=f"category_{feature}"
        )


with col2:

    st.markdown("### Hardware Specifications")

    for feature in category[4:]:

        options = dropdowns[feature]

        user_input[feature] = st.selectbox(
            feature.replace("_", " ").title(),
            options=options,
            key=f"category_{feature}"
        )


# ==========================================
# NUMERICAL INPUTS
# ==========================================

st.divider()

st.subheader("📊 Additional Specifications")

num_col1, num_col2, num_col3 = st.columns(3)

input_columns = [num_col1, num_col2, num_col3]

for i, feature in enumerate(numerical_columns):

    info = numeric_info[feature]

    minimum = float(info["min"])

    maximum = float(info["max"])

    default = float(info["median"])

    step = 1.0 if feature == "Ram" else 0.1

    with input_columns[i % 3]:

        user_input[feature] = st.number_input(
            feature,
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=step,
            key=f"input_{feature}"
        )


st.divider()


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔍 Predict Laptop Price"):

    try:

        # ----------------------------------
        # STEP 1: CREATE INPUT DATAFRAME
        # ----------------------------------

        input_df = pd.DataFrame([user_input])


        # ----------------------------------
        # STEP 2: ONE-HOT ENCODING
        # ----------------------------------

        input_encoded = pd.get_dummies(
            input_df,
            columns=category,
            drop_first=True,
            dtype=int
        )


        # ----------------------------------
        # STEP 3: MATCH TRAINING COLUMNS
        # ----------------------------------

        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )


        # ----------------------------------
        # STEP 4: CONVERT TO INTEGER
        # ----------------------------------

        input_encoded = input_encoded.astype(int)


        # ----------------------------------
        # STEP 5: APPLY SCALER
        # ----------------------------------

        input_encoded[numerical_columns] = scaler.transform(
            input_encoded[numerical_columns]
        )


        # ----------------------------------
        # STEP 6: MAKE PREDICTION
        # ----------------------------------

        prediction = model.predict(
            input_encoded,
            verbose=0
        )

        predicted_price = float(
            np.asarray(prediction).flatten()[0]
        )


        # ----------------------------------
        # STEP 7: DISPLAY RESULT
        # ----------------------------------

        st.divider()

        st.subheader("🎯 Prediction Result")

        if not np.isfinite(predicted_price):

            st.error(
                "Invalid prediction. Please check your trained model."
            )

        elif predicted_price < 0:

            st.warning(
                "The model returned a negative price. "
                "Please check the model performance."
            )

        else:

            st.success("Laptop price predicted successfully!")

            st.metric(
                label="Estimated Laptop Price",
                value=f"€ {predicted_price:,.2f}"
            )

            st.info(
                "This is an estimated laptop price generated "
                "by your trained ANN model."
            )


    except Exception as e:

        st.error(f"Prediction Error: {e}")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <p style="text-align:center; color:#94A3B8;">
        Laptop Price Prediction | ANN | TensorFlow | Streamlit
    </p>
    """,
    unsafe_allow_html=True
)

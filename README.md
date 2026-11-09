#  Laptop Price Prediction Using Artificial Neural Networks (ANN)

##  Project Overview

Laptop Price Prediction is a Machine Learning project that predicts the estimated price of a laptop based on its specifications using an Artificial Neural Network (ANN).

The project uses TensorFlow and Keras to build and train a deep learning regression model. Users can enter laptop specifications through an interactive Streamlit web application and receive an estimated laptop price instantly.

The application applies the same preprocessing techniques used during model training, including One-Hot Encoding and feature scaling, to ensure consistent predictions.

##  Key Features

- Predicts laptop prices using an Artificial Neural Network (ANN).
- Interactive and user-friendly Streamlit interface.
- Supports categorical and numerical laptop specifications.
- Uses One-Hot Encoding for categorical features.
- Applies StandardScaler for numerical feature scaling.
- Loads a pre-trained Keras model for predictions.
- Displays estimated laptop prices in an easy-to-understand format.

##  Technologies Used

- Python
- TensorFlow
- Keras
- Artificial Neural Networks (ANN)
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

##  Project Workflow

1. Collect and load the laptop dataset.
2. Perform data cleaning and preprocessing.
3. Encode categorical features using One-Hot Encoding.
4. Scale numerical features using StandardScaler.
5. Train an Artificial Neural Network regression model.
6. Save the trained model and preprocessing objects.
7. Develop an interactive Streamlit application.
8. Generate laptop price predictions based on user inputs.

##  Project Files

- `app.py` – Streamlit application.
- `laptop_price_model.keras` – Trained ANN model.
- `dropdowns.pkl` – Dropdown options and preprocessing information.
- `model_columns.pkl` – Training feature column information.

##  How to Run the Project Locally

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project folder**

```bash
cd Laptop_Price_Prediction_ANN
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate the environment (Windows)**

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the Streamlit application**

```bash
python -m streamlit run app.py
```

##  Project Objective

The objective of this project is to apply deep learning techniques to a real-world regression problem and develop an interactive application that helps users estimate laptop prices based on hardware and configuration specifications.

## 👨‍💻 Author

Developed as a Machine Learning and Deep Learning project using Python, TensorFlow, Keras, and Streamlit.

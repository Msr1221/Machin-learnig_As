# Diabetes Prediction System

## Overview
The **Diabetes Prediction System** is a machine learning-based web application that predicts whether an individual is diabetic based on medical features. The project is built using **FastAPI** for the backend and **Streamlit** for the user interface.

## Features
- **Machine Learning Model**: Utilizes Logistic Regression for binary classification.
- **User-Friendly Interface**: Allows users to input their health data and receive predictions.
- **REST API with FastAPI**: Provides an endpoint for model inference.
- **Deployed Application**: Accessible as a web application.

## Dataset
- **Source**: [UCI Machine Learning Repository - Pima Indians Diabetes Dataset]
- **Rows**: 768
- **Columns**: 9 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome)
- **Target Variable**: Outcome (0 - Non-Diabetic, 1 - Diabetic)

## Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit (Python)
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Deployment**: Render (or any cloud platform)

## Installation and Setup

### 1. Clone the Repository
```sh
git clone https://github.com/YAWUKAL21/diabetes-prediction.git
cd diabetes-prediction
```

### 2. Create a Virtual Environment
```sh
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Run the FastAPI Backend
```sh
uvicorn app:app --reload
```
- The API will be available at: `http://127.0.0.1:8000`

### 5. Run the Streamlit Frontend
```sh
streamlit run app.py
```
- Open `https://diabetes-prediction-api-3.onrender.com` in your browser to access the web app.

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predicts diabetes based on input features |

### Sample Request
```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 25.0,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 30
}
```

### Sample Response
```json
{
  "prediction": 1,
  "probability": 0.85
}
```

## Future Improvements
- Enhance the model with advanced algorithms.
- Improve UI/UX design for better user experience.
- Integrate real-time data input from wearable devices.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **UCI Machine Learning Repository** for the dataset.
- **Scikit-learn, FastAPI, and Streamlit** for making machine learning deployment seamless.

## Contact
For questions or collaborations, feel free to reach out via [yabkalmelak@gmail.com] or create an issue in this repository.

---

### 🔗 [GitHub Repository](https://github.com/YAWUKAL21/diabetes-prediction)


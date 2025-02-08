from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np  # Ensure numpy is imported
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and scaler
try:
    model = joblib.load("logreg.joblib")  # Load trained model
    scaler = joblib.load("scaler.joblib")  # Load trained scaler
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    raise RuntimeError(f"Error loading model or scaler: {e}")

# Define request schema
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# List of features used during training
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Preprocess input data
def preprocess(data: DiabetesInput):
    try:
        df_input = pd.DataFrame([data.dict()])  # Convert input to DataFrame
        df_input = df_input[FEATURES]  # Ensure feature order matches training
        
        # Check if scaler is loaded correctly
        if scaler is None:
            raise RuntimeError("Scaler not found. Ensure 'scaler.joblib' is present.")

        # Transform input using the pre-trained scaler
        df_input[FEATURES] = scaler.transform(df_input[FEATURES])
        
        return df_input

    except Exception as e:
        raise ValueError(f"Error preprocessing input: {e}")

@app.post("/predict/")
def predict_diabetes(data: DiabetesInput):
    try:
        print(f"📥 Received input: {data}")

        processed_data = preprocess(data)
        print(f"🔄 Processed Data: \n{processed_data}")

        # Check if model is loaded correctly
        if model is None:
            raise RuntimeError("Model not found. Ensure 'logreg.joblib' is present.")

        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        print(f"✅ Prediction: {prediction}, Probability: {probability:.4f}")

        return {"prediction": int(prediction), "probability": float(probability)}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.get("/")
def root():
    return {"message": "🚀 Diabetes Prediction API is running!"}

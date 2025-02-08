from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and scaler with error handling
try:
    model = joblib.load("logreg.joblib")  # Load trained model
    scaler = joblib.load("scaler.joblib")  # Load trained scaler
    print("✅ Model and scaler loaded successfully")
except ModuleNotFoundError as e:
    print(f"❌ Missing module error: {e}")
    raise RuntimeError("❌ Model loading failed: Required module is missing. Try reinstalling numpy and scikit-learn.")
except Exception as e:
    print(f"❌ General error while loading model: {e}")
    raise RuntimeError(f"❌ Model loading failed: {e}")

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
    df_input = pd.DataFrame([data.dict()])  # Convert input to DataFrame
    df_input = df_input[FEATURES]  # Ensure feature order matches training

    # Transform input using the pre-trained scaler
    try:
        df_input[FEATURES] = scaler.transform(df_input[FEATURES])
    except Exception as e:
        raise ValueError(f"Error scaling input: {e}")
    
    return df_input

@app.post("/predict/")
def predict_diabetes(data: DiabetesInput):
    try:
        print(f"📥 Received input: {data}")

        processed_data = preprocess(data)
        print(f"🔄 Processed Data: \n{processed_data}")

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

@app.get("/health/")
def health_check():
    try:
        _ = np.array([1, 2, 3])  # Test NumPy
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

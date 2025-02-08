from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow requests from the frontend (Streamlit)
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

# from fastapi import FastAPI, HTTPException, Depends, Security
# from fastapi.security import APIKeyHeader
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # Initialize FastAPI app
# app = FastAPI()

# # Enable CORS (Allow all origins for testing, restrict in production)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods
#     allow_headers=["*"],  # Allow all headers
# )

# # API Key Authentication
# API_KEY_NAME = "X-API-Key"
# api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# # Replace this with your actual API key (make sure it's the same as in Streamlit)
# API_KEY = "your-secret-api-key"

# # Dependency to validate the API key
# async def get_api_key(api_key: str = Security(api_key_header)):
#     if not api_key:
#         raise HTTPException(status_code=403, detail="API Key missing")
#     if api_key != API_KEY:
#         raise HTTPException(status_code=403, detail="Invalid API Key")
#     return api_key

# # Load trained model and scaler
# try:
#     model = joblib.load("logreg.joblib")  # Load trained model
#     scaler = joblib.load("scaler.joblib")  # Load trained scaler
#     print("✅ Model and scaler loaded successfully")
# except Exception as e:
#     print(f"❌ Error loading model or scaler: {e}")
#     raise RuntimeError(f"Error loading model or scaler: {e}")

# # Define request schema
# class DiabetesInput(BaseModel):
#     Pregnancies: int
#     Glucose: float
#     BloodPressure: float
#     SkinThickness: float
#     Insulin: float
#     BMI: float
#     DiabetesPedigreeFunction: float
#     Age: int

# # List of features used during training
# FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
#             "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# # Preprocess input data
# def preprocess(data: DiabetesInput):
#     df_input = pd.DataFrame([data.dict()])  # Convert input to DataFrame
#     df_input = df_input[FEATURES]  # Ensure feature order matches training

#     # Transform input using the pre-trained scaler
#     try:
#         df_input[FEATURES] = scaler.transform(df_input[FEATURES])
#     except Exception as e:
#         raise ValueError(f"Error scaling input: {e}")
    
#     return df_input

# @app.post("/predict/")
# def predict_diabetes(data: DiabetesInput, api_key: str = Depends(get_api_key)):
#     try:
#         print(f"📥 Received input: {data}")

#         processed_data = preprocess(data)
#         print(f"🔄 Processed Data: \n{processed_data}")

#         prediction = model.predict(processed_data)[0]
#         probability = model.predict_proba(processed_data)[0][1]

#         print(f"✅ Prediction: {prediction}, Probability: {probability:.4f}")

#         return {"prediction": int(prediction), "probability": float(probability)}

#     except Exception as e:
#         print(f"❌ Error: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# @app.get("/")
# def root():
#     return {"message": "🚀 Diabetes Prediction API is running"}

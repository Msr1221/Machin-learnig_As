import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
df = pd.read_csv("diabetes.csv")

# Features for scaling
numerical_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Train scaler on the full dataset
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save the trained scaler
joblib.dump(scaler, "scaler.joblib")
print("Scaler saved successfully!")

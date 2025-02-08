# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("Cleaned_data_for_model.csv")

# Display basic dataset information
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Drop duplicate rows (if any)
df.drop_duplicates(inplace=True)
print("Number of duplicate rows removed:", df.duplicated().sum())

# Define numerical and categorical features
numerical_features = ['price', 'baths', 'bedrooms', 'Area_in_Marla']
categorical_features = ['property_type', 'location', 'city', 'purpose']

# Remove invalid values (e.g., negative numbers)
for feature in numerical_features:
    df = df[df[feature] > 0]

# Handle outliers using the IQR method
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

for feature in numerical_features:
    handle_outliers(df, feature)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Define features (X) and target variable (y)
X = df.drop(columns=['price'])
y = df['price']

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization (optional, but useful for some ML models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Feature Importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'])
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest Model")
plt.show()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

print("Starting grid search...")
grid_search.fit(X_train, y_train)
print("Grid search completed.")

# Retrieve the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R² Score): {r2:.4f}")

# Visualizations

# Actual vs. Predicted Prices Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.show()

# Residuals Distribution
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residuals Distribution")
plt.show()

# Compare with Dummy Baseline Model
dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(X_train, y_train)
dummy_pred = dummy_regressor.predict(X_test)

dummy_mse = mean_squared_error(y_test, dummy_pred)
dummy_r2 = r2_score(y_test, dummy_pred)

print("\nBaseline Model Performance:")
print(f"Baseline MSE: {dummy_mse:.4f}")
print(f"Baseline R² Score: {dummy_r2:.4f}")

print("\nPerformance Comparison:")
print(f"{'Metric':<20} {'Baseline Model':<20} {'Random Forest Model':<20}")
print(f"{'-'*60}")
print(f"{'MSE':<20} {dummy_mse:<20.4f} {mse:<20.4f}")
print(f"{'RMSE':<20} {np.sqrt(dummy_mse):<20.4f} {rmse:<20.4f}")
print(f"{'R² Score':<20} {dummy_r2:<20.4f} {r2:<20.4f}")

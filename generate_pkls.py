import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Example data
data = pd.DataFrame({
    "age": [25, 30, 45, 50],
    "height": [170, 160, 180, 175],
    "weight": [70, 60, 80, 75],
    "activity_level": ["Low", "Medium", "High", "Medium"],
    "region": ["North", "South", "East", "West"],
    "dietary_restriction": ["None", "Vegetarian", "Vegan", "None"],
    "diet_type": ["Balanced Diet", "Low Carb", "High Protein", "Balanced Diet"]
})

categorical_features = ["activity_level", "region", "dietary_restriction"]
numerical_features = ["age", "height", "weight"]

# Fit encoders and scaler
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(data[categorical_features])

scaler = StandardScaler()
X_num = scaler.fit_transform(data[numerical_features])

# Combine features
import numpy as np
X = np.hstack([X_num, X_cat])
feature_names = list(numerical_features) + list(ohe.get_feature_names_out(categorical_features))

# Train a simple model
y = data["diet_type"]
model = RandomForestClassifier()
model.fit(X, y)

# Save files
joblib.dump(model, "diet_recommendation_model.pkl")
joblib.dump(ohe, "ohe_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
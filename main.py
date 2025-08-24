from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import joblib

# Load trained model
with open("house_rent_price_in_Dhaka.pkl", "rb") as f:
    model = pickle.load(f)

# Load the feature columns used during training (includes 68 Location_XXX columns)
feature_columns = joblib.load("feature_columns.pkl")

app = FastAPI(title="House Rent Prediction API")

# Input schema
class HouseFeatures(BaseModel):
    Area: float
    Bed: int
    Bath: int
    Location: str  # user chooses one location

@app.get("/")
def home():
    return {"message": "Welcome to the House Rent Prediction API"}

@app.post("/predict")
def predict(data: HouseFeatures):
    # Start with numeric features
    user_input = pd.DataFrame([{
        "Area": data.Area,
        "Bed": data.Bed,
        "Bath": data.Bath
    }])

    # Handle the location columns dynamically
    location_columns = [col for col in feature_columns if col.startswith("Location_")]
    for loc_col in location_columns:
        user_input[loc_col] = 1 if loc_col == f"Location_{data.Location}" else 0

    # Reindex to match all training features, fill any missing with 0
    user_input = user_input.reindex(columns=feature_columns, fill_value=0)

    # Predict rent
    prediction = model.predict(user_input)
    return {"predicted_rent": float(prediction[0])}

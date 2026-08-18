from fastapi import FastAPI,HTTPException
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field 
import joblib

app=FastAPI(
    title="Used Car Price Prediction API",
    description="Backend API serving XGBoost regression pipeline for car valuations"
)

#Loading model and preprocessing pipeline
model_pipeline=joblib.load("car_price_xgb_pipeline.pkl")
metadata=joblib.load("car_price_metadata.pkl")


class CarFeatures(BaseModel):
    make: str = Field(..., example="Toyota")
    model_name: str = Field(..., example="Corolla")
    model_year: int = Field(..., ge=1970, le=2026, example=2019)
    mileage: float = Field(..., ge=0, le=600000, example=65000)
    engine_capacity: float = Field(..., ge=600, le=6000, example=1600)
    fuel_type: str = Field(..., example="Petrol")
    transmission: str = Field(..., example="Automatic")
    auction_rating: float = Field(default=-1.0, ge=-1.0, le=10.0, example=-1.0)

def clean_fuel_type(fuel: str) -> str:
    if fuel in ["Electric", "PHEV", "REEV"]:
        return "EV_PHEV"
    elif fuel in ["CNG", "LPG"]:
        return "CNG_LPG"
    elif fuel in ["Petrol", "Diesel", "Hybrid"]:
        return fuel
    return "Other"

@app.get("/")
def api_check():
    return{'status':'ok', 'message': 'Car Prediction API is active'}

@app.get("/options")
def get_filter_options():
    return{
        "makes":metadata["top_10_makes"]+["Other"],
        "transmissions": ["Automatic", "Manual"],
        "fuel_types": ["Petrol", "Hybrid", "Diesel", "EV_PHEV", "CNG_LPG", "Other"],
    }
@app.post("/predict")
def predict_price(car: CarFeatures):
    try:
        
        current_year = 2026
        car_age = current_year - car.model_year
        mileage_per_year = car.mileage / (car_age + 1)

        make_clean = (
            car.make if car.make in metadata["top_10_makes"] else "Other"
        )
        model_line = f"{car.make} {car.model_name}".strip()
        model_clean = (
            model_line
            if model_line in metadata["top_25_models"]
            else "Other_Model"
        )
        fuel_clean = clean_fuel_type(car.fuel_type)

        
        input_df = pd.DataFrame(
            [
                {
                    "Car_Age": car_age,
                    "Mileage": car.mileage,
                    "Engine Capacity": car.engine_capacity,
                    "Auction Rating": car.auction_rating,
                    "Mileage_Per_Year": mileage_per_year,
                    "Make_Cleaned": make_clean,
                    "Model_Cleaned": model_clean,
                    "Fuel_Type_Cleaned": fuel_clean,
                    "Transmission": car.transmission,
                }
            ]
        )

        
        prediction = float(model_pipeline.predict(input_df)[0])

        
        return {
            "status": "success",
            "estimated_price_pkr": round(prediction, 2),
            "price_in_lacs": round(prediction / 100000, 2),
            "price_in_crores": round(prediction / 10000000, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

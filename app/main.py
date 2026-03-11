import os
import re
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import CarFeatures
from app.utils import logger

app = FastAPI(title="Car Price Prediction API")

def get_latest_model(model_dir):
    """Scan the model directory and return the path to the newest model version."""
    if not os.path.exists(model_dir):
        return None
        
    pattern = re.compile(r"^model_v(\d+)\.pkl$")
    max_version = 0
    latest_model_path = None
    
    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            version = int(match.group(1))
            if version > max_version:
                max_version = version
                latest_model_path = os.path.join(model_dir, filename)
                
    if latest_model_path:
        logger.info(f"Identified latest model: {os.path.basename(latest_model_path)} (Version {max_version})")
    
    return latest_model_path

# Load the latest model pipeline at startup
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
latest_model_path = get_latest_model(model_dir)

pipeline = None
if latest_model_path and os.path.exists(latest_model_path):
    try:
        pipeline = joblib.load(latest_model_path)
        logger.info(f"API Startup: Successfully loaded model {os.path.basename(latest_model_path)}")
    except Exception as e:
        logger.error(f"API Startup Error: Failed to load model {latest_model_path}. Error: {str(e)}")
else:
    logger.warning("API Startup Warning: No model files found in `model/`. Please run train/train.py first.")

@app.get("/")
def read_root():
    logger.info("Health check endpoint / requested")
    return {"message": "Car Price Prediction API - MLOps Grade"}

@app.post("/predict")
def predict_price(car: CarFeatures):
    logger.info(f"Incoming prediction request: {car.dict()}")
    
    if pipeline is None:
        logger.error("Prediction failed: Model not loaded in memory")
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded")

    try:
        # Convert Pydantic validated schema properties into single-row Pandas DataFrame
        input_data = pd.DataFrame([car.dict()])
        
        # Predict 
        log_price_pred = pipeline.predict(input_data)[0]
        
        # Invert logarithmic transformation
        price_pred = np.expm1(log_price_pred)
        
        result = round(float(price_pred), 2)
        logger.info(f"Prediction successful for {car.Year} {car.Make} {car.Model}: ${result}")
        
        return {"predicted_price": result}
        
    except Exception as e:
        logger.error(f"Prediction error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

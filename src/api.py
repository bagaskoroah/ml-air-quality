# Import the required libraries.
from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
import pandas as pd

import utils
import data_pipeline
import preprocessing


# Constant variables.p
PATH_CONFIG = "../config/config.yaml"

config = utils.load_config()
ohe_stasiun = utils.deserialize_data(config["path_fitted_encoder_stasiun"])
le_encoder = utils.deserialize_data(config["path_fitted_encoder_label"])
scaler = utils.deserialize_data(config["path_fitted_scaler"])
best_model = utils.deserialize_data(config["path_production_model"])


# Define input data structure.
class DataAPI(BaseModel):
    """Represents the user input data structure."""
    stasiun : str
    pm10 : int
    pm25 : int
    so2 : int
    co : int
    o3 : int
    no2 : int

# Create API object.
app = FastAPI()

# Define handlers.
@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict")
def predict(data: DataAPI):
    # Convert DataAPI to Pandas DataFrame.
    data = pd.DataFrame([data.dict()])

    # Convert dtype.
    data[config["features"][1:]] = data[config["features"][1:]].astype(float)

    # Do data defense.
    try:
        data_pipeline.data_defense(data, config, api=True)
    except AssertionError as err:
        return {"res": [], "error_msg": str(err)}

    # Encoding stasiun.
    data = preprocessing.transform_ohe_encoder(data, ohe_stasiun)

    # Scale the data
    data = preprocessing.transform_scaler_inference(data, scaler)

    # Predict data.
    y_pred = best_model.predict(data)

    # Inverse transform.
    y_pred_label = list(le_encoder.inverse_transform(y_pred))[0]

    return {"res": y_pred_label, "error_msg": ""}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080)
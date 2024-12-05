from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from typing import Optional
import logging
import pandas as pd
import joblib
import numpy as np
import io

# Import necessary functions for preprocessing
from Data_preprocessing_for_training import (
    remove_duplicates,
    extract_mileage,
    extract_engine,
    extract_max_power,
    extract_torque_RPM,
    seats2int,
    name2brand,
    encode_names)

app = FastAPI()

model = joblib.load("linear_regression_model_cars_Alan.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")


class Item(BaseModel):
    name: str
    year: int
    selling_price: Optional[int] = 0  # Default to 0
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

class CarFeatures(Item):
    pass

class CarFeaturesCollection(Items):
    pass


def preprocess_dataframe(df):
    df = df.copy()
    # df = remove_duplicates(df)

    df['mileage'] = df['mileage'].apply(extract_mileage)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

    df['engine'] = df['engine'].apply(extract_engine)
    df['engine'] = pd.to_numeric(df['engine'], errors='coerce')

    df['max_power'] = df['max_power'].apply(extract_max_power)
    df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

    torque_rpm = df['torque'].apply(lambda x: pd.Series(extract_torque_RPM(x)))
    df['torque'] = torque_rpm[0]
    df['rpm'] = torque_rpm[1]
    df['torque'] = pd.to_numeric(df['torque'], errors='coerce')
    df['rpm'] = pd.to_numeric(df['rpm'], errors='coerce')

    df['seats'] = df['seats'].apply(seats2int)
    df['seats'] = pd.to_numeric(df['seats'], errors='coerce')

    df['name'] = df['name'].apply(name2brand)
    df = encode_names(df, encoder)
    return df


@app.post("/predict_item")
def predict_item(item: CarFeatures) -> float:
    try:
        logging.debug(f"Received item: {item.dict()}")
        data = pd.DataFrame([item.dict()])
        logging.debug(f"Converted to DataFrame: {data}")

        processed_data = preprocess_dataframe(data)
        logging.debug(f"Preprocessed data: {processed_data}")

        brand_columns = [col for col in processed_data.columns if col.startswith("brand_")]
        numeric_columns = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats", "rpm"]
        processed_numeric = scaler.transform(processed_data[numeric_columns])
        processed_data_scaled = pd.DataFrame(processed_numeric, columns=numeric_columns)
        processed_data_scaled = pd.concat(
            [processed_data_scaled, processed_data[brand_columns]], axis=1
        )
        logging.debug(f"Final processed data for prediction: {processed_data_scaled}")

        prediction = model.predict(processed_data_scaled)
        logging.debug(f"Prediction: {prediction}")

        return prediction[0]
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return {"error": "An error occurred during prediction. Please check the logs."}


# @app.post("/predict_items")
# def predict_items(items: List[CarFeatures]) -> List[float]:
#     data = pd.DataFrame([item.dict() for item in items])
#
#     processed_data = preprocess_dataframe(data)
#     brand_columns = [col for col in processed_data.columns if col.startswith("brand_")]
#     numeric_columns = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats", "rpm"]
#     processed_numeric = scaler.transform(processed_data[numeric_columns])
#     processed_data_scaled = pd.DataFrame(processed_numeric, columns=numeric_columns)
#     processed_data_scaled = pd.concat(
#         [processed_data_scaled, processed_data[brand_columns]], axis=1
#     )
#
#     predictions = model.predict(processed_data_scaled)
#     return predictions.tolist()


@app.post("/predict_from_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        processed_data = preprocess_dataframe(df)

        brand_columns = [col for col in processed_data.columns if col.startswith("brand_")]
        numeric_columns = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats", "rpm"]

        processed_numeric = scaler.transform(processed_data[numeric_columns])
        processed_data_scaled = pd.DataFrame(processed_numeric, columns=numeric_columns)
        processed_data_scaled = pd.concat(
            [processed_data_scaled, processed_data[brand_columns]], axis=1
        )

        predictions = model.predict(processed_data_scaled)
        df["predicted_price"] = predictions

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception as e:
        logging.error(f"Error during CSV prediction: {e}", exc_info=True)
        return {"error": "An error occurred during CSV processing. Please check the logs."}
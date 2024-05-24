from fastapi import FastAPI

from pydantic import BaseModel

import json

import pandas as pd

import dill

# Input features class
class ModelInputFeatures(BaseModel):
    age: int
    balance: int
    duration: int
    campaign: int
    pdays: int
    previous: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    day_of_week: int
    month: str
    poutcome: str

class ModelOutputFeatures(BaseModel):
    prediction_float: float
    prediction_boolean: int

# Load model
    
with open("model/current_model.dill", "rb") as f_:
    pipeline = dill.load(f_)

app = FastAPI()

@app.post("/prediction/")
async def get_prediction(input_features: ModelInputFeatures) -> ModelOutputFeatures:

    input_features = json.loads(input_features.json())

    input_features_df = pd.DataFrame.from_dict({0: input_features}, orient="index")

    predictions = pipeline.transform(input_features_df)

    output_features = ModelOutputFeatures(**predictions.to_dict(orient="index")[0])

    return output_features




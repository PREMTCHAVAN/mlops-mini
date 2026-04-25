from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "MLOps API Running 🚀"}

@app.post("/predict")
def get_prediction(data: InputData):
    result = predict(data.features)
    return {"prediction": int(result)}
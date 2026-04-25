import joblib
import numpy as np

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(data):
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)
    return model.predict(data)[0]
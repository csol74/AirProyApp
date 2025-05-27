from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Cargar el modelo y scaler
model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")

class InputData(BaseModel):
    indicador1: float
    indicador2: float
    indicador3: float
    indicador4: float

@app.post("/predict")
def predict(data: InputData):
    input_values = np.array([[data.indicador1, data.indicador2, data.indicador3, data.indicador4]])
    scaled_values = scaler.transform(input_values)
    prediction = model.predict(scaled_values)[0]
    
    if prediction == 0:
        mensaje = "Calidad del aire buena."
    elif prediction == 1:
        mensaje = "Calidad del aire moderada."
    else:
        mensaje = "¡Advertencia! Calidad del aire peligrosa."
    
    return {"prediccion": int(prediction), "mensaje": mensaje}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


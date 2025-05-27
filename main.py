from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Cargar modelo y scaler
model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")

class InputData(BaseModel):
    PM10: float
    PM2_5: float  
    NO2: float
    O3: float

@app.post("/predict")
def predict(data: InputData):
    # Crear DataFrame con los mismos nombres que se usaron al entrenar el modelo
    input_df = pd.DataFrame([{
        "PM10": data.PM10,
        "PM2,5": data.PM2_5,  
        "NO2": data.NO2,
        "O3": data.O3
    }])

    # Escalar valores
    scaled_values = scaler.transform(input_df)

    # Hacer predicción
    prediction = model.predict(scaled_values)[0]
    mensaje = f"Calidad del aire: {prediction}."

    return {"prediccion": prediction, "mensaje": mensaje}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

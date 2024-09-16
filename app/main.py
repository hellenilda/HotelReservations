from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os
from dotenv import load_dotenv
import tarfile

# Inicializar a aplicação FastAPI
app = FastAPI()

# Modelo de input usando Pydantic
class InferenceRequest(BaseModel):
    no_of_adults: int
    no_of_children: int
    type_of_meal_plan: str

def extract_model(tar_gz_path, extract_path):
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

def load_model_from_local_path(model_path):
    return joblib.load(model_path)

def load_model():
    extract_model('model/model.tar.gz', '/tmp/modelo/')
    model = load_model_from_local_path('model/trained_model/xgboost-model.pkl')
    return model

# Carregar o modelo no startup da aplicação
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

# Endpoint de inferência
@app.post("/api/v1/inference")
async def inference(data: InferenceRequest):
    try:
        features = [
            data.no_of_adults,
            data.no_of_children,
            data.type_of_meal_plan,
        ]

        result = model.predict([features])[0]
        return {"result": int(result)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os
from dotenv import load_dotenv

# Carregar as variáveis do arquivo .env
load_dotenv()

# Inicializar a aplicação FastAPI
app = FastAPI()

# Modelo de input usando Pydantic
class InferenceRequest(BaseModel):
    no_of_adults: int
    no_of_children: int
    type_of_meal_plan: str

# Função para carregar o modelo do S3
def load_model():
    s3 = boto3.client('s3',
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    bucket = os.getenv('S3_BUCKET_NAME')
    model_path = os.getenv('MODEL_PATH')
    local_model_path = '/tmp/modelo_local.pkl'  # Salvar no diretório temporário
    s3.download_file(bucket, model_path, local_model_path)
    model = joblib.load(local_model_path)
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
        # Preparar os dados para inferência
        features = [
            data.no_of_adults,
            data.no_of_children,
            data.type_of_meal_plan,
        ]

        # Fazer a predição
        result = model.predict([features])[0]

        # Retornar o resultado
        return {"result": int(result)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
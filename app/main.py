from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Inicializar a aplicação FastAPI
app = FastAPI()

# Modelo de input usando Pydantic
class InferenceRequest(BaseModel):
    no_of_adults: int
    no_of_children: int
    type_of_meal_plan: int
    room_type_reserved: int
    arrival_year: int
    market_segment_type: int

def load_model_from_local_path(model_path):
    # Carregar o modelo XGBoost treinado
    return joblib.load(model_path)

def load_model():
    # Carregar o modelo treinado
    model = load_model_from_local_path('scripts/modelo_xgboost_treinado.pkl')
    return model

# Função para processar as entradas e gerar as 30 features necessárias
def process_input(data):
    features = []

    # Adicione as variáveis numéricas diretamente
    features.append(data.no_of_adults)
    features.append(data.no_of_children)

    # Exemplo de dummy encoding para 'type_of_meal_plan' (ajustar conforme necessário)
    meal_plan_dummies = [0, 0, 0]  # Exemplo com 3 categorias
    meal_plan_dummies[data.type_of_meal_plan] = 1
    features.extend(meal_plan_dummies)

    # Exemplo de dummy encoding para 'room_type_reserved' (ajustar conforme necessário)
    room_type_dummies = [0, 0, 0, 0, 0]  # Exemplo com 5 categorias
    room_type_dummies[data.room_type_reserved] = 1
    features.extend(room_type_dummies)

    # Exemplo de dummy encoding para 'market_segment_type' (ajustar conforme necessário)
    market_segment_dummies = [0, 0, 0]  # Exemplo com 3 categorias
    market_segment_dummies[data.market_segment_type] = 1
    features.extend(market_segment_dummies)

    # Adicione variáveis contínuas diretamente, como 'arrival_year'
    features.append(data.arrival_year)

    # O código acima resulta em 14 features. A partir daqui, preencha com as variáveis restantes
    # As variáveis extras dependem de como foi feito o pré-processamento no treinamento
    # Suponha que você precise de 16 features a mais para completar 30 (isso varia conforme o modelo)
    extra_features = [0] * (30 - len(features))  # Adicione zeros para completar as 30 features

    features.extend(extra_features)

    # Certifique-se de que a quantidade de variáveis resultantes corresponde às 30 que o modelo espera
    return np.array(features).reshape(1, -1)  # Retorna como um array 2D

# Carregar o modelo no startup da aplicação
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

# Endpoint de inferência
@app.post("/api/v1/inference")
async def inference(data: InferenceRequest):
    try:
        # Processar os dados de entrada para gerar as 30 features
        features = process_input(data)

        # Fazer a predição usando o modelo treinado
        result = model.predict(features)[0]
        return {"result": int(result)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

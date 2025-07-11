# src/api/app_api/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List, Dict, Union

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="API de Predição de Perda de Oportunidades de Venda",
    description="API para prever a probabilidade de uma oportunidade de venda ser perdida usando um modelo de Machine Learning.",
    version="1.0.0"
)

# --- Carregamento do Modelo e Pipeline ---
# Define o caminho do modelo
# Certifique-se de que o caminho está correto em relação à execução da API
# Se você for rodar a API de 'src/api/app_api', o caminho seria '../../models/full_pipeline.joblib'
# Ou se for rodar da raiz do projeto, seria 'models/full_pipeline.joblib'
# Vou assumir que você rodará da pasta raiz do projeto por enquanto.
try:
    FULL_PIPELINE = joblib.load('models/full_pipeline.joblib')
    print("Pipeline 'full_pipeline.joblib' carregado com sucesso!")
except FileNotFoundError:
    print("ERRO: O arquivo 'full_pipeline.joblib' não foi encontrado na pasta 'models/'.")
    print("Certifique-se de que o pipeline foi salvo após a execução do notebook 03_model_experimentation_and_detailed_evaluation.ipynb.")
    FULL_PIPELINE = None # Define como None para evitar erros posteriores

# --- Definição do Modelo de Dados de Entrada (Pydantic) ---
# Este modelo deve espelhar as features que seu pipeline espera.
# Use os tipos de dados Python correspondentes.

class OpportunityData(BaseModel):
    # Features categóricas
    sales_agent: str
    product: str
    sector: str
    office_location: str
    subsidiary_of: str
    series: str
    manager: str
    regional_office: str

    # Features numéricas
    close_value: float
    year_established: int
    revenue: float
    employees: int
    sales_price: float
    opportunity_duration_days: float


# --- Endpoints da API ---

@app.get("/")
async def read_root():
    """
    Endpoint raiz da API. Retorna uma mensagem de boas-vindas e o status da API.
    """
    return {"message": "Bem-vindo à API de Predição de Perda de Oportunidades de Venda!"}

@app.post("/predict")
async def predict_opportunity_loss(data: OpportunityData):
    """
    Endpoint para prever a probabilidade de uma oportunidade de venda ser perdida.

    Recebe os dados de uma oportunidade e retorna a probabilidade de ser perdida (0 a 1).
    """
    if FULL_PIPELINE is None:
        return {"error": "Modelo não carregado. Verifique os logs da aplicação."}

    try:
        # Converte os dados recebidos do Pydantic para um DataFrame pandas
        # É importante que os nomes das colunas e a ordem (se o pipeline for sensível)
        # correspondam ao que o modelo foi treinado.
        input_df = pd.DataFrame([data.model_dump()])

        # Realiza a predição de probabilidade usando o pipeline completo
        # O pipeline já inclui o pré-processamento e o modelo LightGBM
        # predict_proba retorna um array com as probabilidades para cada classe [prob_classe_0, prob_classe_1]
        probabilities = FULL_PIPELINE.predict_proba(input_df)

        # A probabilidade de interesse é a da classe 1 (oportunidade perdida)
        loss_probability = probabilities[0][1] # Pega a probabilidade da primeira (e única) linha, para a classe 1

        return {"loss_probability": float(loss_probability)}

    except Exception as e:
        return {"error": f"Ocorreu um erro durante a predição: {str(e)}"}
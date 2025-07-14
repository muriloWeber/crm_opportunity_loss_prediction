# src/api/app_api/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import sys
import pandas as pd
from datetime import date
from typing import Optional

# Adiciona a raiz do projeto ao PYTHONPATH para importar módulos customizados
# Assumindo que api.py está em src/api/app_api/
# Para chegar na raiz do projeto, precisamos de '../../..'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Adicionado '{project_root}' ao sys.path para importações de módulos do projeto.")

# Importa o custom transformer para que joblib possa carregá-lo
from src.utils.custom_transformers import DateFeatureEngineer

app = FastAPI(
    title="API de Predição de Perda de Oportunidades CRM",
    description="API para prever a probabilidade de uma oportunidade de venda ser perdida."
)

# --- Carregamento do Pipeline Completo ---
# Define o caminho para o arquivo do pipeline
# Assumimos que o pipeline está em project_root/models/full_pipeline.joblib
models_dir = os.path.join(project_root, 'models')
full_pipeline_path = os.path.join(models_dir, 'full_pipeline.joblib')

# Variável global para armazenar o pipeline carregado
full_pipeline = None

@app.on_event("startup")
async def load_pipeline():
    """
    Carrega o pipeline completo (pré-processamento + modelo) na inicialização da API.
    """
    global full_pipeline
    try:
        full_pipeline = joblib.load(full_pipeline_path)
        print(f"Pipeline '{full_pipeline_path}' carregado com sucesso na inicialização da API!")
    except FileNotFoundError:
        print(f"ERRO: Pipeline '{full_pipeline_path}' não encontrado. Certifique-se de que o notebook 03 foi executado e salvou o pipeline.")
        raise HTTPException(status_code=500, detail="Modelo preditivo não encontrado.")
    except Exception as e:
        print(f"ERRO ao carregar o pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao carregar o modelo: {e}")

# --- Definição dos Limiares de Probabilidade ---
# Baseado na análise dos percentis do conjunto de TESTE.
# Estes limiares refletem a natureza polarizada das probabilidades do modelo.
PROB_THRESHOLD_VERY_HIGH = 0.999     # Para probabilidades de perda extremamente altas (P75+)
PROB_THRESHOLD_MEDIUM = 0.0007     # Para probabilidades de perda um pouco mais elevadas (P50 a P75)
PROB_THRESHOLD_LOW = 0.0001        # Para probabilidades de perda baixas (P10 a P50)
# Qualquer coisa abaixo de PROB_THRESHOLD_LOW será "Muito Baixa"

def classify_probability_label(probability: float) -> str:
    """
    Classifica a probabilidade de perda em uma label interpretável para o time de negócio.
    """
    if probability >= PROB_THRESHOLD_VERY_HIGH:
        return "Chance MUITO ALTA de Perda"
    elif probability >= PROB_THRESHOLD_MEDIUM:
        return "Chance MÉDIA de Perda"
    elif probability >= PROB_THRESHOLD_LOW:
        return "Chance BAIXA de Perda"
    else:
        return "Chance MUITO BAIXA de Perda"

# --- Definição do Modelo de Dados para a Predição (Pydantic) ---
class OpportunityData(BaseModel):
    # Campos que esperamos que NUNCA estejam ausentes/nulos
    sales_agent: str = Field(..., example="Ricardo Dantas", description="Nome do agente de vendas.")
    product: str = Field(..., example="GTK 500", description="Nome do produto envolvido na oportunidade.")
    # Colunas que podem ter NaNs e serão imputadas pelo pipeline
    # Use Optional[Tipo] e um valor padrão de None
    sector: Optional[str] = Field(None, example="Technology", description="Setor de atuação da conta. Pode ser nulo.")
    office_location: Optional[str] = Field(None, example="São Paulo", description="Localização do escritório do cliente. Pode ser nulo.")
    subsidiary_of: Optional[str] = Field(None, example="Empresa Grande S.A.", description="Empresa controladora, se aplicável ('Not_Subsidiary' se não). Pode ser nulo.")
    series: Optional[str] = Field(None, example="GTK", description="Série do produto. Pode ser nulo.")
    manager: Optional[str] = Field(None, example="Fernando Costa", description="Nome do gerente do agente de vendas. Pode ser nulo.")
    regional_office: Optional[str] = Field(None, example="Southeast", description="Escritório regional do agente de vendas. Pode ser nulo.")
    
    close_value: Optional[float] = Field(None, example=1500.0, description="Valor potencial de fechamento da oportunidade. Pode ser nulo.")
    year_established: Optional[int] = Field(None, example=2005, description="Ano de fundação da empresa cliente. Pode ser nulo.")
    revenue: Optional[float] = Field(None, example=5000000.0, description="Receita anual da empresa cliente. Pode ser nulo.")
    employees: Optional[int] = Field(None, example=250, description="Número de funcionários da empresa cliente. Pode ser nulo.")
    sales_price: Optional[float] = Field(None, example=1200.0, description="Preço de venda do produto. Pode ser nulo.")
    
    # Datas que podem ser nulas e serão tratadas pelo DateFeatureEngineer
    engage_date: Optional[date] = Field(None, example="2024-01-15", description="Data de engajamento inicial na oportunidade (YYYY-MM-DD). Pode ser nula.")
    close_date: Optional[date] = Field(None, example="2024-03-20", description="Data de fechamento prevista/real da oportunidade (YYYY-MM-DD). Pode ser nula.")

# --- Endpoint de Verificação de Saúde (Health Check) ---
@app.get("/health", summary="Verifica a saúde da API", response_description="Status da API")
async def health_check():
    """
    Endpoint simples para verificar se a API está online e o modelo foi carregado.
    """
    if full_pipeline is not None:
        return {"status": "ok", "message": "API está online e modelo carregado."}
    else:
        raise HTTPException(status_code=503, detail="API está online, mas o modelo ainda não foi carregado ou falhou ao carregar.")

# --- Endpoint de Predição ---
@app.post("/predict", summary="Prever probabilidade de perda de oportunidade", response_description="Probabilidade e classificação de perda")
async def predict_loss(data: OpportunityData):
    """
    Recebe os dados de uma oportunidade de venda e retorna a probabilidade
    de ela ser perdida, juntamente com uma classificação textual.
    """
    global full_pipeline

    if full_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo preditivo não carregado. Tente novamente mais tarde.")

    try:
        input_data = data.model_dump()

        if input_data['engage_date'] is not None:
            input_data['engage_date'] = input_data['engage_date'].strftime('%Y-%m-%d')
        if input_data['close_date'] is not None:
            input_data['close_date'] = input_data['close_date'].strftime('%Y-%m-%d')

        expected_columns = [
            'sales_agent', 'product', 'engage_date', 'close_date', 'close_value',
            'sector', 'year_established', 'revenue', 'employees', 'office_location',
            'subsidiary_of', 'series', 'sales_price', 'manager', 'regional_office'
        ]
        
        df_input = pd.DataFrame([input_data])[expected_columns]

        prediction_proba = full_pipeline.predict_proba(df_input)[:, 1][0]
        
        # --- Classifica a probabilidade em uma label ---
        prediction_label = classify_probability_label(prediction_proba)

        return {
            "prediction_probability_of_loss": float(prediction_proba),
            "prediction_label": prediction_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante a predição: {str(e)}")

# Para rodar localmente (apenas para testes diretos via script Python, não para Uvicorn)
if __name__ == "__main__":
    import uvicorn
    # O host '0.0.0.0' permite acesso de outras máquinas na rede, se necessário.
    # O port 8000 é o padrão do FastAPI/Uvicorn.
    uvicorn.run(app, host="0.0.0.0", port=8000)
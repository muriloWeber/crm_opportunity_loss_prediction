import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# --- 1. Carregar o Modelo Treinado e Componentes de Pré-processamento ---
# Define o caminho para o diretório raiz do projeto a partir do arquivo atual da API
# Isso é importante para que o caminho para o modelo e o scaler funcionem independentemente
# de onde a API é iniciada.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'lightgbm_model.joblib')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.joblib') # Precisaremos salvar o scaler!

try:
    model = joblib.load(MODEL_PATH)
    # Precisamos carregar o scaler também, pois as features numéricas foram escalonadas.
    # Se você ainda não salvou o scaler no 02_data_preparation.ipynb, faremos isso em breve.
    scaler = joblib.load(SCALER_PATH)
    print("Modelo e Scaler carregados com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar o modelo ou scaler: {e}")
    print("Certifique-se de que o '03_model_training.ipynb' foi executado (para o modelo) e que o scaler foi salvo.")
    # Saída ou levantamento de erro para indicar falha crítica
    raise RuntimeError("Falha ao iniciar a API: Modelo ou Scaler não encontrados.")

# --- 2. Definir o Schema de Entrada da API (Validação de Dados) ---
# Usaremos Pydantic para validar os dados de entrada da API.
# Este é um exemplo, você precisará ajustar para TODAS as features usadas pelo seu modelo.
class OpportunityData(BaseModel):
    # Exemplo de algumas features. Adapte isso exatamente às suas features de entrada.
    # Lembre-se: o modelo espera as features APÓS o pré-processamento (OHE e escalonamento).
    # Aqui, definimos as features BRUTAS que a API receberá.
    # Exemplo:
    # opportunity_id: str
    # sales_agent: str
    # product: str
    # close_value: float
    # opportunity_duration_days: float
    # year_established: float
    # revenue: float
    # employees: float
    # subsidiary_of: str
    # sector: str
    # office_location: str
    # manager: str
    # regional_office: str
    # series: str

    # Para fins de demonstração, vou listar algumas que sei que você tem.
    # VOCÊ PRECISARÁ AJUSTAR ISSO PARA TODAS AS 13 FEATURES ORIGINAIS (antes do OHE)
    sales_agent: str
    product: str
    close_value: float
    opportunity_duration_days: float
    year_established: float
    revenue: float
    employees: float
    subsidiary_of: str
    sector: str
    office_location: str
    manager: str
    regional_office: str
    series: str
    # Remova 'opportunity_id' e 'account' aqui, pois não são usados na predição direta.
    # 'target' também não entra aqui.

# --- 3. Inicializar a Aplicação FastAPI ---
app = FastAPI(
    title="API de Predição de Perda de Oportunidades de Venda B2B",
    description="API para prever a probabilidade de uma oportunidade de venda ser perdida."
)

# --- 4. Endpoint de Predição ---
@app.post("/predict_loss_probability/")
async def predict_loss_probability(data: OpportunityData):
    try:
        # Converter os dados de entrada Pydantic para um DataFrame pandas
        input_df = pd.DataFrame([data.model_dump()]) # model_dump() para Pydantic v2+

        # --- Aplicar Pré-processamento Idêntico ao Treinamento ---
        # ATENÇÃO: Esta é a parte mais CRÍTICA e complexa.
        # Precisaremos de um Pipeline ou de funções de pré-processamento que repliquem
        # exatamente o 02_data_preparation.ipynb.
        # Por enquanto, vou deixar um placeholder para o que precisará ser feito:

        # 4.1. Tratamento de colunas categóricas com NaN (se houver no input)
        # Ex: input_df['subsidiary_of'].fillna('Not_Subsidiary', inplace=True)
        # Ex: input_df['sector'].fillna('Unknown_Sector', inplace=True)
        # Faça isso para TODAS as categóricas que podem vir com NaN

        # 4.2. Criar a feature 'opportunity_duration_days' (se ainda não for input direto)
        # O ideal é que 'opportunity_duration_days' já seja um campo direto no OpportunityData
        # se o cliente (Streamlit ou CRM) for responsável por calculá-lo.
        # Se não, você precisará de engage_date e close_date aqui.

        # 4.3. One-Hot Encoding para as features categóricas
        # Esta é a parte mais propensa a erros se não for feita com o mesmo mapeamento do treino.
        # Você precisará salvar as colunas dummy criadas durante o treino ou usar um ColumnTransformer.
        # Por exemplo, se 'sector' tem 5 categorias no treino, o OHE deve criar 5 colunas.
        # A forma mais robusta é usar o "feature engineering" do `02_data_preparation.ipynb`
        # e aplicar aqui. Idealmente, teríamos um mapeamento de colunas.

        # EXEMPLO SIMPLIFICADO E INCOMPLETO PARA OHE:
        categorical_cols = ['sales_agent', 'product', 'subsidiary_of', 'sector',
                            'office_location', 'manager', 'regional_office', 'series']
        # Precisaremos de todas as colunas que foram criadas pelo get_dummies no treinamento.
        # Uma abordagem robusta seria ter uma lista dessas colunas ou um OneHotEncoder já treinado.
        # Por simplicidade inicial, vamos criar as dummys apenas para o input_df.
        # Isso pode falhar se o input tiver categorias novas ou perder categorias existentes.
        input_df_processed = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        # Garantir que todas as colunas que o modelo espera estejam presentes,
        # e na ordem correta, preenchendo com 0 se a categoria não estiver no input.
        # (Isso exige a lista das colunas finais do X_train. Vamos precisar salvar X_train.columns)
        # Por enquanto, vamos carregar as colunas de um X_train salvo para ter a ordem e nomes corretos.
        X_train_cols_path = os.path.join(PROJECT_ROOT, 'models', 'X_train_columns.joblib')
        try:
            X_train_columns = joblib.load(X_train_cols_path)
        except FileNotFoundError:
            raise RuntimeError("Lista de colunas do X_train não encontrada. Salve X_train.columns no 02_data_preparation.ipynb.")

        # Reindexar o DataFrame de entrada para corresponder às colunas de X_train
        # Isso adicionará colunas que não vieram no input_df como 0 e removerá as extras.
        input_df_final = input_df_processed.reindex(columns=X_train_columns, fill_value=0)


        # 4.4. Escalonamento de Features Numéricas
        # As colunas numéricas devem ser as mesmas que foram escalonadas no treinamento.
        numeric_cols_for_scaling = ['year_established', 'revenue', 'employees', 'sales_price', 'close_value', 'opportunity_duration_days']
        # Certifique-se de que essas colunas existem e estão na ordem correta
        # antes de aplicar o scaler.
        input_df_final[numeric_cols_for_scaling] = scaler.transform(input_df_final[numeric_cols_for_scaling])


        # 4.5. Predição da Probabilidade
        # lgbm_model.predict_proba() retorna as probabilidades para as classes [0, 1]
        # Queremos a probabilidade da classe 1 (perdida).
        probability_of_loss = model.predict_proba(input_df_final)[:, 1][0] # [0] para pegar o primeiro (e único) resultado

        return {"probability_of_loss": probability_of_loss}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {str(e)}")

# --- Exemplo de como rodar a API (para teste local, não faz parte do arquivo api.py) ---
# Para rodar:
# cd para a pasta app_api/
# uvicorn api:app --host 0.0.0.0 --port 8000 --reload
#
# Para testar (com curl ou Postman/Insomnia):
# curl -X POST "http://localhost:8000/predict_loss_probability/" -H "Content-Type: application/json" -d '{
#   "sales_agent": "Ricardo Bents",
#   "product": "GTK 100",
#   "close_value": 5000.0,
#   "opportunity_duration_days": 60.0,
#   "year_established": 2005.0,
#   "revenue": 1000000.0,
#   "employees": 500.0,
#   "subsidiary_of": "Not_Subsidiary",
#   "sector": "Marketing",
#   "office_location": "New York",
#   "manager": "Rocco Neubert",
#   "regional_office": "East",
#   "series": "GTK"
# }'


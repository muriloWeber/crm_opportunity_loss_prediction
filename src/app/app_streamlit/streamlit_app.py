import streamlit as st
import requests
import json
from datetime import datetime
import os

# Tenta obter a URL da API da variável de ambiente 'API_URL'.
# Se a variável de ambiente NÃO for definida (como ao rodar localmente),
# ele usa 'http://127.0.0.1:8000/predict' como valor padrão.
# Quando rodar no Docker Compose, a variável 'API_URL' será definida,
# e o Streamlit usará 'http://api:8000/predict'.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# --- Opções para os Dropdowns ---
sales_agent_options = ['Unknown', 'Anna Snelling', 'Boris Faz', 'Cassey Cress', 'Cecily Lampkin', 'Corliss Cosme', 'Daniell Hammack', 'Darcel Schlecht', 'Donn Cantrell', 'Elease Gluck', 'Garret Kinder', 'Gladys Colclough', 'Hayden Neloms', 'James Ascencio', 'Jonathan Berthelot', 'Kami Bicknell', 'Kary Hendrixson', 'Lajuana Vencill', 'Markita Hansen', 'Marty Freudenburg', 'Maureen Marcano', 'Moses Frase', 'Niesha Huffines', 'Reed Clapper', 'Rosalina Dieter', 'Rosie Papadopoulos', 'Versie Hillebrand', 'Vicki Laflamme', 'Violet Mclelland', 'Wilburn Farren', 'Zane Levy']
product_options = ['Unknown', 'GTK 500', 'GTX Basic', 'GTX Plus Basic', 'GTX Plus Pro', 'GTXPro', 'MG Advanced', 'MG Special']
sector_options = ['Unknown', 'employment', 'entertainment', 'finance', 'marketing', 'medical', 'retail', 'services', 'software', 'technolgy', 'telecommunications']
office_location_options = ['Unknown', 'Belgium', 'Brazil', 'China', 'Germany', 'Italy', 'Japan', 'Jordan', 'Kenya', 'Korea', 'Norway', 'Panama', 'Philipines', 'Poland', 'Romania', 'United States']
subsidiary_of_options = ['Not_Subsidiary', 'Acme Corporation', 'Bubba Gump', 'Golddex', 'Inity', 'Massive Dynamic', 'Sonron', 'Warephase']
series_options = ['Unknown', 'GTK', 'GTX', 'MG']
manager_options = ['Unknown', 'Cara Losch', 'Celia Rouche', 'Dustin Brinkmann', 'Melvin Marxen', 'Rocco Neubert', 'Summer Sewald']
regional_office_options = ['Unknown', 'Central', 'East', 'West']
deal_stage_options = ['Unknown', 'Engaging', 'Prospecting']
# --------------------------------------------------------

st.set_page_config(page_title="Simulador de Perda de Oportunidades CRM", layout="wide")

st.title("📊 Simulador de Perda de Oportunidades CRM")
st.markdown("Este aplicativo simula a predição de perda de uma oportunidade de venda B2B usando sua API de Machine Learning.")

st.sidebar.header("Dados da Oportunidade")

# Campos de entrada para o usuário
# Usando st.selectbox para features categóricas
sales_agent = st.sidebar.selectbox("Agente de Vendas", sales_agent_options)
product = st.sidebar.selectbox("Produto", product_options)
sector = st.sidebar.selectbox("Setor", sector_options)
office_location = st.sidebar.selectbox("Localização do Escritório", office_location_options)
subsidiary_of = st.sidebar.selectbox("Subsidiária de", subsidiary_of_options)
series = st.sidebar.selectbox("Série do Produto", series_options)
manager = st.sidebar.selectbox("Gerente", manager_options)
regional_office = st.sidebar.selectbox("Escritório Regional", regional_office_options)
deal_stage = st.sidebar.selectbox("Etapa do Negócio", deal_stage_options, help="Última etapa conhecida antes do fechamento/perda.")

close_value = st.sidebar.number_input("Valor de Fechamento", min_value=0.0, format="%f")
year_established = st.sidebar.number_input("Ano de Fundação da Empresa", min_value=1900, max_value=datetime.now().year, value=2000)
revenue = st.sidebar.number_input("Faturamento da Empresa", min_value=0.0, format="%f")
employees = st.sidebar.number_input("Número de Funcionários", min_value=0)
sales_price = st.sidebar.number_input("Preço de Venda do Produto", min_value=0.0, format="%f")

# Campos de data (permitindo nulos para testar a imputação da API)
# Usando st.date_input para melhor UX e formato correto
# Definindo uma data padrão para o dia atual para facilitar o preenchimento
today = datetime.now().date()
engage_date = st.sidebar.date_input("Data de Engajamento", value=None, help="Formato esperado: AAAA-MM-DD")
close_date = st.sidebar.date_input("Previsão de Fechamento", value=None, help="Formato esperado: AAAA-MM-DD")

# No payload, convertemos para string no formato correto
# Se o usuário não selecionar uma data (valor None), enviamos None para a API
engage_date_str = engage_date.strftime("%Y-%m-%d") if engage_date else None
close_date_str = close_date.strftime("%Y-%m-%d") if close_date else None


if st.button("Prever Probabilidade de Perda"):
    # Construir o payload JSON para a API
    payload = {
        "sales_agent": None if sales_agent == "Unknown" else sales_agent,
        "product": None if product == "Unknown" else product,
        "sector": None if sector == "Unknown" else sector,
        "office_location": None if office_location == "Unknown" else office_location,
        "subsidiary_of": None if subsidiary_of == "Not_Subsidiary" else subsidiary_of,
        "series": None if series == "Unknown" else series,
        "manager": None if manager == "Unknown" else manager,
        "regional_office": None if regional_office == "Unknown" else regional_office,
        "deal_stage": None if deal_stage == "Unknown" else deal_stage,
        "close_value": close_value if close_value > 0 else None,
        "year_established": year_established if year_established > 1900 and year_established < datetime.now().year else None, # Ajuste para considerar 2000 como default e passar None se for valor padrão ou inválido
        "revenue": revenue if revenue > 0 else None,
        "employees": employees if employees > 0 else None,
        "sales_price": sales_price if sales_price > 0 else None,
        "engage_date": engage_date_str if engage_date_str else None,
        "close_date": close_date_str if close_date_str else None
    }

    st.write("---")
    st.subheader("Resultados da Predição")

    try:
        # Fazer a requisição POST para a API
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Lança exceção para códigos de status HTTP de erro (4xx ou 5xx)

        prediction_data = response.json()

        prob_loss = prediction_data.get("prediction_probability_of_loss")
        label = prediction_data.get("prediction_label")

        if prob_loss is not None and label is not None:
            st.success(f"**Probabilidade de Perda:** `{prob_loss:.6f}`")
            st.info(f"**Classificação:** `{label}`")

            # Adicionar algumas dicas visuais baseadas na label
            if "MUITO ALTA" in label:
                st.error("🚨 Atenção: Esta oportunidade tem uma **chance MUITO ALTA de ser perdida**. Intervenção imediata recomendada!")
            elif "MÉDIA" in label:
                st.warning("⚠️ Alerta: Esta oportunidade tem uma **chance MÉDIA de ser perdida**. Considere uma revisão de estratégia.")
            elif "BAIXA" in label:
                st.success("✅ Bom Sinal: Esta oportunidade tem uma **chance BAIXA de ser perdida**.")
            else: # MUITO BAIXA
                st.success("👍 Excelente: Esta oportunidade tem uma **chance MUITO BAIXA de ser perdida**. Ótimo!")

            st.markdown("---")
            st.markdown("Dados enviados para a API:")
            st.json(payload) # Exibe o payload enviado
        else:
            st.error("Erro: A resposta da API não contém as chaves esperadas (prediction_probability_of_loss, prediction_label).")
            st.json(prediction_data) # Exibe a resposta completa para depuração

    except requests.exceptions.ConnectionError:
        st.error(f"Erro de conexão: Não foi possível conectar à API em `{API_URL}`. Certifique-se de que a API está rodando.")
        st.info("Para rodar a API, use no terminal (na raiz do projeto): `uvicorn src.api.app_api.api:app --reload`")
    except requests.exceptions.Timeout:
        st.error("Erro: A requisição para a API excedeu o tempo limite.")
    except requests.exceptions.RequestException as e:
        st.error(f"Ocorreu um erro na requisição à API: {e}")
        st.write(f"Status HTTP: {response.status_code if 'response' in locals() else 'N/A'}")
        st.write(f"Corpo da Resposta: {response.text if 'response' in locals() else 'N/A'}")
    except json.JSONDecodeError:
        st.error("Erro: Resposta inválida da API (não é um JSON válido).")
        st.write(f"Conteúdo da resposta: {response.text if 'response' in locals() else 'N/A'}")
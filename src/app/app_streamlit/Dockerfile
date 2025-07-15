# Usa a mesma imagem base Python que a API para consistência
FROM python:3.13-slim-buster

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de requisitos para o contêiner
# O arquivo requirements.txt DEVE estar na raiz do seu projeto
COPY requirements.txt .

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do aplicativo Streamlit
COPY src/app/app_streamlit/ ./src/app/app_streamlit/

# Define a variável de ambiente para que o Python consiga encontrar os módulos
ENV PYTHONPATH=/app

# Exponha a porta que o Streamlit usa por padrão
EXPOSE 8501

# Comando para rodar o aplicativo Streamlit
# --server.port 8501: Define a porta
# --server.address 0.0.0.0: Permite acesso de qualquer IP dentro do contêiner
CMD ["streamlit", "run", "src/app/app_streamlit/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
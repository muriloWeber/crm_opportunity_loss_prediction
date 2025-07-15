# Usa uma imagem base Python oficial (escolha a versão compatível com seu ambiente)
FROM python:3.13-slim-buster

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de requisitos para o contêiner
# O arquivo requirements.txt DEVE estar na raiz do seu projeto
COPY requirements.txt .

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da API, os utilitários e os modelos
# Certifique-se de que a pasta 'models' está na raiz do seu projeto
COPY src/api/app_api/ ./src/api/app_api/
COPY src/utils/ ./src/utils/
COPY models/ ./models/

# Define a variável de ambiente para que o Python consiga encontrar os módulos
ENV PYTHONPATH=/app

# Comando para rodar a API usando Uvicorn
# '0.0.0.0' é necessário para que a API seja acessível de fora do contêiner
CMD ["uvicorn", "src.api.app_api.api:app", "--host", "0.0.0.0", "--port", "8000"]
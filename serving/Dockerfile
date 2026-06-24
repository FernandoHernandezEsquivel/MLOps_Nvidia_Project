# serving/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY serving/app.py .
COPY mlflow_runs/ ./mlflow_runs/
COPY src/ ./src/

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

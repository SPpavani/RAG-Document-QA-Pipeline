FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
# Mount your data folder at run time so the vector store persists:
#   docker run -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" rag-api
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

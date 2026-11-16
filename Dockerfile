FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Couche dépendances : invalidée uniquement si requirements.txt change
COPY requirements.txt .
RUN pip install -r requirements.txt

# Couche code : seule couche reconstruite lors d'une modification de l'app
COPY app/ ./app/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

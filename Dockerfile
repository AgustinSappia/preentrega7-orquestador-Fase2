# Imagen de la API (Python 3.12, como pide la consigna).
FROM python:3.12-slim

# Sin archivos .pyc y con logs que salen al instante (sin buffer).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Primero solo requirements.txt: si el código cambia pero las dependencias
# no, Docker reutiliza esta capa y no reinstala todo de nuevo.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

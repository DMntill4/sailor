# =============================================================================
# Dockerfile para Tribuna Global - Analisis de Percepcion con Streamlit y LLM
# =============================================================================

FROM python:3.12-slim

# Evitar generacion de archivos .pyc y activar salida sin buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Instalar dependencias del sistema requeridas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar manifiesto de dependencias e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el codigo fuente del proyecto
COPY . .

# Crear directorio de datos con permisos adecuados si no existe
RUN mkdir -p /app/datos

# Exponer el puerto predeterminado de Streamlit
EXPOSE 8501

# Comprobacion de salud (Healthcheck)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de inicio del servidor
CMD ["streamlit", "run", "app.py"]


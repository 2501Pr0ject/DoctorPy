# Dockerfile pour Service RAG - DoctorPy
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="DoctorPy Team"
LABEL service="rag"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SERVICE_NAME=rag
ENV SERVICE_PORT=8002

# Dossier de travail
WORKDIR /app

# Installer les dépendances système (incluant cURL pour health check)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY services/shared/ ./services/shared/
COPY services/rag/ ./services/rag/

# Créer répertoires pour données vectorielles (volumes)
RUN mkdir -p /app/data/vector_stores
RUN mkdir -p /app/data/cache

# Créer utilisateur non-root pour sécurité
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Port exposé
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Commande de démarrage
CMD ["python", "services/rag/app_simple.py"]
# Dockerfile pour Service Auth - DoctorPy
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="DoctorPy Team"
LABEL service="auth"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SERVICE_NAME=auth
ENV SERVICE_PORT=8001

# Dossier de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY services/shared/ ./services/shared/
COPY services/auth/ ./services/auth/

# Créer utilisateur non-root pour sécurité
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Port exposé
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Commande de démarrage
CMD ["python", "services/auth/app_simple.py"]
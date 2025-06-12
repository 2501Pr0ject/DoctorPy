# Dockerfile pour Service Quest - DoctorPy
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="DoctorPy Team"
LABEL service="quest"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SERVICE_NAME=quest
ENV SERVICE_PORT=8004

# Dossier de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY services/shared/ ./services/shared/
COPY services/quest/ ./services/quest/

# Créer répertoires pour données de quêtes
RUN mkdir -p /app/data/quests
RUN mkdir -p /app/data/progress

# Créer utilisateur non-root pour sécurité
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Port exposé
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Commande de démarrage
CMD ["python", "services/quest/app_simple.py"]
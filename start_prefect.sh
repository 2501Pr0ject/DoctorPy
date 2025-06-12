#!/bin/bash

# Script de démarrage Prefect pour DoctorPy

echo "🚀 Démarrage de Prefect pour DoctorPy..."

# Définir les variables d'environnement
export PREFECT_API_URL="http://localhost:4200/api"
export PREFECT_LOGGING_LEVEL="INFO"

# Démarrer le serveur Prefect en arrière-plan
echo "🖥️ Démarrage du serveur Prefect..."
prefect server start --host 0.0.0.0 --port 4200 &
SERVER_PID=$!

# Attendre que le serveur démarre
sleep 10

# Créer l'agent de travail
echo "🤖 Création de l'agent de travail..."
prefect work-pool create default-agent-pool --type process &
POOL_PID=$!

# Attendre que le pool soit créé
sleep 5

# Démarrer l'agent
echo "▶️ Démarrage de l'agent..."
prefect agent start --pool default-agent-pool &
AGENT_PID=$!

echo "✅ Prefect démarré avec succès!"
echo "🌐 Interface web: http://localhost:4200"
echo "📊 Dashboard: http://localhost:4200/dashboard"

# Garder le script en vie
echo "🔄 Prefect en cours d'exécution... (Ctrl+C pour arrêter)"
wait $SERVER_PID $POOL_PID $AGENT_PID
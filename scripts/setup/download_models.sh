#!/bin/bash

# Script de téléchargement des modèles Ollama
set -e

echo "📦 Téléchargement des modèles Ollama..."

# Modèles à télécharger
MODELS=(
    "llama3.1:8b"
    "codellama:7b"
)

# Fonction pour télécharger un modèle
download_model() {
    local model=$1
    echo "⬇️  Téléchargement de $model..."
    
    if ollama pull "$model"; then
        echo "✅ $model téléchargé avec succès"
    else
        echo "❌ Échec du téléchargement de $model"
        return 1
    fi
}

# Vérifier qu'Ollama est démarré
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama n'est pas démarré. Lancez: ollama serve"
    exit 1
fi

# Télécharger chaque modèle
for model in "${MODELS[@]}"; do
    download_model "$model"
done

echo ""
echo "🎉 Tous les modèles ont été téléchargés!"
echo ""
echo "📋 Modèles disponibles:"
ollama list
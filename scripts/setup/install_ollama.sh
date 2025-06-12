#!/bin/bash

# Script d'installation d'Ollama pour Linux/macOS
set -e

echo "🚀 Installation d'Ollama..."

# Détecter l'OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📱 Système détecté: Linux"
    
    # Installation via curl
    curl -fsSL https://ollama.ai/install.sh | sh
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Système détecté: macOS"
    
    # Vérifier si Homebrew est installé
    if command -v brew &> /dev/null; then
        echo "🍺 Installation via Homebrew..."
        brew install ollama
    else
        echo "📥 Installation via curl..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    
else
    echo "❌ Système non supporté: $OSTYPE"
    echo "Veuillez installer Ollama manuellement depuis https://ollama.ai"
    exit 1
fi

# Vérifier l'installation
if command -v ollama &> /dev/null; then
    echo "✅ Ollama installé avec succès!"
    ollama --version
    
    # Démarrer le service
    echo "🔧 Démarrage du service Ollama..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux: utiliser systemctl si disponible
        if command -v systemctl &> /dev/null; then
            sudo systemctl start ollama
            sudo systemctl enable ollama
            echo "✅ Service Ollama démarré et activé"
        else
            echo "⚠️  Démarrez Ollama manuellement avec: ollama serve"
        fi
    else
        # macOS: démarrer en arrière-plan
        ollama serve &
        echo "✅ Ollama démarré en arrière-plan"
    fi
    
    # Attendre que le service soit prêt
    echo "⏳ Attente que le service soit prêt..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Service Ollama prêt!"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo "⚠️  Le service met du temps à démarrer. Vérifiez avec: curl http://localhost:11434/api/tags"
        fi
    done
    
else
    echo "❌ Échec de l'installation d'Ollama"
    exit 1
fi

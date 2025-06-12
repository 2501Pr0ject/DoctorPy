#!/bin/bash

# Script de configuration complète du projet
set -e

echo "🚀 Configuration complète du Python Learning Assistant"
echo "=================================================="

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages colorés
print_step() {
    echo -e "${BLUE}[ÉTAPE]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCÈS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[ATTENTION]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERREUR]${NC} $1"
}

# Vérifier Python
print_step "Vérification de Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION trouvé"
else
    print_error "Python 3 n'est pas installé"
    exit 1
fi

# Créer l'environnement virtuel
print_step "Création de l'environnement virtuel..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Environnement virtuel créé"
else
    print_warning "Environnement virtuel existant trouvé"
fi

# Activer l'environnement virtuel
print_step "Activation de l'environnement virtuel..."
source venv/bin/activate
print_success "Environnement virtuel activé"

# Installer les dépendances
print_step "Installation des dépendances Python..."
pip install --upgrade pip
pip install -r requirements.txt
print_success "Dépendances installées"

# Créer le fichier .env
print_step "Configuration de l'environnement..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_success "Fichier .env créé à partir de .env.example"
    print_warning "Modifiez .env si nécessaire"
else
    print_warning "Fichier .env existant trouvé"
fi

# Installer Ollama
print_step "Installation d'Ollama..."
if command -v ollama &> /dev/null; then
    print_warning "Ollama déjà installé"
else
    chmod +x scripts/setup/install_ollama.sh
    ./scripts/setup/install_ollama.sh
fi

# Télécharger les modèles
print_step "Téléchargement des modèles Ollama..."
chmod +x scripts/setup/download_models.sh
./scripts/setup/download_models.sh

# Initialiser la base de données
print_step "Initialisation de la base de données..."
python scripts/setup/setup_database.py

# Initialiser le vector store
print_step "Initialisation du vector store..."
print_warning "Cette étape peut prendre plusieurs minutes..."
python scripts/setup/init_vector_store.py

# Test final
print_step "Test de l'installation..."
python -c "
import sys
sys.path.insert(0, 'src')
from src.core.config import settings
from src.core.logger import logger
from src.llm.ollama_client import OllamaClient
from src.rag.retriever import DocumentRetriever

logger.info('Test de configuration...')

# Test Ollama
ollama = OllamaClient()
if ollama.is_model_available():
    logger.info('✅ Ollama configuré')
else:
    logger.error('❌ Problème avec Ollama')

# Test RAG
retriever = DocumentRetriever()
docs = retriever.retrieve_relevant_documents('python variables', max_docs=1)
if docs:
    logger.info('✅ Système RAG fonctionnel')
else:
    logger.warning('⚠️ Système RAG vide')

logger.info('🎉 Configuration terminée!')
"

echo ""
echo "🎉 Configuration terminée avec succès!"
echo ""
echo "📋 Prochaines étapes:"
echo "   1. Activez l'environnement virtuel: source venv/bin/activate"
echo "   2. Lancez l'application: streamlit run ui/streamlit_app.py"
echo "   3. Ou testez l'API: uvicorn src.api.main:app --reload"
echo ""
echo "🔧 Commandes utiles:"
echo "   - Vérifier Ollama: curl http://localhost:11434/api/tags"
echo "   - Logs de l'application: tail -f logs/app.log"
echo "   - Tests: python -m pytest tests/"
echo ""
# Makefile pour DoctorPy - Architecture Microservices v2.0

.PHONY: help dev test demo clean install docs docker

# Configuration
PYTHON := python3
SCRIPTS_DIR := scripts

# Aide
help:
	@echo "🤖 DoctorPy - Commandes disponibles:"
	@echo ""
	@echo "🚀 DÉVELOPPEMENT:"
	@echo "  make dev           - Démarrer tous les services de développement"
	@echo "  make dev-single    - Démarrer un service spécifique (SERVICE=auth|rag|quest|analytics|notification)"
	@echo "  make stop          - Arrêter tous les services"
	@echo ""
	@echo "🧪 TESTS:"
	@echo "  make test          - Lancer les tests d'intégration"
	@echo "  make test-quick    - Tests rapides"
	@echo "  make test-load     - Tests de charge"
	@echo ""
	@echo "🎬 DÉMONSTRATIONS:"
	@echo "  make demo          - Démonstration complète"
	@echo "  make demo-basic    - Démonstration basique"
	@echo ""
	@echo "📦 INSTALLATION:"
	@echo "  make install       - Installer les dépendances"
	@echo "  make install-dev   - Installer les dépendances de développement"
	@echo ""
	@echo "📚 DOCUMENTATION:"
	@echo "  make docs          - Générer la documentation"
	@echo "  make docs-serve    - Servir la documentation"
	@echo ""
	@echo "🐳 DOCKER:"
	@echo "  make docker-build  - Construire les images Docker"
	@echo "  make docker-up     - Démarrer avec Docker Compose"
	@echo "  make docker-down   - Arrêter Docker Compose"
	@echo ""
	@echo "🧹 MAINTENANCE:"
	@echo "  make clean         - Nettoyer les fichiers temporaires"
	@echo "  make lint          - Vérifier le code"
	@echo "  make format        - Formater le code"

# Développement
dev:
	@echo "🚀 Démarrage de l'écosystème DoctorPy..."
	$(PYTHON) $(SCRIPTS_DIR)/dev/run_services.py

dev-single:
	@if [ -z "$(SERVICE)" ]; then \
		echo "❌ Erreur: Spécifiez SERVICE=auth|rag|quest|analytics|notification"; \
		exit 1; \
	fi
	@echo "🚀 Démarrage du service $(SERVICE)..."
	$(PYTHON) apps/$(SERVICE)/app_simple.py

stop:
	@echo "🛑 Arrêt des services..."
	@pkill -f "python.*apps.*app_simple.py" || echo "Aucun service à arrêter"

# Tests
test:
	@echo "🧪 Lancement des tests d'intégration..."
	$(PYTHON) $(SCRIPTS_DIR)/test/test_integration_auto.py

test-quick:
	@echo "⚡ Tests rapides..."
	$(PYTHON) $(SCRIPTS_DIR)/test/quick_test.py

test-load:
	@echo "📊 Tests de charge..."
	@echo "Tests de charge non encore implémentés"

# Démonstrations
demo:
	@echo "🎬 Démonstration complète..."
	$(PYTHON) $(SCRIPTS_DIR)/demo/demo_complete.py

demo-basic:
	@echo "🎬 Démonstration basique..."
	$(PYTHON) $(SCRIPTS_DIR)/demo/demo_services.py

# Installation
install:
	@echo "📦 Installation des dépendances..."
	pip install -r requirements.txt

install-dev: install
	@echo "🛠️ Installation des dépendances de développement..."
	pip install pytest black flake8 mypy jupyter

# Documentation
docs:
	@echo "📚 Génération de la documentation..."
	@echo "Documentation non encore configurée"

docs-serve:
	@echo "🌐 Service de documentation..."
	@echo "Service de documentation non encore configuré"

# Docker
docker-build:
	@echo "🐳 Construction des images Docker..."
	docker-compose build

docker-up:
	@echo "🐳 Démarrage avec Docker Compose..."
	docker-compose up -d

docker-down:
	@echo "🐳 Arrêt Docker Compose..."
	docker-compose down

docker-logs:
	@echo "📋 Logs des services..."
	docker-compose logs -f

docker-status:
	@echo "📊 Status des services..."
	docker-compose ps

# Maintenance
clean:
	@echo "🧹 Nettoyage..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".DS_Store" -delete

lint:
	@echo "🔍 Vérification du code..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 apps/ shared/ --max-line-length=100; \
	else \
		echo "⚠️ flake8 non installé. Utilisez 'make install-dev'"; \
	fi

format:
	@echo "✨ Formatage du code..."
	@if command -v black >/dev/null 2>&1; then \
		black apps/ shared/ scripts/ --line-length=100; \
	else \
		echo "⚠️ black non installé. Utilisez 'make install-dev'"; \
	fi

# Raccourcis
dev-auth:
	@make dev-single SERVICE=auth

dev-rag:
	@make dev-single SERVICE=rag

dev-quest:
	@make dev-single SERVICE=quest

dev-analytics:
	@make dev-single SERVICE=analytics

dev-notification:
	@make dev-single SERVICE=notification
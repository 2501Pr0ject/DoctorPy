# 🚀 Microservices DoctorPy

## Vue d'ensemble

DoctorPy utilise maintenant une **architecture microservices** avec 3 services principaux :

### ✅ Services Implémentés

| Service | Port | Description | Statut |
|---------|------|-------------|--------|
| **Auth Service** | 8001 | Authentification et gestion des utilisateurs | ✅ Opérationnel |
| **RAG Service** | 8002 | Récupération et génération assistée (RAG) | ✅ Nouveau |
| **Quest Service** | 8004 | Gamification et gestion des quêtes | ✅ Nouveau |

## 🚀 Démarrage Rapide

### Option 1 : Démarrage automatique (Recommandé)
```bash
# Démarrer tous les services d'un coup
python start_services.py
```

### Option 2 : Démarrage manuel
```bash
# Service Auth (port 8001)
cd services/auth
python app.py

# Service RAG (port 8002) 
cd services/rag
python app.py

# Service Quest (port 8004)
cd services/quest  
python app.py
```

## 🧪 Tests d'Intégration

```bash
# Tester tous les services
python test_integration.py
```

## 📊 Interfaces d'API

### Service Auth (8001)
- **Swagger**: http://localhost:8001/docs
- **Fonctionnalités**: JWT, RBAC, sessions utilisateur

### Service RAG (8002) 
- **Swagger**: http://localhost:8002/docs
- **Fonctionnalités**: 
  - Requêtes RAG intelligentes
  - Indexation de documents
  - Templates de prompts adaptatifs
  - Cache des réponses

### Service Quest (8004)
- **Swagger**: http://localhost:8004/docs
- **Fonctionnalités**:
  - Gestion des quêtes
  - Système de progression
  - Achievements/badges
  - Leaderboard
  - Analytics utilisateur

## 🔧 Configuration

### Variables d'environnement
```bash
# Redis (requis pour tous les services)
export REDIS_URL="redis://localhost:6379"

# Base de données (optionnel, SQLite par défaut)
export DATABASE_URL="postgresql://user:pass@localhost/doctorpy"

# Clés de sécurité
export SECRET_KEY="your-secret-key-here"
export JWT_SECRET="your-jwt-secret"
```

### Dépendances système
```bash
# Redis (requis)
brew install redis       # macOS
sudo apt install redis   # Ubuntu

# Démarrer Redis
redis-server
```

## 📋 API Endpoints Principaux

### RAG Service
```bash
# Requête RAG
POST /api/v1/rag/query
{
  "query": "Comment créer une liste en Python ?",
  "query_type": "code_help",
  "user_id": "user123"
}

# Templates disponibles
GET /api/v1/rag/templates

# Health check
GET /health
```

### Quest Service
```bash
# Lister les quêtes
GET /api/v1/quests?category=python_basics

# Démarrer une quête
POST /api/v1/quests/start
{
  "quest_id": "python_variables_101",
  "user_id": "user123"
}

# Soumettre une réponse
POST /api/v1/quests/submit
{
  "progress_id": "progress123",
  "question_id": "q1",
  "answer": "age = 25"
}

# Leaderboard
GET /api/v1/quests/leaderboard/global

# Stats publiques
GET /stats/public
```

## 🎯 Fonctionnalités Clés

### Service RAG
- ✅ **Requêtes intelligentes** avec types spécialisés
- ✅ **Cache optimisé** pour les réponses IA
- ✅ **Templates adaptatifs** selon le contexte
- ✅ **Indexation de documents** avec métadonnées
- ✅ **Intégration Ollama** (LLM local)

### Service Quest
- ✅ **Quêtes interactives** multi-types
- ✅ **Système de progression** avec scoring
- ✅ **Achievements automatiques** 
- ✅ **Leaderboard temps réel**
- ✅ **Analytics détaillées** par utilisateur
- ✅ **Gamification complète** (niveaux, streaks, badges)

### Architecture Partagée
- ✅ **Event Bus** (Redis) pour communication inter-services
- ✅ **Cache intelligent** avec stratégies spécialisées
- ✅ **Configuration centralisée** par environnement
- ✅ **Logging structuré** avec niveaux
- ✅ **Health checks** complets
- ✅ **Middleware sécurisé** (rate limiting, CORS, etc.)

## 🔄 Communication Inter-Services

Les services communiquent via :

1. **Event Bus** (Redis) pour les événements asynchrones
2. **API REST** pour les requêtes synchrones
3. **Cache partagé** (Redis) pour les données communes

### Événements principaux :
- `USER_AUTHENTICATED` → Auth → Autres services
- `QUEST_COMPLETED` → Quest → Analytics
- `RAG_QUERY_PROCESSED` → RAG → Analytics
- `ACHIEVEMENT_UNLOCKED` → Quest → Notification

## 🛠️ Développement

### Ajout d'un nouveau service
1. Créer le dossier `services/nouveau_service/`
2. Implémenter `models.py`, `manager.py`, `routes.py`, `app.py`
3. Utiliser les composants partagés (`services/shared/`)
4. Ajouter au `start_services.py`
5. Créer les tests d'intégration

### Patterns à suivre
- **Config** : Utiliser `ServiceConfig` centralisée
- **Events** : Publier les événements importants
- **Cache** : Utiliser les stratégies spécialisées
- **Errors** : Exceptions personnalisées avec codes d'erreur
- **Logs** : Logger structuré avec contexte

## 🚀 Prochaines Étapes

### Services à créer :
- [ ] **Analytics Service** (port 8003) - Métriques et reporting
- [ ] **Notification Service** (port 8005) - Emails, alerts, webhooks

### Infrastructure :
- [ ] **API Gateway** (Nginx/Kong) - Routage centralisé
- [ ] **Service Discovery** (Consul) - Registration automatique
- [ ] **Monitoring** (Prometheus/Grafana) - Observabilité

### Déploiement :
- [ ] **Dockerisation** - Containers par service
- [ ] **Kubernetes** - Orchestration (optionnel)
- [ ] **CI/CD** - Pipeline automatisé

---

## 🎉 Statut Actuel

✅ **Phase 1 Complétée** : Architecture microservices de base  
🔄 **Phase 2 En cours** : Tests et optimisations  
📋 **Phase 3 Prévue** : Infrastructure et déploiement  

**Score de complétude** : 70% de l'architecture cible

L'architecture est maintenant **opérationnelle** et prête pour le développement ! 🚀
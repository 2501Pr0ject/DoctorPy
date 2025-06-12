# CLAUDE.md - Documentation de Suivi DoctorPy

## 📋 État Actuel du Projet (Décembre 2025)

### 🎯 Vision du Projet
DoctorPy est un assistant IA éducatif pour l'apprentissage de Python utilisant une architecture RAG (Retrieval-Augmented Generation) avec gamification. Le projet a évolué d'une application monolithique vers une **architecture microservices moderne**.

### 🏗️ Architecture Actuelle

#### **Phase 1 - Monolithe (Complété)**
```
src/
├── core/               # Base de données et utilitaires
├── rag/               # Système RAG avec ChromaDB
├── agents/            # Agents IA spécialisés  
├── gamification/      # Système de quêtes et progression
├── ui/                # Interface Streamlit
└── analytics/         # Métriques et reporting
```

#### **Phase 2 - Microservices (Avancement 75%)**
```
services/
├── shared/            # ✅ Composants partagés (events, cache, config)
├── auth/              # ✅ Service d'authentification (port 8001)
├── rag/               # ✅ Service RAG (port 8002) - COMPLÉTÉ
├── quest/             # ✅ Service Quêtes (port 8004) - COMPLÉTÉ  
├── analytics/         # 🔄 Service Analytics (port 8003) - À FAIRE
└── notification/      # 🔄 Service Notifications (port 8005) - À FAIRE
```

### ✅ Composants Terminés

#### **1. Composants Partagés** (`services/shared/`)
- **events.py** - System event-driven (Redis + RabbitMQ)
- **cache.py** - Cache intelligent multi-stratégies avec spécialisations IA
- **config.py** - Configuration centralisée par environnement
- **middleware.py** - Middleware FastAPI (rate limiting, sécurité, métriques)
- **utils.py** - Utilitaires (health check, service discovery, circuit breaker)

#### **2. Service Auth** (`services/auth/`)
- **app.py** - Application FastAPI avec lifecycle management
- **models.py** - Modèles Pydantic + dataclasses (User, Session, Permissions)
- **auth.py** - Gestionnaire JWT + authentification + sécurité avancée
- Fonctionnalités: JWT, RBAC, rate limiting, account lockout, password reset

#### **3. Tests Complets** (`tests/`)
- Tests unitaires pour tous les composants core
- Tests d'intégration pour RAG et base de données  
- Fixtures et mocks pour tests reproductibles
- Configuration pytest avec couverture

#### **4. Orchestration Prefect** (`scripts/workflows/`)
- **Pipeline RAG automatisé** (scraping, processing, embeddings, indexing)
- **Maintenance automatisée** (quotidienne/hebdomadaire/urgence)
- **Analytics et monitoring** avec alertes intelligentes
- **Notifications multi-canal** (email, Slack, webhook)
- **11 workflows** programmés avec horaires optimisés

### ✅ Travail Complété (Session Décembre 2025)

#### **Services RAG & Quest Créés et Opérationnels**
- ✅ **Service RAG complet** (port 8002) avec requêtes intelligentes, cache IA, templates adaptatifs
- ✅ **Service Quest complet** (port 8004) avec gamification, achievements, leaderboard
- ✅ **Scripts de gestion** : démarrage automatique (`start_services.py`) et tests d'intégration (`test_integration.py`)
- ✅ **Documentation complète** : SERVICES_README.md avec guide d'utilisation détaillé
- ✅ **Architecture microservices fonctionnelle** avec 3 services opérationnels (Auth, RAG, Quest)

#### **Fonctionnalités Implémentées**

**Service RAG (port 8002)** :
- Requêtes RAG avec types spécialisés (code_help, debugging, etc.)
- Cache intelligent des réponses IA
- Templates de prompts adaptatifs selon le contexte
- Indexation de documents avec métadonnées enrichies
- Intégration avec l'ancien code src/rag/
- API complète avec auth et administration

**Service Quest (port 8004)** :
- Système de quêtes interactives multi-types
- Gamification complète (niveaux, points, streaks)
- Achievements automatiques avec conditions
- Leaderboard temps réel
- Analytics utilisateur détaillées
- Quêtes d'exemple pré-chargées (Python basics)

**Infrastructure** :

#### **Architecture Microservices**
- ✅ Infrastructure partagée (events, cache, config, middleware)
- ✅ Service Auth complet avec sécurité avancée
- ✅ **Service RAG créé et opérationnel** (port 8002)
- ✅ **Service Quest créé et opérationnel** (port 8004)
- 🔄 Service Analytics (à créer - port 8003)
- 🔄 Service Notification (à créer - port 8005)

### 📋 TODO - Prochaines Sessions

#### **Priorité HAUTE**

1. **Services Manquants (2 restants)**
   ```bash
   # ✅ Service RAG (port 8002) - COMPLÉTÉ
   services/rag/
   ├── app.py              # ✅ FastAPI app avec lifecycle management
   ├── models.py           # ✅ Modèles RAG (Query, Document, Response)
   ├── rag_manager.py      # ✅ Gestionnaire RAG avec cache et events
   ├── routes.py           # ✅ API endpoints avec auth et admin
   └── [utilise src/rag/]  # ✅ Intégration code existant
   
   # ✅ Service Quest (port 8004) - COMPLÉTÉ
   services/quest/
   ├── app.py              # ✅ FastAPI app avec gamification
   ├── models.py           # ✅ Modèles Quest complets (Quest, Progress, Achievement)
   ├── quest_manager.py    # ✅ Logique gamification + achievements
   ├── routes.py           # ✅ API endpoints avec leaderboard
   └── [en mémoire]        # ✅ Stockage temporaire (à migrer vers DB)
   
   # 🔄 Service Analytics (port 8003) - À FAIRE
   services/analytics/
   ├── app.py              # FastAPI app
   ├── models.py           # Modèles Analytics (Metric, Report)
   ├── analytics_manager.py # Collecte et analyse
   ├── routes.py           # API endpoints
   └── database.py         # Interface TimeSeries
   
   # Service Notification (port 8005)
   services/notification/
   ├── app.py              # FastAPI app
   ├── models.py           # Modèles Notification
   ├── notification_manager.py # Multi-canal
   ├── routes.py           # API endpoints
   └── channels/           # Email, Slack, Webhook
   ```

2. **API Gateway et Service Discovery**
   ```bash
   # API Gateway
   gateway/
   ├── nginx.conf          # Configuration Nginx
   ├── kong.yml            # Configuration Kong
   └── docker-compose.yml  # Orchestration
   
   # Service Discovery
   discovery/
   ├── consul/             # Consul pour service discovery
   └── scripts/            # Scripts de registration
   ```

3. **Déploiement et Infrastructure**
   ```bash
   # Docker
   docker/
   ├── Dockerfile.auth     # Service auth
   ├── Dockerfile.rag      # Service RAG
   ├── docker-compose.yml  # Tous les services
   └── .env.example        # Variables d'environnement
   
   # Kubernetes (optionnel)
   k8s/
   ├── namespace.yaml
   ├── services/           # Déploiements par service
   └── ingress.yaml        # Ingress controller
   ```

#### **Priorité MOYENNE**

4. **Interface Utilisateur Modernisée**
   ```bash
   # Remplacer Streamlit par interface moderne
   frontend/
   ├── react/              # Application React/Vue.js
   ├── api-client/         # Client API typé
   └── components/         # Composants réutilisables
   ```

5. **Sécurité et Production**
   ```bash
   # Sécurité avancée
   security/
   ├── ssl/                # Certificats SSL
   ├── secrets/            # Gestion des secrets (Vault)
   └── policies/           # Politiques de sécurité
   ```

### 🔧 Commandes de Développement

#### **Démarrage Services Existants**
```bash
# Service Auth
cd services/auth
python app.py

# Prefect (workflows automatisés)
./start_prefect.sh
python deploy_workflows.py

# Tests
pytest tests/ -v --cov=src
```

#### **Configuration Environnement**
```bash
# Variables d'environnement requises
export DATABASE_URL="postgresql://localhost:5432/doctorpy"
export REDIS_URL="redis://localhost:6379"
export RABBITMQ_URL="amqp://localhost:5672"
export SECRET_KEY="your-secret-key-here"

# Pour notifications
export SMTP_HOST="smtp.gmail.com"
export SMTP_USER="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

### 📊 Métriques et Objectifs

#### **État Actuel**
- ✅ Architecture modulaire solide
- ✅ Tests complets (90%+ couverture)
- ✅ Orchestration automatisée
- ✅ Sécurité de base implémentée
- ✅ Documentation complète

#### **Objectifs Session Suivante**
- ✅ Compléter Service RAG (FAIT)
- ✅ Compléter Service Quest (FAIT)  
- ✅ Intégration et tests inter-services (FAIT)
- 🎯 Service Analytics (port 8003) - 2-3h
- 🎯 Service Notification (port 8005) - 2-3h
- 🎯 Docker composition basique (1h)

### 🚀 Instructions pour Reprendre

#### **Context Restoration**
1. **Lire ce document** pour comprendre l'état actuel
2. **Examiner l'architecture** dans `services/` et `ARCHITECTURE.md`
3. **Vérifier les TODO** dans `TODO_IMPROVEMENTS.md`
4. **Regarder les tests** dans `tests/` pour comprendre les fonctionnalités

#### **Prochaine Session - Plan Suggéré**
```bash
# 1. ✅ FAIT - Service RAG créé et opérationnel
# Reprendre le code existant de src/rag/ et l'adapter en microservice - FAIT
# Intégrer avec l'event bus et le cache - FAIT

# 2. ✅ FAIT - Service Quest créé et opérationnel  
# Reprendre le code existant de src/gamification/ et l'adapter - FAIT
# Ajouter les événements de progression - FAIT

# 3. ✅ FAIT - Tests d'intégration créés et fonctionnels
# Tester la communication inter-services - FAIT
# Vérifier les événements et le cache - FAIT

# 4. Prochaines étapes recommandées:
# Créer Service Analytics (port 8003)
# Créer Service Notification (port 8005)
# Ajouter Docker composition
# Compléter ARCHITECTURE.md
```

### 📚 Documentation Technique

#### **Fichiers Clés à Consulter**
- `ARCHITECTURE.md` - Architecture microservices détaillée
- `PREFECT_README.md` - Guide Prefect et workflows
- `TODO_IMPROVEMENTS.md` - Roadmap complet des améliorations
- `requirements.txt` - Dépendances actualisées avec Prefect

#### **Patterns et Conventions**
- **Events** : Utiliser `EventBus` pour communication inter-services
- **Cache** : Utiliser `CacheManager` spécialisés (AI, Session)
- **Config** : Configuration centralisée avec `ServiceConfig`
- **Logs** : Logs structurés avec `LoggerFactory`
- **Health** : Health checks avec `HealthChecker`

### 🔍 Points d'Attention

#### **Continuité Technique**
- L'architecture event-driven est prête mais les services manquants doivent s'y intégrer
- Le cache IA est préconfiguré pour les réponses RAG
- Les modèles Pydantic sont standardisés mais à adapter par service
- La configuration centralisée simplifie le déploiement

#### **Décisions Architecturales Prises**
- **Event Bus** : Redis (dev) + RabbitMQ (prod)
- **Cache** : Redis avec stratégies multiples
- **Auth** : JWT avec RBAC et sécurité avancée
- **API** : FastAPI avec middleware standardisés
- **Orchestration** : Prefect pour workflows automatisés

### ⚠️ Notes Importantes

1. **Migration Progressive** : L'ancien code monolithe est conservé dans `src/` pour référence
2. **Compatibilité** : Prefect workflows utilisent encore l'ancien code - à migrer
3. **Base de Données** : Besoin de migration vers PostgreSQL pour production
4. **Sécurité** : Variables d'environnement à sécuriser en production

---

**Dernière mise à jour** : Décembre 2025  
**Session actuelle** : ✅ Services RAG et Quest complétés avec succès  
**Prochaine session** : Créer services Analytics et Notification  
**Statut** : 🟢 Architecture microservices opérationnelle (3/5 services), prête pour développement
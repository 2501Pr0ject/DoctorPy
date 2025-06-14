# = DoctorPy - Votre Assistant IA pour Apprendre Python

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![AI](https://img.shields.io/badge/AI-Ollama-orange.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI+Streamlit-red.svg)

DoctorPy est un assistant IA intelligent con�u pour vous accompagner dans votre apprentissage de Python. Utilisant des technologies d'IA avanc�es et un syst�me de qu�tes gamifi�es, DoctorPy offre une exp�rience d'apprentissage personnalis�e et interactive.

## < Caract�ristiques Principales

### > Assistant IA Conversationnel
- **Chat intelligent** avec compr�hension contextuelle
- **Explications personnalis�es** adapt�es � votre niveau
- **Correction de code en temps r�el** avec suggestions d'am�lioration
- **Syst�me RAG** (Retrieval Augmented Generation) pour des r�ponses pr�cises bas�es sur la documentation Python officielle

### <� Apprentissage Gamifi�
- **Qu�tes interactives** structur�es par difficult� (d�butant, interm�diaire, avanc�)
- **Syst�me d'XP et de niveaux** pour suivre vos progr�s
- **Exercices pratiques** avec validation automatique
- **Indices intelligents** pour vous d�bloquer sans donner la solution

### =� Suivi Personnalis�
- **Tableaux de bord** pour visualiser vos progr�s
- **Recommandations personnalis�es** de qu�tes selon votre niveau
- **Analyse de vos points forts** et axes d'am�lioration
- **Statistiques d�taill�es** de votre apprentissage

### =' Technologies de Pointe
- **Ollama** pour l'IA locale (LLama 3.1, CodeLlama)
- **ChromaDB** pour la recherche s�mantique
- **FastAPI** pour l'API backend robuste
- **Streamlit** pour l'interface utilisateur moderne
- **SQLite** pour la persistance des donn�es

## =� Installation Rapide

### Pr�requis
- Python 3.8 ou sup�rieur
- 8 GB de RAM minimum (pour les mod�les IA)
- 10 GB d'espace disque libre

### Installation Automatique

```bash
# 1. Cloner le projet
git clone https://github.com/votre-username/DoctorPy.git
cd DoctorPy

# 2. Lancer l'installation automatique
chmod +x scripts/setup/complete_setup.sh
./scripts/setup/complete_setup.sh
```

Le script d'installation automatique va :
-  Installer toutes les d�pendances Python
-  T�l�charger et configurer Ollama
-  T�l�charger les mod�les IA n�cessaires
-  Initialiser la base de donn�es
-  Cr�er et indexer la base de connaissances
-  Configurer l'environnement

### Installation Manuelle

Si vous pr�f�rez installer manuellement :

```bash
# 1. Cr�er un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 2. Installer les d�pendances
pip install -r requirements.txt

# 3. Installer Ollama
scripts/setup/install_ollama.sh

# 4. T�l�charger les mod�les IA
scripts/setup/download_models.sh

# 5. Configurer l'environnement
cp .env.example .env
# �diter .env selon vos besoins

# 6. Initialiser la base de donn�es
python scripts/setup/setup_database_simple.py

# 7. Construire la base de connaissances
python scripts/data_processing/scrape_docs.py
python scripts/data_processing/process_documents.py
python scripts/data_processing/create_embeddings.py
python scripts/data_processing/index_documents.py
```

## <� D�marrage Rapide

### Lancer l'Interface Web

```bash
# D�marrer l'interface Streamlit
streamlit run ui/streamlit_app.py
```

Ouvrez votre navigateur � `http://localhost:8501`

### Lancer l'API Backend (Optionnel)

```bash
# D�marrer le serveur FastAPI
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API disponible � `http://localhost:8000` avec documentation automatique � `/docs`

## =� Guide d'Utilisation

### 1. Premier Contact - Interface de Chat

![Chat Interface](docs/images/chat_interface.png)

L'interface de chat est votre point d'entr�e principal :

```
=d Vous : Bonjour, je suis d�butant en Python
> DoctorPy : Bonjour ! Parfait pour commencer. Voulez-vous :
   1. Apprendre les bases avec une qu�te guid�e
   2. Poser une question sp�cifique  
   3. Faire r�viser du code que vous avez �crit
```

**Types de questions que vous pouvez poser :**
- `"Comment cr�er une variable en Python ?"`
- `"Explique-moi les boucles for"`
- `"Corrige ce code : print('hello world'"`
- `"Je veux apprendre les fonctions"`

### 2. Syst�me de Qu�tes

#### Navigation des Qu�tes

Acc�dez aux qu�tes via la barre lat�rale ou en demandant :
```
=d "Je veux faire une qu�te sur les variables"
```

#### Structure d'une Qu�te

Chaque qu�te contient :
- **Objectifs d'apprentissage** clairs
- **3-5 �tapes progressives** avec exercices
- **Exemples de code** interactifs
- **Syst�me d'indices** si vous �tes bloqu�
- **Validation automatique** de vos r�ponses

#### Exemple de Qu�te : "Variables Python"

```python
# �tape 1 : Cr�er votre premi�re variable
nom = "Alice"

# �tape 2 : Diff�rents types de donn�es  
age = 25
taille = 1.65
est_etudiant = True

# �tape 3 : Afficher vos variables
print(f"Je suis {nom}, j'ai {age} ans")
```

### 3. R�vision de Code

DoctorPy peut analyser votre code et sugg�rer des am�liorations :

```python
# Votre code
def calcul(x,y):
result=x+y
print(result)
return result

# Suggestions de DoctorPy :
#  Ajoutez une docstring
#  Corrigez l'indentation  
#  Ajoutez des espaces autour des op�rateurs
#  �vitez les print() dans les fonctions de calcul
```

### 4. Suivi des Progr�s

#### Tableau de Bord Personnel

Visualisez vos statistiques :
- **XP Total** : 450 points
- **Niveau Actuel** : 3
- **Qu�tes Compl�t�es** : 5/12
- **Temps d'�tude** : 2h30 cette semaine
- **Streak** : 7 jours cons�cutifs

#### Recommandations Personnalis�es

Bas�es sur vos progr�s :
```
<� Recommand� pour vous :
   " "Les Listes Python" (20 min, 200 XP)
   " "Gestion d'Erreurs" (25 min, 180 XP)
```

## <� Architecture Technique

### Vue d'Ensemble

```
                                                            
   Interface              Backend             Intelligence  
   Streamlit     �  �     FastAPI      �  �     Ollama      
                                                            
                              
                              �
                                                            
   Base de                 Syst�me            Base de       
   Donn�es       �  �       RAG        �  �  Connaissances  
   SQLite                ChromaDB             (Docs Python) 
                                                            
```

### Composants Principaux

#### 1. Interface Utilisateur (`ui/`)
- **Streamlit App** : Interface web moderne avec th�me sombre
- **Composants r�utilisables** : Chat, qu�tes, tableaux de bord
- **CSS personnalis�** : Design responsive et attrayant

#### 2. Backend API (`src/api/`)
- **FastAPI** : API REST haute performance
- **Authentification** : JWT avec gestion des sessions
- **Endpoints** : `/auth`, `/quests`, `/chat`, `/progress`

#### 3. Agents Conversationnels (`src/agents/`)
- **ChatAgent** : Conversations g�n�rales avec RAG
- **QuestAgent** : Gestion des qu�tes et exercices
- **CodeReviewAgent** : Analyse et suggestions de code
- **StateManager** : Gestion des sessions et historique

#### 4. Syst�me RAG (`src/rag/`)
- **DocumentRetriever** : Recherche s�mantique dans la documentation
- **EmbeddingManager** : G�n�ration d'embeddings avec sentence-transformers
- **VectorStore** : Stockage et requ�te avec ChromaDB

#### 5. Gestion des Qu�tes (`src/quests/`)
- **QuestManager** : Chargement et validation des qu�tes
- **ProgressTracker** : Suivi des progr�s utilisateur
- **QuestValidator** : Validation des structures de qu�tes

#### 6. Int�gration LLM (`src/llm/`)
- **OllamaClient** : Interface avec les mod�les Ollama
- **PromptManager** : G�n�ration de prompts optimis�s
- **ResponseParser** : Analyse et formatage des r�ponses

### Base de Donn�es

#### Structure

```sql
-- Utilisateurs et authentification
users (id, username, email, xp_total, level, streak_days)

-- Syst�me de qu�tes  
quests (id, title, difficulty, category, content, xp_reward)
user_progress (user_id, quest_id, status, completion_percentage)

-- Sessions et messages
chat_sessions (id, user_id, mode, created_at, is_active)
messages (id, session_id, role, content, timestamp)

-- Analytics et m�triques
analytics (id, user_id, event_type, event_data, timestamp)
```

### Pipeline de Donn�es

#### 1. Construction de la Base de Connaissances

```bash
# �tape 1 : Scraping de la documentation Python
python scripts/data_processing/scrape_docs.py
# � 46 documents collect�s

# �tape 2 : Traitement et chunking  
python scripts/data_processing/process_documents.py
# � 1,172 chunks cr��s

# �tape 3 : G�n�ration d'embeddings
python scripts/data_processing/create_embeddings.py  
# � Embeddings 384D avec sentence-transformers

# �tape 4 : Indexation ChromaDB
python scripts/data_processing/index_documents.py
# � Base vectorielle pr�te pour recherche s�mantique
```

#### 2. Flux de Conversation

```
Question Utilisateur
        �
Recherche RAG (similarit� cosinus)
        �  
Prompt contextualis� + Historique
        �
G�n�ration Ollama (LLama 3.1)
        �
Parsing et formatage de la r�ponse
        �
Mise � jour BDD + Analytics
        �
Affichage Interface Utilisateur
```

## >� Tests et Qualit�

### Suite de Tests Compl�te

```bash
# Tests unitaires (mocks et composants isol�s)
pytest tests/unit/ -v

# Tests d'int�gration (base de donn�es, API, LLM)  
pytest tests/integration/ -v

# Tests bout-en-bout (workflows complets)
pytest tests/integration/test_end_to_end.py -v

# Couverture de code
pytest --cov=src --cov-report=html
```

### M�triques de Qualit�

- **Couverture de code** : >90%
- **Tests unitaires** : 150+ tests
- **Tests d'int�gration** : 50+ sc�narios
- **Validation continue** : Pre-commit hooks

### Types de Tests

#### Tests Unitaires (`tests/unit/`)
- **Mod�les** : User, Quest, Session, Message
- **Agents** : Chat, Quest, CodeReview  
- **RAG** : Retriever, Embeddings, VectorStore
- **Utilitaires** : Validation, S�curit�, Performance

#### Tests d'Int�gration (`tests/integration/`)
- **Base de donn�es** : CRUD, contraintes, performance
- **API** : Authentification, endpoints, gestion d'erreurs
- **LLM** : Ollama, g�n�ration, streaming
- **Bout-en-bout** : Workflows utilisateur complets

## =� Contenu P�dagogique

### Qu�tes Disponibles

#### =� Niveau D�butant
- **Variables Python** (20 min, 100 XP)
  - Cr�ation et types de variables
  - R�gles de nommage
  - Affichage avec print()

- **Fonctions** (30 min, 150 XP)
  - D�finition avec `def`
  - Param�tres et valeurs de retour
  - Docstrings et bonnes pratiques

#### =� Niveau Interm�diaire  
- **Listes et Collections** (25 min, 200 XP)
  - Cr�ation et manipulation de listes
  - M�thodes de liste (append, remove, etc.)
  - Parcours avec boucles

- **D�bogage** (20 min, 150 XP)
  - Types d'erreurs courantes
  - Lecture des tracebacks
  - Techniques de d�bogage

#### =4 Niveau Avanc�
- **Classes et Objets** (45 min, 300 XP)
  - Programmation orient�e objet
  - Constructeurs et m�thodes
  - H�ritage et polymorphisme

### Base de Connaissances

#### Sources Officielles
- **Documentation Python 3.12** : Tutorial complet
- **PEP (Python Enhancement Proposals)** : Standards et bonnes pratiques
- **Guide de Style PEP 8** : Conventions de codage
- **Biblioth�que Standard** : Modules essentiels

#### Contenu Enrichi
- **1,172 chunks** de documentation index�s
- **Recherche s�mantique** avec score de pertinence
- **Exemples de code** valid�s et test�s
- **Explications contextuelles** adapt�es au niveau

## � Configuration

### Variables d'Environnement

Copiez `.env.example` vers `.env` et configurez :

```bash
# Application
APP_NAME=DoctorPy
ENVIRONMENT=development
SECRET_KEY=your-super-secret-key-change-this-in-production

# Base de donn�es
DATABASE_URL=sqlite:///./data/databases/doctorpy.db

# Ollama (IA locale)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_CODE_MODEL=codellama:7b

# RAG et Embeddings
RAG_MAX_RESULTS=5
RAG_SIMILARITY_THRESHOLD=0.7
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Interface Utilisateur
UI_THEME=dark
UI_LANGUAGE=fr
ENABLE_ANALYTICS=true
```

### Configuration Avanc�e

#### Mod�les Ollama

```bash
# Mod�les recommand�s
ollama pull llama3.1:8b      # Conversations g�n�rales
ollama pull codellama:7b     # Analyse de code
ollama pull mistral:7b       # Alternative l�g�re

# V�rifier les mod�les install�s
ollama list
```

#### Performance ChromaDB

```python
# src/rag/vector_store.py
COLLECTION_CONFIG = {
    "hnsw_space": "cosine",
    "hnsw_construction_ef": 200,
    "hnsw_M": 16
}
```

## =� D�ploiement

### D�ploiement Local

```bash
# Production locale avec Docker
docker-compose up -d

# Ou d�ploiement manuel
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
streamlit run ui/streamlit_app.py --server.port 8501
```

### D�ploiement Cloud

#### Streamlit Cloud
1. Forkez le repository
2. Connectez votre compte Streamlit Cloud
3. D�ployez depuis `ui/streamlit_app.py`

#### Serveur VPS
```bash
# Configuration nginx
server {
    listen 80;
    server_name votre-domaine.com;
    
    location /api/ {
        proxy_pass http://localhost:8000;
    }
    
    location / {
        proxy_pass http://localhost:8501;
    }
}
```

### Monitoring

#### M�triques Applicatives
- Temps de r�ponse Ollama
- Utilisation m�moire  
- Taux d'erreur API
- Sessions utilisateur actives

#### Logs Structur�s
```bash
# Logs applicatifs
tail -f logs/doctorpy.log

# Logs Ollama
tail -f ~/.ollama/logs/server.log
```

## > Contribution

### Pour les D�veloppeurs

#### Setup D�veloppement

```bash
# Installation mode d�veloppement
pip install -e .
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Tests en continu
pytest-watch tests/
```

#### Standards de Code

- **Formatage** : Black, isort
- **Linting** : Flake8, Pylint  
- **Type hints** : mypy
- **Tests** : pytest, coverage >90%

#### Contribution de Contenu

##### Ajouter une Nouvelle Qu�te

```json
{
  "id": "ma_nouvelle_quete",
  "title": "Titre de la Qu�te", 
  "description": "Description claire",
  "difficulty": "beginner|intermediate|advanced",
  "category": "python|debugging|best_practices",
  "estimated_time": 25,
  "xp_reward": 200,
  "prerequisites": ["quete_prerequise"],
  "content": {
    "steps": [
      {
        "title": "Nom de l'�tape",
        "content": "Explication th�orique",
        "exercise": "Instruction de l'exercice", 
        "solution": "Code solution",
        "tips": ["Conseil 1", "Conseil 2"]
      }
    ]
  }
}
```

##### Enrichir la Base de Connaissances

```bash
# Ajouter de nouveaux documents
mkdir data/raw/ma_source/
# Placer les fichiers .md dans le dossier

# Reprocesser la base
python scripts/data_processing/process_documents.py
python scripts/data_processing/create_embeddings.py  
python scripts/data_processing/index_documents.py
```

### Pour les �ducateurs

#### Personnalisation P�dagogique

- **Cr�ez vos propres qu�tes** avec le format JSON
- **Adaptez les niveaux** de difficult�
- **Ajoutez du contenu sp�cialis�** (data science, web, etc.)
- **Configurez les recommandations** selon vos objectifs

#### Analytics �tudiants

```python
# Exporter les progr�s �tudiants
from src.core.database import db_manager

stats = db_manager.get_user_analytics(user_id=123)
progress = db_manager.get_user_progress(user_id=123)
```

## = D�pannage

### Probl�mes Courants

#### Ollama ne d�marre pas

```bash
# V�rifier le service
systemctl status ollama

# Red�marrer  
sudo systemctl restart ollama

# Logs de d�bogage
journalctl -u ollama -f
```

#### Erreurs de m�moire

```bash
# V�rifier l'utilisation m�moire
ps aux | grep ollama
nvidia-smi  # Si GPU

# R�duire la taille du mod�le
ollama pull llama3.1:7b  # Au lieu de 8b
```

#### Base de donn�es corrompue

```bash
# Sauvegarder
cp data/databases/doctorpy.db data/databases/doctorpy.db.backup

# Recr�er
python scripts/setup/setup_database_simple.py
```

#### Performance lente

```bash
# V�rifier les index ChromaDB
python -c "from src.rag.vector_store import VectorStore; vs = VectorStore(); print(vs.get_stats())"

# Re-indexer si n�cessaire  
python scripts/data_processing/index_documents.py --rebuild
```

### Logs de D�bogage

```bash
# Activer le mode debug
export LOG_LEVEL=DEBUG

# Logs d�taill�s
tail -f logs/doctorpy.log | grep ERROR
```

## =� M�triques et Performance

### Benchmarks

#### Temps de R�ponse
- **Chat simple** : <2 secondes
- **Qu�te avec RAG** : <5 secondes  
- **R�vision de code** : <3 secondes
- **Recherche s�mantique** : <500ms

#### Utilisation Ressources
- **RAM** : 4-8 GB (selon mod�le Ollama)
- **CPU** : 2-4 cores (g�n�ration IA)
- **Stockage** : 10 GB (mod�les + donn�es)
- **R�seau** : Minimal (IA locale)

### Scalabilit�

#### Utilisateurs Simultan�s
- **Configuration de base** : 5-10 utilisateurs
- **Serveur d�di�** : 50-100 utilisateurs
- **Cluster distribu�** : 500+ utilisateurs

## = S�curit�

### Mesures Impl�ment�es

#### Authentification
- **JWT tokens** avec expiration
- **Hachage bcrypt** pour les mots de passe
- **Sessions s�curis�es** avec timeout

#### Protection des Donn�es
- **Validation d'entr�es** stricte
- **Sanitisation** des codes utilisateur
- **Isolation** des ex�cutions de code
- **Chiffrement** des donn�es sensibles

#### S�curit� IA
- **Filtrage des prompts** malicieux
- **Limite de tokens** par requ�te
- **Rate limiting** par utilisateur
- **Audit trails** des interactions

## =� Roadmap

### Version 2.0 (Q2 2024)
- <� **Support multi-langages** (JavaScript, Java)
- <� **Mode collaboratif** (qu�tes en �quipe)
- <� **Projets guid�s** (applications compl�tes)
- <� **Int�gration IDE** (VS Code extension)

### Version 2.5 (Q3 2024)  
- <� **IA vocale** (reconnaissance/synth�se)
- <� **R�alit� augment�e** (visualisation code)
- <� **Marketplace** de qu�tes communautaires
- <� **Certification** officielle

### Version 3.0 (Q4 2024)
- <� **Plateforme �ducative** compl�te
- <� **Outils enseignants** avanc�s
- <� **Analytics pr�dictifs** 
- <� **Infrastructure cloud** native

## =� Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de d�tails.

### Utilisation Commerciale

 **Autoris�** : Utilisation, modification, distribution  
 **�ducation** : Libre utilisation en milieu scolaire  
 **Entreprise** : Formation interne des �quipes  
L **Marque** : Pas d'utilisation du nom "DoctorPy" sans permission

## =e Communaut�

### Support et Questions
- =� **Discussions** : [GitHub Discussions](https://github.com/votre-username/DoctorPy/discussions)
- = **Issues** : [GitHub Issues](https://github.com/votre-username/DoctorPy/issues)  
- =� **Email** : support@doctorpy.ai

### R�seaux Sociaux
- =& **Twitter** : [@DoctorPy_AI](https://twitter.com/DoctorPy_AI)
- =� **LinkedIn** : [DoctorPy Official](https://linkedin.com/company/doctorpy)

### Contributeurs

Un grand merci � tous les contributeurs qui rendent ce projet possible ! =O

[![Contributors](https://contrib.rocks/image?repo=votre-username/DoctorPy)](https://github.com/votre-username/DoctorPy/graphs/contributors)

---

<div align="center">

**= Fait avec d pour la communaut� Python =**

[P Star ce repo](https://github.com/votre-username/DoctorPy) " [<t Fork](https://github.com/votre-username/DoctorPy/fork) " [=� Documentation](https://docs.doctorpy.ai)

</div>
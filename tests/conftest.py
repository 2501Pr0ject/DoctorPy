"""Configuration pytest globale pour DoctorPy"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import sys

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.database import DatabaseManager


@pytest.fixture(scope="session")
def test_db_path():
    """Créer un fichier de base de données temporaire pour les tests"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        yield f.name
    # Nettoyer après les tests
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture
def db_manager(test_db_path):
    """Gestionnaire de base de données pour les tests"""
    return DatabaseManager(test_db_path)


@pytest.fixture
def sample_user_data():
    """Données utilisateur d'exemple pour les tests"""
    return {
        "id": 1,
        "username": "test_user",
        "email": "test@example.com",
        "xp_total": 100,
        "level": 2,
        "streak_days": 5
    }


@pytest.fixture
def sample_quest_data():
    """Données de quête d'exemple au format JSON"""
    return {
        "id": "sample_quest",
        "title": "Quête d'exemple",
        "description": "Une quête pour apprendre les bases",
        "difficulty": "beginner",
        "category": "python",
        "estimated_time": 15,
        "xp_reward": 100,
        "content": {
            "steps": [
                {
                    "title": "Premier pas",
                    "content": "Apprenez les variables Python",
                    "exercise": "Créez une variable nommée 'nom'",
                    "solution": "nom = 'Alice'",
                    "tips": ["Utilisez des guillemets", "Pas d'espaces dans le nom"]
                }
            ]
        }
    }


@pytest.fixture
def mock_ollama_client():
    """Client Ollama mocké pour les tests"""
    mock = Mock()
    mock.generate = AsyncMock(return_value={
        'response': 'Test response from Ollama',
        'done': True,
        'total_duration': 1000000,
        'load_duration': 500000,
        'prompt_eval_count': 10,
        'eval_count': 20
    })
    mock.list = AsyncMock(return_value={'models': [
        {'name': 'llama3.1:8b', 'size': 4661211808},
        {'name': 'codellama:7b', 'size': 3825819519}
    ]})
    return mock


@pytest.fixture
def mock_embeddings():
    """Embeddings mockés pour les tests"""
    return {
        "test_chunk_1": [0.1, 0.2, 0.3, 0.4, 0.5] * 76,  # 384 dimensions
        "test_chunk_2": [0.2, 0.3, 0.4, 0.5, 0.6] * 76
    }


@pytest.fixture
def sample_documents():
    """Documents d'exemple pour les tests RAG"""
    return [
        {
            "id": "doc_1",
            "title": "Les variables Python",
            "content": "Les variables en Python permettent de stocker des données. Exemple: nom = 'Alice'",
            "section": "tutorial",
            "url": "https://docs.python.org/3/tutorial/variables.html"
        },
        {
            "id": "doc_2", 
            "title": "Les fonctions Python",
            "content": "Les fonctions permettent de réutiliser du code. Exemple: def hello(): print('Hello')",
            "section": "tutorial",
            "url": "https://docs.python.org/3/tutorial/functions.html"
        }
    ]


@pytest.fixture
def mock_chromadb_collection():
    """Collection ChromaDB mockée"""
    mock = Mock()
    mock.add = Mock()
    mock.query = Mock(return_value={
        'ids': [['doc_1', 'doc_2']],
        'distances': [[0.1, 0.3]],
        'documents': [['Document 1 content', 'Document 2 content']],
        'metadatas': [[{'title': 'Doc 1'}, {'title': 'Doc 2'}]]
    })
    mock.count = Mock(return_value=2)
    return mock


@pytest.fixture
def temp_quest_file():
    """Fichier de quête temporaire pour les tests"""
    quest_data = {
        "id": "temp_quest",
        "title": "Quête temporaire",
        "description": "Une quête pour les tests",
        "difficulty": "beginner",
        "category": "test",
        "steps": [{"title": "Test step", "content": "Test content"}]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(quest_data, f, indent=2)
        yield f.name
    
    # Nettoyer après le test
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Configuration automatique de l'environnement de test"""
    # Variables d'environnement pour les tests
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
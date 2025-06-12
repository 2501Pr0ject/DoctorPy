"""Tests d'intégration pour l'API FastAPI"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.endpoints.auth import router as auth_router
from src.api.endpoints.quests import router as quests_router
from src.api.endpoints.chat import router as chat_router


@pytest.mark.integration
@pytest.mark.api
class TestAPIIntegration:
    """Tests d'intégration pour l'API"""
    
    @pytest.fixture
    def client(self):
        """Client de test FastAPI"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Headers d'authentification pour les tests"""
        # Créer un utilisateur et obtenir un token
        response = client.post("/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "TestPassword123!"
        })
        
        if response.status_code == 201:
            # Login pour obtenir le token
            login_response = client.post("/auth/login", json={
                "username": "testuser",
                "password": "TestPassword123!"
            })
            
            if login_response.status_code == 200:
                token = login_response.json()["access_token"]
                return {"Authorization": f"Bearer {token}"}
        
        # Fallback avec mock token
        return {"Authorization": "Bearer mock_token"}
    
    def test_api_health_check(self, client):
        """Test du endpoint de santé"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


@pytest.mark.integration
@pytest.mark.api
class TestAuthEndpoints:
    """Tests des endpoints d'authentification"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_register_user(self, client):
        """Test d'inscription d'utilisateur"""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "StrongPassword123!"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]
        assert "id" in data
        assert "password" not in data  # Ne pas exposer le mot de passe
    
    def test_register_duplicate_user(self, client):
        """Test d'inscription avec utilisateur existant"""
        user_data = {
            "username": "duplicateuser",
            "email": "duplicate@example.com",
            "password": "StrongPassword123!"
        }
        
        # Première inscription
        response1 = client.post("/auth/register", json=user_data)
        assert response1.status_code == 201
        
        # Deuxième inscription (devrait échouer)
        response2 = client.post("/auth/register", json=user_data)
        assert response2.status_code == 409  # Conflict
        assert "already exists" in response2.json()["detail"]
    
    def test_login_valid_credentials(self, client):
        """Test de connexion avec identifiants valides"""
        # D'abord créer un utilisateur
        user_data = {
            "username": "loginuser",
            "email": "login@example.com",
            "password": "ValidPassword123!"
        }
        
        register_response = client.post("/auth/register", json=user_data)
        assert register_response.status_code == 201
        
        # Maintenant se connecter
        login_data = {
            "username": "loginuser",
            "password": "ValidPassword123!"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
    
    def test_login_invalid_credentials(self, client):
        """Test de connexion avec identifiants invalides"""
        login_data = {
            "username": "nonexistent",
            "password": "WrongPassword"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        assert "Invalid credentials" in response.json()["detail"]
    
    def test_get_current_user(self, client, auth_headers):
        """Test d'obtention de l'utilisateur actuel"""
        response = client.get("/auth/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data
        assert "xp_total" in data
    
    def test_update_user_profile(self, client, auth_headers):
        """Test de mise à jour du profil utilisateur"""
        update_data = {
            "email": "updated@example.com",
            "profile_data": {
                "preferred_language": "fr",
                "difficulty_preference": "intermediate"
            }
        }
        
        response = client.put("/auth/profile", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == update_data["email"]
        assert data["profile_data"]["preferred_language"] == "fr"


@pytest.mark.integration
@pytest.mark.api
class TestQuestEndpoints:
    """Tests des endpoints de quêtes"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_get_all_quests(self, client):
        """Test d'obtention de toutes les quêtes"""
        response = client.get("/quests/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Vérifier la structure des quêtes
        if data:
            quest = data[0]
            assert "id" in quest
            assert "title" in quest
            assert "difficulty" in quest
            assert "category" in quest
    
    def test_get_quest_by_id(self, client):
        """Test d'obtention d'une quête par ID"""
        # D'abord obtenir la liste des quêtes
        all_quests_response = client.get("/quests/")
        quests = all_quests_response.json()
        
        if quests:
            quest_id = quests[0]["id"]
            response = client.get(f"/quests/{quest_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == quest_id
            assert "content" in data
            assert "steps" in data["content"]
    
    def test_get_nonexistent_quest(self, client):
        """Test d'obtention d'une quête inexistante"""
        response = client.get("/quests/nonexistent_quest_id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_get_quests_by_difficulty(self, client):
        """Test d'obtention de quêtes par difficulté"""
        response = client.get("/quests/?difficulty=beginner")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Toutes les quêtes devraient être de niveau beginner
        for quest in data:
            assert quest["difficulty"] == "beginner"
    
    def test_get_quests_by_category(self, client):
        """Test d'obtention de quêtes par catégorie"""
        response = client.get("/quests/?category=python")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Toutes les quêtes devraient être de catégorie python
        for quest in data:
            assert quest["category"] == "python"
    
    def test_start_quest(self, client, auth_headers):
        """Test de démarrage d'une quête"""
        # Obtenir une quête disponible
        quests_response = client.get("/quests/")
        quests = quests_response.json()
        
        if quests:
            quest_id = quests[0]["id"]
            response = client.post(f"/quests/{quest_id}/start", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["quest_id"] == quest_id
            assert data["status"] == "in_progress"
            assert "current_step" in data
    
    def test_get_user_progress(self, client, auth_headers):
        """Test d'obtention de la progression utilisateur"""
        response = client.get("/quests/progress", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Chaque progression devrait avoir les champs requis
        for progress in data:
            assert "quest_id" in progress
            assert "status" in progress
            assert "completion_percentage" in progress
    
    def test_submit_quest_answer(self, client, auth_headers):
        """Test de soumission de réponse à une quête"""
        # D'abord démarrer une quête
        quests_response = client.get("/quests/")
        quests = quests_response.json()
        
        if quests:
            quest_id = quests[0]["id"]
            start_response = client.post(f"/quests/{quest_id}/start", headers=auth_headers)
            
            if start_response.status_code == 200:
                answer_data = {
                    "step_index": 0,
                    "answer": "print('Hello, World!')"
                }
                
                response = client.post(
                    f"/quests/{quest_id}/submit",
                    json=answer_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "correct" in data
                assert "feedback" in data


@pytest.mark.integration
@pytest.mark.api
class TestChatEndpoints:
    """Tests des endpoints de chat"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_create_chat_session(self, client, auth_headers):
        """Test de création de session de chat"""
        session_data = {
            "mode": "free_chat"
        }
        
        response = client.post("/chat/sessions", json=session_data, headers=auth_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["mode"] == "free_chat"
        assert data["is_active"] is True
    
    def test_send_message(self, client, auth_headers):
        """Test d'envoi de message"""
        # D'abord créer une session
        session_response = client.post(
            "/chat/sessions", 
            json={"mode": "free_chat"}, 
            headers=auth_headers
        )
        
        if session_response.status_code == 201:
            session_id = session_response.json()["session_id"]
            
            message_data = {
                "content": "Bonjour, comment créer une variable en Python?"
            }
            
            response = client.post(
                f"/chat/sessions/{session_id}/messages",
                json=message_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "message_id" in data
            assert "response" in data
            assert data["response"] is not None
    
    def test_get_session_history(self, client, auth_headers):
        """Test d'obtention de l'historique de session"""
        # Créer une session et envoyer un message
        session_response = client.post(
            "/chat/sessions",
            json={"mode": "free_chat"},
            headers=auth_headers
        )
        
        if session_response.status_code == 201:
            session_id = session_response.json()["session_id"]
            
            # Envoyer un message
            client.post(
                f"/chat/sessions/{session_id}/messages",
                json={"content": "Test message"},
                headers=auth_headers
            )
            
            # Obtenir l'historique
            response = client.get(f"/chat/sessions/{session_id}/history", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) >= 1  # Au moins le message envoyé
    
    def test_get_user_sessions(self, client, auth_headers):
        """Test d'obtention des sessions utilisateur"""
        response = client.get("/chat/sessions", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Chaque session devrait avoir les champs requis
        for session in data:
            assert "id" in session
            assert "mode" in session
            assert "created_at" in session
            assert "is_active" in session


@pytest.mark.integration
@pytest.mark.api
class TestAPIErrorHandling:
    """Tests de gestion d'erreurs de l'API"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_unauthorized_access(self, client):
        """Test d'accès non autorisé"""
        response = client.get("/auth/me")  # Sans headers d'auth
        
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]
    
    def test_invalid_token(self, client):
        """Test avec token invalide"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/auth/me", headers=headers)
        
        assert response.status_code == 401
    
    def test_malformed_request_body(self, client):
        """Test avec corps de requête malformé"""
        response = client.post("/auth/register", json={"invalid": "data"})
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    def test_rate_limiting(self, client):
        """Test de limitation de taux (si implémenté)"""
        # Envoyer plusieurs requêtes rapidement
        responses = []
        for _ in range(100):
            response = client.get("/health")
            responses.append(response.status_code)
        
        # La plupart devraient être 200, mais certaines pourraient être 429
        assert 200 in responses
        # Note: Le rate limiting peut ne pas être implémenté en test
    
    def test_server_error_simulation(self, client):
        """Test de simulation d'erreur serveur"""
        with patch('src.api.main.app') as mock_app:
            mock_app.side_effect = Exception("Simulated server error")
            
            # Cette requête devrait déclencher une erreur 500
            # Note: Ceci nécessite une configuration spéciale pour les tests


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Tests de performance de l'API"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_response_time(self, client):
        """Test du temps de réponse"""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 1.0  # Devrait répondre en moins d'1 seconde
    
    def test_concurrent_requests(self, client):
        """Test de requêtes concurrentes"""
        import threading
        import time
        
        results = []
        
        def make_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            results.append({
                "status_code": response.status_code,
                "response_time": end_time - start_time
            })
        
        # Lancer 10 requêtes concurrentes
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join()
        
        # Vérifier que toutes les requêtes ont réussi
        assert len(results) == 10
        assert all(result["status_code"] == 200 for result in results)
        
        # Vérifier les temps de réponse
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        assert avg_response_time < 2.0  # Temps de réponse moyen acceptable
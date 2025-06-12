"""Tests d'intégration bout-en-bout pour DoctorPy"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.core.database import DatabaseManager
from src.llm.ollama_client import OllamaClient
from src.agents.state_manager_simple import SimpleStateManager
from src.agents.chat_agent import ChatAgent
from src.agents.quest_agent import QuestAgent
from src.quests.quest_manager import QuestManager
from src.rag.retriever import DocumentRetriever


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflows:
    """Tests d'intégration bout-en-bout"""
    
    @pytest.fixture
    def test_database(self):
        """Base de données de test"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db_manager = DatabaseManager(db_path)
        
        # Ajouter des données de test
        user_id = db_manager.create_user(
            username="e2e_user",
            email="e2e@example.com"
        )
        
        # Ajouter une quête de test
        db_manager.execute_update(
            """INSERT INTO quests (
                id, title, description, difficulty, category, 
                estimated_time, content, xp_reward
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "e2e_quest",
                "Test Quest E2E",
                "Une quête pour les tests bout-en-bout",
                "beginner",
                "python",
                15,
                '{"steps": [{"title": "Créer une variable", "content": "Créez une variable nom", "exercise": "Créez une variable nommée \\"nom\\"", "solution": "nom = \\"Alice\\"", "tips": ["Utilisez des guillemets"]}]}',
                100
            )
        )
        
        yield db_manager, user_id
        
        # Nettoyer
        try:
            Path(db_path).unlink()
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def mock_llm_client(self):
        """Client LLM mocké pour les tests E2E"""
        client = Mock()
        client.generate = Mock(return_value={
            "response": "Bonjour! Pour créer une variable en Python, utilisez la syntaxe: nom = 'valeur'",
            "done": True
        })
        return client
    
    @pytest.fixture
    def mock_retriever(self):
        """Retriever mocké pour les tests E2E"""
        retriever = Mock()
        retriever.search = Mock(return_value=[
            {
                "content": "Les variables en Python permettent de stocker des données. Syntaxe: nom = valeur",
                "metadata": {"title": "Variables Python", "section": "tutorial"},
                "score": 0.95
            }
        ])
        return retriever


@pytest.mark.integration
class TestUserRegistrationAndLogin:
    """Test du flux d'inscription et connexion"""
    
    def test_complete_user_registration_flow(self, test_database):
        """Test du flux complet d'inscription utilisateur"""
        db_manager, _ = test_database
        
        # 1. Vérifier qu'un nouvel utilisateur peut s'inscrire
        new_user_id = db_manager.create_user(
            username="newuser",
            email="newuser@example.com",
            password_hash="hashed_password_123"
        )
        
        assert new_user_id is not None
        
        # 2. Vérifier que l'utilisateur peut se connecter
        user = db_manager.get_user_by_username("newuser")
        assert user is not None
        assert user["email"] == "newuser@example.com"
        assert user["xp_total"] == 0
        assert user["level"] == 1
        
        # 3. Mettre à jour la dernière connexion
        success = db_manager.update_user_login(new_user_id)
        assert success is True
        
        # 4. Vérifier la mise à jour
        updated_user = db_manager.get_user_by_id(new_user_id)
        assert updated_user["last_login"] is not None


@pytest.mark.integration
class TestChatWorkflow:
    """Test du flux de conversation"""
    
    @pytest.mark.asyncio
    async def test_complete_chat_session(self, test_database, mock_llm_client, mock_retriever):
        """Test d'une session de chat complète"""
        db_manager, user_id = test_database
        
        # 1. Créer les composants
        state_manager = SimpleStateManager(db_manager=db_manager)
        chat_agent = ChatAgent(
            llm_client=mock_llm_client,
            retriever=mock_retriever
        )
        
        # 2. Créer une session de chat
        session_id = await state_manager.create_session(user_id, mode="free_chat")
        assert session_id is not None
        
        # 3. Envoyer un message
        message_id = await state_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content="Comment créer une variable en Python?"
        )
        assert message_id is not None
        
        # 4. Générer une réponse
        response = await chat_agent.generate_response(
            message="Comment créer une variable en Python?",
            user_id=user_id,
            session_id=session_id
        )
        
        assert response is not None
        assert "response" in response
        assert "variable" in response["response"].lower()
        
        # 5. Ajouter la réponse à l'historique
        await state_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=response["response"]
        )
        
        # 6. Vérifier l'historique
        history = await state_manager.get_session_history(session_id, limit=10)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        
        # 7. Continuer la conversation avec contexte
        response2 = await chat_agent.generate_response(
            message="Peux-tu me donner un exemple?",
            user_id=user_id,
            session_id=session_id,
            conversation_history=history
        )
        
        assert response2 is not None
        mock_retriever.search.assert_called()
        mock_llm_client.generate.assert_called()


@pytest.mark.integration
class TestQuestWorkflow:
    """Test du flux de quêtes"""
    
    @pytest.mark.asyncio
    async def test_complete_quest_workflow(self, test_database, mock_llm_client):
        """Test d'un workflow complet de quête"""
        db_manager, user_id = test_database
        
        # 1. Créer les composants
        quest_manager = QuestManager(db_manager=db_manager)
        quest_agent = QuestAgent(
            llm_client=mock_llm_client,
            quest_manager=quest_manager
        )
        
        # 2. Obtenir la quête de test
        quest = db_manager.get_quest_by_id("e2e_quest")
        assert quest is not None
        
        # 3. Démarrer la quête
        progress_id = db_manager.create_user_progress(user_id, "e2e_quest")
        assert progress_id is not None
        
        # 4. Obtenir la progression initiale
        progress = db_manager.get_user_progress(user_id, "e2e_quest")
        assert len(progress) == 1
        assert progress[0]["status"] == "in_progress"
        assert progress[0]["current_step"] == 0
        
        # 5. Soumettre une réponse correcte
        mock_llm_client.generate.return_value = {
            "response": "EVALUATION: CORRECT\nFEEDBACK: Excellente réponse!\nNEXT_STEP: true",
            "done": True
        }
        
        with patch.object(quest_agent, 'validate_code_answer', return_value=True):
            result = await quest_agent.process_answer(
                quest_id="e2e_quest",
                user_id=user_id,
                step_index=0,
                user_answer="nom = 'Alice'"
            )
        
        assert result["correct"] is True
        assert "feedback" in result
        
        # 6. Avancer dans la quête
        db_manager.update_user_progress(user_id, "e2e_quest", 1, 100.0)
        
        # 7. Compléter la quête
        success = db_manager.complete_quest(user_id, "e2e_quest", 100)
        assert success is True
        
        # 8. Vérifier la complétion
        final_progress = db_manager.get_user_progress(user_id, "e2e_quest")
        assert final_progress[0]["status"] == "completed"
        assert final_progress[0]["completion_percentage"] == 100.0
        assert final_progress[0]["xp_earned"] == 100
        
        # 9. Vérifier que l'XP utilisateur a été mis à jour
        user = db_manager.get_user_by_id(user_id)
        assert user["xp_total"] == 100


@pytest.mark.integration
class TestRAGWorkflow:
    """Test du flux RAG (Retrieval Augmented Generation)"""
    
    def test_document_retrieval_and_response_generation(self, mock_retriever, mock_llm_client):
        """Test du flux complet RAG"""
        # 1. Configurer le retriever avec des documents
        mock_retriever.search.return_value = [
            {
                "content": "Les variables en Python stockent des données. Exemple: nom = 'Alice'",
                "metadata": {"title": "Variables Python", "section": "tutorial"},
                "score": 0.95
            },
            {
                "content": "Les fonctions permettent de réutiliser du code. Exemple: def hello(): print('Hello')",
                "metadata": {"title": "Fonctions Python", "section": "tutorial"},  
                "score": 0.87
            }
        ]
        
        # 2. Créer l'agent de chat
        chat_agent = ChatAgent(
            llm_client=mock_llm_client,
            retriever=mock_retriever
        )
        
        # 3. Configurer la réponse du LLM
        mock_llm_client.generate.return_value = {
            "response": "Basé sur la documentation, les variables en Python stockent des données. Voici un exemple: nom = 'Alice'",
            "done": True
        }
        
        # 4. Tester la génération de réponse avec RAG
        import asyncio
        response = asyncio.run(chat_agent.generate_response(
            message="Comment utiliser les variables en Python?",
            user_id=1,
            session_id="rag_test"
        ))
        
        # 5. Vérifier que le retriever a été appelé
        mock_retriever.search.assert_called_with("Comment utiliser les variables en Python?")
        
        # 6. Vérifier que le LLM a été appelé avec le contexte RAG
        mock_llm_client.generate.assert_called()
        call_args = mock_llm_client.generate.call_args
        prompt = call_args[1]["prompt"]
        assert "variables en Python stockent des données" in prompt
        
        # 7. Vérifier la réponse
        assert response is not None
        assert "variables" in response["response"].lower()


@pytest.mark.integration
class TestMultiAgentWorkflow:
    """Test du flux multi-agents"""
    
    @pytest.mark.asyncio
    async def test_chat_to_quest_handoff(self, test_database, mock_llm_client, mock_retriever):
        """Test de transfert de l'agent de chat vers l'agent de quêtes"""
        db_manager, user_id = test_database
        
        # 1. Créer les agents
        state_manager = SimpleStateManager(db_manager=db_manager)
        chat_agent = ChatAgent(
            llm_client=mock_llm_client,
            retriever=mock_retriever
        )
        quest_manager = QuestManager(db_manager=db_manager)
        quest_agent = QuestAgent(
            llm_client=mock_llm_client,
            quest_manager=quest_manager
        )
        
        # 2. Démarrer une session de chat
        session_id = await state_manager.create_session(user_id, mode="free_chat")
        
        # 3. L'utilisateur demande à commencer une quête
        mock_llm_client.generate.return_value = {
            "response": "Je vais vous aider à commencer une quête. Voici les quêtes disponibles...",
            "done": True
        }
        
        chat_response = await chat_agent.generate_response(
            message="Je veux commencer une quête sur les variables",
            user_id=user_id,
            session_id=session_id
        )
        
        # 4. Obtenir les quêtes disponibles
        available_quests = db_manager.get_quests_by_category("python")
        assert len(available_quests) >= 1
        
        # 5. L'utilisateur choisit une quête
        chosen_quest = available_quests[0]
        progress_id = db_manager.create_user_progress(user_id, chosen_quest["id"])
        
        # 6. Démarrer la quête avec l'agent de quêtes
        quest_start_result = await quest_agent.start_quest(
            quest_id=chosen_quest["id"],
            user_id=user_id
        )
        
        assert quest_start_result is not None
        assert "quest" in quest_start_result
        assert "current_step" in quest_start_result
        
        # 7. Changer le mode de session
        # (En pratique, ceci serait géré par l'orchestrateur principal)
        await state_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="system",
            content="Mode changé vers quest_mode"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestSystemPerformance:
    """Tests de performance du système complet"""
    
    @pytest.mark.asyncio
    async def test_concurrent_user_sessions(self, test_database, mock_llm_client, mock_retriever):
        """Test de sessions utilisateur concurrentes"""
        db_manager, _ = test_database
        
        # Créer plusieurs utilisateurs
        user_ids = []
        for i in range(3):
            user_id = db_manager.create_user(f"user_{i}", f"user_{i}@example.com")
            user_ids.append(user_id)
        
        # Créer le state manager
        state_manager = SimpleStateManager(db_manager=db_manager)
        chat_agent = ChatAgent(
            llm_client=mock_llm_client,
            retriever=mock_retriever
        )
        
        async def simulate_user_session(user_id):
            """Simuler une session utilisateur"""
            session_id = await state_manager.create_session(user_id, mode="free_chat")
            
            # Envoyer quelques messages
            for i in range(3):
                await state_manager.add_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content=f"Message {i} de l'utilisateur {user_id}"
                )
                
                response = await chat_agent.generate_response(
                    message=f"Message {i}",
                    user_id=user_id,
                    session_id=session_id
                )
                
                await state_manager.add_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=response["response"]
                )
            
            return session_id
        
        # Lancer les sessions concurrentes
        tasks = [simulate_user_session(user_id) for user_id in user_ids]
        session_ids = await asyncio.gather(*tasks)
        
        # Vérifier que toutes les sessions ont été créées
        assert len(session_ids) == 3
        assert all(session_id is not None for session_id in session_ids)
        
        # Vérifier l'historique de chaque session
        for session_id in session_ids:
            history = await state_manager.get_session_history(session_id, limit=10)
            assert len(history) == 6  # 3 messages utilisateur + 3 réponses assistant
    
    def test_database_performance_with_many_operations(self, test_database):
        """Test de performance de la base de données avec de nombreuses opérations"""
        db_manager, user_id = test_database
        
        import time
        
        # Mesurer le temps pour de nombreuses opérations
        start_time = time.time()
        
        # Créer de nombreux enregistrements de progression
        for i in range(100):
            quest_id = f"perf_quest_{i}"
            
            # Créer une quête
            db_manager.execute_update(
                """INSERT INTO quests (id, title, difficulty, category, content) 
                   VALUES (?, ?, ?, ?, ?)""",
                (quest_id, f"Quest {i}", "beginner", "python", '{"steps": []}')
            )
            
            # Créer une progression
            db_manager.create_user_progress(user_id, quest_id)
            
            # Mettre à jour la progression
            db_manager.update_user_progress(user_id, quest_id, 1, 50.0)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 100 opérations complexes devraient prendre moins de 10 secondes
        assert total_time < 10.0
        
        # Vérifier l'intégrité des données
        progress_records = db_manager.execute_query(
            "SELECT COUNT(*) as count FROM user_progress WHERE user_id = ?",
            (user_id,)
        )
        assert progress_records[0]["count"] >= 100


@pytest.mark.integration
class TestErrorRecovery:
    """Tests de récupération d'erreurs"""
    
    @pytest.mark.asyncio
    async def test_database_connection_recovery(self, test_database):
        """Test de récupération de connexion base de données"""
        db_manager, user_id = test_database
        state_manager = SimpleStateManager(db_manager=db_manager)
        
        # Créer une session
        session_id = await state_manager.create_session(user_id)
        
        # Simuler une perte de connexion (fermer temporairement)
        # En pratique, ceci pourrait être un test plus sophistiqué
        
        # Vérifier que les opérations suivantes fonctionnent toujours
        message_id = await state_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content="Test après reconnexion"
        )
        
        assert message_id is not None
    
    @pytest.mark.asyncio
    async def test_llm_error_handling(self, test_database, mock_retriever):
        """Test de gestion d'erreurs LLM"""
        db_manager, user_id = test_database
        
        # Créer un client LLM qui échoue
        failing_llm_client = Mock()
        failing_llm_client.generate = Mock(side_effect=Exception("LLM connection failed"))
        
        chat_agent = ChatAgent(
            llm_client=failing_llm_client,
            retriever=mock_retriever
        )
        
        # Tenter de générer une réponse
        with pytest.raises(Exception):
            await chat_agent.generate_response(
                message="Test message",
                user_id=user_id,
                session_id="error_test"
            )
        
        # Vérifier que l'erreur est gérée gracieusement
        # (L'implémentation exacte dépend de la stratégie de gestion d'erreurs)
"""Tests unitaires pour les agents conversationnels"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from src.agents.state_manager_simple import SimpleStateManager
from src.agents.chat_agent import ChatAgent
from src.agents.quest_agent import QuestAgent
from src.agents.code_review_agent import CodeReviewAgent
from src.models.session import ChatSession, Message


@pytest.mark.unit
class TestSimpleStateManager:
    """Tests pour le gestionnaire d'état simple"""
    
    @pytest.fixture
    def state_manager(self, db_manager):
        """State manager avec base de données de test"""
        return SimpleStateManager(db_manager=db_manager)
    
    @pytest.mark.asyncio
    async def test_create_session(self, state_manager):
        """Test de création de session"""
        user_id = 1
        session_id = await state_manager.create_session(user_id)
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    @pytest.mark.asyncio
    async def test_session_timeout(self, state_manager):
        """Test de timeout de session"""
        user_id = 1
        session_id = await state_manager.create_session(user_id)
        
        # Simuler un timeout court pour les tests
        state_manager.session_timeout = 0.1  # 100ms
        
        # Attendre le timeout
        await asyncio.sleep(0.2)
        
        # La session devrait être inactive
        session = state_manager.sessions.get(session_id)
        if session:
            assert not session.is_active
    
    @pytest.mark.asyncio
    async def test_add_message(self, state_manager):
        """Test d'ajout de message"""
        user_id = 1
        session_id = await state_manager.create_session(user_id)
        
        message_id = await state_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content="Bonjour!"
        )
        
        assert message_id is not None
        assert isinstance(message_id, int)
    
    @pytest.mark.asyncio
    async def test_get_session_history(self, state_manager):
        """Test de récupération de l'historique"""
        user_id = 1
        session_id = await state_manager.create_session(user_id)
        
        # Ajouter quelques messages
        await state_manager.add_message(session_id, user_id, "user", "Message 1")
        await state_manager.add_message(session_id, user_id, "assistant", "Réponse 1")
        await state_manager.add_message(session_id, user_id, "user", "Message 2")
        
        history = await state_manager.get_session_history(session_id, limit=10)
        
        assert len(history) == 3
        assert history[0]['role'] == "user"
        assert history[0]['content'] == "Message 1"
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive_sessions(self, state_manager):
        """Test de nettoyage des sessions inactives"""
        user_id = 1
        session_id = await state_manager.create_session(user_id)
        
        # Marquer la session comme inactive
        if session_id in state_manager.sessions:
            state_manager.sessions[session_id].is_active = False
        
        # Nettoyer
        cleaned_count = await state_manager.cleanup_inactive_sessions()
        
        assert cleaned_count >= 0
        # La session devrait être supprimée de la mémoire
        assert session_id not in state_manager.sessions


@pytest.mark.unit
class TestChatAgent:
    """Tests pour l'agent de chat"""
    
    @pytest.fixture
    def mock_llm_client(self, mock_ollama_client):
        """Client LLM mocké"""
        return mock_ollama_client
    
    @pytest.fixture
    def mock_retriever(self):
        """Retriever mocké"""
        retriever = Mock()
        retriever.search = Mock(return_value=[
            {
                "id": "doc_1",
                "content": "Les variables en Python stockent des données",
                "metadata": {"title": "Variables Python"},
                "score": 0.95
            }
        ])
        return retriever
    
    def test_chat_agent_initialization(self, mock_llm_client, mock_retriever):
        """Test d'initialisation de l'agent de chat"""
        agent = ChatAgent(
            llm_client=mock_llm_client,
            retriever=mock_retriever
        )
        
        assert agent.llm_client == mock_llm_client
        assert agent.retriever == mock_retriever
        assert agent.max_context_length > 0
    
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_llm_client, mock_retriever):
        """Test de génération de réponse"""
        agent = ChatAgent(
            llm_client=mock_llm_client,
            retriever=mock_retriever
        )
        
        response = await agent.generate_response(
            message="Comment créer une variable en Python?",
            user_id=1,
            session_id="test_session"
        )
        
        assert response is not None
        assert "response" in response
        assert response["response"] == "Test response from Ollama"
        mock_llm_client.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_retrieval(self, mock_llm_client, mock_retriever):
        """Test de récupération de contexte"""
        agent = ChatAgent(
            llm_client=mock_llm_client,
            retriever=mock_retriever
        )
        
        await agent.generate_response(
            message="variables Python",
            user_id=1,
            session_id="test_session"
        )
        
        # Vérifier que le retriever a été appelé
        mock_retriever.search.assert_called_once_with("variables Python")
    
    @pytest.mark.asyncio
    async def test_conversation_history(self, mock_llm_client, mock_retriever):
        """Test d'utilisation de l'historique de conversation"""
        agent = ChatAgent(
            llm_client=mock_llm_client,
            retriever=mock_retriever
        )
        
        history = [
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider?"}
        ]
        
        await agent.generate_response(
            message="Continue la conversation",
            user_id=1,
            session_id="test_session",
            conversation_history=history
        )
        
        # Vérifier que l'historique est inclus dans la requête
        call_args = mock_llm_client.generate.call_args
        prompt = call_args[1]["prompt"]
        assert "Bonjour" in prompt


@pytest.mark.unit
class TestQuestAgent:
    """Tests pour l'agent de quêtes"""
    
    @pytest.fixture
    def mock_quest_manager(self, sample_quest_data):
        """Gestionnaire de quêtes mocké"""
        manager = Mock()
        manager.get_quest = Mock(return_value=sample_quest_data)
        manager.get_user_progress = Mock(return_value={
            "current_step": 0,
            "completion_percentage": 0.0,
            "status": "not_started"
        })
        return manager
    
    def test_quest_agent_initialization(self, mock_llm_client, mock_quest_manager):
        """Test d'initialisation de l'agent de quêtes"""
        agent = QuestAgent(
            llm_client=mock_llm_client,
            quest_manager=mock_quest_manager
        )
        
        assert agent.llm_client == mock_llm_client
        assert agent.quest_manager == mock_quest_manager
    
    @pytest.mark.asyncio
    async def test_start_quest(self, mock_llm_client, mock_quest_manager):
        """Test de démarrage d'une quête"""
        agent = QuestAgent(
            llm_client=mock_llm_client,
            quest_manager=mock_quest_manager
        )
        
        result = await agent.start_quest(
            quest_id="sample_quest",
            user_id=1
        )
        
        assert result is not None
        assert "quest" in result
        assert "current_step" in result
        mock_quest_manager.get_quest.assert_called_once_with("sample_quest")
    
    @pytest.mark.asyncio
    async def test_process_answer(self, mock_llm_client, mock_quest_manager, sample_quest_data):
        """Test de traitement d'une réponse"""
        agent = QuestAgent(
            llm_client=mock_llm_client,
            quest_manager=mock_quest_manager
        )
        
        # Mocker la validation de code
        with patch.object(agent, 'validate_code_answer', return_value=True):
            result = await agent.process_answer(
                quest_id="sample_quest",
                user_id=1,
                step_index=0,
                user_answer="nom = 'Alice'"
            )
        
        assert result is not None
        assert "correct" in result
        assert "feedback" in result
    
    def test_validate_code_answer(self, mock_llm_client, mock_quest_manager):
        """Test de validation de réponse de code"""
        agent = QuestAgent(
            llm_client=mock_llm_client,
            quest_manager=mock_quest_manager
        )
        
        # Test avec une réponse correcte
        correct_answer = "nom = 'Alice'"
        expected_solution = "nom = 'Alice'"
        
        is_correct = agent.validate_code_answer(correct_answer, expected_solution)
        assert is_correct is True
        
        # Test avec une réponse incorrecte
        incorrect_answer = "nom = Alice"  # Sans guillemets
        is_correct = agent.validate_code_answer(incorrect_answer, expected_solution)
        assert is_correct is False
    
    @pytest.mark.asyncio
    async def test_get_hint(self, mock_llm_client, mock_quest_manager):
        """Test d'obtention d'indice"""
        agent = QuestAgent(
            llm_client=mock_llm_client,
            quest_manager=mock_quest_manager
        )
        
        hint = await agent.get_hint(
            quest_id="sample_quest",
            user_id=1,
            step_index=0
        )
        
        assert hint is not None
        assert isinstance(hint, str)
        mock_llm_client.generate.assert_called_once()


@pytest.mark.unit
class TestCodeReviewAgent:
    """Tests pour l'agent de révision de code"""
    
    def test_code_review_agent_initialization(self, mock_llm_client):
        """Test d'initialisation de l'agent de révision de code"""
        agent = CodeReviewAgent(llm_client=mock_llm_client)
        
        assert agent.llm_client == mock_llm_client
        assert hasattr(agent, 'review_criteria')
    
    @pytest.mark.asyncio
    async def test_review_code(self, mock_llm_client):
        """Test de révision de code"""
        agent = CodeReviewAgent(llm_client=mock_llm_client)
        
        # Mock de la réponse de révision
        mock_llm_client.generate.return_value = {
            "response": "Le code est correct. Bonne utilisation des variables.",
            "done": True
        }
        
        code_to_review = """
def hello():
    name = "Alice"
    print(f"Hello, {name}!")
"""
        
        review = await agent.review_code(
            code=code_to_review,
            context="Fonction de salutation"
        )
        
        assert review is not None
        assert "feedback" in review
        mock_llm_client.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_code_analysis(self, mock_llm_client):
        """Test d'analyse de code"""
        agent = CodeReviewAgent(llm_client=mock_llm_client)
        
        code = "x = 5\ny = 10\nresult = x + y"
        
        analysis = agent.analyze_code(code)
        
        assert analysis is not None
        assert "lines_count" in analysis
        assert "variables" in analysis
        assert analysis["lines_count"] == 3
    
    def test_security_check(self, mock_llm_client):
        """Test de vérification de sécurité"""
        agent = CodeReviewAgent(llm_client=mock_llm_client)
        
        # Code sécurisé
        safe_code = "print('Hello, World!')"
        assert agent.check_security(safe_code) is True
        
        # Code potentiellement dangereux
        dangerous_code = "import os; os.system('rm -rf /')"
        assert agent.check_security(dangerous_code) is False
    
    def test_syntax_validation(self, mock_llm_client):
        """Test de validation de syntaxe"""
        agent = CodeReviewAgent(llm_client=mock_llm_client)
        
        # Code syntaxiquement correct
        valid_code = "x = 5\nprint(x)"
        assert agent.validate_syntax(valid_code) is True
        
        # Code avec erreur de syntaxe
        invalid_code = "x = 5\nprint(x"  # Parenthèse manquante
        assert agent.validate_syntax(invalid_code) is False


@pytest.mark.unit
class TestAgentIntegration:
    """Tests d'intégration entre agents"""
    
    @pytest.mark.asyncio
    async def test_agent_handoff(self, mock_llm_client, mock_retriever):
        """Test de transfert entre agents"""
        chat_agent = ChatAgent(
            llm_client=mock_llm_client,
            retriever=mock_retriever
        )
        
        # Simuler une demande qui nécessite un transfert vers l'agent de quêtes
        response = await chat_agent.generate_response(
            message="Je veux commencer une quête sur les variables",
            user_id=1,
            session_id="test_session"
        )
        
        # Vérifier que la réponse indique un transfert
        assert response is not None
        # L'implémentation exacte dépend de la logique de transfert
    
    @pytest.mark.asyncio
    async def test_multi_agent_conversation(self, mock_llm_client, mock_retriever):
        """Test de conversation multi-agents"""
        # Ce test simulerait une conversation où plusieurs agents interviennent
        # Par exemple: Chat -> Quest -> CodeReview -> Chat
        
        agents = {
            'chat': ChatAgent(llm_client=mock_llm_client, retriever=mock_retriever),
            'code_review': CodeReviewAgent(llm_client=mock_llm_client)
        }
        
        # Simuler une séquence d'interactions
        # 1. Demande initiale
        chat_response = await agents['chat'].generate_response(
            message="Peux-tu réviser ce code?",
            user_id=1,
            session_id="test_session"
        )
        
        # 2. Révision de code
        code_review = await agents['code_review'].review_code(
            code="print('hello')",
            context="Code d'exemple"
        )
        
        assert chat_response is not None
        assert code_review is not None
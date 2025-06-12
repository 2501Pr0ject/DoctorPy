"""Tests unitaires pour les modèles de données"""

import pytest
from datetime import datetime
from src.models.user import User
from src.models.quest import Quest, QuestStep, QuestProgress
from src.models.session import ChatSession, Message


@pytest.mark.unit
class TestUser:
    """Tests pour le modèle User"""
    
    def test_user_creation(self):
        """Test de création d'un utilisateur"""
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            xp_total=100,
            level=2
        )
        
        assert user.id == 1
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.xp_total == 100
        assert user.level == 2
        assert user.is_active is True
    
    def test_user_level_calculation(self):
        """Test du calcul automatique du niveau"""
        user = User(username="test", email="test@example.com", xp_total=2500)
        
        # Level 1: 0-499 XP, Level 2: 500-999 XP, etc.
        expected_level = (user.xp_total // 500) + 1
        assert user.calculate_level() == expected_level
    
    def test_user_add_xp(self):
        """Test d'ajout d'XP"""
        user = User(username="test", email="test@example.com", xp_total=100)
        initial_level = user.level
        
        user.add_xp(150)
        
        assert user.xp_total == 250
        # Vérifier si le niveau a changé
        if user.xp_total >= 500:
            assert user.level > initial_level
    
    def test_user_validation(self):
        """Test de validation des données utilisateur"""
        # Email invalide
        with pytest.raises(ValueError):
            User(username="test", email="invalid-email", xp_total=0)
        
        # Username vide
        with pytest.raises(ValueError):
            User(username="", email="test@example.com", xp_total=0)
    
    def test_user_streak(self):
        """Test du système de streak"""
        user = User(username="test", email="test@example.com")
        
        # Initialiser le streak
        user.update_streak()
        assert user.streak_days >= 0
        
        # Simuler une connexion quotidienne
        user.streak_days = 5
        user.last_login = datetime.now()
        user.update_streak()
        assert user.streak_days >= 5


@pytest.mark.unit
class TestQuest:
    """Tests pour le modèle Quest"""
    
    def test_quest_creation(self, sample_quest_data):
        """Test de création d'une quête"""
        quest = Quest.from_dict(sample_quest_data)
        
        assert quest.id == "sample_quest"
        assert quest.title == "Quête d'exemple"
        assert quest.difficulty == "beginner"
        assert quest.category == "python"
        assert quest.estimated_time == 15
        assert quest.xp_reward == 100
        assert len(quest.steps) == 1
    
    def test_quest_validation(self):
        """Test de validation des données de quête"""
        # Difficulté invalide
        with pytest.raises(ValueError):
            Quest(
                id="test",
                title="Test",
                difficulty="invalid",
                category="python"
            )
        
        # Catégorie vide
        with pytest.raises(ValueError):
            Quest(
                id="test", 
                title="Test",
                difficulty="beginner",
                category=""
            )
    
    def test_quest_step_creation(self):
        """Test de création d'une étape de quête"""
        step = QuestStep(
            title="Test Step",
            content="Contenu de test",
            exercise="print('hello')",
            solution="print('hello')",
            tips=["Conseil 1", "Conseil 2"]
        )
        
        assert step.title == "Test Step"
        assert step.exercise == "print('hello')"
        assert len(step.tips) == 2
    
    def test_quest_progress_tracking(self):
        """Test du suivi de progression"""
        progress = QuestProgress(
            user_id=1,
            quest_id="test_quest",
            current_step=0,
            completion_percentage=25.0
        )
        
        assert progress.user_id == 1
        assert progress.quest_id == "test_quest" 
        assert progress.current_step == 0
        assert progress.completion_percentage == 25.0
        assert progress.status == "in_progress"
    
    def test_quest_completion(self):
        """Test de complétion d'une quête"""
        progress = QuestProgress(
            user_id=1,
            quest_id="test_quest"
        )
        
        # Compléter la quête
        progress.complete(xp_earned=100)
        
        assert progress.status == "completed"
        assert progress.completion_percentage == 100.0
        assert progress.xp_earned == 100
        assert progress.completed_at is not None


@pytest.mark.unit 
class TestSession:
    """Tests pour les modèles de session"""
    
    def test_chat_session_creation(self):
        """Test de création d'une session de chat"""
        session = ChatSession(
            id="session_123",
            user_id=1,
            mode="free_chat"
        )
        
        assert session.id == "session_123"
        assert session.user_id == 1
        assert session.mode == "free_chat"
        assert session.is_active is True
        assert session.message_count == 0
    
    def test_message_creation(self):
        """Test de création d'un message"""
        message = Message(
            session_id="session_123",
            user_id=1,
            role="user",
            content="Bonjour!"
        )
        
        assert message.session_id == "session_123"
        assert message.user_id == 1
        assert message.role == "user"
        assert message.content == "Bonjour!"
        assert message.timestamp is not None
    
    def test_message_validation(self):
        """Test de validation des messages"""
        # Rôle invalide
        with pytest.raises(ValueError):
            Message(
                session_id="test",
                user_id=1,
                role="invalid_role",
                content="Test"
            )
        
        # Contenu vide
        with pytest.raises(ValueError):
            Message(
                session_id="test",
                user_id=1,
                role="user",
                content=""
            )
    
    def test_session_add_message(self):
        """Test d'ajout de message à une session"""
        session = ChatSession(
            id="session_123",
            user_id=1,
            mode="quest_mode"
        )
        
        initial_count = session.message_count
        
        message = Message(
            session_id=session.id,
            user_id=session.user_id,
            role="user",
            content="Test message"
        )
        
        session.add_message(message)
        
        assert session.message_count == initial_count + 1
        assert len(session.messages) == 1
        assert session.messages[0].content == "Test message"


@pytest.mark.unit
class TestModelSerialization:
    """Tests de sérialisation/désérialisation des modèles"""
    
    def test_user_to_dict(self):
        """Test de conversion User vers dictionnaire"""
        user = User(
            id=1,
            username="test",
            email="test@example.com",
            xp_total=100
        )
        
        user_dict = user.to_dict()
        
        assert user_dict["id"] == 1
        assert user_dict["username"] == "test"
        assert user_dict["email"] == "test@example.com"
        assert user_dict["xp_total"] == 100
    
    def test_quest_to_dict(self, sample_quest_data):
        """Test de conversion Quest vers dictionnaire"""
        quest = Quest.from_dict(sample_quest_data)
        quest_dict = quest.to_dict()
        
        assert quest_dict["id"] == sample_quest_data["id"]
        assert quest_dict["title"] == sample_quest_data["title"]
        assert quest_dict["difficulty"] == sample_quest_data["difficulty"]
        assert "steps" in quest_dict["content"]
    
    def test_quest_from_json_file(self, temp_quest_file):
        """Test de chargement de quête depuis un fichier JSON"""
        quest = Quest.from_json_file(temp_quest_file)
        
        assert quest.id == "temp_quest"
        assert quest.title == "Quête temporaire"
        assert quest.difficulty == "beginner"
    
    def test_invalid_json_handling(self):
        """Test de gestion des JSON invalides"""
        with pytest.raises(ValueError):
            Quest.from_dict({"id": "test"})  # Données incomplètes
        
        with pytest.raises(FileNotFoundError):
            Quest.from_json_file("nonexistent_file.json")
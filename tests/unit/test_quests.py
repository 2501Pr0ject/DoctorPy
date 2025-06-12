"""Tests unitaires pour le système de quêtes"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.quests.quest_manager import QuestManager
from src.quests.quest_loader import QuestLoader
from src.quests.progress_tracker import ProgressTracker
from src.quests.validator import QuestValidator
from src.models.quest import Quest, QuestProgress


@pytest.mark.unit
class TestQuestLoader:
    """Tests pour le chargeur de quêtes"""
    
    def test_load_quest_from_file(self, temp_quest_file):
        """Test de chargement d'une quête depuis un fichier"""
        loader = QuestLoader()
        quest = loader.load_quest(temp_quest_file)
        
        assert quest is not None
        assert quest.id == "temp_quest"
        assert quest.title == "Quête temporaire"
        assert quest.difficulty == "beginner"
    
    def test_load_quest_from_dict(self, sample_quest_data):
        """Test de chargement d'une quête depuis un dictionnaire"""
        loader = QuestLoader()
        quest = loader.load_quest_from_dict(sample_quest_data)
        
        assert quest.id == sample_quest_data["id"]
        assert quest.title == sample_quest_data["title"]
        assert len(quest.steps) == len(sample_quest_data["content"]["steps"])
    
    def test_load_quests_from_directory(self, temp_quest_file):
        """Test de chargement de quêtes depuis un répertoire"""
        loader = QuestLoader()
        quest_dir = Path(temp_quest_file).parent
        
        quests = loader.load_quests_from_directory(str(quest_dir))
        
        assert len(quests) >= 1
        assert any(quest.id == "temp_quest" for quest in quests)
    
    def test_invalid_quest_file(self):
        """Test de gestion de fichier de quête invalide"""
        loader = QuestLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_quest("nonexistent_file.json")
    
    def test_malformed_quest_data(self):
        """Test de gestion de données malformées"""
        loader = QuestLoader()
        
        invalid_data = {"id": "test"}  # Données incomplètes
        
        with pytest.raises(ValueError):
            loader.load_quest_from_dict(invalid_data)


@pytest.mark.unit
class TestQuestValidator:
    """Tests pour le validateur de quêtes"""
    
    def test_validate_valid_quest(self, sample_quest_data):
        """Test de validation d'une quête valide"""
        validator = QuestValidator()
        
        is_valid, errors = validator.validate_quest(sample_quest_data)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_missing_required_fields(self):
        """Test de validation avec champs requis manquants"""
        validator = QuestValidator()
        
        invalid_quest = {
            "id": "test_quest"
            # Champs manquants: title, difficulty, category, content
        }
        
        is_valid, errors = validator.validate_quest(invalid_quest)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("title" in error for error in errors)
        assert any("difficulty" in error for error in errors)
    
    def test_validate_invalid_difficulty(self):
        """Test de validation avec difficulté invalide"""
        validator = QuestValidator()
        
        invalid_quest = {
            "id": "test_quest",
            "title": "Test Quest",
            "difficulty": "expert",  # Invalide
            "category": "python",
            "content": {"steps": []}
        }
        
        is_valid, errors = validator.validate_quest(invalid_quest)
        
        assert is_valid is False
        assert any("difficulty" in error for error in errors)
    
    def test_validate_quest_steps(self):
        """Test de validation des étapes de quête"""
        validator = QuestValidator()
        
        quest_with_invalid_steps = {
            "id": "test_quest",
            "title": "Test Quest",
            "difficulty": "beginner",
            "category": "python",
            "content": {
                "steps": [
                    {
                        "title": "Step 1"
                        # Champs manquants: content, exercise, solution
                    }
                ]
            }
        }
        
        is_valid, errors = validator.validate_quest(quest_with_invalid_steps)
        
        assert is_valid is False
        assert any("step" in error.lower() for error in errors)
    
    def test_validate_xp_reward(self):
        """Test de validation de la récompense XP"""
        validator = QuestValidator()
        
        quest_with_negative_xp = {
            "id": "test_quest",
            "title": "Test Quest",
            "difficulty": "beginner",
            "category": "python",
            "xp_reward": -50,  # Invalide
            "content": {"steps": [{"title": "Step", "content": "Content", "exercise": "Ex", "solution": "Sol"}]}
        }
        
        is_valid, errors = validator.validate_quest(quest_with_negative_xp)
        
        assert is_valid is False
        assert any("xp" in error.lower() for error in errors)


@pytest.mark.unit
class TestProgressTracker:
    """Tests pour le suivi de progression"""
    
    @pytest.fixture
    def progress_tracker(self, db_manager):
        """Tracker de progression avec base de données de test"""
        return ProgressTracker(db_manager=db_manager)
    
    def test_start_quest_progress(self, progress_tracker):
        """Test de démarrage de progression de quête"""
        user_id = 1
        quest_id = "test_quest"
        
        progress = progress_tracker.start_quest(user_id, quest_id)
        
        assert progress is not None
        assert progress.user_id == user_id
        assert progress.quest_id == quest_id
        assert progress.status == "in_progress"
        assert progress.current_step == 0
    
    def test_update_progress(self, progress_tracker):
        """Test de mise à jour de progression"""
        user_id = 1
        quest_id = "test_quest"
        
        # Démarrer la progression
        progress_tracker.start_quest(user_id, quest_id)
        
        # Mettre à jour
        updated = progress_tracker.update_progress(
            user_id=user_id,
            quest_id=quest_id,
            current_step=1,
            completion_percentage=50.0
        )
        
        assert updated is True
        
        # Vérifier la mise à jour
        progress = progress_tracker.get_progress(user_id, quest_id)
        assert progress.current_step == 1
        assert progress.completion_percentage == 50.0
    
    def test_complete_quest(self, progress_tracker):
        """Test de complétion de quête"""
        user_id = 1
        quest_id = "test_quest"
        xp_earned = 100
        
        # Démarrer la progression
        progress_tracker.start_quest(user_id, quest_id)
        
        # Compléter la quête
        completed = progress_tracker.complete_quest(
            user_id=user_id,
            quest_id=quest_id,
            xp_earned=xp_earned
        )
        
        assert completed is True
        
        # Vérifier la complétion
        progress = progress_tracker.get_progress(user_id, quest_id)
        assert progress.status == "completed"
        assert progress.completion_percentage == 100.0
        assert progress.xp_earned == xp_earned
    
    def test_get_user_progress_summary(self, progress_tracker):
        """Test d'obtention du résumé de progression"""
        user_id = 1
        
        # Créer plusieurs progressions
        progress_tracker.start_quest(user_id, "quest_1")
        progress_tracker.start_quest(user_id, "quest_2")
        progress_tracker.complete_quest(user_id, "quest_1", 100)
        
        summary = progress_tracker.get_user_progress_summary(user_id)
        
        assert summary is not None
        assert "total_quests" in summary
        assert "completed_quests" in summary
        assert "total_xp" in summary
        assert summary["total_quests"] >= 2
        assert summary["completed_quests"] >= 1
    
    def test_calculate_completion_percentage(self, progress_tracker):
        """Test de calcul du pourcentage de complétion"""
        total_steps = 5
        current_step = 2
        
        percentage = progress_tracker.calculate_completion_percentage(
            current_step, total_steps
        )
        
        assert percentage == 40.0  # 2/5 * 100
    
    def test_get_next_quest_recommendations(self, progress_tracker):
        """Test de recommandations de prochaines quêtes"""
        user_id = 1
        
        # Compléter une quête de base
        progress_tracker.start_quest(user_id, "python_basics")
        progress_tracker.complete_quest(user_id, "python_basics", 100)
        
        # Mocker le gestionnaire de quêtes pour les recommandations
        with patch.object(progress_tracker, 'quest_manager') as mock_quest_manager:
            mock_quest_manager.get_recommended_quests.return_value = [
                {"id": "python_variables", "difficulty": "beginner"},
                {"id": "python_functions", "difficulty": "beginner"}
            ]
            
            recommendations = progress_tracker.get_next_quest_recommendations(user_id)
            
            assert len(recommendations) > 0
            assert all("id" in quest for quest in recommendations)


@pytest.mark.unit
class TestQuestManager:
    """Tests pour le gestionnaire de quêtes"""
    
    @pytest.fixture
    def quest_manager(self, db_manager):
        """Gestionnaire de quêtes avec base de données de test"""
        return QuestManager(db_manager=db_manager)
    
    def test_quest_manager_initialization(self, quest_manager):
        """Test d'initialisation du gestionnaire de quêtes"""
        assert quest_manager.db_manager is not None
        assert hasattr(quest_manager, 'quest_loader')
        assert hasattr(quest_manager, 'validator')
    
    def test_load_quest(self, quest_manager, sample_quest_data):
        """Test de chargement d'une quête"""
        with patch.object(quest_manager.quest_loader, 'load_quest_from_dict', return_value=Quest.from_dict(sample_quest_data)):
            quest = quest_manager.get_quest("sample_quest")
            
            assert quest is not None
            assert quest.id == "sample_quest"
    
    def test_get_quests_by_difficulty(self, quest_manager):
        """Test d'obtention de quêtes par difficulté"""
        quests = quest_manager.get_quests_by_difficulty("beginner")
        
        assert isinstance(quests, list)
        # Les quêtes retournées devraient toutes être de niveau beginner
        assert all(quest.get("difficulty") == "beginner" for quest in quests)
    
    def test_get_quests_by_category(self, quest_manager):
        """Test d'obtention de quêtes par catégorie"""
        quests = quest_manager.get_quests_by_category("python")
        
        assert isinstance(quests, list)
        # Les quêtes retournées devraient toutes être de catégorie python
        assert all(quest.get("category") == "python" for quest in quests)
    
    def test_validate_quest_data(self, quest_manager, sample_quest_data):
        """Test de validation de données de quête"""
        is_valid = quest_manager.validate_quest(sample_quest_data)
        
        assert is_valid is True
    
    def test_get_recommended_quests(self, quest_manager):
        """Test d'obtention de quêtes recommandées"""
        user_id = 1
        
        # Mocker les données de progression
        with patch.object(quest_manager, 'get_user_progress', return_value=[]):
            recommendations = quest_manager.get_recommended_quests(user_id)
            
            assert isinstance(recommendations, list)
            # Pour un nouvel utilisateur, devrait recommander des quêtes débutant
            if recommendations:
                assert all(quest.get("difficulty") == "beginner" for quest in recommendations[:3])


@pytest.mark.unit
class TestQuestIntegration:
    """Tests d'intégration du système de quêtes"""
    
    def test_full_quest_lifecycle(self, db_manager, sample_quest_data):
        """Test du cycle de vie complet d'une quête"""
        # Créer les composants
        quest_manager = QuestManager(db_manager=db_manager)
        progress_tracker = ProgressTracker(db_manager=db_manager)
        
        user_id = 1
        quest_id = sample_quest_data["id"]
        
        # 1. Charger la quête
        with patch.object(quest_manager.quest_loader, 'load_quest_from_dict', return_value=Quest.from_dict(sample_quest_data)):
            quest = quest_manager.get_quest(quest_id)
            assert quest is not None
        
        # 2. Démarrer la progression
        progress = progress_tracker.start_quest(user_id, quest_id)
        assert progress.status == "in_progress"
        
        # 3. Progresser dans les étapes
        progress_tracker.update_progress(user_id, quest_id, 1, 100.0)
        
        # 4. Compléter la quête
        completed = progress_tracker.complete_quest(user_id, quest_id, 100)
        assert completed is True
        
        # 5. Vérifier la complétion
        final_progress = progress_tracker.get_progress(user_id, quest_id)
        assert final_progress.status == "completed"
    
    def test_quest_prerequisites(self, db_manager):
        """Test de gestion des prérequis de quêtes"""
        quest_manager = QuestManager(db_manager=db_manager)
        progress_tracker = ProgressTracker(db_manager=db_manager)
        
        user_id = 1
        
        # Créer une quête avec prérequis
        advanced_quest = {
            "id": "advanced_quest",
            "title": "Quête avancée",
            "difficulty": "intermediate",
            "category": "python",
            "prerequisites": ["basic_quest"],
            "content": {"steps": [{"title": "Step", "content": "Content", "exercise": "Ex", "solution": "Sol"}]}
        }
        
        # Vérifier que l'utilisateur ne peut pas commencer sans prérequis
        can_start = quest_manager.can_user_start_quest(user_id, "advanced_quest")
        assert can_start is False
        
        # Compléter le prérequis
        progress_tracker.start_quest(user_id, "basic_quest")
        progress_tracker.complete_quest(user_id, "basic_quest", 50)
        
        # Maintenant l'utilisateur devrait pouvoir commencer
        can_start = quest_manager.can_user_start_quest(user_id, "advanced_quest")
        assert can_start is True
    
    def test_quest_statistics(self, db_manager):
        """Test des statistiques de quêtes"""
        quest_manager = QuestManager(db_manager=db_manager)
        progress_tracker = ProgressTracker(db_manager=db_manager)
        
        user_id = 1
        
        # Créer quelques progressions
        progress_tracker.start_quest(user_id, "quest_1")
        progress_tracker.start_quest(user_id, "quest_2")
        progress_tracker.complete_quest(user_id, "quest_1", 100)
        
        # Obtenir les statistiques
        stats = quest_manager.get_quest_statistics()
        
        assert "total_quests" in stats
        assert "total_completions" in stats
        assert "average_completion_rate" in stats
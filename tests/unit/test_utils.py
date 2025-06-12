"""Tests unitaires pour les utilitaires"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from src.utils.file_manager import FileManager
from src.utils.text_processor import TextProcessor
from src.utils.validators import EmailValidator, PasswordValidator, CodeValidator
from src.utils.logger_utils import LoggerUtils
from src.utils.security import SecurityUtils
from src.utils.performance import PerformanceMonitor


@pytest.mark.unit
class TestFileManager:
    """Tests pour le gestionnaire de fichiers"""
    
    def test_read_json_file(self):
        """Test de lecture de fichier JSON"""
        test_data = {"key": "value", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            manager = FileManager()
            data = manager.read_json(temp_path)
            
            assert data == test_data
            assert data["key"] == "value"
            assert data["number"] == 42
        finally:
            Path(temp_path).unlink()
    
    def test_write_json_file(self):
        """Test d'écriture de fichier JSON"""
        test_data = {"test": True, "items": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = FileManager()
            manager.write_json(temp_path, test_data)
            
            # Vérifier que le fichier a été écrit correctement
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == test_data
        finally:
            Path(temp_path).unlink()
    
    def test_file_exists(self):
        """Test de vérification d'existence de fichier"""
        manager = FileManager()
        
        # Fichier existant
        with tempfile.NamedTemporaryFile() as f:
            assert manager.file_exists(f.name) is True
        
        # Fichier inexistant
        assert manager.file_exists("/nonexistent/path/file.txt") is False
    
    def test_create_directory(self):
        """Test de création de répertoire"""
        manager = FileManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "test_directory"
            
            result = manager.create_directory(str(new_dir))
            
            assert result is True
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_get_file_size(self):
        """Test d'obtention de taille de fichier"""
        manager = FileManager()
        test_content = "Hello, World!" * 100
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            size = manager.get_file_size(temp_path)
            assert size > 0
            assert size == len(test_content.encode('utf-8'))
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_json_handling(self):
        """Test de gestion de JSON invalide"""
        manager = FileManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                manager.read_json(temp_path)
        finally:
            Path(temp_path).unlink()


@pytest.mark.unit
class TestTextProcessor:
    """Tests pour le processeur de texte"""
    
    def test_clean_text(self):
        """Test de nettoyage de texte"""
        processor = TextProcessor()
        
        dirty_text = "  Hello,   World!  \n\n  How are you?  \t\t"
        cleaned = processor.clean_text(dirty_text)
        
        assert cleaned == "Hello, World! How are you?"
        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")
    
    def test_remove_html_tags(self):
        """Test de suppression des balises HTML"""
        processor = TextProcessor()
        
        html_text = "<p>Hello <strong>World</strong>!</p><br><a href='#'>Link</a>"
        cleaned = processor.remove_html_tags(html_text)
        
        assert cleaned == "Hello World! Link"
        assert "<" not in cleaned
        assert ">" not in cleaned
    
    def test_extract_code_blocks(self):
        """Test d'extraction de blocs de code"""
        processor = TextProcessor()
        
        text_with_code = """
        Voici du code Python:
        ```python
        def hello():
            print("Hello!")
        ```
        Et du code JavaScript:
        ```javascript
        console.log("Hello!");
        ```
        """
        
        code_blocks = processor.extract_code_blocks(text_with_code)
        
        assert len(code_blocks) == 2
        assert code_blocks[0]["language"] == "python"
        assert "def hello():" in code_blocks[0]["code"]
        assert code_blocks[1]["language"] == "javascript"
        assert "console.log" in code_blocks[1]["code"]
    
    def test_truncate_text(self):
        """Test de troncature de texte"""
        processor = TextProcessor()
        
        long_text = "This is a very long text that should be truncated."
        truncated = processor.truncate_text(long_text, max_length=20)
        
        assert len(truncated) <= 23  # 20 + "..."
        assert truncated.endswith("...")
    
    def test_word_count(self):
        """Test de comptage de mots"""
        processor = TextProcessor()
        
        text = "Hello world, this is a test sentence."
        count = processor.count_words(text)
        
        assert count == 8
    
    def test_extract_keywords(self):
        """Test d'extraction de mots-clés"""
        processor = TextProcessor()
        
        text = "Python programming language variables functions classes objects"
        keywords = processor.extract_keywords(text, min_length=3)
        
        assert "Python" in keywords
        assert "programming" in keywords
        assert "variables" in keywords
        assert len(keywords) > 0


@pytest.mark.unit
class TestValidators:
    """Tests pour les validateurs"""
    
    def test_email_validator(self):
        """Test du validateur d'email"""
        validator = EmailValidator()
        
        # Emails valides
        assert validator.is_valid("user@example.com") is True
        assert validator.is_valid("test.email+tag@domain.co.uk") is True
        assert validator.is_valid("user123@test-domain.org") is True
        
        # Emails invalides
        assert validator.is_valid("invalid-email") is False
        assert validator.is_valid("@domain.com") is False
        assert validator.is_valid("user@") is False
        assert validator.is_valid("") is False
    
    def test_password_validator(self):
        """Test du validateur de mot de passe"""
        validator = PasswordValidator()
        
        # Mots de passe valides
        assert validator.is_valid("StrongPassword123!") is True
        assert validator.is_valid("MyPass123$") is True
        
        # Mots de passe invalides
        assert validator.is_valid("weak") is False  # Trop court
        assert validator.is_valid("nouppercaseordigit") is False  # Pas de majuscule/chiffre
        assert validator.is_valid("NOLOWERCASE123") is False  # Pas de minuscule
        assert validator.is_valid("NoDigitsHere!") is False  # Pas de chiffre
    
    def test_password_strength(self):
        """Test d'évaluation de force de mot de passe"""
        validator = PasswordValidator()
        
        weak_password = "123456"
        medium_password = "MyPassword"
        strong_password = "MyStr0ngP@ssw0rd!"
        
        assert validator.get_strength(weak_password) == "weak"
        assert validator.get_strength(medium_password) == "medium"
        assert validator.get_strength(strong_password) == "strong"
    
    def test_code_validator(self):
        """Test du validateur de code"""
        validator = CodeValidator()
        
        # Code Python valide
        valid_python = """
def hello():
    print("Hello, World!")
    return True
"""
        assert validator.is_valid_python(valid_python) is True
        
        # Code Python invalide
        invalid_python = """
def hello(:
    print("Hello, World!"
    return True
"""
        assert validator.is_valid_python(invalid_python) is False
    
    def test_dangerous_code_detection(self):
        """Test de détection de code dangereux"""
        validator = CodeValidator()
        
        # Code sûr
        safe_code = "print('Hello, World!')"
        assert validator.is_safe(safe_code) is True
        
        # Code dangereux
        dangerous_code = "import os; os.system('rm -rf /')"
        assert validator.is_safe(dangerous_code) is False
        
        dangerous_code2 = "exec('malicious code')"
        assert validator.is_safe(dangerous_code2) is False


@pytest.mark.unit
class TestLoggerUtils:
    """Tests pour les utilitaires de logging"""
    
    def test_logger_creation(self):
        """Test de création de logger"""
        logger = LoggerUtils.get_logger("test_logger")
        
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_log_performance(self):
        """Test de logging de performance"""
        with patch('src.utils.logger_utils.logging') as mock_logging:
            mock_logger = Mock()
            mock_logging.getLogger.return_value = mock_logger
            
            @LoggerUtils.log_performance
            def test_function():
                return "result"
            
            result = test_function()
            
            assert result == "result"
            mock_logger.info.assert_called()
    
    def test_log_exception(self):
        """Test de logging d'exception"""
        with patch('src.utils.logger_utils.logging') as mock_logging:
            mock_logger = Mock()
            mock_logging.getLogger.return_value = mock_logger
            
            @LoggerUtils.log_exceptions
            def failing_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                failing_function()
            
            mock_logger.error.assert_called()


@pytest.mark.unit
class TestSecurityUtils:
    """Tests pour les utilitaires de sécurité"""
    
    def test_hash_password(self):
        """Test de hachage de mot de passe"""
        password = "MySecretPassword123!"
        hashed = SecurityUtils.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert isinstance(hashed, str)
    
    def test_verify_password(self):
        """Test de vérification de mot de passe"""
        password = "MySecretPassword123!"
        hashed = SecurityUtils.hash_password(password)
        
        # Mot de passe correct
        assert SecurityUtils.verify_password(password, hashed) is True
        
        # Mot de passe incorrect
        assert SecurityUtils.verify_password("WrongPassword", hashed) is False
    
    def test_generate_token(self):
        """Test de génération de token"""
        token = SecurityUtils.generate_token()
        
        assert len(token) > 0
        assert isinstance(token, str)
        
        # Chaque token devrait être unique
        token2 = SecurityUtils.generate_token()
        assert token != token2
    
    def test_sanitize_input(self):
        """Test de sanitisation d'entrée"""
        malicious_input = "<script>alert('xss')</script>Hello"
        sanitized = SecurityUtils.sanitize_input(malicious_input)
        
        assert "<script>" not in sanitized
        assert "Hello" in sanitized
    
    def test_validate_session_token(self):
        """Test de validation de token de session"""
        # Token valide (format UUID)
        valid_token = "123e4567-e89b-12d3-a456-426614174000"
        assert SecurityUtils.is_valid_session_token(valid_token) is True
        
        # Token invalide
        invalid_token = "invalid-token-format"
        assert SecurityUtils.is_valid_session_token(invalid_token) is False


@pytest.mark.unit
class TestPerformanceMonitor:
    """Tests pour le moniteur de performance"""
    
    def test_timing_context_manager(self):
        """Test du gestionnaire de contexte de timing"""
        monitor = PerformanceMonitor()
        
        with monitor.timer("test_operation") as timer:
            # Simuler une opération
            import time
            time.sleep(0.01)  # 10ms
        
        assert timer.elapsed_time > 0
        assert timer.elapsed_time >= 0.01
    
    def test_memory_usage(self):
        """Test de mesure d'utilisation mémoire"""
        monitor = PerformanceMonitor()
        
        initial_memory = monitor.get_memory_usage()
        
        # Créer des objets pour augmenter l'utilisation mémoire
        large_list = [i for i in range(10000)]
        
        current_memory = monitor.get_memory_usage()
        
        assert current_memory >= initial_memory
        assert isinstance(current_memory, (int, float))
    
    def test_profile_function(self):
        """Test de profilage de fonction"""
        monitor = PerformanceMonitor()
        
        @monitor.profile
        def test_function():
            total = 0
            for i in range(1000):
                total += i
            return total
        
        result = test_function()
        
        assert result == sum(range(1000))
        # Vérifier que les métriques ont été collectées
        assert hasattr(monitor, 'metrics')
    
    def test_get_stats(self):
        """Test d'obtention des statistiques"""
        monitor = PerformanceMonitor()
        
        # Effectuer quelques opérations
        with monitor.timer("operation1"):
            pass
        
        with monitor.timer("operation2"):
            pass
        
        stats = monitor.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_operations" in stats
        assert "average_time" in stats


@pytest.mark.unit
class TestUtilsIntegration:
    """Tests d'intégration des utilitaires"""
    
    def test_file_and_text_processing(self):
        """Test d'intégration fichier et traitement de texte"""
        file_manager = FileManager()
        text_processor = TextProcessor()
        
        # Créer un fichier avec du contenu texte
        test_content = """
        <p>Hello  World!</p>
        
        This is a test with   extra spaces.
        
        ```python
        def test():
            pass
        ```
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Lire le fichier
            content = file_manager.read_text(temp_path)
            
            # Traiter le texte
            cleaned = text_processor.clean_text(content)
            no_html = text_processor.remove_html_tags(cleaned)
            code_blocks = text_processor.extract_code_blocks(content)
            
            assert "Hello World!" in no_html
            assert len(code_blocks) == 1
            assert code_blocks[0]["language"] == "python"
        finally:
            Path(temp_path).unlink()
    
    def test_security_and_validation(self):
        """Test d'intégration sécurité et validation"""
        email_validator = EmailValidator()
        password_validator = PasswordValidator()
        security_utils = SecurityUtils()
        
        # Scénario complet de validation et sécurisation
        email = "user@example.com"
        password = "MyStr0ngP@ssw0rd!"
        
        # Valider l'email
        assert email_validator.is_valid(email) is True
        
        # Valider le mot de passe
        assert password_validator.is_valid(password) is True
        assert password_validator.get_strength(password) == "strong"
        
        # Hacher le mot de passe
        hashed = security_utils.hash_password(password)
        assert security_utils.verify_password(password, hashed) is True
        
        # Générer un token de session
        token = security_utils.generate_token()
        assert security_utils.is_valid_session_token(token) is True
"""Exceptions personnalisées pour l'application"""


class PythonLearningAssistantError(Exception):
    """Exception de base pour l'application"""
    pass


class ConfigurationError(PythonLearningAssistantError):
    """Erreur de configuration"""
    pass


class DatabaseError(PythonLearningAssistantError):
    """Erreur de base de données"""
    pass


class RAGError(PythonLearningAssistantError):
    """Erreur dans le système RAG"""
    pass


class DocumentLoadingError(RAGError):
    """Erreur lors du chargement de documents"""
    pass


class EmbeddingError(RAGError):
    """Erreur lors de la création d'embeddings"""
    pass


class VectorStoreError(RAGError):
    """Erreur du vector store"""
    pass


class LLMError(PythonLearningAssistantError):
    """Erreur du modèle de langage"""
    pass


class OllamaError(LLMError):
    """Erreur spécifique à Ollama"""
    pass


class QuestError(PythonLearningAssistantError):
    """Erreur dans le système de quêtes"""
    pass


class QuestNotFoundError(QuestError):
    """Quête non trouvée"""
    pass


class InvalidQuestError(QuestError):
    """Quête invalide"""
    pass


class CodeExecutionError(PythonLearningAssistantError):
    """Erreur d'exécution de code"""
    pass


class SecurityError(CodeExecutionError):
    """Erreur de sécurité dans l'exécution de code"""
    pass


class ValidationError(PythonLearningAssistantError):
    """Erreur de validation de données"""
    pass


class NotFoundError(PythonLearningAssistantError):
    """Ressource non trouvée"""
    pass
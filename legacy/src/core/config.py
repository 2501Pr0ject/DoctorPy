import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum

# variables d'environnement
load_dotenv()

class Environment(Enum):
    """Énumération des environnements"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class DatabaseConfig:
    """Configuration de la base de données"""
    url: str
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class LLMConfig:
    """Configuration des modèles de langage"""
    provider: str  # "ollama" ou "openai"
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30

@dataclass
class RAGConfig:
    """Configuration du système RAG"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: str = "vector_stores/chroma_db"
    similarity_threshold: float = 0.7
    max_results: int = 5

@dataclass
class SecurityConfig:
    """Configuration de sécurité"""
    secret_key: str
    session_lifetime: int = 3600  # 1 heure
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 heure
    allowed_file_types: list = None
    max_file_size: int = 10  # MB

@dataclass
class UIConfig:
    """Configuration de l'interface utilisateur"""
    title: str = "Assistant Pédagogique IA"
    version: str = "1.0.0"
    port: int = 8501
    address: str = "localhost"
    theme: Dict[str, str] = None

class Config:
    """Classe principale de configuration"""
    
    def __init__(self, environment: Optional[str] = None):
        # Déterminer l'environnement
        self.environment = Environment(environment or os.getenv("ENVIRONMENT", "development"))
        
        # Chemins de base
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.base_dir / "config"
        self.data_dir = self.base_dir / "data"
        self.vector_stores_dir = self.base_dir / "vector_stores"
        self.logs_dir = self.base_dir / "logs"
        
        # Créer les répertoires nécessaires
        self._create_directories()
        
        # Charger la configuration
        self._load_configuration()
        
        # Valider la configuration
        self._validate_configuration()
    
    def _create_directories(self):
        """Crée les répertoires nécessaires"""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "databases",
            self.vector_stores_dir,
            self.logs_dir,
            self.data_dir / "uploads",
            self.data_dir / "exports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_configuration(self):
        """Charge la configuration depuis les fichiers YAML"""
        # Charger la configuration de base
        base_config_file = self.config_dir / f"{self.environment.value}.yaml"
        
        if base_config_file.exists():
            with open(base_config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = {}
            print(f"Attention: Fichier de configuration {base_config_file} non trouvé")
        
        # Charger la configuration des modèles Ollama
        ollama_config_file = self.config_dir / "ollama_models.yaml"
        if ollama_config_file.exists():
            with open(ollama_config_file, 'r', encoding='utf-8') as f:
                ollama_config = yaml.safe_load(f)
                config_data.setdefault('llm', {}).update(ollama_config)
        
        # Configuration de la base de données
        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", 
                         config_data.get("database", {}).get("url", 
                                                            f"sqlite:///{self.data_dir}/databases/users.db")),
            echo=config_data.get("database", {}).get("echo", False),
            pool_size=config_data.get("database", {}).get("pool_size", 5),
            max_overflow=config_data.get("database", {}).get("max_overflow", 10)
        )
        
        # Configuration LLM
        llm_config = config_data.get("llm", {})
        self.llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", llm_config.get("provider", "ollama")),
            model_name=os.getenv("LLM_MODEL", llm_config.get("model_name", "llama3.1")),
            base_url=os.getenv("OLLAMA_BASE_URL", llm_config.get("base_url", "http://localhost:11434")),
            api_key=os.getenv("OPENAI_API_KEY", llm_config.get("api_key")),
            temperature=float(os.getenv("LLM_TEMPERATURE", llm_config.get("temperature", 0.7))),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", llm_config.get("max_tokens", 2000))),
            timeout=int(os.getenv("LLM_TIMEOUT", llm_config.get("timeout", 30)))
        )
        
        # Configuration RAG
        rag_config = config_data.get("rag", {})
        self.rag = RAGConfig(
            chunk_size=rag_config.get("chunk_size", 1000),
            chunk_overlap=rag_config.get("chunk_overlap", 200),
            embedding_model=rag_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            vector_store_path=str(self.vector_stores_dir / rag_config.get("vector_store_path", "chroma_db")),
            similarity_threshold=rag_config.get("similarity_threshold", 0.7),
            max_results=rag_config.get("max_results", 5)
        )
        
        # Configuration de sécurité
        security_config = config_data.get("security", {})
        self.security = SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", security_config.get("secret_key", "dev-secret-key")),
            session_lifetime=int(os.getenv("SESSION_LIFETIME", security_config.get("session_lifetime", 3600))),
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", security_config.get("rate_limit_requests", 100))),
            rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", security_config.get("rate_limit_window", 3600))),
            allowed_file_types=security_config.get("allowed_file_types", [".pdf", ".txt", ".docx", ".md"]),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", security_config.get("max_file_size", 10)))
        )
        
        # Configuration UI
        ui_config = config_data.get("ui", {})
        self.ui = UIConfig(
            title=ui_config.get("title", "Assistant Pédagogique IA"),
            version=ui_config.get("version", "1.0.0"),
            port=int(os.getenv("PORT", ui_config.get("port", 8501))),
            address=os.getenv("ADDRESS", ui_config.get("address", "localhost")),
            theme=ui_config.get("theme", {
                "primaryColor": "#2E86AB",
                "backgroundColor": "#FFFFFF",
                "secondaryBackgroundColor": "#F0F2F6",
                "textColor": "#262730"
            })
        )
        
        # Configuration des logs
        self.logging_config = config_data.get("logging", {})
        
        # Configuration des quêtes
        self.quests_config = config_data.get("quests", {})
        
        # Configuration debug
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
    
    def _validate_configuration(self):
        """Valide la configuration"""
        errors = []
        
        # Validation LLM
        if self.llm.provider == "openai" and not self.llm.api_key:
            errors.append("OPENAI_API_KEY est requis pour le provider OpenAI")
        
        # Validation sécurité
        if (self.environment == Environment.PRODUCTION and 
            self.security.secret_key == "dev-secret-key"):
            errors.append("SECRET_KEY doit être définie en production")
        
        # Validation répertoires
        for directory in [self.data_dir, self.vector_stores_dir, self.logs_dir]:
            if not os.access(directory, os.W_OK):
                errors.append(f"Répertoire {directory} non accessible en écriture")
        
        # Validation modèle LLM
        if self.llm.provider == "ollama":
            # Vérifier si Ollama est accessible (optionnel en dev)
            if self.environment == Environment.PRODUCTION:
                try:
                    import requests
                    response = requests.get(f"{self.llm.base_url}/api/tags", timeout=5)
                    if response.status_code != 200:
                        errors.append("Ollama n'est pas accessible à l'URL configurée")
                except Exception:
                    if self.environment == Environment.PRODUCTION:
                        errors.append("Impossible de vérifier la connexion à Ollama")
        
        if errors:
            error_msg = f"Erreurs de configuration: {', '.join(errors)}"
            if self.environment == Environment.PRODUCTION:
                raise ValueError(error_msg)
            else:
                print(f"Avertissements de configuration: {error_msg}")
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Retourne la configuration Streamlit"""
        return {
            "server": {
                "port": self.ui.port,
                "address": self.ui.address,
                "runOnSave": self.debug,
                "fileWatcherType": "auto" if self.debug else "none"
            },
            "browser": {
                "gatherUsageStats": False
            },
            "theme": self.ui.theme
        }
    
    def get_database_url(self) -> str:
        """Retourne l'URL de la base de données"""
        return self.database.url
    
    def get_vector_store_path(self) -> str:
        """Retourne le chemin du vector store"""
        return self.rag.vector_store_path
    
    def get_uploads_dir(self) -> Path:
        """Retourne le répertoire des uploads"""
        return self.data_dir / "uploads"
    
    def get_exports_dir(self) -> Path:
        """Retourne le répertoire des exports"""
        return self.data_dir / "exports"
    
    def get_quests_dir(self) -> Path:
        """Retourne le répertoire des quêtes"""
        return self.data_dir / "quests"
    
    def is_file_allowed(self, filename: str) -> bool:
        """Vérifie si un type de fichier est autorisé"""
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.security.allowed_file_types
    
    def get_log_config(self) -> Dict[str, Any]:
        """Retourne la configuration des logs"""
        default_config = {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": str(self.logs_dir / "app.log"),
            "max_size": "10MB",
            "backup_count": 5
        }
        
        # Charger la config depuis logging.yaml si disponible
        logging_config_file = self.config_dir / "logging.yaml"
        if logging_config_file.exists():
            with open(logging_config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        
        return default_config

# Instance globale de configuration
_config_instance = None

def get_config(environment: Optional[str] = None) -> Config:
    """Retourne l'instance de configuration (singleton)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(environment)
    return _config_instance

def reload_config(environment: Optional[str] = None):
    """Recharge la configuration"""
    global _config_instance
    _config_instance = Config(environment)
    return _config_instance

# Templates de prompts par défaut
DEFAULT_PROMPTS = {
    "system_prompt": """Tu es un assistant pédagogique expert spécialisé dans l'aide à l'apprentissage du Python. 
    Tu dois fournir des explications claires, structurées et adaptées au niveau de l'utilisateur. 
    Utilise des exemples concrets et encourage l'apprenant dans sa démarche d'apprentissage.
    
    Tes réponses doivent être :
    - Claires et pédagogiques
    - Adaptées au niveau de l'utilisateur
    - Accompagnées d'exemples pratiques
    - Encourageantes et motivantes
    """,
    
    "quest_generation_prompt": """Génère une quête pédagogique pour l'apprentissage du Python avec les caractéristiques suivantes:
    - Sujet: {subject}
    - Niveau: {level}
    - Difficulté: {difficulty}
    - Objectifs: {objectives}
    
    La quête doit inclure :
    1. Une description claire de l'objectif
    2. Des étapes progressives
    3. Des exemples de code
    4. Des exercices pratiques
    5. Des critères de validation
    """,
    
    "code_evaluation_prompt": """Évalue ce code Python selon les critères suivants :
    
    Code à évaluer :
    ```python
    {code}
    ```
    
    Critères d'évaluation :
    - Correction syntaxique
    - Logique algorithmique
    - Bonnes pratiques Python
    - Lisibilité et documentation
    - Performance
    
    Fournis des commentaires constructifs et des suggestions d'amélioration.
    """,
    
    "tutor_prompt": """Tu es un tuteur Python bienveillant et expert. 
    L'utilisateur a le niveau : {user_level}
    Contexte de la conversation : {context}
    
    Réponds à sa question en :
    1. Expliquant clairement le concept
    2. Donnant un exemple pratique
    3. Proposant un exercice simple si approprié
    4. Encourageant l'apprentissage
    
    Question de l'utilisateur : {question}
    """
}

# Validation automatique au chargement (seulement si pas en mode test)
if __name__ != "__main__" and os.getenv("TESTING") != "true":
    try:
        config = get_config()
    except Exception as e:
        print(f"Erreur lors du chargement de la configuration: {e}")
        print("Vérifiez vos fichiers de configuration et variables d'environnement.")
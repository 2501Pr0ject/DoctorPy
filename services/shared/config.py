"""
Configuration centralisée pour tous les microservices DoctorPy
"""

import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
from enum import Enum


class Environment(Enum):
    """Environnements d'exécution"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Configuration base de données"""
    url: str = "postgresql://localhost:5432/doctorpy"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False


@dataclass
class RedisConfig:
    """Configuration Redis"""
    url: str = "redis://localhost:6379"
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True


@dataclass
class RabbitMQConfig:
    """Configuration RabbitMQ"""
    url: str = "amqp://localhost:5672"
    heartbeat: int = 600
    blocked_connection_timeout: int = 300


@dataclass
class AuthConfig:
    """Configuration authentification"""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    bcrypt_rounds: int = 12


@dataclass
class AIConfig:
    """Configuration IA et RAG"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama2"
    ollama_url: str = "http://localhost:11434"
    chromadb_path: str = "./data/vector_store"
    chunk_size: int = 500
    chunk_overlap: int = 100
    max_tokens: int = 2048
    temperature: float = 0.7


@dataclass
class NotificationConfig:
    """Configuration notifications"""
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_email: str = "doctorpy@localhost"
    slack_webhook_url: str = ""
    webhook_url: str = ""
    webhook_token: str = ""


@dataclass
class MonitoringConfig:
    """Configuration monitoring"""
    prometheus_port: int = 8090
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    jaeger_url: str = "http://localhost:14268"


@dataclass
class ServiceConfig:
    """Configuration complète d'un service"""
    service_name: str
    service_port: int
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Configurations des composants
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    rabbitmq: RabbitMQConfig = field(default_factory=RabbitMQConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Configurations spécifiques
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Charger la configuration depuis les variables d'environnement"""
        self._load_from_env()
        self._load_from_file()
        self._validate_config()
    
    def _load_from_env(self) -> None:
        """Charger depuis les variables d'environnement"""
        # Environment
        env_str = os.getenv("DOCTORPY_ENV", "development")
        try:
            self.environment = Environment(env_str)
        except ValueError:
            self.environment = Environment.DEVELOPMENT
        
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Database
        if db_url := os.getenv("DATABASE_URL"):
            self.database.url = db_url
        
        # Redis
        if redis_url := os.getenv("REDIS_URL"):
            self.redis.url = redis_url
        
        # RabbitMQ
        if rabbitmq_url := os.getenv("RABBITMQ_URL"):
            self.rabbitmq.url = rabbitmq_url
        
        # Auth
        if secret_key := os.getenv("SECRET_KEY"):
            self.auth.secret_key = secret_key
        
        if token_expire := os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"):
            try:
                self.auth.access_token_expire_minutes = int(token_expire)
            except ValueError:
                pass
        
        # AI
        if embedding_model := os.getenv("EMBEDDING_MODEL"):
            self.ai.embedding_model = embedding_model
        
        if llm_model := os.getenv("LLM_MODEL"):
            self.ai.llm_model = llm_model
        
        if ollama_url := os.getenv("OLLAMA_URL"):
            self.ai.ollama_url = ollama_url
        
        # Notifications
        if smtp_host := os.getenv("SMTP_HOST"):
            self.notification.smtp_host = smtp_host
        
        if smtp_port := os.getenv("SMTP_PORT"):
            try:
                self.notification.smtp_port = int(smtp_port)
            except ValueError:
                pass
        
        if smtp_user := os.getenv("SMTP_USER"):
            self.notification.smtp_user = smtp_user
        
        if smtp_password := os.getenv("SMTP_PASSWORD"):
            self.notification.smtp_password = smtp_password
        
        if slack_webhook := os.getenv("SLACK_WEBHOOK_URL"):
            self.notification.slack_webhook_url = slack_webhook
        
        # Monitoring
        if log_level := os.getenv("LOG_LEVEL"):
            self.monitoring.log_level = log_level.upper()
    
    def _load_from_file(self) -> None:
        """Charger depuis un fichier de configuration"""
        config_files = [
            f"config/{self.service_name}.yaml",
            f"config/{self.service_name}.yml",
            f"config/{self.service_name}.json",
            "config/default.yaml",
            "config/default.yml"
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        if config_path.suffix in ['.yaml', '.yml']:
                            file_config = yaml.safe_load(f)
                        else:
                            file_config = json.load(f)
                    
                    self._merge_config(file_config)
                    break
                    
                except Exception as e:
                    print(f"Erreur lecture config {config_file}: {e}")
    
    def _merge_config(self, file_config: Dict[str, Any]) -> None:
        """Fusionner la configuration depuis un fichier"""
        if not file_config:
            return
        
        # Fusionner les configurations par section
        for section, values in file_config.items():
            if hasattr(self, section) and isinstance(values, dict):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _validate_config(self) -> None:
        """Valider la configuration"""
        # Vérifications de sécurité pour la production
        if self.environment == Environment.PRODUCTION:
            if self.auth.secret_key == "your-secret-key-change-in-production":
                raise ValueError("SECRET_KEY doit être définie en production")
            
            if self.debug:
                print("WARNING: DEBUG activé en production")
            
            if "localhost" in self.database.url:
                print("WARNING: Base de données locale en production")
    
    def get_database_url(self) -> str:
        """Obtenir l'URL de base de données"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Obtenir l'URL Redis"""
        return self.redis.url
    
    def get_rabbitmq_url(self) -> str:
        """Obtenir l'URL RabbitMQ"""
        return self.rabbitmq.url
    
    def is_production(self) -> bool:
        """Vérifier si on est en production"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Vérifier si on est en développement"""
        return self.environment == Environment.DEVELOPMENT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        return {
            "service_name": self.service_name,
            "service_port": self.service_port,
            "environment": self.environment.value,
            "debug": self.debug,
            "database": {
                "url": self.database.url,
                "pool_size": self.database.pool_size,
                "echo": self.database.echo
            },
            "redis": {
                "url": self.redis.url,
                "max_connections": self.redis.max_connections
            },
            "auth": {
                "algorithm": self.auth.algorithm,
                "access_token_expire_minutes": self.auth.access_token_expire_minutes
            },
            "ai": {
                "embedding_model": self.ai.embedding_model,
                "llm_model": self.ai.llm_model,
                "ollama_url": self.ai.ollama_url
            },
            "monitoring": {
                "log_level": self.monitoring.log_level,
                "metrics_enabled": self.monitoring.metrics_enabled
            }
        }


class ConfigManager:
    """Gestionnaire de configuration centralisé"""
    
    _instance: Optional['ConfigManager'] = None
    _configs: Dict[str, ServiceConfig] = {}
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_config(cls, service_name: str, service_port: int) -> ServiceConfig:
        """Obtenir la configuration d'un service"""
        if service_name not in cls._configs:
            cls._configs[service_name] = ServiceConfig(
                service_name=service_name,
                service_port=service_port
            )
        return cls._configs[service_name]
    
    @classmethod
    def reload_config(cls, service_name: str) -> ServiceConfig:
        """Recharger la configuration d'un service"""
        if service_name in cls._configs:
            config = cls._configs[service_name]
            config._load_from_env()
            config._load_from_file()
            config._validate_config()
        return cls._configs[service_name]
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, ServiceConfig]:
        """Obtenir toutes les configurations"""
        return cls._configs.copy()


# Configurations prédéfinies pour chaque service
def get_auth_config() -> ServiceConfig:
    """Configuration pour le service d'authentification"""
    return ConfigManager.get_config("auth", 8001)


def get_rag_config() -> ServiceConfig:
    """Configuration pour le service RAG"""
    return ConfigManager.get_config("rag", 8002)


def get_analytics_config() -> ServiceConfig:
    """Configuration pour le service Analytics"""
    return ConfigManager.get_config("analytics", 8003)


def get_quest_config() -> ServiceConfig:
    """Configuration pour le service Quest"""
    return ConfigManager.get_config("quest", 8004)


def get_notification_config() -> ServiceConfig:
    """Configuration pour le service Notification"""
    return ConfigManager.get_config("notification", 8005)


# Fonction utilitaire pour créer des fichiers de configuration
def create_default_config_files():
    """Créer des fichiers de configuration par défaut"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Configuration par défaut
    default_config = {
        "database": {
            "url": "postgresql://localhost:5432/doctorpy",
            "pool_size": 10,
            "echo": False
        },
        "redis": {
            "url": "redis://localhost:6379",
            "max_connections": 10
        },
        "rabbitmq": {
            "url": "amqp://localhost:5672"
        },
        "auth": {
            "secret_key": "change-this-in-production",
            "algorithm": "HS256",
            "access_token_expire_minutes": 30
        },
        "ai": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "llama2",
            "ollama_url": "http://localhost:11434",
            "chunk_size": 500,
            "chunk_overlap": 100
        },
        "notification": {
            "smtp_host": "localhost",
            "smtp_port": 587,
            "from_email": "doctorpy@localhost"
        },
        "monitoring": {
            "log_level": "INFO",
            "metrics_enabled": True
        }
    }
    
    # Fichier de configuration par défaut
    with open(config_dir / "default.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    # Configurations spécifiques par service
    services_config = {
        "auth": {
            **default_config,
            "custom_config": {
                "max_login_attempts": 5,
                "lockout_duration_minutes": 15,
                "password_min_length": 8
            }
        },
        "rag": {
            **default_config,
            "custom_config": {
                "max_query_length": 1000,
                "max_results": 10,
                "similarity_threshold": 0.7
            }
        },
        "analytics": {
            **default_config,
            "custom_config": {
                "retention_days": 90,
                "batch_size": 1000,
                "aggregation_interval_minutes": 5
            }
        },
        "quest": {
            **default_config,
            "custom_config": {
                "max_active_quests": 10,
                "difficulty_levels": ["beginner", "intermediate", "advanced"],
                "points_multiplier": 1.0
            }
        },
        "notification": {
            **default_config,
            "custom_config": {
                "max_retries": 3,
                "retry_delay_seconds": 60,
                "batch_size": 100
            }
        }
    }
    
    for service, config in services_config.items():
        with open(config_dir / f"{service}.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Fichiers de configuration créés dans {config_dir}")


if __name__ == "__main__":
    # Créer les fichiers de configuration par défaut
    create_default_config_files()
    
    # Tester la configuration
    auth_config = get_auth_config()
    print("Configuration Auth:", auth_config.to_dict())
"""
DoctorPy Microservices Architecture

Cette package contient tous les microservices de DoctorPy :
- Auth Service (Authentification & Autorisation)
- RAG Service (Knowledge Base & IA)
- Quest Service (Gamification & Quêtes)
- Analytics Service (Métriques & Monitoring)
- Notification Service (Alertes & Notifications)
"""

__version__ = "2.0.0"
__author__ = "DoctorPy Team"

# Services disponibles
SERVICES = {
    "auth": {
        "name": "Auth Service",
        "port": 8001,
        "description": "Authentication & Authorization"
    },
    "rag": {
        "name": "RAG Service", 
        "port": 8002,
        "description": "Knowledge Base & AI Processing"
    },
    "analytics": {
        "name": "Analytics Service",
        "port": 8003,
        "description": "Metrics & Monitoring"
    },
    "quest": {
        "name": "Quest Service",
        "port": 8004,
        "description": "Gamification & Progress Tracking"
    },
    "notification": {
        "name": "Notification Service",
        "port": 8005,
        "description": "Multi-channel Notifications"
    }
}

# Configuration par défaut
DEFAULT_CONFIG = {
    "redis_url": "redis://localhost:6379",
    "rabbitmq_url": "amqp://localhost:5672",
    "database_url": "postgresql://localhost:5432/doctorpy",
    "secret_key": "your-secret-key-here",
    "algorithm": "HS256",
    "access_token_expire_minutes": 30
}
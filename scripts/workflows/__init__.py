"""
Workflows d'orchestration avec Prefect pour DoctorPy

Ce module contient tous les workflows automatisés :
- Pipeline de données RAG
- Tâches de maintenance
- Analytics et monitoring
- Notifications et alertes
- Déploiement et configuration
"""

from .data_pipeline import update_knowledge_base, rag_quick_update, rag_full_pipeline
from .maintenance import daily_maintenance, weekly_maintenance, emergency_maintenance
from .analytics import generate_analytics, health_check_flow
from .deployment import deploy_all_workflows, setup_prefect_environment
from .tasks import *

__all__ = [
    # Pipeline RAG
    "update_knowledge_base",
    "rag_quick_update", 
    "rag_full_pipeline",
    
    # Maintenance
    "daily_maintenance",
    "weekly_maintenance",
    "emergency_maintenance",
    
    # Analytics et monitoring
    "generate_analytics",
    "health_check_flow",
    
    # Déploiement
    "deploy_all_workflows",
    "setup_prefect_environment"
]
"""
Tâches Prefect réutilisables pour DoctorPy

Organisation des tâches par domaine :
- scraping : Collecte de données
- processing : Traitement de documents  
- embedding : Génération d'embeddings
- indexing : Indexation ChromaDB
- maintenance : Nettoyage et optimisation
- monitoring : Surveillance et alertes
- notification : Notifications et rapports
"""

from .scraping import scrape_python_docs, validate_scraped_docs
from .processing import process_documents, validate_chunks
from .embedding import create_embeddings, validate_embeddings
from .indexing import index_documents, validate_index
from .maintenance import cleanup_sessions, cleanup_logs, backup_database, optimize_vector_store
from .monitoring import collect_system_metrics, check_health, generate_report
from .notification import send_notification, send_alert

__all__ = [
    # Data Pipeline
    "scrape_python_docs",
    "validate_scraped_docs",
    "process_documents", 
    "validate_chunks",
    "create_embeddings",
    "validate_embeddings",
    "index_documents",
    "validate_index",
    
    # Maintenance
    "cleanup_sessions",
    "cleanup_logs", 
    "backup_database",
    "optimize_vector_store",
    
    # Monitoring
    "collect_system_metrics",
    "check_health",
    "generate_report",
    
    # Notifications
    "send_notification",
    "send_alert"
]
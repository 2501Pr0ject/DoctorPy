"""
Gestionnaire de base de données partagé pour les microservices
"""

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Optional
import os
from .utils import LoggerFactory

Base = declarative_base()
logger = LoggerFactory.get_logger("database")


class DatabaseManager:
    """Gestionnaire de base de données centralisé"""
    
    def __init__(self, database_url: Optional[str] = None):
        if database_url is None:
            database_url = os.getenv("DATABASE_URL", "sqlite:///./data/databases/doctorpy.db")
        
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        
    def initialize(self):
        """Initialise la connexion à la base de données"""
        try:
            if self.database_url.startswith("sqlite"):
                # Configuration SQLite
                self.engine = create_engine(
                    self.database_url,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                )
            else:
                # Configuration PostgreSQL ou autres
                self.engine = create_engine(self.database_url)
                
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Créer les tables
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"✅ Base de données initialisée: {self.database_url}")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation base de données: {e}")
            raise
    
    def get_session(self) -> Session:
        """Obtient une session de base de données"""
        if self.SessionLocal is None:
            self.initialize()
        return self.SessionLocal()
    
    def close(self):
        """Ferme les connexions"""
        if self.engine:
            self.engine.dispose()
            logger.info("✅ Connexions base de données fermées")


# Instance globale
db_manager = DatabaseManager()


def get_database_session():
    """Dependency injection pour FastAPI"""
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()
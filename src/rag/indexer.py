import asyncio
from pathlib import Path
from typing import List, Optional
from langchain.docstore.document import Document

from ..core.config import settings
from ..core.logger import logger
from ..core.exceptions import RAGError
from .document_loader import DocumentLoader
from .vector_store import ChromaVectorStore


class DocumentIndexer:
    """Indexeur de documents pour le système RAG"""
    
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.vector_store = ChromaVectorStore()
    
    def index_python_documentation(self, force_refresh: bool = False) -> None:
        """Indexe la documentation Python officielle"""
        try:
            logger.info("Début de l'indexation de la documentation Python")
            
            # Vérifier si la collection existe déjà
            doc_count = self.vector_store.get_collection_count()
            
            if doc_count > 0 and not force_refresh:
                logger.info(f"Collection existante avec {doc_count} documents. Utilisez force_refresh=True pour réindexer.")
                return
            
            if force_refresh and doc_count > 0:
                logger.info("Suppression de la collection existante")
                self.vector_store.delete_collection()
                self.vector_store._initialize_chroma()
            
            # Charger les documents
            documents = self.document_loader.load_python_docs()
            
            if not documents:
                raise RAGError("Aucun document chargé")
            
            # Indexer dans le vector store
            self.vector_store.add_documents(documents)
            
            final_count = self.vector_store.get_collection_count()
            logger.info(f"Indexation terminée. {final_count} documents indexés.")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation: {e}")
            raise RAGError(f"Impossible d'indexer la documentation: {e}")
    
    def index_local_documents(self, directory: Path, force_refresh: bool = False) -> None:
        """Indexe des documents locaux"""
        try:
            logger.info(f"Indexation des documents locaux depuis: {directory}")
            
            if not directory.exists():
                raise RAGError(f"Le répertoire {directory} n'existe pas")
            
            # Charger les documents markdown
            documents = self.document_loader.load_local_markdown(directory)
            
            if not documents:
                logger.warning("Aucun document trouvé dans le répertoire")
                return
            
            # Ajouter au vector store
            self.vector_store.add_documents(documents)
            
            logger.info(f"Indexation terminée. {len(documents)} documents ajoutés.")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation locale: {e}")
            raise RAGError(f"Impossible d'indexer les documents locaux: {e}")
    
    def get_index_stats(self) -> dict:
        """Retourne des statistiques sur l'index"""
        try:
            return {
                'total_documents': self.vector_store.get_collection_count(),
                'collection_name': self.vector_store.collection_name,
                'embedding_model': self.vector_store.embedding_model.model_name
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {'error': str(e)}


# Exemple d'utilisation simple
def setup_rag_system() -> DocumentRetriever:
    """Configure le système RAG complet"""
    try:
        logger.info("Configuration du système RAG")
        
        # Créer l'indexeur
        indexer = DocumentIndexer()
        
        # Indexer la documentation Python (seulement si pas déjà fait)
        indexer.index_python_documentation()
        
        # Créer le retriever
        retriever = DocumentRetriever()
        
        logger.info("Système RAG configuré avec succès")
        return retriever
        
    except Exception as e:
        logger.error(f"Erreur lors de la configuration du RAG: {e}")
        raise RAGError(f"Impossible de configurer le système RAG: {e}")
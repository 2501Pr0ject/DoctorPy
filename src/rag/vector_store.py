from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.docstore.document import Document

from ..core.config import settings
from ..core.logger import logger
from ..core.exceptions import VectorStoreError
from .embeddings import EmbeddingModel


class ChromaVectorStore:
    """Gestionnaire du vector store Chroma"""
    
    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = settings.chroma_persist_directory
        self.embedding_model = EmbeddingModel()
        
        # Initialiser Chroma
        self._client = None
        self._collection = None
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialise le client et la collection Chroma"""
        try:
            logger.info("Initialisation de Chroma")
            
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Créer ou récupérer la collection
            try:
                self._collection = self._client.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' récupérée")
            except:
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Python documentation embeddings"}
                )
                logger.info(f"Collection '{self.collection_name}' créée")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de Chroma: {e}")
            raise VectorStoreError(f"Impossible d'initialiser Chroma: {e}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Ajoute des documents au vector store"""
        try:
            logger.info(f"Ajout de {len(documents)} documents au vector store")
            
            # Préparer les données
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Créer les embeddings
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Ajouter à Chroma en batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                
                self._collection.add(
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx],
                    embeddings=embeddings[i:end_idx]
                )
                
                logger.info(f"Batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ajouté")
            
            logger.info(f"Tous les documents ajoutés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout des documents: {e}")
            raise VectorStoreError(f"Impossible d'ajouter les documents: {e}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None, 
        score_threshold: float = None
    ) -> List[Tuple[Document, float]]:
        """Recherche par similarité"""
        try:
            k = k or settings.max_retrieved_docs
            score_threshold = score_threshold or settings.similarity_threshold
            
            # Créer l'embedding de la requête
            query_embedding = self.embedding_model.embed_query(query)
            
            # Recherche dans Chroma
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convertir les résultats
            documents_with_scores = []
            if results['documents'][0]:  # Vérifier qu'il y a des résultats
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Convertir distance en score de similarité (plus proche = score plus élevé)
                    similarity_score = 1.0 - distance
                    
                    if similarity_score >= score_threshold:
                        document = Document(
                            page_content=doc,
                            metadata=metadata
                        )
                        documents_with_scores.append((document, similarity_score))
            
            logger.info(f"Trouvé {len(documents_with_scores)} documents pertinents pour: '{query}'")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            raise VectorStoreError(f"Impossible d'effectuer la recherche: {e}")
    
    def get_collection_count(self) -> int:
        """Retourne le nombre de documents dans la collection"""
        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"Erreur lors du comptage: {e}")
            return 0
    
    def delete_collection(self) -> None:
        """Supprime la collection"""
        try:
            self._client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' supprimée")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {e}")
            raise VectorStoreError(f"Impossible de supprimer la collection: {e}")
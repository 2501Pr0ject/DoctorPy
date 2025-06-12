from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from ..core.config import settings
from ..core.logger import logger
from ..core.exceptions import EmbeddingError


class EmbeddingModel:
    """Gestionnaire des embeddings avec sentence-transformers"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self.device = settings.embedding_device
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Charge le modèle de manière paresseuse"""
        if self._model is None:
            try:
                logger.info(f"Chargement du modèle d'embedding: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
                logger.info("Modèle d'embedding chargé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle: {e}")
                raise EmbeddingError(f"Impossible de charger le modèle: {e}")
        
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Crée des embeddings pour une liste de documents"""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Erreur lors de la création des embeddings: {e}")
            raise EmbeddingError(f"Impossible de créer les embeddings: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """Crée un embedding pour une requête"""
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Erreur lors de l'embedding de la requête: {e}")
            raise EmbeddingError(f"Impossible d'embedder la requête: {e}")
    
    def get_dimension(self) -> int:
        """Retourne la dimension des embeddings"""
        return self.model.get_sentence_embedding_dimension()
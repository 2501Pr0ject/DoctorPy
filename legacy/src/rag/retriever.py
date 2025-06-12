from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document

from ..core.config import settings
from ..core.logger import logger
from ..core.exceptions import RAGError
from .vector_store import ChromaVectorStore


class DocumentRetriever:
    """Récupérateur de documents avec post-traitement"""
    
    def __init__(self, vector_store: Optional[ChromaVectorStore] = None):
        self.vector_store = vector_store or ChromaVectorStore()
    
    def retrieve_relevant_documents(
        self, 
        query: str, 
        max_docs: int = None,
        min_score: float = None
    ) -> List[Document]:
        """Récupère les documents les plus pertinents pour une requête"""
        try:
            max_docs = max_docs or settings.max_retrieved_docs
            min_score = min_score or settings.similarity_threshold
            
            # Recherche par similarité
            docs_with_scores = self.vector_store.similarity_search(
                query=query,
                k=max_docs * 2,  # Récupérer plus pour filtrer ensuite
                score_threshold=min_score
            )
            
            if not docs_with_scores:
                logger.warning(f"Aucun document trouvé pour la requête: '{query}'")
                return []
            
            # Post-traitement : déduplication et filtrage
            filtered_docs = self._post_process_documents(docs_with_scores, max_docs)
            
            logger.info(f"Récupéré {len(filtered_docs)} documents pour: '{query}'")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération: {e}")
            raise RAGError(f"Impossible de récupérer les documents: {e}")
    
    def _post_process_documents(
        self, 
        docs_with_scores: List[tuple], 
        max_docs: int
    ) -> List[Document]:
        """Post-traite les documents récupérés"""
        
        # Trier par score décroissant
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Déduplication basée sur le contenu
        seen_contents = set()
        unique_docs = []
        
        for doc, score in docs_with_scores:
            # Utiliser un hash du début du contenu pour la déduplication
            content_hash = hash(doc.page_content[:200])
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                
                # Ajouter le score aux métadonnées
                doc.metadata['similarity_score'] = score
                unique_docs.append(doc)
                
                if len(unique_docs) >= max_docs:
                    break
        
        return unique_docs
    
    def retrieve_by_topic(self, topic: str, subtopic: str = None) -> List[Document]:
        """Récupère des documents par sujet spécifique"""
        query_parts = [topic]
        if subtopic:
            query_parts.append(subtopic)
        
        query = " ".join(query_parts)
        return self.retrieve_relevant_documents(query)
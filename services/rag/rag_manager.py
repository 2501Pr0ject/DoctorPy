"""
Gestionnaire principal du service RAG
"""

import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

from ..shared.cache import CacheManager
from ..shared.events import EventBus, EventType
from ..shared.utils import LoggerFactory

from .models import (
    RAGQueryRequest, RAGResponse, DocumentChunk, IndexDocumentRequest,
    IndexDocumentResponse, QueryType, DocumentType, RAGServiceError,
    QueryAnalytics, RAGServiceConfig
)


class RAGManager:
    """Gestionnaire principal des opérations RAG"""
    
    def __init__(self, config: RAGServiceConfig, cache: CacheManager, event_bus: EventBus):
        self.config = config
        self.cache = cache
        self.event_bus = event_bus
        self.logger = LoggerFactory.get_logger("rag_manager")
        
        # Initialiser les composants
        self._init_vector_store()
        self._init_llm()
        self._init_text_splitter()
        
    def _init_vector_store(self):
        """Initialise le vector store"""
        try:
            from ...src.rag.vector_store import ChromaVectorStore
            self.vector_store = ChromaVectorStore()
            self.logger.info("✅ Vector store initialisé")
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation vector store: {e}")
            raise RAGServiceError("Impossible d'initialiser le vector store", "VECTOR_STORE_ERROR")
    
    def _init_llm(self):
        """Initialise le LLM"""
        try:
            self.llm = Ollama(
                model="llama2:7b",
                temperature=0.1,
                num_predict=512
            )
            self.logger.info("✅ LLM initialisé")
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation LLM: {e}")
            raise RAGServiceError("Impossible d'initialiser le LLM", "LLM_ERROR")
    
    def _init_text_splitter(self):
        """Initialise le text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\\n\\n", "\\n", " ", ""]
        )
        self.logger.info("✅ Text splitter initialisé")
    
    async def query(self, request: RAGQueryRequest) -> RAGResponse:
        """Traite une requête RAG"""
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"🔍 Requête RAG reçue: {request.query[:100]}...")
            
            # Vérifier le cache
            cache_key = f"rag_query:{hash(request.query)}:{request.query_type}"
            cached_response = await self.cache.get(cache_key)
            
            if cached_response:
                self.logger.info("📦 Réponse trouvée en cache")
                return RAGResponse.parse_obj(cached_response)
            
            # Récupérer les documents pertinents
            relevant_docs = await self._retrieve_documents(request)
            
            if not relevant_docs:
                return await self._handle_no_results(request, start_time)
            
            # Générer la réponse
            answer = await self._generate_answer(request, relevant_docs)
            
            # Calculer le score de confiance
            confidence_score = self._calculate_confidence(relevant_docs, answer)
            
            # Créer la réponse
            response = RAGResponse(
                answer=answer,
                query=request.query,
                query_type=request.query_type,
                sources=relevant_docs,
                confidence_score=confidence_score,
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={
                    "query_id": query_id,
                    "model_used": "llama2:7b",
                    "vector_store": "chroma"
                }
            )
            
            # Mettre en cache
            await self.cache.set(cache_key, response.dict(), ttl=self.config.cache_ttl_seconds)
            
            # Envoyer l'événement analytics
            await self._send_analytics_event(request, response, query_id)
            
            self.logger.info(f"✅ Requête traitée en {response.processing_time_ms}ms")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Erreur traitement requête: {e}")
            raise RAGServiceError(f"Erreur traitement requête: {e}", "QUERY_ERROR")
    
    async def _retrieve_documents(self, request: RAGQueryRequest) -> List[DocumentChunk]:
        """Récupère les documents pertinents"""
        try:
            # Adapter la requête selon le type
            enhanced_query = self._enhance_query(request.query, request.query_type)
            
            # Recherche dans le vector store
            docs_with_scores = self.vector_store.similarity_search(
                query=enhanced_query,
                k=request.max_results * 2,  # Récupérer plus pour filtrer
                score_threshold=self.config.confidence_threshold
            )
            
            # Convertir en DocumentChunk
            chunks = []
            for doc, score in docs_with_scores[:request.max_results]:
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    content=doc.page_content,
                    document_id=doc.metadata.get("document_id", "unknown"),
                    document_title=doc.metadata.get("title", "Document sans titre"),
                    document_type=DocumentType(doc.metadata.get("document_type", "documentation")),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    total_chunks=doc.metadata.get("total_chunks", 1),
                    metadata=doc.metadata,
                    similarity_score=float(score)
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Erreur récupération documents: {e}")
            return []
    
    def _enhance_query(self, query: str, query_type: QueryType) -> str:
        """Améliore la requête selon son type"""
        enhancements = {
            QueryType.CODE_HELP: f"Python code help: {query}",
            QueryType.CONCEPT_EXPLANATION: f"Explain Python concept: {query}",
            QueryType.DEBUGGING: f"Python debugging help: {query}",
            QueryType.BEST_PRACTICES: f"Python best practices: {query}",
            QueryType.GENERAL: query
        }
        return enhancements.get(query_type, query)
    
    async def _generate_answer(self, request: RAGQueryRequest, docs: List[DocumentChunk]) -> str:
        """Génère une réponse basée sur les documents"""
        try:
            # Construire le contexte
            context = "\\n\\n".join([
                f"Document: {doc.document_title}\\n{doc.content}"
                for doc in docs
            ])
            
            # Template de prompt selon le type de requête
            prompt_template = self._get_prompt_template(request.query_type)
            prompt = prompt_template.format(
                query=request.query,
                context=context
            )
            
            # Générer la réponse
            response = self.llm(prompt)
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Erreur génération réponse: {e}")
            return "Désolé, je n'ai pas pu générer une réponse pour cette requête."
    
    def _get_prompt_template(self, query_type: QueryType) -> str:
        """Retourne le template de prompt selon le type"""
        templates = {
            QueryType.CODE_HELP: """
Basé sur le contexte suivant, aide l'utilisateur avec son problème de code Python.

Contexte:
{context}

Question: {query}

Réponds de manière claire et pratique avec des exemples de code si nécessaire.

Réponse:
""",
            QueryType.CONCEPT_EXPLANATION: """
Explique ce concept Python en te basant sur le contexte fourni.

Contexte:
{context}

Concept à expliquer: {query}

Fournis une explication claire avec des exemples pratiques.

Réponse:
""",
            QueryType.DEBUGGING: """
Aide à déboguer ce problème Python en utilisant le contexte.

Contexte:
{context}

Problème: {query}

Identifie la cause probable et propose une solution.

Réponse:
""",
            QueryType.BEST_PRACTICES: """
Fournis les meilleures pratiques Python pour cette situation.

Contexte:
{context}

Situation: {query}

Explique les bonnes pratiques avec des exemples.

Réponse:
""",
            QueryType.GENERAL: """
Réponds à cette question Python en utilisant le contexte fourni.

Contexte:
{context}

Question: {query}

Réponse:
"""
        }
        return templates.get(query_type, templates[QueryType.GENERAL])
    
    def _calculate_confidence(self, docs: List[DocumentChunk], answer: str) -> float:
        """Calcule le score de confiance de la réponse"""
        if not docs:
            return 0.0
        
        # Score basé sur la similarité moyenne des documents
        avg_similarity = sum(doc.similarity_score or 0 for doc in docs) / len(docs)
        
        # Score basé sur la longueur de la réponse
        length_score = min(len(answer) / 500, 1.0)
        
        # Score composite
        confidence = (avg_similarity * 0.7) + (length_score * 0.3)
        return round(confidence, 3)
    
    async def _handle_no_results(self, request: RAGQueryRequest, start_time: float) -> RAGResponse:
        """Gère le cas où aucun document n'est trouvé"""
        return RAGResponse(
            answer="Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question. Pourriez-vous reformuler ou être plus spécifique ?",
            query=request.query,
            query_type=request.query_type,
            sources=[],
            confidence_score=0.0,
            processing_time_ms=int((time.time() - start_time) * 1000),
            metadata={"no_results": True}
        )
    
    async def _send_analytics_event(self, request: RAGQueryRequest, response: RAGResponse, query_id: str):
        """Envoie un événement analytics"""
        try:
            analytics = QueryAnalytics(
                query_id=query_id,
                user_id=request.user_id,
                query=request.query,
                query_type=request.query_type,
                response_time_ms=response.processing_time_ms,
                confidence_score=response.confidence_score,
                sources_count=len(response.sources),
                timestamp=datetime.now()
            )
            
            await self.event_bus.publish(
                EventType.RAG_QUERY_PROCESSED,
                analytics.dict()
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur envoi analytics: {e}")
    
    async def index_document(self, request: IndexDocumentRequest) -> IndexDocumentResponse:
        """Indexe un nouveau document"""
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"📄 Indexation document: {request.metadata.title}")
            
            # Découper le document
            documents = self.text_splitter.split_text(request.content)
            
            # Créer les documents avec métadonnées
            docs_to_index = []
            for i, chunk in enumerate(documents):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "document_id": document_id,
                        "title": request.metadata.title,
                        "document_type": request.metadata.document_type.value,
                        "chunk_index": i,
                        "total_chunks": len(documents),
                        "source_url": request.metadata.source_url,
                        "author": request.metadata.author,
                        "created_at": request.metadata.created_at.isoformat(),
                        "tags": request.metadata.tags,
                        "language": request.metadata.language,
                        "difficulty_level": request.metadata.difficulty_level
                    }
                )
                docs_to_index.append(doc)
            
            # Indexer dans le vector store
            self.vector_store.add_documents(docs_to_index)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Envoyer l'événement
            await self.event_bus.publish(
                EventType.DOCUMENT_INDEXED,
                {
                    "document_id": document_id,
                    "title": request.metadata.title,
                    "chunks_count": len(documents),
                    "processing_time_ms": processing_time
                }
            )
            
            self.logger.info(f"✅ Document indexé: {len(documents)} chunks en {processing_time}ms")
            
            return IndexDocumentResponse(
                document_id=document_id,
                chunks_created=len(documents),
                status="success",
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ Erreur indexation: {e}")
            return IndexDocumentResponse(
                document_id=document_id,
                chunks_created=0,
                status="error",
                processing_time_ms=int((time.time() - start_time) * 1000),
                error=str(e)
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Retourne le statut de santé du service"""
        try:
            # Vérifier le vector store
            vector_store_status = "healthy"
            total_docs = 0
            total_chunks = 0
            
            try:
                # Test simple du vector store
                test_results = self.vector_store.similarity_search("test", k=1)
                total_chunks = len(test_results) if test_results else 0
            except Exception:
                vector_store_status = "unhealthy"
            
            # Vérifier le LLM
            llm_status = "healthy"
            try:
                self.llm("test")
            except Exception:
                llm_status = "unhealthy"
            
            # Vérifier le cache
            cache_status = "healthy"
            try:
                await self.cache.get("test")
            except Exception:
                cache_status = "unhealthy"
            
            return {
                "status": "healthy" if all([
                    vector_store_status == "healthy",
                    llm_status == "healthy",
                    cache_status == "healthy"
                ]) else "unhealthy",
                "timestamp": datetime.now(),
                "vector_store_status": vector_store_status,
                "embedding_model_status": llm_status,
                "cache_status": cache_status,
                "total_documents": total_docs,
                "total_chunks": total_chunks
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erreur health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now()
            }
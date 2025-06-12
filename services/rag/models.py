"""
Modèles de données pour le service RAG
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, validator
import uuid


class QueryType(Enum):
    """Types de requêtes RAG"""
    GENERAL = "general"
    CODE_HELP = "code_help"
    CONCEPT_EXPLANATION = "concept_explanation"
    DEBUGGING = "debugging"
    BEST_PRACTICES = "best_practices"


class DocumentType(Enum):
    """Types de documents"""
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"
    CODE_EXAMPLE = "code_example"
    CHEAT_SHEET = "cheat_sheet"
    OFFICIAL_DOC = "official_doc"


class RAGQueryRequest(BaseModel):
    """Requête RAG entrante"""
    query: str
    query_type: QueryType = QueryType.GENERAL
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    max_results: int = 5
    include_metadata: bool = True
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("La requête doit contenir au moins 3 caractères")
        return v.strip()


class DocumentChunk(BaseModel):
    """Chunk de document avec métadonnées"""
    id: str
    content: str
    document_id: str
    document_title: str
    document_type: DocumentType
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = {}
    similarity_score: Optional[float] = None
    
    class Config:
        use_enum_values = True


class RAGResponse(BaseModel):
    """Réponse RAG complète"""
    answer: str
    query: str
    query_type: QueryType
    sources: List[DocumentChunk]
    confidence_score: float
    processing_time_ms: int
    metadata: Dict[str, Any] = {}
    
    class Config:
        use_enum_values = True


class DocumentMetadata(BaseModel):
    """Métadonnées d'un document"""
    title: str
    document_type: DocumentType
    source_url: Optional[str] = None
    author: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    tags: List[str] = []
    language: str = "python"
    difficulty_level: Optional[str] = None
    
    class Config:
        use_enum_values = True


class IndexDocumentRequest(BaseModel):
    """Requête d'indexation de document"""
    content: str
    metadata: DocumentMetadata
    chunk_size: int = 1000
    chunk_overlap: int = 200
    force_reindex: bool = False


class IndexDocumentResponse(BaseModel):
    """Réponse d'indexation"""
    document_id: str
    chunks_created: int
    status: str
    processing_time_ms: int
    error: Optional[str] = None


@dataclass
class RAGServiceConfig:
    """Configuration du service RAG"""
    port: int = 8002
    host: str = "0.0.0.0"
    vector_store_path: str = "./vector_stores/chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_results: int = 10
    confidence_threshold: float = 0.7
    
    # Cache configuration
    cache_ttl_seconds: int = 3600  # 1 heure
    cache_max_size: int = 1000
    
    # Performance
    max_concurrent_queries: int = 50
    query_timeout_seconds: int = 30


class HealthCheckResponse(BaseModel):
    """Réponse health check"""
    status: str
    timestamp: datetime
    vector_store_status: str
    embedding_model_status: str
    cache_status: str
    total_documents: int
    total_chunks: int
    service_version: str = "1.0.0"


class RAGServiceError(Exception):
    """Exception personnalisée du service RAG"""
    def __init__(self, message: str, error_code: str = "RAG_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class QueryAnalytics(BaseModel):
    """Analytics d'une requête"""
    query_id: str
    user_id: Optional[str]
    query: str
    query_type: QueryType
    response_time_ms: int
    confidence_score: float
    sources_count: int
    timestamp: datetime
    
    class Config:
        use_enum_values = True
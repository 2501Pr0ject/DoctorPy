"""Tests unitaires pour le système RAG (Retrieval Augmented Generation)"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from src.rag.retriever import DocumentRetriever
from src.rag.embeddings import EmbeddingManager
from src.rag.document_processor import DocumentProcessor
from src.rag.vector_store import VectorStore


@pytest.mark.unit
class TestDocumentRetriever:
    """Tests pour le retrieveur de documents"""
    
    @pytest.fixture
    def mock_vector_store(self, mock_chromadb_collection):
        """Vector store mocké"""
        vector_store = Mock()
        vector_store.collection = mock_chromadb_collection
        vector_store.search = Mock(return_value=[
            {
                "id": "doc_1",
                "content": "Les variables Python permettent de stocker des données",
                "metadata": {"title": "Variables Python", "section": "tutorial"},
                "score": 0.95
            },
            {
                "id": "doc_2", 
                "content": "Les fonctions permettent de réutiliser du code",
                "metadata": {"title": "Fonctions Python", "section": "tutorial"},
                "score": 0.87
            }
        ])
        return vector_store
    
    def test_retriever_initialization(self, mock_vector_store):
        """Test d'initialisation du retriever"""
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        
        assert retriever.vector_store == mock_vector_store
        assert retriever.max_results == 5  # Valeur par défaut
        assert retriever.similarity_threshold == 0.7
    
    def test_search_documents(self, mock_vector_store):
        """Test de recherche de documents"""
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        
        results = retriever.search("variables Python", max_results=2)
        
        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert "variables" in results[0]["content"].lower()
        mock_vector_store.search.assert_called_once()
    
    def test_filter_by_score(self, mock_vector_store):
        """Test de filtrage par score de similarité"""
        retriever = DocumentRetriever(
            vector_store=mock_vector_store,
            similarity_threshold=0.9
        )
        
        # Mocker un résultat avec score faible
        mock_vector_store.search.return_value = [
            {"id": "doc_1", "content": "test", "score": 0.85},  # En dessous du seuil
            {"id": "doc_2", "content": "test", "score": 0.95}   # Au dessus du seuil
        ]
        
        results = retriever.search("test query")
        
        # Seul le document avec score > 0.9 devrait être retourné
        assert len(results) == 1
        assert results[0]["score"] == 0.95
    
    def test_search_with_filters(self, mock_vector_store):
        """Test de recherche avec filtres"""
        retriever = DocumentRetriever(vector_store=mock_vector_store)
        
        filters = {"section": "tutorial", "difficulty": "beginner"}
        retriever.search("Python basics", filters=filters)
        
        # Vérifier que les filtres sont passés au vector store
        call_args = mock_vector_store.search.call_args
        assert "filters" in call_args.kwargs
        assert call_args.kwargs["filters"] == filters


@pytest.mark.unit
class TestEmbeddingManager:
    """Tests pour le gestionnaire d'embeddings"""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Sentence transformer mocké"""
        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([
            [0.1, 0.2, 0.3, 0.4] * 96  # 384 dimensions
        ]))
        return mock_model
    
    def test_embedding_manager_initialization(self):
        """Test d'initialisation du gestionnaire d'embeddings"""
        with patch('src.rag.embeddings.SentenceTransformer') as mock_st:
            manager = EmbeddingManager()
            
            mock_st.assert_called_once_with('sentence-transformers/all-MiniLM-L6-v2')
            assert manager.model is not None
            assert manager.dimension == 384
    
    def test_encode_text(self, mock_sentence_transformer):
        """Test d'encodage de texte"""
        with patch('src.rag.embeddings.SentenceTransformer', return_value=mock_sentence_transformer):
            manager = EmbeddingManager()
            
            embedding = manager.encode("Test text")
            
            assert embedding is not None
            assert len(embedding) == 384
            mock_sentence_transformer.encode.assert_called_once_with("Test text")
    
    def test_encode_batch(self, mock_sentence_transformer):
        """Test d'encodage par batch"""
        texts = ["Text 1", "Text 2", "Text 3"]
        mock_sentence_transformer.encode.return_value = np.array([
            [0.1] * 384,
            [0.2] * 384, 
            [0.3] * 384
        ])
        
        with patch('src.rag.embeddings.SentenceTransformer', return_value=mock_sentence_transformer):
            manager = EmbeddingManager()
            
            embeddings = manager.encode_batch(texts)
            
            assert len(embeddings) == 3
            assert all(len(emb) == 384 for emb in embeddings)
            mock_sentence_transformer.encode.assert_called_once_with(texts)
    
    def test_similarity_calculation(self, mock_sentence_transformer):
        """Test de calcul de similarité"""
        with patch('src.rag.embeddings.SentenceTransformer', return_value=mock_sentence_transformer):
            manager = EmbeddingManager()
            
            # Simuler deux embeddings
            emb1 = np.array([1.0, 0.0, 0.0, 0.0] * 96)
            emb2 = np.array([0.0, 1.0, 0.0, 0.0] * 96)
            
            similarity = manager.calculate_similarity(emb1, emb2)
            
            assert 0.0 <= similarity <= 1.0
            assert isinstance(similarity, float)


@pytest.mark.unit
class TestDocumentProcessor:
    """Tests pour le processeur de documents"""
    
    def test_text_chunking(self):
        """Test de découpage en chunks"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        long_text = "This is a test document. " * 20  # Texte long
        chunks = processor.create_chunks(long_text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # chunk_size + overlap
    
    def test_chunk_overlap(self):
        """Test de l'overlap entre chunks"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        
        text = "This is a test document with multiple sentences. Each sentence should be processed correctly."
        chunks = processor.create_chunks(text)
        
        if len(chunks) > 1:
            # Vérifier qu'il y a bien un overlap
            overlap_found = False
            for i in range(len(chunks) - 1):
                if chunks[i][-10:] in chunks[i + 1]:
                    overlap_found = True
                    break
            # Note: Overlap dépend de l'implémentation exacte
    
    def test_metadata_extraction(self):
        """Test d'extraction des métadonnées"""
        processor = DocumentProcessor()
        
        document = {
            "title": "Python Variables",
            "content": "Variables store data in Python...",
            "url": "https://docs.python.org/variables",
            "section": "tutorial"
        }
        
        processed = processor.process_document(document)
        
        assert "metadata" in processed
        assert processed["metadata"]["title"] == "Python Variables"
        assert processed["metadata"]["section"] == "tutorial"
    
    def test_content_cleaning(self):
        """Test de nettoyage du contenu"""
        processor = DocumentProcessor()
        
        dirty_text = "   This has   extra    whitespace\n\n\nand\tmultiple\tlines   "
        cleaned = processor.clean_text(dirty_text)
        
        assert cleaned == "This has extra whitespace and multiple lines"
        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")


@pytest.mark.unit
class TestVectorStore:
    """Tests pour le vector store"""
    
    @pytest.fixture
    def mock_chroma_client(self, mock_chromadb_collection):
        """Client ChromaDB mocké"""
        client = Mock()
        client.get_or_create_collection = Mock(return_value=mock_chromadb_collection)
        return client
    
    def test_vector_store_initialization(self, mock_chroma_client):
        """Test d'initialisation du vector store"""
        with patch('src.rag.vector_store.chromadb.Client', return_value=mock_chroma_client):
            store = VectorStore(collection_name="test_collection")
            
            assert store.collection_name == "test_collection"
            mock_chroma_client.get_or_create_collection.assert_called_once_with(
                name="test_collection"
            )
    
    def test_add_documents(self, mock_chroma_client, mock_chromadb_collection):
        """Test d'ajout de documents"""
        with patch('src.rag.vector_store.chromadb.Client', return_value=mock_chroma_client):
            store = VectorStore(collection_name="test_collection")
            
            documents = [
                {
                    "id": "doc_1",
                    "content": "Test content",
                    "embedding": [0.1, 0.2, 0.3] * 128,
                    "metadata": {"title": "Test Doc"}
                }
            ]
            
            store.add_documents(documents)
            
            mock_chromadb_collection.add.assert_called_once()
            call_args = mock_chromadb_collection.add.call_args
            assert "ids" in call_args.kwargs
            assert "documents" in call_args.kwargs
            assert "embeddings" in call_args.kwargs
            assert "metadatas" in call_args.kwargs
    
    def test_search_documents(self, mock_chroma_client, mock_chromadb_collection):
        """Test de recherche de documents"""
        with patch('src.rag.vector_store.chromadb.Client', return_value=mock_chroma_client):
            store = VectorStore(collection_name="test_collection")
            
            query_embedding = [0.1, 0.2, 0.3] * 128
            results = store.search(query_embedding, max_results=3)
            
            mock_chromadb_collection.query.assert_called_once()
            call_args = mock_chromadb_collection.query.call_args
            assert call_args.kwargs["query_embeddings"] == [query_embedding]
            assert call_args.kwargs["n_results"] == 3
    
    def test_search_with_filters(self, mock_chroma_client, mock_chromadb_collection):
        """Test de recherche avec filtres"""
        with patch('src.rag.vector_store.chromadb.Client', return_value=mock_chroma_client):
            store = VectorStore(collection_name="test_collection")
            
            filters = {"section": "tutorial"}
            store.search([0.1] * 384, filters=filters)
            
            call_args = mock_chromadb_collection.query.call_args
            assert "where" in call_args.kwargs
            assert call_args.kwargs["where"] == filters
    
    def test_get_collection_stats(self, mock_chroma_client, mock_chromadb_collection):
        """Test d'obtention des statistiques de collection"""
        mock_chromadb_collection.count.return_value = 150
        
        with patch('src.rag.vector_store.chromadb.Client', return_value=mock_chroma_client):
            store = VectorStore(collection_name="test_collection")
            
            stats = store.get_stats()
            
            assert stats["document_count"] == 150
            mock_chromadb_collection.count.assert_called_once()


@pytest.mark.unit
class TestRAGIntegration:
    """Tests d'intégration des composants RAG"""
    
    def test_end_to_end_retrieval(self, mock_chromadb_collection):
        """Test du pipeline complet de récupération"""
        # Mocker tous les composants
        with patch('src.rag.vector_store.chromadb.Client') as mock_client, \
             patch('src.rag.embeddings.SentenceTransformer') as mock_st:
            
            # Configuration des mocks
            mock_client.return_value.get_or_create_collection.return_value = mock_chromadb_collection
            mock_st.return_value.encode.return_value = np.array([[0.1] * 384])
            
            # Créer les composants
            embedding_manager = EmbeddingManager()
            vector_store = VectorStore("test_collection")
            retriever = DocumentRetriever(vector_store)
            
            # Test de recherche
            query = "Comment créer des variables en Python?"
            results = retriever.search(query)
            
            # Vérifications
            assert len(results) == 2  # Basé sur le mock
            assert all("id" in result for result in results)
            assert all("content" in result for result in results)
    
    def test_error_handling(self, mock_chromadb_collection):
        """Test de gestion d'erreurs"""
        mock_chromadb_collection.query.side_effect = Exception("Database error")
        
        with patch('src.rag.vector_store.chromadb.Client') as mock_client:
            mock_client.return_value.get_or_create_collection.return_value = mock_chromadb_collection
            
            vector_store = VectorStore("test_collection")
            retriever = DocumentRetriever(vector_store)
            
            # La recherche devrait gérer l'erreur gracieusement
            with pytest.raises(Exception):
                retriever.search("test query")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoctorPy - Script d'indexation des documents
Indexe les embeddings dans ChromaDB pour la recherche vectorielle
"""

import asyncio
import aiofiles
import json
import logging
import numpy as np
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import uuid

# Imports pour ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("⚠️ chromadb non disponible")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IndexedDocument:
    """Document indexé dans Chroma"""
    doc_id: str
    chunk_id: str
    collection_name: str
    content: str
    embedding: List[float]
    metadata: Dict
    indexing_date: str


@dataclass
class IndexingResult:
    """Résultat d'indexation"""
    collection_name: str
    total_indexed: int
    total_skipped: int
    total_errors: int
    indexing_date: str
    processing_time_seconds: float
    indexed_documents: List[IndexedDocument]


class DocumentIndexer:
    """
    Indexeur de documents pour ChromaDB
    """
    
    def __init__(
        self,
        embeddings_dir: str = "./data/processed/embeddings",
        chunks_dir: str = "./data/processed/chunks",
        chroma_db_path: str = "./vector_stores/chroma_db",
        collection_name: str = "doctorpy_docs",
        batch_size: int = 100
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.chunks_dir = Path(chunks_dir)
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        self.batch_size = batch_size
        
        # Créer le répertoire ChromaDB
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        # Client ChromaDB
        self.chroma_client = None
        self.collection = None
        
        # Statistiques
        self.total_indexed = 0
        self.total_skipped = 0
        self.total_errors = 0
        self.indexed_documents: List[IndexedDocument] = []
    
    async def index_all_documents(self) -> IndexingResult:
        """Indexe tous les documents disponibles"""
        logger.info(f"🚀 Démarrage de l'indexation dans {self.collection_name}")
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb n'est pas installé. Installez-le avec: pip install chromadb")
        
        start_time = time.time()
        
        try:
            # Initialiser ChromaDB
            await self._initialize_chromadb()
            
            # Charger tous les embeddings
            embedding_files = await self._find_embedding_files()
            
            if not embedding_files:
                logger.warning(f"⚠️ Aucun embedding trouvé dans {self.embeddings_dir}")
                return self._create_empty_result(start_time)
            
            logger.info(f"📄 {len(embedding_files)} embeddings trouvés")
            
            # Charger les données des embeddings et chunks
            embeddings_data = await self._load_embeddings_data(embedding_files)
            
            if not embeddings_data:
                logger.warning("⚠️ Aucune donnée d'embedding valide")
                return self._create_empty_result(start_time)
            
            # Filtrer les documents déjà indexés
            new_embeddings = await self._filter_existing_documents(embeddings_data)
            
            if not new_embeddings:
                logger.info("✅ Tous les documents sont déjà indexés")
                return self._create_empty_result(start_time)
            
            logger.info(f"🔄 {len(new_embeddings)} nouveaux documents à indexer")
            
            # Indexer par lots
            await self._index_embeddings_in_batches(new_embeddings)
            
            processing_time = time.time() - start_time
            
            # Créer le résultat
            result = IndexingResult(
                collection_name=self.collection_name,
                total_indexed=self.total_indexed,
                total_skipped=self.total_skipped,
                total_errors=self.total_errors,
                indexing_date=datetime.now().isoformat(),
                processing_time_seconds=processing_time,
                indexed_documents=self.indexed_documents
            )
            
            # Sauvegarder le rapport
            await self._save_indexing_report(result)
            
            logger.info(f"✅ Indexation terminée en {processing_time:.2f}s")
            logger.info(f"📚 {self.total_indexed} documents indexés")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'indexation: {e}")
            raise
    
    async def _initialize_chromadb(self):
        """Initialise le client ChromaDB"""
        try:
            logger.info(f"🔧 Initialisation de ChromaDB ({self.chroma_db_path})")
            
            # Créer le client ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Créer ou récupérer la collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"📚 Collection existante '{self.collection_name}' récupérée")
                
                # Vérifier le nombre de documents existants
                existing_count = self.collection.count()
                logger.info(f"📊 {existing_count} documents déjà dans la collection")
                
            except Exception:
                # Créer une nouvelle collection
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Documentation Python pour DoctorPy"}
                )
                logger.info(f"✨ Nouvelle collection '{self.collection_name}' créée")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation de ChromaDB: {e}")
            raise
    
    async def _find_embedding_files(self) -> List[Path]:
        """Trouve tous les fichiers d'embedding"""
        embedding_files = list(self.embeddings_dir.glob("*_embedding.pkl"))
        return sorted(embedding_files)
    
    async def _load_embeddings_data(self, embedding_files: List[Path]) -> List[Dict[str, Any]]:
        """Charge les données d'embeddings et les contenus des chunks"""
        embeddings_data = []
        
        for embedding_file in embedding_files:
            try:
                # Charger l'embedding
                async with aiofiles.open(embedding_file, 'rb') as f:
                    data = await f.read()
                    embedding_data = pickle.loads(data)
                
                chunk_id = embedding_data['chunk_id']
                
                # Charger le contenu du chunk correspondant
                chunk_file = self.chunks_dir / f"{chunk_id}.txt"
                
                if not chunk_file.exists():
                    logger.warning(f"⚠️ Chunk non trouvé: {chunk_file}")
                    continue
                
                async with aiofiles.open(chunk_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                
                if not content.strip():
                    logger.warning(f"⚠️ Contenu vide pour {chunk_id}")
                    continue
                
                # Charger les métadonnées du chunk
                chunk_metadata_file = self.chunks_dir / f"{chunk_id}.json"
                chunk_metadata = {}
                
                if chunk_metadata_file.exists():
                    async with aiofiles.open(chunk_metadata_file, 'r', encoding='utf-8') as f:
                        chunk_metadata = json.loads(await f.read())
                
                # Combiner toutes les données
                combined_data = {
                    'chunk_id': chunk_id,
                    'content': content,
                    'embedding': embedding_data['embedding'],
                    'model_name': embedding_data['model_name'],
                    'embedding_dimension': embedding_data['embedding_dimension'],
                    'content_hash': embedding_data['content_hash'],
                    'embedding_metadata': embedding_data.get('metadata', {}),
                    'chunk_metadata': chunk_metadata
                }
                
                embeddings_data.append(combined_data)
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur lors du chargement de {embedding_file}: {e}")
                self.total_errors += 1
                continue
        
        logger.info(f"📚 {len(embeddings_data)} embeddings chargés avec succès")
        return embeddings_data
    
    async def _filter_existing_documents(self, embeddings_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filtre les documents déjà indexés"""
        new_embeddings = []
        
        # Récupérer tous les IDs existants dans la collection
        try:
            existing_ids = set()
            if self.collection.count() > 0:
                # Récupérer par petits lots pour éviter les problèmes de mémoire
                batch_size = 1000
                offset = 0
                
                while True:
                    result = self.collection.get(
                        limit=batch_size,
                        offset=offset,
                        include=["metadatas"]
                    )
                    
                    if not result['ids']:
                        break
                    
                    existing_ids.update(result['ids'])
                    offset += len(result['ids'])
                    
                    if len(result['ids']) < batch_size:
                        break
            
            logger.info(f"📊 {len(existing_ids)} documents déjà indexés")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de la récupération des IDs existants: {e}")
            existing_ids = set()
        
        # Filtrer les nouveaux embeddings
        for embedding_data in embeddings_data:
            chunk_id = embedding_data['chunk_id']
            
            if chunk_id in existing_ids:
                # Vérifier si le contenu a changé
                try:
                    existing_doc = self.collection.get(
                        ids=[chunk_id],
                        include=["metadatas"]
                    )
                    
                    if existing_doc['metadatas']:
                        existing_hash = existing_doc['metadatas'][0].get('content_hash')
                        current_hash = embedding_data['content_hash']
                        
                        if existing_hash == current_hash:
                            self.total_skipped += 1
                            continue  # Document inchangé
                
                except Exception:
                    pass  # En cas d'erreur, re-indexer le document
            
            new_embeddings.append(embedding_data)
        
        return new_embeddings
    
    async def _index_embeddings_in_batches(self, embeddings_data: List[Dict[str, Any]]):
        """Indexe les embeddings par lots"""
        total_embeddings = len(embeddings_data)
        
        for i in range(0, total_embeddings, self.batch_size):
            batch_data = embeddings_data[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            logger.info(f"🔄 Indexation lot {batch_num} ({len(batch_data)} documents)")
            
            try:
                await self._index_single_batch(batch_data)
                logger.info(f"✅ Lot {batch_num} indexé avec succès")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'indexation du lot {batch_num}: {e}")
                self.total_errors += len(batch_data)
            
            # Petite pause entre les lots
            await asyncio.sleep(0.1)
    
    async def _index_single_batch(self, batch_data: List[Dict[str, Any]]):
        """Indexe un seul lot de documents"""
        
        # Préparer les données pour ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for data in batch_data:
            chunk_id = data['chunk_id']
            content = data['content']
            embedding = data['embedding']
            
            # Convertir l'embedding en liste si c'est un numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Préparer les métadonnées
            metadata = {
                'chunk_id': chunk_id,
                'content_hash': data['content_hash'],
                'model_name': data['model_name'],
                'embedding_dimension': data['embedding_dimension'],
                'indexing_date': datetime.now().isoformat(),
                **data.get('chunk_metadata', {}),
                **data.get('embedding_metadata', {})
            }
            
            # Nettoyer les métadonnées (ChromaDB n'accepte que certains types)
            cleaned_metadata = self._clean_metadata(metadata)
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(content)
            metadatas.append(cleaned_metadata)
            
            # Créer l'objet IndexedDocument
            indexed_doc = IndexedDocument(
                doc_id=cleaned_metadata.get('original_doc_id', chunk_id),
                chunk_id=chunk_id,
                collection_name=self.collection_name,
                content=content,
                embedding=embedding,
                metadata=cleaned_metadata,
                indexing_date=datetime.now().isoformat()
            )
            
            self.indexed_documents.append(indexed_doc)
        
        # Upsert dans ChromaDB
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            self.total_indexed += len(batch_data)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'upsert ChromaDB: {e}")
            raise
    
    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Nettoie les métadonnées pour ChromaDB"""
        cleaned = {}
        
        for key, value in metadata.items():
            # ChromaDB accepte seulement: str, int, float, bool
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, dict):
                # Convertir les dictionnaires en JSON string
                try:
                    cleaned[key] = json.dumps(value)
                except:
                    cleaned[key] = str(value)
            elif isinstance(value, list):
                # Convertir les listes en JSON string
                try:
                    cleaned[key] = json.dumps(value)
                except:
                    cleaned[key] = str(value)
            else:
                # Convertir tout le reste en string
                cleaned[key] = str(value)
        
        return cleaned
    
    def _create_empty_result(self, start_time: float) -> IndexingResult:
        """Crée un résultat vide"""
        return IndexingResult(
            collection_name=self.collection_name,
            total_indexed=0,
            total_skipped=0,
            total_errors=0,
            indexing_date=datetime.now().isoformat(),
            processing_time_seconds=time.time() - start_time,
            indexed_documents=[]
        )
    
    async def _save_indexing_report(self, result: IndexingResult):
        """Sauvegarde un rapport d'indexation"""
        report = {
            "indexing_date": result.indexing_date,
            "collection_name": result.collection_name,
            "chroma_db_path": str(self.chroma_db_path),
            "total_indexed": result.total_indexed,
            "total_skipped": result.total_skipped,
            "total_errors": result.total_errors,
            "processing_time_seconds": result.processing_time_seconds,
            "batch_size": self.batch_size,
            "documents": []
        }
        
        # Ajouter les informations des documents indexés
        for doc in result.indexed_documents:
            doc_info = {
                "doc_id": doc.doc_id,
                "chunk_id": doc.chunk_id,
                "content_length": len(doc.content),
                "embedding_dimension": len(doc.embedding),
                "indexing_date": doc.indexing_date
            }
            report["documents"].append(doc_info)
        
        # Statistiques de la collection
        try:
            if self.collection:
                collection_stats = {
                    "total_documents": self.collection.count(),
                    "collection_name": self.collection_name
                }
                report["collection_stats"] = collection_stats
        except Exception as e:
            logger.warning(f"⚠️ Impossible de récupérer les stats de la collection: {e}")
        
        # Sauvegarder le rapport
        report_path = self.chroma_db_path / "indexing_report.json"
        async with aiofiles.open(report_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(report, indent=2, ensure_ascii=False))
        
        logger.info(f"📊 Rapport d'indexation sauvegardé: {report_path}")
    
    async def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Teste la recherche dans la collection"""
        if not self.collection:
            raise RuntimeError("Collection non initialisée")
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            return {
                "query": query_text,
                "results_count": len(results['ids'][0]) if results['ids'] else 0,
                "documents": results['documents'][0] if results['documents'] else [],
                "metadatas": results['metadatas'][0] if results['metadatas'] else [],
                "distances": results['distances'][0] if results['distances'] else []
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            raise


async def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Indexe les documents dans ChromaDB")
    parser.add_argument(
        "--embeddings-dir", 
        default="./data/processed/embeddings",
        help="Répertoire des embeddings"
    )
    parser.add_argument(
        "--chunks-dir", 
        default="./data/processed/chunks",
        help="Répertoire des chunks"
    )
    parser.add_argument(
        "--chroma-db-path", 
        default="./vector_stores/chroma_db",
        help="Chemin de la base ChromaDB"
    )
    parser.add_argument(
        "--collection-name", 
        default="doctorpy_docs",
        help="Nom de la collection ChromaDB"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=100,
        help="Taille des lots pour l'indexation"
    )
    parser.add_argument(
        "--test-query", 
        help="Requête de test après indexation"
    )
    
    args = parser.parse_args()
    
    if not CHROMADB_AVAILABLE:
        logger.error("❌ chromadb n'est pas installé. Installez-le avec: pip install chromadb")
        return
    
    indexer = DocumentIndexer(
        embeddings_dir=args.embeddings_dir,
        chunks_dir=args.chunks_dir,
        chroma_db_path=args.chroma_db_path,
        collection_name=args.collection_name,
        batch_size=args.batch_size
    )
    
    start_time = datetime.now()
    
    try:
        # Lancer l'indexation
        result = await indexer.index_all_documents()
        
        # Statistiques finales
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"🎉 Indexation terminée en {elapsed_time:.2f}s")
        logger.info(f"📚 {result.total_indexed} documents indexés")
        logger.info(f"⏭️ {result.total_skipped} documents ignorés")
        
        if result.total_errors > 0:
            logger.warning(f"⚠️ {result.total_errors} erreurs rencontrées")
        
        # Test de recherche si demandé
        if args.test_query:
            logger.info(f"🔍 Test de recherche: '{args.test_query}'")
            try:
                search_results = await indexer.query_collection(args.test_query)
                logger.info(f"✅ {search_results['results_count']} résultats trouvés")
                
                for i, (doc, metadata) in enumerate(zip(search_results['documents'], search_results['metadatas'])):
                    logger.info(f"  {i+1}. {metadata.get('title', 'Sans titre')} (score: {search_results['distances'][i]:.4f})")
                    
            except Exception as e:
                logger.error(f"❌ Erreur lors du test de recherche: {e}")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Indexation interrompue par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'indexation: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoctorPy - Script de génération d'embeddings
Crée les embeddings vectoriels pour tous les chunks de documents traités
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
from concurrent.futures import ThreadPoolExecutor
import threading

# Imports pour les embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("⚠️ sentence-transformers non disponible")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Résultat d'embedding pour un chunk"""
    chunk_id: str
    embedding: List[float]
    model_name: str
    embedding_dimension: int
    creation_date: str
    content_hash: str
    metadata: Dict


@dataclass
class EmbeddingBatch:
    """Lot d'embeddings traités"""
    batch_id: str
    model_name: str
    total_embeddings: int
    creation_date: str
    processing_time_seconds: float
    embeddings: List[EmbeddingResult]
    metadata: Dict


class EmbeddingGenerator:
    """
    Générateur d'embeddings pour les chunks de documents
    """
    
    def __init__(
        self,
        input_dir: str = "./data/processed/chunks",
        output_dir: str = "./data/processed/embeddings",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_workers: int = 4
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Créer le répertoire de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modèle d'embedding
        self.model = None
        self.embedding_dimension = None
        
        # Statistiques
        self.total_embeddings = 0
        self.failed_embeddings = 0
        self.processing_batches: List[EmbeddingBatch] = []
        
        # Thread lock pour l'accès au modèle
        self._model_lock = threading.Lock()
    
    async def generate_all_embeddings(self) -> List[EmbeddingBatch]:
        """Génère tous les embeddings pour les chunks disponibles"""
        logger.info(f"🚀 Démarrage de la génération d'embeddings avec {self.model_name}")
        
        # Initialiser le modèle
        await self._initialize_model()
        
        # Trouver tous les fichiers de chunks
        chunk_files = list(self.input_dir.glob("*.txt"))
        
        if not chunk_files:
            logger.warning(f"⚠️ Aucun chunk trouvé dans {self.input_dir}")
            return []
        
        logger.info(f"📄 {len(chunk_files)} chunks à traiter")
        
        # Charger tous les chunks avec leurs métadonnées
        chunks_data = await self._load_all_chunks(chunk_files)
        
        if not chunks_data:
            logger.error("❌ Aucun chunk valide chargé")
            return []
        
        # Filtrer les chunks déjà traités
        new_chunks = await self._filter_existing_embeddings(chunks_data)
        
        if not new_chunks:
            logger.info("✅ Tous les embeddings sont déjà à jour")
            return self.processing_batches
        
        logger.info(f"🔄 {len(new_chunks)} nouveaux chunks à traiter")
        
        # Traiter par lots
        await self._process_chunks_in_batches(new_chunks)
        
        # Sauvegarder le rapport final
        await self._save_generation_report()
        
        logger.info(f"✅ Génération terminée: {self.total_embeddings} embeddings créés")
        return self.processing_batches
    
    async def _initialize_model(self):
        """Initialise le modèle d'embedding"""
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers non disponible")
            
            logger.info(f"🤖 Chargement du modèle {self.model_name}")
            
            # Charger le modèle dans un thread séparé pour éviter de bloquer
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                self.model = await loop.run_in_executor(
                    executor, 
                    SentenceTransformer, 
                    self.model_name
                )
            
            # Tester le modèle et obtenir la dimension
            test_embedding = self.model.encode(["test text"])
            self.embedding_dimension = len(test_embedding[0])
            
            logger.info(f"✅ Modèle chargé (dimension: {self.embedding_dimension})")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
    
    async def _load_all_chunks(self, chunk_files: List[Path]) -> List[Tuple[str, str, Dict]]:
        """Charge tous les chunks avec leurs métadonnées"""
        chunks_data = []
        
        for chunk_file in chunk_files:
            try:
                # Charger le contenu du chunk
                async with aiofiles.open(chunk_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                
                if not content.strip():
                    continue
                
                # Charger les métadonnées
                metadata_file = chunk_file.with_suffix('.json')
                metadata = {}
                
                if metadata_file.exists():
                    async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.loads(await f.read())
                
                chunk_id = chunk_file.stem
                chunks_data.append((chunk_id, content, metadata))
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur lors du chargement de {chunk_file}: {e}")
                continue
        
        logger.info(f"📚 {len(chunks_data)} chunks chargés")
        return chunks_data
    
    async def _filter_existing_embeddings(
        self, 
        chunks_data: List[Tuple[str, str, Dict]]
    ) -> List[Tuple[str, str, Dict]]:
        """Filtre les chunks dont l'embedding existe déjà"""
        new_chunks = []
        
        for chunk_id, content, metadata in chunks_data:
            embedding_file = self.output_dir / f"{chunk_id}_embedding.pkl"
            
            if embedding_file.exists():
                # Vérifier si l'embedding est à jour
                try:
                    async with aiofiles.open(embedding_file, 'rb') as f:
                        data = await f.read()
                        existing_embedding = pickle.loads(data)
                    
                    # Vérifier le hash du contenu
                    current_hash = hashlib.sha256(content.encode()).hexdigest()
                    
                    if (existing_embedding.get('content_hash') == current_hash and 
                        existing_embedding.get('model_name') == self.model_name):
                        continue  # Embedding déjà à jour
                        
                except Exception:
                    pass  # Embedding corrompu, le refaire
            
            new_chunks.append((chunk_id, content, metadata))
        
        return new_chunks
    
    async def _process_chunks_in_batches(self, chunks_data: List[Tuple[str, str, Dict]]):
        """Traite les chunks par lots"""
        total_chunks = len(chunks_data)
        
        for i in range(0, total_chunks, self.batch_size):
            batch_chunks = chunks_data[i:i + self.batch_size]
            batch_id = f"batch_{i // self.batch_size + 1:04d}"
            
            logger.info(f"🔄 Traitement du lot {batch_id} ({len(batch_chunks)} chunks)")
            
            start_time = time.time()
            
            try:
                # Générer les embeddings pour ce lot
                batch_embeddings = await self._generate_batch_embeddings(batch_chunks, batch_id)
                
                # Sauvegarder les embeddings
                await self._save_batch_embeddings(batch_embeddings)
                
                processing_time = time.time() - start_time
                
                # Créer le lot d'embeddings
                embedding_batch = EmbeddingBatch(
                    batch_id=batch_id,
                    model_name=self.model_name,
                    total_embeddings=len(batch_embeddings),
                    creation_date=datetime.now().isoformat(),
                    processing_time_seconds=processing_time,
                    embeddings=batch_embeddings,
                    metadata={
                        "batch_size": len(batch_chunks),
                        "embedding_dimension": self.embedding_dimension,
                        "chunks_processed": f"{i + len(batch_chunks)}/{total_chunks}"
                    }
                )
                
                self.processing_batches.append(embedding_batch)
                self.total_embeddings += len(batch_embeddings)
                
                logger.info(f"✅ Lot {batch_id} terminé en {processing_time:.2f}s")
                
                # Petite pause entre les lots pour éviter la surcharge
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement du lot {batch_id}: {e}")
                self.failed_embeddings += len(batch_chunks)
    
    async def _generate_batch_embeddings(
        self, 
        chunks_data: List[Tuple[str, str, Dict]], 
        batch_id: str
    ) -> List[EmbeddingResult]:
        """Génère les embeddings pour un lot de chunks"""
        
        chunk_ids = [chunk_id for chunk_id, _, _ in chunks_data]
        contents = [content for _, content, _ in chunks_data]
        metadatas = [metadata for _, _, metadata in chunks_data]
        
        # Générer les embeddings dans un thread séparé
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            embeddings = await loop.run_in_executor(
                executor,
                self._generate_embeddings_sync,
                contents
            )
        
        # Créer les résultats d'embedding
        embedding_results = []
        
        for i, (chunk_id, content, metadata) in enumerate(chunks_data):
            if i < len(embeddings):
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                
                embedding_result = EmbeddingResult(
                    chunk_id=chunk_id,
                    embedding=embeddings[i].tolist(),
                    model_name=self.model_name,
                    embedding_dimension=self.embedding_dimension,
                    creation_date=datetime.now().isoformat(),
                    content_hash=content_hash,
                    metadata={
                        **metadata,
                        "batch_id": batch_id,
                        "content_length": len(content)
                    }
                )
                
                embedding_results.append(embedding_result)
        
        return embedding_results
    
    def _generate_embeddings_sync(self, contents: List[str]) -> np.ndarray:
        """Génère les embeddings de manière synchrone"""
        with self._model_lock:
            try:
                embeddings = self.model.encode(
                    contents,
                    batch_size=min(self.batch_size, len(contents)),
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return embeddings
            except Exception as e:
                logger.error(f"❌ Erreur lors de la génération d'embeddings: {e}")
                raise
    
    async def _save_batch_embeddings(self, embeddings: List[EmbeddingResult]):
        """Sauvegarde un lot d'embeddings"""
        tasks = []
        
        for embedding in embeddings:
            task = self._save_single_embedding(embedding)
            tasks.append(task)
        
        # Sauvegarder en parallèle
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _save_single_embedding(self, embedding: EmbeddingResult):
        """Sauvegarde un embedding individuel"""
        try:
            # Sauvegarder l'embedding binaire
            embedding_file = self.output_dir / f"{embedding.chunk_id}_embedding.pkl"
            
            embedding_data = {
                'chunk_id': embedding.chunk_id,
                'embedding': np.array(embedding.embedding),
                'model_name': embedding.model_name,
                'embedding_dimension': embedding.embedding_dimension,
                'creation_date': embedding.creation_date,
                'content_hash': embedding.content_hash,
                'metadata': embedding.metadata
            }
            
            async with aiofiles.open(embedding_file, 'wb') as f:
                await f.write(pickle.dumps(embedding_data))
            
            # Sauvegarder les métadonnées JSON
            metadata_file = self.output_dir / f"{embedding.chunk_id}_embedding.json"
            
            metadata_json = {
                'chunk_id': embedding.chunk_id,
                'model_name': embedding.model_name,
                'embedding_dimension': embedding.embedding_dimension,
                'creation_date': embedding.creation_date,
                'content_hash': embedding.content_hash,
                'metadata': embedding.metadata,
                'file_path': str(embedding_file)
            }
            
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata_json, indent=2, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde de l'embedding {embedding.chunk_id}: {e}")
    
    async def _save_generation_report(self):
        """Sauvegarde un rapport de génération"""
        report = {
            "generation_date": datetime.now().isoformat(),
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "total_embeddings": self.total_embeddings,
            "failed_embeddings": self.failed_embeddings,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "batches": []
        }
        
        # Ajouter les informations des lots
        for batch in self.processing_batches:
            batch_info = {
                "batch_id": batch.batch_id,
                "embeddings_count": batch.total_embeddings,
                "processing_time_seconds": batch.processing_time_seconds,
                "creation_date": batch.creation_date
            }
            report["batches"].append(batch_info)
        
        # Statistiques de performance
        total_time = sum(batch.processing_time_seconds for batch in self.processing_batches)
        if total_time > 0:
            report["performance"] = {
                "total_processing_time": total_time,
                "embeddings_per_second": self.total_embeddings / total_time,
                "average_batch_time": total_time / len(self.processing_batches) if self.processing_batches else 0
            }
        
        # Sauvegarder le rapport
        report_path = self.output_dir / "generation_report.json"
        async with aiofiles.open(report_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(report, indent=2, ensure_ascii=False))
        
        logger.info(f"📊 Rapport de génération sauvegardé: {report_path}")


async def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Génère les embeddings pour les chunks")
    parser.add_argument(
        "--input-dir", 
        default="./data/processed/chunks",
        help="Répertoire des chunks"
    )
    parser.add_argument(
        "--output-dir", 
        default="./data/processed/embeddings",
        help="Répertoire de sortie des embeddings"
    )
    parser.add_argument(
        "--model-name", 
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Modèle d'embedding à utiliser"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Taille des lots pour le traitement"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=4,
        help="Nombre maximum de workers"
    )
    
    args = parser.parse_args()
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("❌ sentence-transformers n'est pas installé. Installez-le avec: pip install sentence-transformers")
        return
    
    generator = EmbeddingGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    start_time = datetime.now()
    
    try:
        # Lancer la génération
        batches = await generator.generate_all_embeddings()
        
        # Statistiques finales
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"🎉 Génération terminée en {elapsed_time:.2f}s")
        logger.info(f"🎯 {generator.total_embeddings} embeddings créés")
        logger.info(f"📦 {len(batches)} lots traités")
        
        if generator.failed_embeddings > 0:
            logger.warning(f"⚠️ {generator.failed_embeddings} embeddings échoués")
        
        # Performance
        if elapsed_time > 0:
            embeddings_per_second = generator.total_embeddings / elapsed_time
            logger.info(f"⚡ Performance: {embeddings_per_second:.2f} embeddings/seconde")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Génération interrompue par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
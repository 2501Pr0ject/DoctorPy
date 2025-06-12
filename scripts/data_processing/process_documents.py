#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoctorPy - Script de traitement des documents
Nettoie, structure et prépare les documents pour l'indexation
"""

import asyncio
import aiofiles
import json
import logging
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import tiktoken
from markdown import markdown
from bs4 import BeautifulSoup

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """Chunk de document traité"""
    chunk_id: str
    original_doc_id: str
    content: str
    title: str
    section: str
    subsection: Optional[str]
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict
    file_path: str


@dataclass
class ProcessedDocument:
    """Document traité complet"""
    doc_id: str
    original_path: str
    title: str
    section: str
    subsection: Optional[str]
    original_content: str
    cleaned_content: str
    chunks: List[ProcessedChunk]
    total_tokens: int
    processing_date: str
    metadata: Dict


class DocumentProcessor:
    """
    Processeur de documents pour préparer les données au système RAG
    """
    
    def __init__(
        self, 
        input_dir: str = "./data/raw/documentation",
        output_dir: str = "./data/processed",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Créer les répertoires de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "chunks").mkdir(exist_ok=True)
        (self.output_dir / "documents").mkdir(exist_ok=True)
        
        # Initialiser l'encodeur de tokens
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception:
            logger.warning("⚠️ Impossible de charger l'encodeur tiktoken, utilisation d'une approximation")
            self.encoding = None
        
        # Statistiques de traitement
        self.processed_documents: List[ProcessedDocument] = []
        self.total_chunks = 0
        self.skipped_files = 0
    
    async def process_all_documents(self) -> List[ProcessedDocument]:
        """Traite tous les documents dans le répertoire d'entrée"""
        logger.info(f"🚀 Démarrage du traitement des documents depuis {self.input_dir}")
        
        # Trouver tous les fichiers Markdown
        markdown_files = list(self.input_dir.rglob("*.md"))
        
        if not markdown_files:
            logger.warning(f"⚠️ Aucun fichier Markdown trouvé dans {self.input_dir}")
            return []
        
        logger.info(f"📄 {len(markdown_files)} fichiers trouvés")
        
        # Traiter chaque fichier
        for file_path in markdown_files:
            try:
                processed_doc = await self._process_single_document(file_path)
                if processed_doc:
                    self.processed_documents.append(processed_doc)
                    await self._save_processed_document(processed_doc)
                else:
                    self.skipped_files += 1
                    
            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement de {file_path}: {e}")
                self.skipped_files += 1
        
        # Sauvegarder le rapport de traitement
        await self._save_processing_report()
        
        logger.info(f"✅ Traitement terminé: {len(self.processed_documents)} documents, {self.total_chunks} chunks")
        return self.processed_documents
    
    async def _process_single_document(self, file_path: Path) -> Optional[ProcessedDocument]:
        """Traite un seul document"""
        try:
            # Lire le contenu du fichier
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if not content.strip():
                logger.warning(f"⚠️ Fichier vide: {file_path}")
                return None
            
            # Extraire les métadonnées YAML
            metadata, clean_content = self._extract_frontmatter(content)
            
            if not clean_content.strip():
                logger.warning(f"⚠️ Contenu vide après extraction des métadonnées: {file_path}")
                return None
            
            # Générer un ID unique pour le document
            doc_id = self._generate_doc_id(file_path, clean_content)
            
            # Nettoyer le contenu
            cleaned_content = self._clean_content(clean_content)
            
            # Créer les chunks
            chunks = await self._create_chunks(
                doc_id, 
                cleaned_content, 
                metadata.get('title', file_path.stem),
                metadata.get('section', 'unknown'),
                metadata.get('subsection'),
                file_path
            )
            
            if not chunks:
                logger.warning(f"⚠️ Aucun chunk créé pour: {file_path}")
                return None
            
            # Calculer le nombre total de tokens
            total_tokens = sum(chunk.token_count for chunk in chunks)
            
            # Créer le document traité
            processed_doc = ProcessedDocument(
                doc_id=doc_id,
                original_path=str(file_path),
                title=metadata.get('title', file_path.stem),
                section=metadata.get('section', 'unknown'),
                subsection=metadata.get('subsection'),
                original_content=content,
                cleaned_content=cleaned_content,
                chunks=chunks,
                total_tokens=total_tokens,
                processing_date=datetime.now().isoformat(),
                metadata={
                    **metadata,
                    'file_size': len(content),
                    'chunk_count': len(chunks),
                    'processing_version': '1.0'
                }
            )
            
            self.total_chunks += len(chunks)
            logger.info(f"✅ Traité: {processed_doc.title} ({len(chunks)} chunks, {total_tokens} tokens)")
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de {file_path}: {e}")
            return None
    
    def _extract_frontmatter(self, content: str) -> Tuple[Dict, str]:
        """Extrait les métadonnées YAML du début du fichier"""
        metadata = {}
        
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    metadata = yaml.safe_load(parts[1]) or {}
                    content = parts[2].strip()
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lors de l'extraction des métadonnées: {e}")
                    content = content.split('---', 2)[-1].strip()
        
        return metadata, content
    
    def _generate_doc_id(self, file_path: Path, content: str) -> str:
        """Génère un ID unique pour le document"""
        # Utiliser le chemin relatif et un hash du contenu
        relative_path = file_path.relative_to(self.input_dir)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        
        # Créer un ID lisible
        path_parts = str(relative_path).replace('/', '_').replace('.md', '')
        return f"{path_parts}_{content_hash}"
    
    def _clean_content(self, content: str) -> str:
        """Nettoie le contenu du document"""
        
        # Convertir le Markdown en texte brut si nécessaire
        if self._contains_markdown(content):
            content = self._markdown_to_text(content)
        
        # Nettoyer les espaces et caractères indésirables
        content = self._normalize_whitespace(content)
        
        # Supprimer les liens externes non pertinents
        content = self._clean_links(content)
        
        # Nettoyer les blocs de code
        content = self._clean_code_blocks(content)
        
        return content.strip()
    
    def _contains_markdown(self, content: str) -> bool:
        """Vérifie si le contenu contient du Markdown"""
        markdown_indicators = [
            r'#{1,6}\s',  # Headers
            r'\*\*.*?\*\*',  # Bold
            r'\*.*?\*',  # Italic
            r'```',  # Code blocks
            r'\[.*?\]\(.*?\)',  # Links
        ]
        
        for pattern in markdown_indicators:
            if re.search(pattern, content):
                return True
        return False
    
    def _markdown_to_text(self, content: str) -> str:
        """Convertit le Markdown en texte brut"""
        try:
            # Convertir en HTML puis en texte
            html = markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extraire le texte en préservant la structure
            text_parts = []
            
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'pre', 'code']):
                if element.name.startswith('h'):
                    text_parts.append(f"\n{element.get_text().strip()}\n")
                elif element.name == 'pre':
                    text_parts.append(f"\n{element.get_text().strip()}\n")
                elif element.name == 'code' and element.parent.name != 'pre':
                    text_parts.append(element.get_text().strip())
                else:
                    text = element.get_text().strip()
                    if text:
                        text_parts.append(text)
            
            return ' '.join(text_parts)
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de la conversion Markdown: {e}")
            return content
    
    def _normalize_whitespace(self, content: str) -> str:
        """Normalise les espaces dans le contenu"""
        # Remplacer les espaces multiples par des espaces simples
        content = re.sub(r'\s+', ' ', content)
        
        # Normaliser les sauts de ligne
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Supprimer les espaces en début et fin de ligne
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(lines)
        
        return content
    
    def _clean_links(self, content: str) -> str:
        """Nettoie les liens dans le contenu"""
        # Remplacer les liens Markdown par le texte du lien
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # Supprimer les URLs orphelines
        content = re.sub(r'https?://[^\s]+', '', content)
        
        return content
    
    def _clean_code_blocks(self, content: str) -> str:
        """Nettoie les blocs de code"""
        # Conserver les blocs de code mais les marquer clairement
        content = re.sub(r'```(\w+)?\n(.*?)\n```', r'Code:\n\2\n', content, flags=re.DOTALL)
        
        # Nettoyer les codes inline
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        return content
    
    async def _create_chunks(
        self, 
        doc_id: str, 
        content: str, 
        title: str, 
        section: str, 
        subsection: Optional[str],
        original_file: Path
    ) -> List[ProcessedChunk]:
        """Crée des chunks à partir du contenu"""
        chunks = []
        
        # Diviser par paragraphes d'abord
        paragraphs = self._split_into_paragraphs(content)
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        start_char = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_tokens = self._count_tokens(paragraph)
            
            # Si le paragraphe est trop long, le diviser
            if paragraph_tokens > self.chunk_size:
                # Sauvegarder le chunk actuel s'il existe
                if current_chunk.strip():
                    chunk = await self._create_single_chunk(
                        doc_id, current_chunk, title, section, subsection,
                        chunk_index, start_char, start_char + len(current_chunk),
                        original_file
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Diviser le paragraphe long
                sub_chunks = self._split_long_paragraph(paragraph)
                for sub_chunk in sub_chunks:
                    chunk = await self._create_single_chunk(
                        doc_id, sub_chunk, title, section, subsection,
                        chunk_index, start_char, start_char + len(sub_chunk),
                        original_file
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    start_char += len(sub_chunk)
                
                current_chunk = ""
                current_tokens = 0
                
            elif current_tokens + paragraph_tokens > self.chunk_size:
                # Sauvegarder le chunk actuel
                if current_chunk.strip():
                    chunk = await self._create_single_chunk(
                        doc_id, current_chunk, title, section, subsection,
                        chunk_index, start_char, start_char + len(current_chunk),
                        original_file
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Commencer un nouveau chunk avec overlap
                if chunks and self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + "\n\n" + paragraph
                    start_char = start_char + len(current_chunk) - len(overlap_text) - len(paragraph) - 2
                else:
                    current_chunk = paragraph
                    start_char += len(current_chunk)
                
                current_tokens = self._count_tokens(current_chunk)
                
            else:
                # Ajouter au chunk actuel
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
        
        # Sauvegarder le dernier chunk
        if current_chunk.strip():
            chunk = await self._create_single_chunk(
                doc_id, current_chunk, title, section, subsection,
                chunk_index, start_char, start_char + len(current_chunk),
                original_file
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """Divise le contenu en paragraphes"""
        # Diviser par double saut de ligne
        paragraphs = content.split('\n\n')
        
        # Nettoyer et filtrer
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 10:  # Ignorer les paragraphes trop courts
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Divise un paragraphe trop long en sous-chunks"""
        # Diviser par phrases d'abord
        sentences = re.split(r'[.!?]+', paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence += "."  # Remettre la ponctuation
            
            if self._count_tokens(current_chunk + sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_chars: int) -> str:
        """Récupère le texte de chevauchement"""
        if len(text) <= overlap_chars:
            return text
        
        # Prendre les derniers caractères en essayant de couper à un mot
        overlap = text[-overlap_chars:]
        
        # Trouver le premier espace pour éviter de couper un mot
        first_space = overlap.find(' ')
        if first_space > 0:
            overlap = overlap[first_space + 1:]
        
        return overlap
    
    async def _create_single_chunk(
        self, 
        doc_id: str, 
        content: str, 
        title: str, 
        section: str, 
        subsection: Optional[str],
        chunk_index: int, 
        start_char: int, 
        end_char: int,
        original_file: Path
    ) -> ProcessedChunk:
        """Crée un seul chunk"""
        
        # Générer un ID pour le chunk
        chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"
        
        # Compter les tokens
        token_count = self._count_tokens(content)
        
        # Créer le chemin de fichier pour le chunk
        chunk_file = self.output_dir / "chunks" / f"{chunk_id}.txt"
        
        # Métadonnées du chunk
        metadata = {
            "original_file": str(original_file),
            "chunk_method": "paragraph_based",
            "overlap_chars": self.chunk_overlap,
            "processing_date": datetime.now().isoformat()
        }
        
        chunk = ProcessedChunk(
            chunk_id=chunk_id,
            original_doc_id=doc_id,
            content=content,
            title=title,
            section=section,
            subsection=subsection,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            token_count=token_count,
            metadata=metadata,
            file_path=str(chunk_file)
        )
        
        # Sauvegarder le chunk
        await self._save_chunk(chunk)
        
        return chunk
    
    def _count_tokens(self, text: str) -> int:
        """Compte le nombre de tokens dans le texte"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                pass
        
        # Approximation simple si tiktoken n'est pas disponible
        return len(text.split()) * 1.3  # Approximation grossière
    
    async def _save_chunk(self, chunk: ProcessedChunk):
        """Sauvegarde un chunk sur le disque"""
        try:
            # Sauvegarder le contenu du chunk
            async with aiofiles.open(chunk.file_path, 'w', encoding='utf-8') as f:
                await f.write(chunk.content)
            
            # Sauvegarder les métadonnées du chunk
            metadata_path = Path(chunk.file_path).with_suffix('.json')
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(asdict(chunk), indent=2, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde du chunk {chunk.chunk_id}: {e}")
    
    async def _save_processed_document(self, doc: ProcessedDocument):
        """Sauvegarde un document traité"""
        try:
            doc_file = self.output_dir / "documents" / f"{doc.doc_id}.json"
            
            async with aiofiles.open(doc_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(asdict(doc), indent=2, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde du document {doc.doc_id}: {e}")
    
    async def _save_processing_report(self):
        """Sauvegarde un rapport du traitement"""
        report = {
            "processing_date": datetime.now().isoformat(),
            "total_documents": len(self.processed_documents),
            "total_chunks": self.total_chunks,
            "skipped_files": self.skipped_files,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "sections": {},
            "documents": []
        }
        
        # Statistiques par section
        for doc in self.processed_documents:
            section = doc.section
            if section not in report["sections"]:
                report["sections"][section] = {
                    "document_count": 0,
                    "chunk_count": 0,
                    "total_tokens": 0
                }
            
            report["sections"][section]["document_count"] += 1
            report["sections"][section]["chunk_count"] += len(doc.chunks)
            report["sections"][section]["total_tokens"] += doc.total_tokens
            
            # Ajouter le document au rapport
            report["documents"].append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "section": doc.section,
                "subsection": doc.subsection,
                "chunk_count": len(doc.chunks),
                "total_tokens": doc.total_tokens
            })
        
        # Sauvegarder le rapport
        report_path = self.output_dir / "processing_report.json"
        async with aiofiles.open(report_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(report, indent=2, ensure_ascii=False))
        
        logger.info(f"📊 Rapport de traitement sauvegardé: {report_path}")


async def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Traite les documents pour le RAG")
    parser.add_argument(
        "--input-dir", 
        default="./data/raw/documentation",
        help="Répertoire d'entrée"
    )
    parser.add_argument(
        "--output-dir", 
        default="./data/processed",
        help="Répertoire de sortie"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000,
        help="Taille des chunks en tokens"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=200,
        help="Chevauchement entre chunks en caractères"
    )
    
    args = parser.parse_args()
    
    processor = DocumentProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    start_time = datetime.now()
    
    try:
        # Lancer le traitement
        documents = await processor.process_all_documents()
        
        # Statistiques finales
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"🎉 Traitement terminé en {elapsed_time:.2f}s")
        logger.info(f"📚 {len(documents)} documents traités")
        logger.info(f"🧩 {processor.total_chunks} chunks créés")
        
        if processor.skipped_files > 0:
            logger.warning(f"⚠️ {processor.skipped_files} fichiers ignorés")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Traitement interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
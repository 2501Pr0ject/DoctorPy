"""
Utilitaires pour le traitement de texte
"""

import re
import string
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from collections import Counter
import tiktoken

class TextProcessor:
    """Classe pour le traitement avancé de texte"""
    
    def __init__(self, language: str = "french"):
        self.language = language
        self.stemmer = SnowballStemmer(language)
        
        # Télécharger les ressources NLTK nécessaires
        self._download_nltk_resources()
        
        # Charger le modèle spaCy si disponible
        try:
            if language == "french":
                self.nlp = spacy.load("fr_core_news_sm")
            else:
                self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(f"Modèle spaCy pour {language} non installé. Fonctionnalités limitées.")
            self.nlp = None
        
        # Mots vides
        try:
            if language == "french":
                self.stop_words = set(stopwords.words('french'))
            else:
                self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()
        
        # Encodeur pour compter les tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _download_nltk_resources(self):
        """Télécharge les ressources NLTK nécessaires"""
        resources = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger', 
            'maxent_ne_chunker', 'words'
        ]
        
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
    
    def clean_text(self, text: str, remove_accents: bool = False) -> str:
        """
        Nettoie un texte en supprimant les caractères indésirables
        
        Args:
            text: Texte à nettoyer
            remove_accents: Supprimer les accents
            
        Returns:
            Texte nettoyé
        """
        if not text:
            return ""
        
        # Supprimer les caractères de contrôle
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les accents si demandé
        if remove_accents:
            text = unicodedata.normalize('NFD', text)
            text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extrait les phrases d'un texte
        
        Args:
            text: Texte source
            
        Returns:
            Liste des phrases
        """
        if not text:
            return []
        
        # Nettoyer le texte
        clean_text = self.clean_text(text)
        
        # Segmentation en phrases
        sentences = sent_tokenize(clean_text, language=self.language)
        
        # Filtrer les phrases trop courtes
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def tokenize(self, text: str, remove_punctuation: bool = True, 
                remove_stopwords: bool = True, lowercase: bool = True) -> List[str]:
        """
        Tokenise un texte
        
        Args:
            text: Texte à tokeniser
            remove_punctuation: Supprimer la ponctuation
            remove_stopwords: Supprimer les mots vides
            lowercase: Convertir en minuscules
            
        Returns:
            Liste des tokens
        """
        if not text:
            return []
        
        # Convertir en minuscules
        if lowercase:
            text = text.lower()
        
        # Tokenisation
        tokens = word_tokenize(text, language=self.language)
        
        # Supprimer la ponctuation
        if remove_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]
        
        # Supprimer les mots vides
        if remove_stopwords and self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Filtrer les tokens trop courts
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Tuple[str, int]]:
        """
        Extrait les mots-clés d'un texte
        
        Args:
            text: Texte source
            max_keywords: Nombre maximum de mots-clés
            
        Returns:
            Liste des mots-clés avec leur fréquence
        """
        tokens = self.tokenize(text)
        
        # Compter les fréquences
        word_freq = Counter(tokens)
        
        # Retourner les plus fréquents
        return word_freq.most_common(max_keywords)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrait les entités nommées du texte
        
        Args:
            text: Texte source
            
        Returns:
            Dictionnaire des entités par type
        """
        entities = {
            'PERSON': [],
            'ORGANIZATION': [],
            'LOCATION': [],
            'MISCELLANEOUS': []
        }
        
        if self.nlp:
            # Utiliser spaCy si disponible
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PER', 'PERSON']:
                    entities['PERSON'].append(ent.text)
                elif ent.label_ in ['ORG', 'ORGANIZATION']:
                    entities['ORGANIZATION'].append(ent.text)
                elif ent.label_ in ['LOC', 'LOCATION', 'GPE']:
                    entities['LOCATION'].append(ent.text)
                else:
                    entities['MISCELLANEOUS'].append(ent.text)
        else:
            # Fallback avec NLTK
            try:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join([token for token, pos in chunk.leaves()])
                        label = chunk.label()
                        
                        if label in entities:
                            entities[label].append(entity_text)
                        else:
                            entities['MISCELLANEOUS'].append(entity_text)
            except:
                pass
        
        # Supprimer les doublons
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def stem_text(self, text: str) -> str:
        """
        Applique la racinisation au texte
        
        Args:
            text: Texte source
            
        Returns:
            Texte avec mots racinisés
        """
        tokens = self.tokenize(text, remove_punctuation=False, remove_stopwords=False)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)
    
    def count_tokens(self, text: str) -> int:
        """
        Compte le nombre de tokens pour les modèles LLM
        
        Args:
            text: Texte à analyser
            
        Returns:
            Nombre de tokens
        """
        try:
            return len(self.encoding.encode(text))
        except:
            # Fallback : estimation approximative
            return len(text.split()) * 1.3
    
    def truncate_text(self, text: str, max_tokens: int, preserve_sentences: bool = True) -> str:
        """
        Tronque un texte pour respecter une limite de tokens
        
        Args:
            text: Texte source
            max_tokens: Nombre maximum de tokens
            preserve_sentences: Préserver les phrases complètes
            
        Returns:
            Texte tronqué
        """
        if self.count_tokens(text) <= max_tokens:
            return text
        
        if preserve_sentences:
            sentences = self.extract_sentences(text)
            result = ""
            
            for sentence in sentences:
                test_text = result + " " + sentence if result else sentence
                if self.count_tokens(test_text) <= max_tokens:
                    result = test_text
                else:
                    break
            
            return result
        else:
            # Tronquer mot par mot
            words = text.split()
            result = ""
            
            for word in words:
                test_text = result + " " + word if result else word
                if self.count_tokens(test_text) <= max_tokens:
                    result = test_text
                else:
                    break
            
            return result
    
    def split_into_chunks(self, text: str, chunk_size: int = 1000, 
                         overlap: int = 200) -> List[Dict[str, any]]:
        """
        Divise un texte en chunks pour le RAG
        
        Args:
            text: Texte source
            chunk_size: Taille des chunks en tokens
            overlap: Chevauchement entre chunks
            
        Returns:
            Liste des chunks avec métadonnées
        """
        sentences = self.extract_sentences(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # Si la phrase est trop longue, la diviser
            if sentence_tokens > chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'tokens': current_tokens,
                        'start_sentence': len(chunks) * 10,  # Approximation
                        'end_sentence': i
                    })
                    current_chunk = ""
                    current_tokens = 0
                
                # Diviser la phrase longue
                words = sentence.split()
                sub_chunk = ""
                sub_tokens = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word)
                    if sub_tokens + word_tokens <= chunk_size:
                        sub_chunk += " " + word if sub_chunk else word
                        sub_tokens += word_tokens
                    else:
                        if sub_chunk:
                            chunks.append({
                                'text': sub_chunk.strip(),
                                'tokens': sub_tokens,
                                'start_sentence': i,
                                'end_sentence': i
                            })
                        sub_chunk = word
                        sub_tokens = word_tokens
                
                if sub_chunk:
                    current_chunk = sub_chunk
                    current_tokens = sub_tokens
            
            # Ajouter la phrase au chunk courant
            elif current_tokens + sentence_tokens <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
            else:
                # Sauvegarder le chunk courant
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'tokens': current_tokens,
                        'start_sentence': max(0, len(chunks) * 10 - overlap // 50),
                        'end_sentence': i - 1
                    })
                
                # Commencer un nouveau chunk avec chevauchement
                if overlap > 0 and chunks:
                    # Prendre les dernières phrases pour le chevauchement
                    overlap_sentences = sentences[max(0, i - overlap // 100):i]
                    overlap_text = " ".join(overlap_sentences)
                    overlap_tokens = self.count_tokens(overlap_text)
                    
                    if overlap_tokens < chunk_size:
                        current_chunk = overlap_text + " " + sentence
                        current_tokens = overlap_tokens + sentence_tokens
                    else:
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
        
        # Ajouter le dernier chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                'start_sentence': max(0, len(sentences) - 10),
                'end_sentence': len(sentences) - 1
            })
        
        return chunks
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité entre deux textes (Jaccard)
        
        Args:
            text1: Premier texte
            text2: Deuxième texte
            
        Returns:
            Score de similarité entre 0 et 1
        """
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 and not tokens2:
            return 1.0
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def extract_programming_concepts(self, text: str) -> Dict[str, List[str]]:
        """
        Extrait les concepts de programmation d'un texte
        
        Args:
            text: Texte source
            
        Returns:
            Dictionnaire des concepts par catégorie
        """
        concepts = {
            'data_types': [],
            'control_structures': [],
            'functions': [],
            'classes': [],
            'modules': [],
            'keywords': []
        }
        
        # Patterns pour différents concepts
        patterns = {
            'data_types': r'\b(int|float|str|string|list|dict|tuple|set|bool|boolean)\b',
            'control_structures': r'\b(if|else|elif|for|while|loop|break|continue|try|except|finally)\b',
            'functions': r'\b(def|function|return|lambda|yield|generator)\b',
            'classes': r'\b(class|object|inheritance|polymorphism|encapsulation)\b',
            'modules': r'\b(import|from|module|package|library)\b',
            'keywords': r'\b(print|input|len|range|enumerate|zip|map|filter|sorted)\b'
        }
        
        text_lower = text.lower()
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text_lower)
            concepts[category] = list(set(matches))
        
        return concepts


def preprocess_code_text(code: str) -> str:
    """
    Prétraite le code pour l'analyse
    
    Args:
        code: Code source
        
    Returns:
        Code nettoyé
    """
    # Supprimer les commentaires Python
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    
    # Supprimer les docstrings
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    
    # Normaliser les espaces
    code = re.sub(r'\s+', ' ', code)
    
    return code.strip()


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extrait les blocs de code d'un texte markdown
    
    Args:
        text: Texte source
        
    Returns:
        Liste des blocs de code avec métadonnées
    """
    # Pattern pour les blocs de code avec langue
    pattern = r'```(\w+)?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    code_blocks = []
    for language, code in matches:
        code_blocks.append({
            'language': language or 'text',
            'code': code.strip(),
            'length': len(code.strip().split('\n'))
        })
    
    return code_blocks


def format_text_for_llm(text: str, max_tokens: int = 3000) -> str:
    """
    Formate un texte pour un modèle LLM
    
    Args:
        text: Texte source
        max_tokens: Limite de tokens
        
    Returns:
        Texte formaté
    """
    processor = TextProcessor()
    
    # Nettoyer le texte
    clean_text = processor.clean_text(text)
    
    # Tronquer si nécessaire
    truncated_text = processor.truncate_text(clean_text, max_tokens)
    
    return truncated_text


def analyze_text_complexity(text: str) -> Dict[str, any]:
    """
    Analyse la complexité d'un texte
    
    Args:
        text: Texte à analyser
        
    Returns:
        Métriques de complexité
    """
    processor = TextProcessor()
    
    sentences = processor.extract_sentences(text)
    tokens = processor.tokenize(text, remove_punctuation=False, remove_stopwords=False)
    
    # Métriques de base
    metrics = {
        'num_sentences': len(sentences),
        'num_words': len(tokens),
        'num_characters': len(text),
        'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
        'num_tokens': processor.count_tokens(text)
    }
    
    # Complexité lexicale
    unique_words = set(processor.tokenize(text))
    metrics['lexical_diversity'] = len(unique_words) / len(tokens) if tokens else 0
    
    # Entités et concepts
    entities = processor.extract_entities(text)
    metrics['num_entities'] = sum(len(ent_list) for ent_list in entities.values())
    
    programming_concepts = processor.extract_programming_concepts(text)
    metrics['num_programming_concepts'] = sum(len(concept_list) for concept_list in programming_concepts.values())
    
    return metrics
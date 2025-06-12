# src/utils/helpers.py
"""
Fonctions utilitaires génériques pour l'assistant pédagogique
"""

import re
import json
import hashlib
import secrets
import string
import unicodedata
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import functools
import time
import uuid
from collections import defaultdict, Counter
import random

logger = logging.getLogger(__name__)

# ========================
# Utilitaires de chaînes
# ========================

def slugify(text: str, max_length: int = 50) -> str:
    """
    Convertit un texte en slug URL-safe
    
    Args:
        text: Texte à convertir
        max_length: Longueur maximale
        
    Returns:
        Slug généré
    """
    if not text:
        return ""
    
    # Normaliser les caractères Unicode
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Convertir en minuscules et remplacer espaces/caractères spéciaux
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '-', text)
    
    # Limiter la longueur
    if len(text) > max_length:
        text = text[:max_length].rstrip('-')
    
    return text.strip('-')


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Tronque un texte intelligemment
    
    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        suffix: Suffixe à ajouter
        
    Returns:
        Texte tronqué
    """
    if not text or len(text) <= max_length:
        return text
    
    # Essayer de couper sur un espace
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.6:  # Si l'espace n'est pas trop loin
        truncated = truncated[:last_space]
    
    return truncated + suffix


def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
    """
    Extrait les mots-clés d'un texte
    
    Args:
        text: Texte source
        min_length: Longueur minimale des mots
        max_keywords: Nombre maximum de mots-clés
        
    Returns:
        Liste des mots-clés
    """
    if not text:
        return []
    
    # Nettoyer et tokeniser
    words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text.lower())
    
    # Filtrer par longueur
    words = [word for word in words if len(word) >= min_length]
    
    # Compter les fréquences
    word_freq = Counter(words)
    
    # Retourner les plus fréquents
    return [word for word, _ in word_freq.most_common(max_keywords)]


def clean_whitespace(text: str) -> str:
    """
    Nettoie les espaces d'un texte
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte nettoyé
    """
    if not text:
        return ""
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Nettoyer les sauts de ligne multiples
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def format_code_snippet(code: str, language: str = "python") -> str:
    """
    Formate un snippet de code pour l'affichage
    
    Args:
        code: Code à formater
        language: Langage de programmation
        
    Returns:
        Code formaté
    """
    if not code:
        return ""
    
    # Indenter uniformément
    lines = code.split('\n')
    if lines:
        # Supprimer l'indentation commune
        min_indent = min(len(line) - len(line.lstrip()) 
                        for line in lines if line.strip())
        
        if min_indent > 0:
            lines = [line[min_indent:] if line.strip() else line 
                    for line in lines]
    
    return '\n'.join(lines)

# ========================
# Utilitaires de temps
# ========================

def get_current_timestamp() -> str:
    """Retourne un timestamp ISO formaté"""
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse un timestamp ISO
    
    Args:
        timestamp_str: Timestamp en format ISO
        
    Returns:
        Objet datetime
    """
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        # Fallback pour différents formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Format de timestamp non reconnu: {timestamp_str}")


def format_duration(seconds: int) -> str:
    """
    Formate une durée en texte lisible
    
    Args:
        seconds: Durée en secondes
        
    Returns:
        Durée formatée
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        return f"{minutes}m"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        if remaining_minutes > 0:
            return f"{hours}h {remaining_minutes}m"
        return f"{hours}h"


def time_ago(timestamp: datetime) -> str:
    """
    Calcule le temps écoulé depuis un timestamp
    
    Args:
        timestamp: Timestamp de référence
        
    Returns:
        Texte décrivant le temps écoulé
    """
    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    diff = now - timestamp
    seconds = int(diff.total_seconds())
    
    if seconds < 60:
        return "à l'instant"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"il y a {minutes} minute{'s' if minutes > 1 else ''}"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"il y a {hours} heure{'s' if hours > 1 else ''}"
    elif seconds < 2592000:  # 30 jours
        days = seconds // 86400
        return f"il y a {days} jour{'s' if days > 1 else ''}"
    elif seconds < 31536000:  # 365 jours
        months = seconds // 2592000
        return f"il y a {months} mois"
    else:
        years = seconds // 31536000
        return f"il y a {years} an{'s' if years > 1 else ''}"


# ========================
# Utilitaires de sécurité
# ========================

def generate_secure_token(length: int = 32) -> str:
    """
    Génère un token sécurisé
    
    Args:
        length: Longueur du token
        
    Returns:
        Token hexadécimal
    """
    return secrets.token_hex(length)


def generate_password(length: int = 12, include_symbols: bool = True) -> str:
    """
    Génère un mot de passe sécurisé
    
    Args:
        length: Longueur du mot de passe
        include_symbols: Inclure des symboles
        
    Returns:
        Mot de passe généré
    """
    characters = string.ascii_letters + string.digits
    if include_symbols:
        characters += "!@#$%^&*"
    
    # Assurer au moins un caractère de chaque type
    password = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits)
    ]
    
    if include_symbols:
        password.append(secrets.choice("!@#$%^&*"))
    
    # Compléter avec des caractères aléatoires
    for _ in range(length - len(password)):
        password.append(secrets.choice(characters))
    
    # Mélanger
    random.shuffle(password)
    return ''.join(password)


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    Hash un texte
    
    Args:
        text: Texte à hasher
        algorithm: Algorithme de hash
        
    Returns:
        Hash hexadécimal
    """
    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Algorithme non supporté: {algorithm}")
    
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def generate_uuid() -> str:
    """Génère un UUID unique"""
    return str(uuid.uuid4())


def mask_sensitive_data(text: str, patterns: List[str] = None) -> str:
    """
    Masque les données sensibles dans un texte
    
    Args:
        text: Texte à traiter
        patterns: Patterns regex à masquer
        
    Returns:
        Texte avec données masquées
    """
    if not text:
        return text
    
    if patterns is None:
        patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Cartes de crédit
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',  # SSN US
        ]
    
    masked_text = text
    for pattern in patterns:
        masked_text = re.sub(pattern, "***", masked_text)
    
    return masked_text


# ========================
# Utilitaires de données
# ========================

def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionne profondément deux dictionnaires
    
    Args:
        dict1: Premier dictionnaire
        dict2: Deuxième dictionnaire
        
    Returns:
        Dictionnaire fusionné
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(nested_dict: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Aplatit un dictionnaire imbriqué
    
    Args:
        nested_dict: Dictionnaire imbriqué
        separator: Séparateur pour les clés
        
    Returns:
        Dictionnaire aplati
    """
    def _flatten(obj: Any, parent_key: str = '') -> Dict[str, Any]:
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                items.extend(_flatten(value, new_key).items())
        else:
            return {parent_key: obj}
        
        return dict(items)
    
    return _flatten(nested_dict)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Divise une liste en chunks
    
    Args:
        lst: Liste à diviser
        chunk_size: Taille des chunks
        
    Returns:
        Liste de chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def remove_duplicates(lst: List[Any], key: Optional[Callable] = None) -> List[Any]:
    """
    Supprime les doublons d'une liste
    
    Args:
        lst: Liste source
        key: Fonction pour extraire la clé de comparaison
        
    Returns:
        Liste sans doublons
    """
    if not lst:
        return []
    
    if key is None:
        # Préserver l'ordre avec dict.fromkeys()
        return list(dict.fromkeys(lst))
    else:
        seen = set()
        result = []
        for item in lst:
            item_key = key(item)
            if item_key not in seen:
                seen.add(item_key)
                result.append(item)
        return result


def group_by(lst: List[Any], key: Callable) -> Dict[Any, List[Any]]:
    """
    Groupe une liste par clé
    
    Args:
        lst: Liste à grouper
        key: Fonction pour extraire la clé de groupement
        
    Returns:
        Dictionnaire groupé
    """
    groups = defaultdict(list)
    for item in lst:
        groups[key(item)].append(item)
    return dict(groups)


def safe_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Accès sécurisé aux données imbriquées
    
    Args:
        data: Dictionnaire source
        path: Chemin vers la valeur (ex: "user.profile.name")
        default: Valeur par défaut
        
    Returns:
        Valeur trouvée ou valeur par défaut
    """
    keys = path.split('.')
    current = data
    
    try:
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    except (KeyError, TypeError):
        return default


# ========================
# Utilitaires de validation
# ========================

def is_valid_json(text: str) -> bool:
    """Vérifie si un texte est un JSON valide"""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def is_valid_email(email: str) -> bool:
    """Vérifie si une adresse email est valide"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """Vérifie si une URL est valide"""
    pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?
    return bool(re.match(pattern, url))


def is_strong_password(password: str) -> Tuple[bool, List[str]]:
    """
    Vérifie la force d'un mot de passe
    
    Args:
        password: Mot de passe à vérifier
        
    Returns:
        Tuple (est_fort, liste_des_problèmes)
    """
    issues = []
    
    if len(password) < 8:
        issues.append("Moins de 8 caractères")
    
    if not re.search(r'[a-z]', password):
        issues.append("Aucune minuscule")
    
    if not re.search(r'[A-Z]', password):
        issues.append("Aucune majuscule")
    
    if not re.search(r'\d', password):
        issues.append("Aucun chiffre")
    
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{};:"\\|,.<>\?]', password):
        issues.append("Aucun caractère spécial")
    
    return len(issues) == 0, issues


# ========================
# Décorateurs utilitaires
# ========================

def retry(max_attempts: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """
    Décorateur pour réessayer une fonction en cas d'échec
    
    Args:
        max_attempts: Nombre maximum de tentatives
        delay: Délai initial entre les tentatives
        exponential_backoff: Utiliser un backoff exponentiel
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Échec final de {func.__name__} après {max_attempts} tentatives: {e}")
                        raise
                    
                    logger.warning(f"Tentative {attempt + 1} échouée pour {func.__name__}: {e}")
                    time.sleep(current_delay)
                    
                    if exponential_backoff:
                        current_delay *= 2
            
            return None  # Ne devrait jamais arriver
        
        return wrapper
    return decorator


def measure_time(func):
    """Décorateur pour mesurer le temps d'exécution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} exécuté en {execution_time:.4f}s")
        
        return result
    
    return wrapper


def cache_result(ttl_seconds: int = 300):
    """
    Décorateur pour mettre en cache le résultat d'une fonction
    
    Args:
        ttl_seconds: Durée de vie du cache en secondes
    """
    cache = {}
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Créer une clé de cache
            cache_key = hash_text(f"{func.__name__}:{str(args)}:{str(kwargs)}")
            current_time = time.time()
            
            # Vérifier si le résultat est en cache et valide
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp < ttl_seconds:
                    return result
            
            # Exécuter la fonction et mettre en cache
            result = func(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            
            # Nettoyer les entrées expirées
            expired_keys = [
                key for key, (_, timestamp) in cache.items()
                if current_time - timestamp >= ttl_seconds
            ]
            for key in expired_keys:
                del cache[key]
            
            return result
        
        return wrapper
    return decorator


# ========================
# Utilitaires de formatage
# ========================

def format_number(number: Union[int, float], decimal_places: int = 2) -> str:
    """
    Formate un nombre avec des séparateurs
    
    Args:
        number: Nombre à formater
        decimal_places: Nombre de décimales
        
    Returns:
        Nombre formaté
    """
    if isinstance(number, int):
        return f"{number:,}".replace(',', ' ')
    else:
        return f"{number:,.{decimal_places}f}".replace(',', ' ')


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Formate un pourcentage
    
    Args:
        value: Valeur entre 0 et 1
        decimal_places: Nombre de décimales
        
    Returns:
        Pourcentage formaté
    """
    percentage = value * 100
    return f"{percentage:.{decimal_places}f}%"


def format_file_size(size_bytes: int) -> str:
    """
    Formate une taille de fichier
    
    Args:
        size_bytes: Taille en bytes
        
    Returns:
        Taille formatée
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024
        i += 1
    
    if i == 0:
        return f"{int(size)} {size_names[i]}"
    else:
        return f"{size:.1f} {size_names[i]}"


def format_code_for_display(code: str, max_lines: int = 20) -> str:
    """
    Formate du code pour l'affichage
    
    Args:
        code: Code source
        max_lines: Nombre maximum de lignes à afficher
        
    Returns:
        Code formaté
    """
    if not code:
        return ""
    
    lines = code.split('\n')
    
    # Limiter le nombre de lignes
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} lignes supplémentaires)"]
    
    # Numéroter les lignes
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        if not line.startswith("..."):
            numbered_lines.append(f"{i:3d} | {line}")
        else:
            numbered_lines.append(f"    | {line}")
    
    return '\n'.join(numbered_lines)


# ========================
# Utilitaires de recherche
# ========================

def fuzzy_search(query: str, candidates: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
    """
    Recherche floue dans une liste de candidats
    
    Args:
        query: Requête de recherche
        candidates: Liste des candidats
        threshold: Seuil de similarité
        
    Returns:
        Liste des correspondances avec score
    """
    def similarity(a: str, b: str) -> float:
        """Calcule la similarité de Jaccard entre deux chaînes"""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        
        if not a_words and not b_words:
            return 1.0
        if not a_words or not b_words:
            return 0.0
        
        intersection = a_words.intersection(b_words)
        union = a_words.union(b_words)
        
        return len(intersection) / len(union)
    
    results = []
    query_lower = query.lower()
    
    for candidate in candidates:
        candidate_lower = candidate.lower()
        
        # Correspondance exacte
        if query_lower == candidate_lower:
            score = 1.0
        # Correspondance de substring
        elif query_lower in candidate_lower:
            score = len(query_lower) / len(candidate_lower)
        # Similarité de mots
        else:
            score = similarity(query, candidate)
        
        if score >= threshold:
            results.append((candidate, score))
    
    # Trier par score décroissant
    return sorted(results, key=lambda x: x[1], reverse=True)


def highlight_text(text: str, query: str, tag: str = "mark") -> str:
    """
    Surligne les occurrences d'une requête dans un texte
    
    Args:
        text: Texte source
        query: Terme à surligner
        tag: Balise HTML pour le surlignage
        
    Returns:
        Texte avec surlignage
    """
    if not query or not text:
        return text
    
    # Échapper les caractères spéciaux regex
    escaped_query = re.escape(query)
    
    # Surligner (insensible à la casse)
    pattern = re.compile(escaped_query, re.IGNORECASE)
    highlighted = pattern.sub(f'<{tag}>\\g<0></{tag}>', text)
    
    return highlighted


# ========================
# Utilitaires de configuration
# ========================

def load_config_from_env(prefix: str = "", default_values: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Charge la configuration depuis les variables d'environnement
    
    Args:
        prefix: Préfixe des variables d'environnement
        default_values: Valeurs par défaut
        
    Returns:
        Configuration chargée
    """
    import os
    
    config = default_values.copy() if default_values else {}
    
    for key, value in os.environ.items():
        if prefix and not key.startswith(prefix):
            continue
        
        # Enlever le préfixe et convertir en minuscules
        config_key = key[len(prefix):].lower() if prefix else key.lower()
        
        # Essayer de convertir la valeur
        if value.lower() in ('true', 'false'):
            config[config_key] = value.lower() == 'true'
        elif value.isdigit():
            config[config_key] = int(value)
        elif value.replace('.', '').isdigit():
            config[config_key] = float(value)
        else:
            config[config_key] = value
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionne plusieurs configurations
    
    Args:
        configs: Configurations à fusionner
        
    Returns:
        Configuration fusionnée
    """
    result = {}
    
    for config in configs:
        if config:
            result = deep_merge(result, config)
    
    return result


# ========================
# Utilitaires de debug
# ========================

def pretty_print_dict(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Affichage formaté d'un dictionnaire
    
    Args:
        data: Dictionnaire à afficher
        indent: Niveau d'indentation
        
    Returns:
        Représentation formatée
    """
    return json.dumps(data, indent=indent, ensure_ascii=False, default=str)


def debug_function_call(func_name: str, args: tuple, kwargs: Dict[str, Any]) -> str:
    """
    Génère une représentation debug d'un appel de fonction
    
    Args:
        func_name: Nom de la fonction
        args: Arguments positionnels
        kwargs: Arguments nommés
        
    Returns:
        Représentation de l'appel
    """
    parts = [func_name, "("]
    
    # Arguments positionnels
    if args:
        parts.extend([", ".join(repr(arg) for arg in args)])
    
    # Arguments nommés
    if kwargs:
        if args:
            parts.append(", ")
        parts.append(", ".join(f"{k}={repr(v)}" for k, v in kwargs.items()))
    
    parts.append(")")
    return "".join(parts)


def get_object_size(obj: Any) -> int:
    """
    Calcule la taille approximative d'un objet en mémoire
    
    Args:
        obj: Objet à mesurer
        
    Returns:
        Taille en bytes
    """
    import sys
    
    size = sys.getsizeof(obj)
    
    # Pour les conteneurs, ajouter la taille des éléments
    if isinstance(obj, dict):
        size += sum(get_object_size(k) + get_object_size(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_object_size(item) for item in obj)
    
    return size
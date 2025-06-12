# src/utils/__init__.py
"""
Module utilitaires pour l'assistant pédagogique IA

Ce module contient des fonctions et classes utilitaires pour :
- Le traitement de texte et l'analyse linguistique
- La gestion et validation des fichiers
- La validation de données et de code
- Les fonctions d'aide génériques

Usage:
    from src.utils import TextProcessor, FileHandler, CodeValidator
    from src.utils.helpers import slugify, format_duration
"""

from .text_processing import (
    TextProcessor,
    preprocess_code_text,
    extract_code_blocks,
    format_text_for_llm,
    analyze_text_complexity
)

from .file_utils import (
    FileHandler,
    safe_filename,
    get_file_encoding,
    format_file_size,
    create_temp_file
)

from .validation import (
    ValidationResult,
    CodeValidator,
    DataValidator,
    QuestValidator,
    UserInputValidator,
    ConfigValidator,
    validate_file_upload,
    sanitize_input,
    validate_quiz_answer
)

from .helpers import (
    # Utilitaires de chaînes
    slugify,
    truncate_text,
    extract_keywords,
    clean_whitespace,
    format_code_snippet,
    
    # Utilitaires de temps
    get_current_timestamp,
    parse_timestamp,
    format_duration,
    time_ago,
    
    # Utilitaires de sécurité
    generate_secure_token,
    generate_password,
    hash_text,
    generate_uuid,
    mask_sensitive_data,
    
    # Utilitaires de données
    deep_merge,
    flatten_dict,
    chunk_list,
    remove_duplicates,
    group_by,
    safe_get,
    
    # Utilitaires de validation
    is_valid_json,
    is_valid_email,
    is_valid_url,
    is_strong_password,
    
    # Décorateurs
    retry,
    measure_time,
    cache_result,
    
    # Utilitaires de formatage
    format_number,
    format_percentage,
    format_code_for_display,
    
    # Utilitaires de recherche
    fuzzy_search,
    highlight_text,
    
    # Utilitaires de configuration
    load_config_from_env,
    merge_configs,
    
    # Utilitaires de debug
    pretty_print_dict,
    debug_function_call,
    get_object_size
)

# Version du module
__version__ = "1.0.0"

# Exports principaux
__all__ = [
    # Classes principales
    "TextProcessor",
    "FileHandler", 
    "ValidationResult",
    "CodeValidator",
    "DataValidator",
    "QuestValidator",
    "UserInputValidator",
    "ConfigValidator",
    
    # Fonctions de traitement de texte
    "preprocess_code_text",
    "extract_code_blocks", 
    "format_text_for_llm",
    "analyze_text_complexity",
    
    # Fonctions de gestion de fichiers
    "safe_filename",
    "get_file_encoding",
    "create_temp_file",
    
    # Fonctions de validation
    "validate_file_upload",
    "sanitize_input",
    "validate_quiz_answer",
    
    # Fonctions utilitaires courantes
    "slugify",
    "truncate_text",
    "extract_keywords",
    "clean_whitespace",
    "format_code_snippet",
    "get_current_timestamp",
    "parse_timestamp",
    "format_duration",
    "time_ago",
    "generate_secure_token",
    "generate_password", 
    "hash_text",
    "generate_uuid",
    "mask_sensitive_data",
    "deep_merge",
    "flatten_dict",
    "chunk_list",
    "remove_duplicates",
    "group_by",
    "safe_get",
    "is_valid_json",
    "is_valid_email",
    "is_valid_url",
    "is_strong_password",
    "retry",
    "measure_time",
    "cache_result",
    "format_number",
    "format_percentage",
    "format_file_size",
    "format_code_for_display",
    "fuzzy_search",
    "highlight_text",
    "load_config_from_env",
    "merge_configs",
    "pretty_print_dict",
    "debug_function_call",
    "get_object_size"
]

# Configuration par défaut pour le module
DEFAULT_CONFIG = {
    "text_processing": {
        "language": "french",
        "max_chunk_size": 1000,
        "chunk_overlap": 200,
        "min_keyword_length": 3,
        "max_keywords": 10
    },
    "file_handling": {
        "max_file_size_mb": 50,
        "allowed_extensions": [".txt", ".md", ".pdf", ".docx", ".json", ".csv", ".yaml", ".yml"],
        "backup_retention_days": 30,
        "temp_file_cleanup": True
    },
    "validation": {
        "code_security_level": "strict",
        "password_min_length": 8,
        "username_min_length": 3,
        "max_text_length": 10000
    },
    "formatting": {
        "date_format": "%Y-%m-%d %H:%M:%S",
        "number_decimal_places": 2,
        "truncate_suffix": "...",
        "max_display_lines": 20
    }
}

def get_utils_info():
    """
    Retourne des informations sur le module utils
    
    Returns:
        Dict avec les informations du module
    """
    return {
        "version": __version__,
        "modules": {
            "text_processing": "Traitement et analyse de texte",
            "file_utils": "Gestion et manipulation de fichiers", 
            "validation": "Validation de données et de code",
            "helpers": "Fonctions utilitaires génériques"
        },
        "main_classes": [
            "TextProcessor",
            "FileHandler",
            "CodeValidator", 
            "QuestValidator"
        ],
        "config": DEFAULT_CONFIG
    }

def configure_utils(config: dict = None):
    """
    Configure le module utils avec des paramètres personnalisés
    
    Args:
        config: Configuration personnalisée
    """
    if config:
        DEFAULT_CONFIG.update(config)
        print(f"Configuration utils mise à jour avec {len(config)} paramètres")

# Initialisation automatique si nécessaire
def _init_module():
    """Initialisation du module au chargement"""
    import logging
    
    # Configurer le logging pour le module utils
    logger = logging.getLogger(__name__)
    
    # Ne pas propager vers le root logger si déjà configuré
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    logger.debug(f"Module utils initialisé (version {__version__})")

# Exécuter l'initialisation
_init_module()
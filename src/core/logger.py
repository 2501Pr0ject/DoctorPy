import logging
import logging.handlers
from pathlib import Path
from .config import settings


def setup_logger(name: str = "python_learning_assistant") -> logging.Logger:
    """Configure et retourne un logger avec rotation des fichiers"""
    
    # Créer le répertoire de logs s'il n'existe pas
    log_dir = Path(settings.log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer le logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Éviter les doublons de handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler pour fichier avec rotation
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file_path,
        maxBytes=_parse_size(settings.log_max_size),
        backupCount=settings.log_backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Handler pour console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Ajouter les handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def _parse_size(size_str: str) -> int:
    """Parse une taille de fichier (ex: '10MB' -> 10485760)"""
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


# Logger global
logger = setup_logger()

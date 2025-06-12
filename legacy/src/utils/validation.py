# src/utils/validation.py
"""
Utilitaires de validation pour l'assistant pédagogique
"""

import re
import ast
import json
import yaml
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from datetime import datetime, date
from pathlib import Path
import validators
from pydantic import BaseModel, ValidationError, validator
import logging

logger = logging.getLogger(__name__)

class ValidationResult:
    """Résultat d'une validation"""
    
    def __init__(self, is_valid: bool = True, errors: List[str] = None, 
                 warnings: List[str] = None, data: Any = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.data = data
    
    def add_error(self, error: str):
        """Ajoute une erreur"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Ajoute un avertissement"""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'data': self.data
        }

class CodeValidator:
    """Validateur de code Python"""
    
    def __init__(self):
        # Mots-clés Python dangereux
        self.dangerous_keywords = {
            'exec', 'eval', 'compile', '__import__', 'open', 'file',
            'input', 'raw_input', 'reload', 'globals', 'locals',
            'vars', 'dir', 'delattr', 'setattr', 'getattr', 'hasattr'
        }
        
        # Modules potentiellement dangereux
        self.dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'tempfile',
            'socket', 'urllib', 'requests', 'http', 'ftplib',
            'pickle', 'cPickle', 'marshal', 'shelve'
        }
        
        # Modules autorisés pour l'apprentissage
        self.allowed_modules = {
            'math', 'random', 'datetime', 'json', 'csv',
            'collections', 'itertools', 'functools', 'operator',
            'string', 're', 'time', 'decimal', 'fractions',
            'statistics', 'numpy', 'pandas', 'matplotlib', 'seaborn'
        }
    
    def validate_syntax(self, code: str) -> ValidationResult:
        """
        Valide la syntaxe du code Python
        
        Args:
            code: Code à valider
            
        Returns:
            Résultat de validation
        """
        result = ValidationResult()
        
        if not code.strip():
            result.add_error("Code vide")
            return result
        
        try:
            # Parser le code avec AST
            tree = ast.parse(code)
            result.data = tree
            
        except SyntaxError as e:
            result.add_error(f"Erreur de syntaxe ligne {e.lineno}: {e.msg}")
        except Exception as e:
            result.add_error(f"Erreur lors de l'analyse: {str(e)}")
        
        return result
    
    def validate_security(self, code: str) -> ValidationResult:
        """
        Valide la sécurité du code
        
        Args:
            code: Code à valider
            
        Returns:
            Résultat de validation
        """
        result = ValidationResult()
        
        # Vérifier la syntaxe d'abord
        syntax_result = self.validate_syntax(code)
        if not syntax_result.is_valid:
            return syntax_result
        
        tree = syntax_result.data
        
        # Analyser l'AST pour détecter les constructions dangereuses
        for node in ast.walk(tree):
            
            # Vérifier les appels de fonction
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                if func_name in self.dangerous_keywords:
                    result.add_error(f"Fonction dangereuse détectée: {func_name}")
            
            # Vérifier les imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.dangerous_modules:
                        result.add_error(f"Import dangereux: {alias.name}")
                    elif alias.name not in self.allowed_modules:
                        result.add_warning(f"Module non recommandé: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.dangerous_modules:
                    result.add_error(f"Import dangereux: {node.module}")
                elif node.module and node.module not in self.allowed_modules:
                    result.add_warning(f"Module non recommandé: {node.module}")
            
            # Vérifier les accès aux attributs dangereux
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    result.add_warning(f"Accès à un attribut spécial: {node.attr}")
        
        return result
    
    def _get_function_name(self, func_node) -> str:
        """Extrait le nom d'une fonction depuis un nœud AST"""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr
        else:
            return ""
    
    def validate_style(self, code: str) -> ValidationResult:
        """
        Valide le style du code (PEP 8 basique)
        
        Args:
            code: Code à valider
            
        Returns:
            Résultat de validation
        """
        result = ValidationResult()
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Ligne trop longue
            if len(line) > 79:
                result.add_warning(f"Ligne {line_num} trop longue ({len(line)} caractères)")
            
            # Espaces en fin de ligne
            if line.endswith(' ') or line.endswith('\t'):
                result.add_warning(f"Espaces en fin de ligne {line_num}")
            
            # Mélange tabs/espaces
            if '\t' in line and '    ' in line:
                result.add_warning(f"Mélange tabs/espaces ligne {line_num}")
            
            # Conventions de nommage basiques
            if '=' in line and not line.strip().startswith('#'):
                # Variables en CamelCase (à éviter)
                var_match = re.search(r'(\w+)\s*=', line)
                if var_match:
                    var_name = var_match.group(1)
                    if re.match(r'^[A-Z][a-z]+[A-Z]', var_name):
                        result.add_warning(f"Variable en CamelCase ligne {line_num}: {var_name}")
        
        return result
    
    def validate_logic(self, code: str) -> ValidationResult:
        """
        Valide la logique du code (détections basiques)
        
        Args:
            code: Code à valider
            
        Returns:
            Résultat de validation
        """
        result = ValidationResult()
        
        # Vérifier la syntaxe d'abord
        syntax_result = self.validate_syntax(code)
        if not syntax_result.is_valid:
            return syntax_result
        
        tree = syntax_result.data
        
        # Compteurs pour l'analyse
        has_return = False
        loop_count = 0
        condition_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                has_return = True
            
            elif isinstance(node, (ast.For, ast.While)):
                loop_count += 1
                # Détecter les boucles infinies potentielles
                if isinstance(node, ast.While):
                    if isinstance(node.test, ast.Constant) and node.test.value is True:
                        result.add_warning("Boucle while True détectée (risque de boucle infinie)")
            
            elif isinstance(node, ast.If):
                condition_count += 1
            
            # Détecter les divisions par zéro potentielles
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    result.add_error("Division par zéro détectée")
        
        # Analyser la structure générale
        if loop_count > 5:
            result.add_warning(f"Beaucoup de boucles ({loop_count}), vérifiez la complexité")
        
        if condition_count > 10:
            result.add_warning(f"Beaucoup de conditions ({condition_count}), considérez refactoriser")
        
        return result
    
    def validate_complete(self, code: str) -> ValidationResult:
        """
        Validation complète du code
        
        Args:
            code: Code à valider
            
        Returns:
            Résultat de validation combiné
        """
        result = ValidationResult()
        
        # Validation syntaxique
        syntax_result = self.validate_syntax(code)
        result.errors.extend(syntax_result.errors)
        result.warnings.extend(syntax_result.warnings)
        
        if not syntax_result.is_valid:
            return result
        
        # Validation sécurité
        security_result = self.validate_security(code)
        result.errors.extend(security_result.errors)
        result.warnings.extend(security_result.warnings)
        
        # Validation style
        style_result = self.validate_style(code)
        result.warnings.extend(style_result.warnings)
        
        # Validation logique
        logic_result = self.validate_logic(code)
        result.errors.extend(logic_result.errors)
        result.warnings.extend(logic_result.warnings)
        
        # Déterminer si le résultat global est valide
        result.is_valid = len(result.errors) == 0
        result.data = {
            'syntax_tree': syntax_result.data,
            'total_errors': len(result.errors),
            'total_warnings': len(result.warnings)
        }
        
        return result


class DataValidator:
    """Validateur de données génériques"""
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        """Valide une adresse email"""
        result = ValidationResult()
        
        if not email:
            result.add_error("Email vide")
            return result
        
        if not validators.email(email):
            result.add_error("Format d'email invalide")
        
        return result
    
    @staticmethod
    def validate_url(url: str) -> ValidationResult:
        """Valide une URL"""
        result = ValidationResult()
        
        if not url:
            result.add_error("URL vide")
            return result
        
        if not validators.url(url):
            result.add_error("Format d'URL invalide")
        
        return result
    
    @staticmethod
    def validate_json(json_str: str) -> ValidationResult:
        """Valide un JSON"""
        result = ValidationResult()
        
        if not json_str.strip():
            result.add_error("JSON vide")
            return result
        
        try:
            parsed_json = json.loads(json_str)
            result.data = parsed_json
        except json.JSONDecodeError as e:
            result.add_error(f"JSON invalide: {e.msg} à la position {e.pos}")
        
        return result
    
    @staticmethod
    def validate_yaml(yaml_str: str) -> ValidationResult:
        """Valide un YAML"""
        result = ValidationResult()
        
        if not yaml_str.strip():
            result.add_error("YAML vide")
            return result
        
        try:
            parsed_yaml = yaml.safe_load(yaml_str)
            result.data = parsed_yaml
        except yaml.YAMLError as e:
            result.add_error(f"YAML invalide: {str(e)}")
        
        return result
    
    @staticmethod
    def validate_date(date_str: str, format_str: str = "%Y-%m-%d") -> ValidationResult:
        """Valide une date"""
        result = ValidationResult()
        
        if not date_str:
            result.add_error("Date vide")
            return result
        
        try:
            parsed_date = datetime.strptime(date_str, format_str)
            result.data = parsed_date
        except ValueError:
            result.add_error(f"Format de date invalide. Attendu: {format_str}")
        
        return result
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None,
                      max_val: Optional[Union[int, float]] = None) -> ValidationResult:
        """Valide qu'une valeur est dans une plage"""
        result = ValidationResult()
        
        if min_val is not None and value < min_val:
            result.add_error(f"Valeur {value} inférieure au minimum {min_val}")
        
        if max_val is not None and value > max_val:
            result.add_error(f"Valeur {value} supérieure au maximum {max_val}")
        
        result.data = value
        return result
    
    @staticmethod
    def validate_length(text: str, min_len: Optional[int] = None, 
                       max_len: Optional[int] = None) -> ValidationResult:
        """Valide la longueur d'un texte"""
        result = ValidationResult()
        
        if min_len is not None and len(text) < min_len:
            result.add_error(f"Texte trop court ({len(text)} caractères, minimum {min_len})")
        
        if max_len is not None and len(text) > max_len:
            result.add_error(f"Texte trop long ({len(text)} caractères, maximum {max_len})")
        
        result.data = text
        return result


class QuestValidator:
    """Validateur spécifique aux quêtes pédagogiques"""
    
    def __init__(self):
        self.required_fields = {
            'id', 'title', 'description', 'level', 'difficulty', 'objectives', 'steps'
        }
        
        self.valid_levels = {'beginner', 'intermediate', 'advanced'}
        self.valid_difficulties = {1, 2, 3, 4, 5}
    
    def validate_quest(self, quest_data: Dict[str, Any]) -> ValidationResult:
        """
        Valide une quête complète
        
        Args:
            quest_data: Données de la quête
            
        Returns:
            Résultat de validation
        """
        result = ValidationResult()
        
        # Vérifier les champs obligatoires
        missing_fields = self.required_fields - set(quest_data.keys())
        if missing_fields:
            result.add_error(f"Champs manquants: {', '.join(missing_fields)}")
            return result
        
        # Valider l'ID
        if not quest_data['id'] or not isinstance(quest_data['id'], str):
            result.add_error("ID de quête invalide")
        elif not re.match(r'^[a-z0-9_]+, quest_data['id']):
            result.add_error("ID doit contenir uniquement des lettres minuscules, chiffres et underscores")
        
        # Valider le titre
        if not quest_data['title'] or len(quest_data['title'].strip()) < 3:
            result.add_error("Titre trop court (minimum 3 caractères)")
        elif len(quest_data['title']) > 100:
            result.add_error("Titre trop long (maximum 100 caractères)")
        
        # Valider la description
        if not quest_data['description'] or len(quest_data['description'].strip()) < 10:
            result.add_error("Description trop courte (minimum 10 caractères)")
        elif len(quest_data['description']) > 1000:
            result.add_warning("Description très longue (plus de 1000 caractères)")
        
        # Valider le niveau
        if quest_data['level'] not in self.valid_levels:
            result.add_error(f"Niveau invalide. Valeurs autorisées: {', '.join(self.valid_levels)}")
        
        # Valider la difficulté
        if quest_data['difficulty'] not in self.valid_difficulties:
            result.add_error(f"Difficulté invalide. Valeurs autorisées: {', '.join(map(str, self.valid_difficulties))}")
        
        # Valider les objectifs
        objectives = quest_data.get('objectives', [])
        if not objectives or not isinstance(objectives, list):
            result.add_error("Les objectifs doivent être une liste non vide")
        elif len(objectives) < 1:
            result.add_error("Au moins un objectif est requis")
        elif len(objectives) > 10:
            result.add_warning("Beaucoup d'objectifs (plus de 10)")
        
        # Valider chaque objectif
        for i, objective in enumerate(objectives):
            if not objective or not isinstance(objective, str):
                result.add_error(f"Objectif {i+1} invalide")
            elif len(objective.strip()) < 5:
                result.add_error(f"Objectif {i+1} trop court")
        
        # Valider les étapes
        steps = quest_data.get('steps', [])
        if not steps or not isinstance(steps, list):
            result.add_error("Les étapes doivent être une liste non vide")
        elif len(steps) < 1:
            result.add_error("Au moins une étape est requise")
        
        # Valider chaque étape
        for i, step in enumerate(steps):
            step_result = self.validate_quest_step(step, i + 1)
            result.errors.extend(step_result.errors)
            result.warnings.extend(step_result.warnings)
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def validate_quest_step(self, step_data: Dict[str, Any], step_number: int) -> ValidationResult:
        """
        Valide une étape de quête
        
        Args:
            step_data: Données de l'étape
            step_number: Numéro de l'étape
            
        Returns:
            Résultat de validation
        """
        result = ValidationResult()
        
        required_step_fields = {'title', 'description'}
        
        # Vérifier les champs obligatoires
        missing_fields = required_step_fields - set(step_data.keys())
        if missing_fields:
            result.add_error(f"Étape {step_number} - Champs manquants: {', '.join(missing_fields)}")
            return result
        
        # Valider le titre de l'étape
        if not step_data['title'] or len(step_data['title'].strip()) < 3:
            result.add_error(f"Étape {step_number} - Titre trop court")
        
        # Valider la description de l'étape
        if not step_data['description'] or len(step_data['description'].strip()) < 10:
            result.add_error(f"Étape {step_number} - Description trop courte")
        
        # Valider le code si présent
        if 'code_template' in step_data:
            if step_data['code_template'] and len(step_data['code_template'].strip()) > 5000:
                result.add_warning(f"Étape {step_number} - Template de code très long")
        
        if 'solution' in step_data:
            solution = step_data['solution']
            if solution:
                # Valider la syntaxe de la solution
                code_validator = CodeValidator()
                code_result = code_validator.validate_syntax(solution)
                if not code_result.is_valid:
                    result.add_error(f"Étape {step_number} - Solution avec erreur de syntaxe: {', '.join(code_result.errors)}")
        
        # Valider les indices
        if 'hints' in step_data:
            hints = step_data['hints']
            if hints and isinstance(hints, list):
                if len(hints) > 5:
                    result.add_warning(f"Étape {step_number} - Beaucoup d'indices (plus de 5)")
                
                for j, hint in enumerate(hints):
                    if not hint or not isinstance(hint, str):
                        result.add_error(f"Étape {step_number} - Indice {j+1} invalide")
                    elif len(hint.strip()) < 5:
                        result.add_error(f"Étape {step_number} - Indice {j+1} trop court")
        
        return result


class UserInputValidator:
    """Validateur pour les entrées utilisateur"""
    
    @staticmethod
    def validate_username(username: str) -> ValidationResult:
        """Valide un nom d'utilisateur"""
        result = ValidationResult()
        
        if not username:
            result.add_error("Nom d'utilisateur vide")
            return result
        
        if len(username) < 3:
            result.add_error("Nom d'utilisateur trop court (minimum 3 caractères)")
        elif len(username) > 50:
            result.add_error("Nom d'utilisateur trop long (maximum 50 caractères)")
        
        if not re.match(r'^[a-zA-Z0-9_-]+, username):
            result.add_error("Nom d'utilisateur contient des caractères invalides")
        
        return result
    
    @staticmethod
    def validate_password(password: str) -> ValidationResult:
        """Valide un mot de passe"""
        result = ValidationResult()
        
        if not password:
            result.add_error("Mot de passe vide")
            return result
        
        if len(password) < 8:
            result.add_error("Mot de passe trop court (minimum 8 caractères)")
        
        # Vérifications de complexité
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        if not has_upper:
            result.add_warning("Mot de passe sans majuscule")
        if not has_lower:
            result.add_warning("Mot de passe sans minuscule")
        if not has_digit:
            result.add_warning("Mot de passe sans chiffre")
        if not has_special:
            result.add_warning("Mot de passe sans caractère spécial")
        
        return result
    
    @staticmethod
    def validate_user_level(level: str) -> ValidationResult:
        """Valide un niveau utilisateur"""
        result = ValidationResult()
        
        valid_levels = {'beginner', 'intermediate', 'advanced'}
        
        if level not in valid_levels:
            result.add_error(f"Niveau invalide. Valeurs autorisées: {', '.join(valid_levels)}")
        
        return result
    
    @staticmethod
    def validate_programming_language(language: str) -> ValidationResult:
        """Valide un langage de programmation"""
        result = ValidationResult()
        
        supported_languages = {'python', 'javascript', 'java', 'cpp', 'sql'}
        
        if language.lower() not in supported_languages:
            result.add_warning(f"Langage non officiellement supporté: {language}")
        
        return result


class ConfigValidator:
    """Validateur pour les fichiers de configuration"""
    
    @staticmethod
    def validate_config_file(file_path: Union[str, Path]) -> ValidationResult:
        """Valide un fichier de configuration"""
        result = ValidationResult()
        file_path = Path(file_path)
        
        if not file_path.exists():
            result.add_error(f"Fichier de configuration introuvable: {file_path}")
            return result
        
        try:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                result.add_error("Format de configuration non supporté (yaml/json uniquement)")
                return result
            
            result.data = config_data
            
            # Validations spécifiques selon le type de config
            if 'database' in config_data:
                db_result = ConfigValidator._validate_database_config(config_data['database'])
                result.errors.extend(db_result.errors)
                result.warnings.extend(db_result.warnings)
            
            if 'llm' in config_data:
                llm_result = ConfigValidator._validate_llm_config(config_data['llm'])
                result.errors.extend(llm_result.errors)
                result.warnings.extend(llm_result.warnings)
            
        except Exception as e:
            result.add_error(f"Erreur lors de la lecture du fichier: {str(e)}")
        
        result.is_valid = len(result.errors) == 0
        return result
    
    @staticmethod
    def _validate_database_config(db_config: Dict[str, Any]) -> ValidationResult:
        """Valide la configuration de base de données"""
        result = ValidationResult()
        
        if 'url' not in db_config:
            result.add_error("URL de base de données manquante")
        elif not isinstance(db_config['url'], str):
            result.add_error("URL de base de données doit être une chaîne")
        
        if 'pool_size' in db_config:
            pool_size = db_config['pool_size']
            if not isinstance(pool_size, int) or pool_size < 1:
                result.add_error("pool_size doit être un entier positif")
        
        return result
    
    @staticmethod
    def _validate_llm_config(llm_config: Dict[str, Any]) -> ValidationResult:
        """Valide la configuration LLM"""
        result = ValidationResult()
        
        if 'provider' not in llm_config:
            result.add_error("Provider LLM manquant")
        elif llm_config['provider'] not in ['openai', 'ollama']:
            result.add_warning(f"Provider LLM non standard: {llm_config['provider']}")
        
        if 'model_name' not in llm_config:
            result.add_error("Nom de modèle manquant")
        
        if 'temperature' in llm_config:
            temp = llm_config['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                result.add_error("Temperature doit être entre 0 et 2")
        
        return result


def validate_file_upload(file_data: Dict[str, Any]) -> ValidationResult:
    """
    Valide un upload de fichier
    
    Args:
        file_data: Données du fichier uploadé
        
    Returns:
        Résultat de validation
    """
    result = ValidationResult()
    
    required_fields = {'name', 'size', 'type'}
    missing_fields = required_fields - set(file_data.keys())
    
    if missing_fields:
        result.add_error(f"Informations manquantes: {', '.join(missing_fields)}")
        return result
    
    # Valider le nom de fichier
    filename = file_data['name']
    if not filename or len(filename.strip()) == 0:
        result.add_error("Nom de fichier vide")
    elif len(filename) > 255:
        result.add_error("Nom de fichier trop long (maximum 255 caractères)")
    elif not re.match(r'^[a-zA-Z0-9._-]+, filename):
        result.add_warning("Nom de fichier contient des caractères spéciaux")
    
    # Valider la taille
    file_size = file_data['size']
    if not isinstance(file_size, int) or file_size < 0:
        result.add_error("Taille de fichier invalide")
    elif file_size == 0:
        result.add_warning("Fichier vide")
    elif file_size > 50 * 1024 * 1024:  # 50MB
        result.add_error("Fichier trop volumineux (maximum 50MB)")
    
    # Valider le type MIME
    mime_type = file_data['type']
    allowed_types = [
        'text/plain', 'text/markdown', 'application/pdf',
        'application/json', 'text/csv',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]
    
    if mime_type not in allowed_types:
        result.add_warning(f"Type de fichier non standard: {mime_type}")
    
    result.is_valid = len(result.errors) == 0
    return result


def sanitize_input(text: str, max_length: int = 1000, 
                  allow_html: bool = False) -> str:
    """
    Nettoie et sécurise une entrée utilisateur
    
    Args:
        text: Texte à nettoyer
        max_length: Longueur maximale
        allow_html: Autoriser le HTML
        
    Returns:
        Texte nettoyé
    """
    if not text:
        return ""
    
    # Limiter la longueur
    if len(text) > max_length:
        text = text[:max_length]
    
    # Supprimer les caractères de contrôle
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Supprimer le HTML si non autorisé
    if not allow_html:
        text = re.sub(r'<[^>]+>', '', text)
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def validate_quiz_answer(user_answer: str, correct_answer: str, 
                        question_type: str = "multiple_choice") -> ValidationResult:
    """
    Valide une réponse de quiz
    
    Args:
        user_answer: Réponse de l'utilisateur
        correct_answer: Réponse correcte
        question_type: Type de question
        
    Returns:
        Résultat de validation
    """
    result = ValidationResult()
    
    if not user_answer:
        result.add_error("Réponse vide")
        return result
    
    # Nettoyer les réponses pour la comparaison
    user_clean = user_answer.strip().lower()
    correct_clean = correct_answer.strip().lower()
    
    if question_type == "multiple_choice":
        is_correct = user_clean == correct_clean
    elif question_type == "true_false":
        is_correct = user_clean in ['true', 'false', 'vrai', 'faux'] and user_clean == correct_clean
    elif question_type == "short_answer":
        # Comparaison plus flexible pour les réponses courtes
        is_correct = user_clean == correct_clean or user_answer.strip() == correct_answer.strip()
    else:
        result.add_error(f"Type de question non supporté: {question_type}")
        return result
    
    result.data = {
        'is_correct': is_correct,
        'user_answer': user_answer,
        'correct_answer': correct_answer,
        'question_type': question_type
    }
    
    if not is_correct:
        result.add_warning("Réponse incorrecte")
    
    return result
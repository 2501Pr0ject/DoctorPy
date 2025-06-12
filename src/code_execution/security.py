"""
Module de sécurité pour l'exécution de code Python.
Fournit des mécanismes de validation et protection avancés.
"""

import ast
import re
import hashlib
import time
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import keyword
import builtins

from src.core.logger import get_logger
from src.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class ThreatLevel(Enum):
    """Niveaux de menace."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityThreat:
    """Menace de sécurité détectée."""
    threat_type: str
    level: ThreatLevel
    line: Optional[int]
    column: Optional[int]
    description: str
    recommendation: str
    code_snippet: Optional[str] = None


@dataclass
class SecurityReport:
    """Rapport de sécurité complet."""
    is_safe: bool
    threat_level: ThreatLevel
    threats: List[SecurityThreat]
    blocked_operations: List[str]
    allowed_operations: List[str]
    risk_score: float
    recommendations: List[str]


class BlacklistManager:
    """Gestionnaire des listes noires."""
    
    # Modules complètement interdits
    FORBIDDEN_MODULES = {
        # Système et fichiers
        'os', 'sys', 'subprocess', 'shutil', 'glob', 'tempfile',
        'pathlib', 'fileinput', 'stat', 'statvfs', 'fnmatch',
        
        # Réseau
        'socket', 'ssl', 'urllib', 'urllib2', 'httplib', 'ftplib',
        'poplib', 'imaplib', 'nntplib', 'smtplib', 'telnetlib',
        'xmlrpc', 'SimpleHTTPServer', 'CGIHTTPServer',
        
        # Processus et threads
        'multiprocessing', 'threading', 'thread', '_thread',
        'subprocess', 'pty', 'pipes',
        
        # Sérialisation dangereuse
        'pickle', 'cPickle', 'shelve', 'marshal', 'dill',
        
        # Base de données
        'sqlite3', 'dbm', 'gdbm', 'anydbm', 'whichdb',
        
        # Introspection et compilation
        'imp', 'importlib', 'pkgutil', 'modulefinder',
        'compileall', 'py_compile',
        
        # Système Windows/Unix spécifiques
        'winreg', 'winsound', 'msvcrt', 'msilib',
        'pwd', 'grp', 'crypt', 'termios', 'tty',
        'fcntl', 'posix', 'resource', 'nis', 'syslog',
        
        # Autres modules sensibles
        'ctypes', 'cffi', 'webbrowser', 'antigravity'
    }
    
    # Fonctions built-in interdites
    FORBIDDEN_BUILTINS = {
        # Exécution dynamique
        'eval', 'exec', 'compile',
        
        # Import dynamique
        '__import__', 'reload',
        
        # Fichiers et entrées
        'open', 'file', 'input', 'raw_input',
        
        # Introspection dangereuse
        'vars', 'globals', 'locals', 'dir',
        'hasattr', 'getattr', 'setattr', 'delattr',
        
        # Autres
        'help', 'quit', 'exit', 'copyright', 'license', 'credits'
    }
    
    # Attributs dangereux
    FORBIDDEN_ATTRIBUTES = {
        '__globals__', '__locals__', '__code__', '__closure__',
        '__func__', '__self__', '__dict__', '__class__',
        '__bases__', '__mro__', '__subclasses__',
        'func_globals', 'func_code', 'func_closure'
    }
    
    # Patterns de code dangereux
    DANGEROUS_PATTERNS = [
        # Boucles infinies
        r'while\s+True\s*:(?!\s*#.*break)',
        r'for\s+\w+\s+in\s+itertools\.count\s*\(',
        
        # Récursion profonde
        r'def\s+\w+\([^)]*\):[^}]*\1\s*\(',
        
        # Grandes boucles
        r'for\s+\w+\s+in\s+range\s*\(\s*\d{6,}\s*\)',
        r'range\s*\(\s*\d{6,}\s*\)',
        
        # Opérations sur de grandes structures
        r'\[\s*\d+\s*\]\s*\*\s*\d{6,}',
        r'\d{6,}\s*\*\s*\[',
        
        # Allocation mémoire excessive
        r'bytearray\s*\(\s*\d{6,}\s*\)',
        r'b\s*["\'][^"\']*["\']\\s*\*\s*\d{6,}',
    ]


class CodeAnalyzer:
    """Analyseur de code pour détecter les menaces."""
    
    def __init__(self):
        self.blacklist = BlacklistManager()
    
    def analyze_imports(self, code: str) -> List[SecurityThreat]:
        """Analyse les imports pour détecter les modules dangereux."""
        threats = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.blacklist.FORBIDDEN_MODULES:
                            threat = SecurityThreat(
                                threat_type="forbidden_import",
                                level=ThreatLevel.CRITICAL,
                                line=node.lineno,
                                column=node.col_offset,
                                description=f"Import de module interdit: {alias.name}",
                                recommendation=f"Retirez 'import {alias.name}' - ce module n'est pas autorisé"
                            )
                            threats.append(threat)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.blacklist.FORBIDDEN_MODULES:
                        threat = SecurityThreat(
                            threat_type="forbidden_import_from",
                            level=ThreatLevel.CRITICAL,
                            line=node.lineno,
                            column=node.col_offset,
                            description=f"Import depuis module interdit: {node.module}",
                            recommendation=f"Retirez 'from {node.module} import ...' - ce module n'est pas autorisé"
                        )
                        threats.append(threat)
                    
                    # Vérifier les imports avec *
                    for alias in node.names:
                        if alias.name == '*':
                            threat = SecurityThreat(
                                threat_type="wildcard_import",
                                level=ThreatLevel.MEDIUM,
                                line=node.lineno,
                                column=node.col_offset,
                                description="Import avec * détecté",
                                recommendation="Utilisez des imports explicites au lieu de '*'"
                            )
                            threats.append(threat)
        
        except SyntaxError:
            pass  # Les erreurs de syntaxe sont gérées ailleurs
        
        return threats
    
    def analyze_function_calls(self, code: str) -> List[SecurityThreat]:
        """Analyse les appels de fonction dangereux."""
        threats = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = self._get_function_name(node.func)
                    
                    if func_name in self.blacklist.FORBIDDEN_BUILTINS:
                        threat = SecurityThreat(
                            threat_type="forbidden_builtin",
                            level=ThreatLevel.CRITICAL,
                            line=node.lineno,
                            column=node.col_offset,
                            description=f"Appel de fonction interdite: {func_name}()",
                            recommendation=f"Retirez l'appel à {func_name}() - cette fonction n'est pas autorisée"
                        )
                        threats.append(threat)
                    
                    # Cas spéciaux
                    elif func_name == 'range' and len(node.args) > 0:
                        # Vérifier les grandes plages
                        if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
                            if node.args[0].value > 1000000:  # Plus d'un million
                                threat = SecurityThreat(
                                    threat_type="large_range",
                                    level=ThreatLevel.HIGH,
                                    line=node.lineno,
                                    column=node.col_offset,
                                    description=f"Range avec valeur très élevée: {node.args[0].value}",
                                    recommendation="Utilisez des valeurs plus raisonnables pour éviter les problèmes de performance"
                                )
                                threats.append(threat)
        
        except SyntaxError:
            pass
        
        return threats
    
    def analyze_attribute_access(self, code: str) -> List[SecurityThreat]:
        """Analyse l'accès aux attributs dangereux."""
        threats = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if node.attr in self.blacklist.FORBIDDEN_ATTRIBUTES:
                        threat = SecurityThreat(
                            threat_type="forbidden_attribute",
                            level=ThreatLevel.HIGH,
                            line=node.lineno,
                            column=node.col_offset,
                            description=f"Accès à attribut dangereux: {node.attr}",
                            recommendation=f"L'attribut '{node.attr}' peut permettre d'accéder aux internals Python"
                        )
                        threats.append(threat)
        
        except SyntaxError:
            pass
        
        return threats
    
    def analyze_control_flow(self, code: str) -> List[SecurityThreat]:
        """Analyse le flux de contrôle pour détecter les problèmes."""
        threats = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Boucles while True sans break évident
                if isinstance(node, ast.While):
                    if self._is_infinite_while(node):
                        threat = SecurityThreat(
                            threat_type="infinite_loop",
                            level=ThreatLevel.HIGH,
                            line=node.lineno,
                            column=node.col_offset,
                            description="Boucle infinie potentielle détectée",
                            recommendation="Ajoutez une condition de sortie explicite ou un break"
                        )
                        threats.append(threat)
                
                # Récursion profonde potentielle
                elif isinstance(node, ast.FunctionDef):
                    if self._has_deep_recursion(node):
                        threat = SecurityThreat(
                            threat_type="deep_recursion",
                            level=ThreatLevel.MEDIUM,
                            line=node.lineno,
                            column=node.col_offset,
                            description=f"Récursion potentiellement profonde dans '{node.name}'",
                            recommendation="Vérifiez que la récursion a une condition d'arrêt appropriée"
                        )
                        threats.append(threat)
        
        except SyntaxError:
            pass
        
        return threats
    
    def analyze_patterns(self, code: str) -> List[SecurityThreat]:
        """Analyse les patterns dangereux dans le code source."""
        threats = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern in self.blacklist.DANGEROUS_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    threat = SecurityThreat(
                        threat_type="dangerous_pattern",
                        level=ThreatLevel.MEDIUM,
                        line=i,
                        column=1,
                        description=f"Pattern potentiellement dangereux détecté",
                        recommendation="Vérifiez que ce code ne causera pas de problèmes de performance ou sécurité",
                        code_snippet=line.strip()
                    )
                    threats.append(threat)
        
        return threats
    
    def _get_function_name(self, node: ast.AST) -> Optional[str]:
        """Extrait le nom d'une fonction depuis un nœud AST."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None
    
    def _is_infinite_while(self, node: ast.While) -> bool:
        """Vérifie si une boucle while est potentiellement infinie."""
        # while True sans break évident
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            # Chercher un break dans le corps
            for child in ast.walk(node):
                if isinstance(child, ast.Break):
                    return False
            return True
        
        # while 1
        if isinstance(node.test, ast.Constant) and node.test.value == 1:
            for child in ast.walk(node):
                if isinstance(child, ast.Break):
                    return False
            return True
        
        return False
    
    def _has_deep_recursion(self, func_node: ast.FunctionDef) -> bool:
        """Vérifie si une fonction a une récursion potentiellement profonde."""
        func_name = func_node.name
        
        # Chercher des appels récursifs
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    return True
        
        return False


class SecurityValidator:
    """Validateur de sécurité principal."""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.threat_cache = {}
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Valide la sécurité d'un code Python.
        
        Args:
            code: Code Python à valider
            
        Returns:
            Dictionnaire avec les résultats de validation
        """
        # Cache basé sur le hash du code
        code_hash = hashlib.md5(code.encode()).hexdigest()
        if code_hash in self.threat_cache:
            return self.threat_cache[code_hash]
        
        threats = []
        
        # Analyse des différents aspects
        threats.extend(self.analyzer.analyze_imports(code))
        threats.extend(self.analyzer.analyze_function_calls(code))
        threats.extend(self.analyzer.analyze_attribute_access(code))
        threats.extend(self.analyzer.analyze_control_flow(code))
        threats.extend(self.analyzer.analyze_patterns(code))
        
        # Évaluation du niveau de menace global
        threat_level = self._evaluate_threat_level(threats)
        
        # Déterminer si le code est sûr
        is_safe = threat_level not in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        
        # Calculer le score de risque
        risk_score = self._calculate_risk_score(threats)
        
        # Générer les recommandations
        recommendations = self._generate_recommendations(threats)
        
        result = {
            'is_safe': is_safe,
            'threat_level': threat_level.value,
            'threats': [self._threat_to_dict(t) for t in threats],
            'risk_score': risk_score,
            'recommendations': recommendations,
            'blocked_operations': self._get_blocked_operations(threats),
            'total_threats': len(threats),
            'by_level': self._count_by_level(threats)
        }
        
        # Cache du résultat
        self.threat_cache[code_hash] = result
        
        return result
    
    def create_security_report(self, code: str) -> SecurityReport:
        """Crée un rapport de sécurité détaillé."""
        validation_result = self.validate_code(code)
        
        threats = []
        for threat_dict in validation_result['threats']:
            threat = SecurityThreat(
                threat_type=threat_dict['type'],
                level=ThreatLevel(threat_dict['level']),
                line=threat_dict.get('line'),
                column=threat_dict.get('column'),
                description=threat_dict['description'],
                recommendation=threat_dict['recommendation'],
                code_snippet=threat_dict.get('code_snippet')
            )
            threats.append(threat)
        
        return SecurityReport(
            is_safe=validation_result['is_safe'],
            threat_level=ThreatLevel(validation_result['threat_level']),
            threats=threats,
            blocked_operations=validation_result['blocked_operations'],
            allowed_operations=self._get_allowed_operations(),
            risk_score=validation_result['risk_score'],
            recommendations=validation_result['recommendations']
        )
    
    def _evaluate_threat_level(self, threats: List[SecurityThreat]) -> ThreatLevel:
        """Évalue le niveau de menace global."""
        if not threats:
            return ThreatLevel.SAFE
        
        max_level = max(t.level for t in threats)
        return max_level
    
    def _calculate_risk_score(self, threats: List[SecurityThreat]) -> float:
        """Calcule un score de risque de 0 à 100."""
        if not threats:
            return 0.0
        
        score = 0.0
        for threat in threats:
            if threat.level == ThreatLevel.CRITICAL:
                score += 40
            elif threat.level == ThreatLevel.HIGH:
                score += 25
            elif threat.level == ThreatLevel.MEDIUM:
                score += 10
            elif threat.level == ThreatLevel.LOW:
                score += 5
        
        return min(100.0, score)
    
    def _generate_recommendations(self, threats: List[SecurityThreat]) -> List[str]:
        """Génère des recommandations basées sur les menaces."""
        recommendations = set()
        
        for threat in threats:
            recommendations.add(threat.recommendation)
        
        # Recommandations générales
        if any(t.threat_type == "forbidden_import" for t in threats):
            recommendations.add("Utilisez seulement les modules autorisés pour l'apprentissage")
        
        if any(t.threat_type == "infinite_loop" for t in threats):
            recommendations.add("Assurez-vous que toutes les boucles ont une condition de sortie")
        
        if any(t.threat_type == "large_range" for t in threats):
            recommendations.add("Utilisez des valeurs raisonnables pour éviter les problèmes de performance")
        
        return list(recommendations)
    
    def _get_blocked_operations(self, threats: List[SecurityThreat]) -> List[str]:
        """Retourne la liste des opérations bloquées."""
        blocked = set()
        
        for threat in threats:
            if threat.level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                if threat.threat_type == "forbidden_import":
                    blocked.add(f"Import: {threat.description.split(': ')[1]}")
                elif threat.threat_type == "forbidden_builtin":
                    blocked.add(f"Fonction: {threat.description.split(': ')[1]}")
                elif threat.threat_type == "forbidden_attribute":
                    blocked.add(f"Attribut: {threat.description.split(': ')[1]}")
        
        return list(blocked)
    
    def _get_allowed_operations(self) -> List[str]:
        """Retourne la liste des opérations autorisées."""
        return [
            "Modules: math, random, datetime, time, json, string, itertools, functools",
            "Fonctions built-in: print, len, range, enumerate, zip, map, filter, sorted",
            "Structures de données: list, dict, tuple, set",
            "Contrôle de flux: if, for, while (avec conditions de sortie)",
            "Définition de fonctions et classes",
            "Modules éducatifs: numpy, pandas, matplotlib (mode éducatif)"
        ]
    
    def _count_by_level(self, threats: List[SecurityThreat]) -> Dict[str, int]:
        """Compte les menaces par niveau."""
        counts = {level.value: 0 for level in ThreatLevel}
        
        for threat in threats:
            counts[threat.level.value] += 1
        
        return counts
    
    def _threat_to_dict(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Convertit une menace en dictionnaire."""
        return {
            'type': threat.threat_type,
            'level': threat.level.value,
            'line': threat.line,
            'column': threat.column,
            'description': threat.description,
            'recommendation': threat.recommendation,
            'code_snippet': threat.code_snippet
        }
    
    def clear_cache(self):
        """Vide le cache des menaces."""
        self.threat_cache.clear()


class SecurityMonitor:
    """Moniteur de sécurité pour l'exécution de code."""
    
    def __init__(self):
        self.validator = SecurityValidator()
        self.execution_stats = {
            'total_validations': 0,
            'blocked_executions': 0,
            'security_violations': 0,
            'threat_history': []
        }
    
    def monitor_execution(self, code: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Monitore l'exécution d'un code pour la sécurité.
        
        Args:
            code: Code à monitorer
            user_id: ID de l'utilisateur
            
        Returns:
            Résultats du monitoring
        """
        self.execution_stats['total_validations'] += 1
        
        # Validation de sécurité
        validation_result = self.validator.validate_code(code)
        
        # Enregistrer les statistiques
        if not validation_result['is_safe']:
            self.execution_stats['blocked_executions'] += 1
            
        if validation_result['total_threats'] > 0:
            self.execution_stats['security_violations'] += 1
        
        # Historique des menaces
        threat_entry = {
            'timestamp': time.time(),
            'user_id': user_id,
            'threat_level': validation_result['threat_level'],
            'threats_count': validation_result['total_threats'],
            'risk_score': validation_result['risk_score']
        }
        self.execution_stats['threat_history'].append(threat_entry)
        
        # Limiter l'historique
        if len(self.execution_stats['threat_history']) > 1000:
            self.execution_stats['threat_history'] = self.execution_stats['threat_history'][-1000:]
        
        # Logger les violations importantes
        if validation_result['threat_level'] in ['high', 'critical']:
            logger.warning(
                f"Violation de sécurité détectée - User: {user_id}, "
                f"Level: {validation_result['threat_level']}, "
                f"Threats: {validation_result['total_threats']}"
            )
        
        return {
            **validation_result,
            'monitoring_stats': self.get_stats()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de monitoring."""
        return {
            'total_validations': self.execution_stats['total_validations'],
            'blocked_executions': self.execution_stats['blocked_executions'],
            'security_violations': self.execution_stats['security_violations'],
            'block_rate': (
                self.execution_stats['blocked_executions'] / 
                max(1, self.execution_stats['total_validations'])
            ),
            'violation_rate': (
                self.execution_stats['security_violations'] / 
                max(1, self.execution_stats['total_validations'])
            )
        }
    
    def get_threat_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyse les tendances des menaces."""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_threats = [
            t for t in self.execution_stats['threat_history'] 
            if t['timestamp'] > cutoff_time
        ]
        
        if not recent_threats:
            return {'total': 0, 'trends': {}}
        
        # Compter par niveau
        level_counts = {}
        for threat in recent_threats:
            level = threat['threat_level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'total': len(recent_threats),
            'by_level': level_counts,
            'average_risk_score': sum(t['risk_score'] for t in recent_threats) / len(recent_threats),
            'unique_users': len(set(t['user_id'] for t in recent_threats if t['user_id']))
        }


# Instances globales
security_validator = SecurityValidator()
security_monitor = SecurityMonitor()


# Fonctions utilitaires
def validate_code_security(code: str) -> Dict[str, Any]:
    """
    Valide la sécurité d'un code Python.
    
    Args:
        code: Code Python à valider
        
    Returns:
        Résultats de validation
    """
    return security_validator.validate_code(code)


def is_code_safe(code: str) -> bool:
    """
    Vérifie rapidement si un code est sûr.
    
    Args:
        code: Code à vérifier
        
    Returns:
        True si le code est sûr
    """
    result = security_validator.validate_code(code)
    return result['is_safe']


def get_security_recommendations(code: str) -> List[str]:
    """
    Obtient les recommandations de sécurité pour un code.
    
    Args:
        code: Code à analyser
        
    Returns:
        Liste des recommandations
    """
    result = security_validator.validate_code(code)
    return result['recommendations']


def monitor_code_execution(code: str, user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Monitore l'exécution d'un code pour la sécurité.
    
    Args:
        code: Code à monitorer
        user_id: ID de l'utilisateur
        
    Returns:
        Résultats du monitoring
    """
    return security_monitor.monitor_execution(code, user_id)


if __name__ == "__main__":
    # Test du module de sécurité
    dangerous_code = """
import os
import sys
import subprocess

def dangerous_function():
    eval("print('Hello')")
    exec("x = 1")
    os.system("ls")
    
    while True:
        print("Infinite loop")
        
    subprocess.call(["rm", "-rf", "/"])
"""
    
    safe_code = """
import math
import random

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def calculate_area(radius):
    return math.pi * radius ** 2

print(f"Area: {calculate_area(5)}")
print(f"Fibonacci: {fibonacci(10)}")
"""
    
    print("=== Code Dangereux ===")
    result = validate_code_security(dangerous_code)
    print(f"Sûr: {result['is_safe']}")
    print(f"Niveau de menace: {result['threat_level']}")
    print(f"Nombre de menaces: {result['total_threats']}")
    print(f"Score de risque: {result['risk_score']}")
    
    print("\n=== Code Sûr ===")
    result = validate_code_security(safe_code)
    print(f"Sûr: {result['is_safe']}")
    print(f"Niveau de menace: {result['threat_level']}")
    print(f"Nombre de menaces: {result['total_threats']}")
    print(f"Score de risque: {result['risk_score']}")
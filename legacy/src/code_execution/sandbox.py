"""
Environnement d'exécution sécurisé pour le code Python.
Fournit un sandbox isolé avec limitations de ressources et sécurité renforcée.
"""

import os
import sys
import subprocess
import tempfile
import threading
import time
import resource
import signal
import contextlib
import io
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import shutil
import json
import ast
import builtins
from dataclasses import dataclass
from enum import Enum

from src.core.logger import get_logger
from src.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class SandboxMode(Enum):
    """Modes d'exécution du sandbox."""
    SAFE = "safe"           # Mode sécurisé par défaut
    EDUCATIONAL = "educational"  # Mode éducatif avec plus de modules
    RESTRICTED = "restricted"    # Mode très restrictif


@dataclass
class SandboxLimits:
    """Limites de ressources pour le sandbox."""
    max_execution_time: float = 5.0    # Temps max en secondes
    max_memory_mb: int = 64             # Mémoire max en MB
    max_output_size: int = 10000        # Taille max de sortie
    max_file_size: int = 1024           # Taille max fichier temporaire
    max_processes: int = 1              # Nombre max de processus


@dataclass
class ExecutionResult:
    """Résultat d'exécution dans le sandbox."""
    success: bool
    output: str
    error: str
    execution_time: float
    memory_used: int
    exit_code: int
    timeout: bool
    security_violation: bool
    warnings: List[str]


class SecurityChecker:
    """Vérificateur de sécurité pour le code Python."""
    
    # Modules interdits
    FORBIDDEN_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'glob', 'tempfile',
        'multiprocessing', 'threading', 'asyncio', 'socket', 'urllib',
        'requests', 'http', 'ftplib', 'smtplib', 'telnetlib',
        'webbrowser', 'pickle', 'shelve', 'dbm', 'sqlite3',
        'ctypes', 'winreg', 'msilib', 'msvcrt', 'winsound',
        'pwd', 'grp', 'termios', 'tty', 'pty', 'fcntl',
        'pipes', 'posix', 'resource', 'nis', 'syslog',
        'commands', 'imp', 'importlib', '__import__',
        'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input'
    }
    
    # Fonctions built-in interdites
    FORBIDDEN_BUILTINS = {
        'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input',
        '__import__', 'reload', 'help', 'quit', 'exit', 'copyright',
        'license', 'credits', 'vars', 'dir', 'globals', 'locals',
        'hasattr', 'getattr', 'setattr', 'delattr'
    }
    
    # Patterns dangereux dans le code
    DANGEROUS_PATTERNS = [
        'import os', 'import sys', 'from os', 'from sys',
        '__import__', 'eval(', 'exec(', 'compile(',
        'open(', 'file(', 'input(', 'raw_input(',
        'subprocess', 'system(', 'popen(', 'spawn(',
        'fork(', 'kill(', 'signal(', 'alarm(',
        'while True:', 'for i in range(999999)',
        'recursion', 'lambda', 'yield', 'generator'
    ]
    
    @classmethod
    def check_code_safety(cls, code: str) -> Tuple[bool, List[str]]:
        """
        Vérifie la sécurité du code avant exécution.
        
        Args:
            code: Code Python à vérifier
            
        Returns:
            Tuple (is_safe, warnings)
        """
        warnings = []
        
        try:
            # Analyse AST
            tree = ast.parse(code)
            
            # Vérifier les imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in cls.FORBIDDEN_MODULES:
                            warnings.append(f"Import interdit: {alias.name}")
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module in cls.FORBIDDEN_MODULES:
                        warnings.append(f"Import interdit: {node.module}")
                        
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in cls.FORBIDDEN_BUILTINS:
                            warnings.append(f"Fonction interdite: {node.func.id}")
                            
                elif isinstance(node, ast.While):
                    # Vérifier les boucles infinies potentielles
                    if isinstance(node.test, ast.Constant) and node.test.value is True:
                        warnings.append("Boucle infinie détectée: while True")
                        
        except SyntaxError as e:
            warnings.append(f"Erreur de syntaxe: {e}")
            return False, warnings
            
        # Vérifier les patterns dangereux
        code_lower = code.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.lower() in code_lower:
                warnings.append(f"Pattern potentiellement dangereux: {pattern}")
                
        # Code considéré comme sûr si pas de warnings critiques
        critical_warnings = [w for w in warnings if "interdit" in w or "infinie" in w]
        is_safe = len(critical_warnings) == 0
        
        return is_safe, warnings


class PythonSandbox:
    """Sandbox d'exécution Python sécurisé."""
    
    def __init__(
        self,
        mode: SandboxMode = SandboxMode.SAFE,
        limits: Optional[SandboxLimits] = None
    ):
        self.mode = mode
        self.limits = limits or SandboxLimits()
        self.temp_dir = None
        self.allowed_modules = self._get_allowed_modules()
        
    def _get_allowed_modules(self) -> set:
        """Retourne les modules autorisés selon le mode."""
        base_modules = {
            'math', 'random', 'datetime', 'time', 'json', 'string',
            'itertools', 'functools', 'operator', 'collections',
            'heapq', 'bisect', 'array', 'copy', 'pprint'
        }
        
        if self.mode == SandboxMode.EDUCATIONAL:
            base_modules.update({
                'numpy', 'pandas', 'matplotlib', 'seaborn',
                'sklearn', 'scipy', 'statistics'
            })
        elif self.mode == SandboxMode.RESTRICTED:
            base_modules = {'math', 'random', 'string'}
            
        return base_modules
    
    def _setup_environment(self) -> Dict[str, Any]:
        """Configure l'environnement d'exécution."""
        # Créer environnement limité
        safe_env = {
            '__builtins__': {
                # Fonctions de base autorisées
                'abs', 'all', 'any', 'bool', 'chr', 'dict', 'enumerate',
                'filter', 'float', 'format', 'frozenset', 'int', 'len',
                'list', 'map', 'max', 'min', 'ord', 'pow', 'print', 'range',
                'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum',
                'tuple', 'type', 'zip',
                # Exceptions
                'Exception', 'ValueError', 'TypeError', 'IndexError',
                'KeyError', 'AttributeError', 'ZeroDivisionError'
            }
        }
        
        # Ajouter modules autorisés
        for module_name in self.allowed_modules:
            try:
                safe_env[module_name] = __import__(module_name)
            except ImportError:
                logger.warning(f"Module {module_name} non disponible")
                
        return safe_env
    
    def _create_temp_workspace(self) -> Path:
        """Crée un espace de travail temporaire."""
        self.temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")
        return Path(self.temp_dir)
    
    def _cleanup_temp_workspace(self):
        """Nettoie l'espace de travail temporaire."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Erreur nettoyage workspace: {e}")
    
    def _set_resource_limits(self):
        """Configure les limites de ressources."""
        try:
            # Limite mémoire
            memory_limit = self.limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Limite temps CPU
            cpu_limit = int(self.limits.max_execution_time)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            
            # Limite nombre de processus
            resource.setrlimit(resource.RLIMIT_NPROC, (self.limits.max_processes, self.limits.max_processes))
            
        except Exception as e:
            logger.warning(f"Impossible de définir les limites de ressources: {e}")
    
    def execute_code(
        self,
        code: str,
        test_cases: Optional[List[Dict]] = None,
        input_data: Optional[str] = None
    ) -> ExecutionResult:
        """
        Exécute du code Python dans le sandbox.
        
        Args:
            code: Code Python à exécuter
            test_cases: Cas de test optionnels
            input_data: Données d'entrée pour le programme
            
        Returns:
            ExecutionResult avec les détails d'exécution
        """
        start_time = time.time()
        result = ExecutionResult(
            success=False,
            output="",
            error="",
            execution_time=0.0,
            memory_used=0,
            exit_code=0,
            timeout=False,
            security_violation=False,
            warnings=[]
        )
        
        try:
            # Vérification sécurité
            is_safe, warnings = SecurityChecker.check_code_safety(code)
            result.warnings = warnings
            
            if not is_safe:
                result.security_violation = True
                result.error = "Code non sécurisé détecté"
                return result
            
            # Créer workspace temporaire
            workspace = self._create_temp_workspace()
            
            # Configurer environnement
            safe_env = self._setup_environment()
            
            # Capturer stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            captured_output = io.StringIO()
            captured_error = io.StringIO()
            
            sys.stdout = captured_output
            sys.stderr = captured_error
            
            # Exécuter avec timeout
            execution_thread = threading.Thread(
                target=self._execute_in_thread,
                args=(code, safe_env, workspace)
            )
            
            execution_thread.start()
            execution_thread.join(timeout=self.limits.max_execution_time)
            
            # Vérifier timeout
            if execution_thread.is_alive():
                result.timeout = True
                result.error = "Timeout d'exécution dépassé"
                # Note: En production, utiliser un processus séparé
                return result
            
            # Récupérer résultats
            result.output = captured_output.getvalue()
            result.error = captured_error.getvalue()
            
            # Limiter taille de sortie
            if len(result.output) > self.limits.max_output_size:
                result.output = result.output[:self.limits.max_output_size] + "\n[Output tronqué]"
            
            result.success = len(result.error) == 0
            result.execution_time = time.time() - start_time
            
        except Exception as e:
            result.error = f"Erreur d'exécution: {str(e)}"
            result.execution_time = time.time() - start_time
            
        finally:
            # Restaurer stdout/stderr
            try:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            except:
                pass
                
            # Nettoyer
            self._cleanup_temp_workspace()
            
        return result
    
    def _execute_in_thread(self, code: str, safe_env: Dict, workspace: Path):
        """Exécute le code dans un thread séparé."""
        try:
            # Changer vers workspace
            old_cwd = os.getcwd()
            os.chdir(workspace)
            
            # Configurer limites
            self._set_resource_limits()
            
            # Exécuter le code
            exec(code, safe_env)
            
        except Exception as e:
            print(f"Erreur: {e}", file=sys.stderr)
        finally:
            try:
                os.chdir(old_cwd)
            except:
                pass
    
    def execute_with_tests(
        self,
        code: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Exécute le code avec des cas de test.
        
        Args:
            code: Code à tester
            test_cases: Liste de cas de test
            
        Returns:
            Résultats détaillés des tests
        """
        results = {
            'execution_result': None,
            'test_results': [],
            'success_rate': 0.0,
            'total_tests': len(test_cases),
            'passed_tests': 0
        }
        
        # Exécuter le code principal
        main_result = self.execute_code(code)
        results['execution_result'] = main_result
        
        if not main_result.success:
            return results
        
        # Exécuter chaque test
        for i, test_case in enumerate(test_cases):
            test_code = f"{code}\n\n# Test case {i+1}\n{test_case.get('code', '')}"
            
            test_result = self.execute_code(test_code, input_data=test_case.get('input'))
            
            test_info = {
                'test_id': i + 1,
                'description': test_case.get('description', f'Test {i+1}'),
                'expected': test_case.get('expected'),
                'actual': test_result.output.strip(),
                'passed': False,
                'execution_time': test_result.execution_time,
                'error': test_result.error
            }
            
            # Vérifier si le test passe
            if test_result.success:
                if test_case.get('expected') is not None:
                    test_info['passed'] = str(test_case['expected']).strip() == test_result.output.strip()
                else:
                    test_info['passed'] = True
            
            if test_info['passed']:
                results['passed_tests'] += 1
                
            results['test_results'].append(test_info)
        
        # Calculer taux de réussite
        if results['total_tests'] > 0:
            results['success_rate'] = results['passed_tests'] / results['total_tests']
        
        return results


class CodeExecutionManager:
    """Gestionnaire d'exécution de code avec cache et métriques."""
    
    def __init__(self):
        self.sandbox = PythonSandbox()
        self.execution_cache = {}
        self.metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'security_violations': 0,
            'timeouts': 0
        }
    
    def execute_user_code(
        self,
        code: str,
        user_id: Optional[int] = None,
        quest_id: Optional[str] = None,
        test_cases: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Exécute le code d'un utilisateur avec logging et métriques.
        
        Args:
            code: Code à exécuter
            user_id: ID de l'utilisateur
            quest_id: ID de la quête
            test_cases: Cas de test
            
        Returns:
            Résultats d'exécution enrichis
        """
        execution_id = f"{user_id}_{quest_id}_{hash(code)}"
        
        # Vérifier cache
        if execution_id in self.execution_cache:
            logger.info(f"Utilisation du cache pour l'exécution {execution_id}")
            return self.execution_cache[execution_id]
        
        # Exécuter
        if test_cases:
            result = self.sandbox.execute_with_tests(code, test_cases)
        else:
            execution_result = self.sandbox.execute_code(code)
            result = {'execution_result': execution_result}
        
        # Mettre à jour métriques
        self._update_metrics(result)
        
        # Logger
        self._log_execution(user_id, quest_id, result)
        
        # Cache (limité en taille)
        if len(self.execution_cache) < 100:
            self.execution_cache[execution_id] = result
        
        return result
    
    def _update_metrics(self, result: Dict[str, Any]):
        """Met à jour les métriques d'exécution."""
        self.metrics['total_executions'] += 1
        
        execution_result = result.get('execution_result')
        if execution_result:
            if execution_result.success:
                self.metrics['successful_executions'] += 1
            else:
                self.metrics['failed_executions'] += 1
                
            if execution_result.security_violation:
                self.metrics['security_violations'] += 1
                
            if execution_result.timeout:
                self.metrics['timeouts'] += 1
    
    def _log_execution(self, user_id: Optional[int], quest_id: Optional[str], result: Dict[str, Any]):
        """Log les détails d'exécution."""
        execution_result = result.get('execution_result')
        if execution_result:
            logger.info(
                f"Exécution code - User: {user_id}, Quest: {quest_id}, "
                f"Success: {execution_result.success}, Time: {execution_result.execution_time:.2f}s"
            )
            
            if execution_result.security_violation:
                logger.warning(f"Violation sécurité détectée - User: {user_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques d'exécution."""
        return {
            **self.metrics,
            'success_rate': (
                self.metrics['successful_executions'] / max(1, self.metrics['total_executions'])
            ),
            'security_violation_rate': (
                self.metrics['security_violations'] / max(1, self.metrics['total_executions'])
            )
        }
    
    def clear_cache(self):
        """Vide le cache d'exécution."""
        self.execution_cache.clear()
        logger.info("Cache d'exécution vidé")


# Instance globale
execution_manager = CodeExecutionManager()


# Fonctions utilitaires
def execute_code_safely(
    code: str,
    test_cases: Optional[List[Dict]] = None,
    mode: SandboxMode = SandboxMode.SAFE
) -> ExecutionResult:
    """
    Fonction utilitaire pour exécuter du code de manière sécurisée.
    
    Args:
        code: Code Python à exécuter
        test_cases: Cas de test optionnels
        mode: Mode de sandbox
        
    Returns:
        ExecutionResult
    """
    sandbox = PythonSandbox(mode=mode)
    return sandbox.execute_code(code, test_cases)


def validate_code_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Valide la syntaxe Python d'un code.
    
    Args:
        code: Code à valider
        
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Erreur de syntaxe ligne {e.lineno}: {e.msg}"


if __name__ == "__main__":
    # Test du sandbox
    test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(f"Fibonacci de 10: {fibonacci(10)}")
"""
    
    result = execute_code_safely(test_code)
    print(f"Succès: {result.success}")
    print(f"Sortie: {result.output}")
    print(f"Temps: {result.execution_time:.2f}s")
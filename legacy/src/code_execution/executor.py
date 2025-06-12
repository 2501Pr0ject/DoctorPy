"""
Exécuteur de code Python avec gestion avancée des tests et validation.
"""

import ast
import sys
import time
import traceback
import contextlib
import io
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import json

from src.code_execution.sandbox import PythonSandbox, ExecutionResult, SandboxMode
from src.code_execution.security import SecurityValidator
from src.core.logger import get_logger

logger = get_logger(__name__)


class TestType(Enum):
    """Types de tests supportés."""
    UNIT = "unit"                   # Tests unitaires simples
    ASSERTION = "assertion"         # Tests avec assertions
    OUTPUT = "output"               # Comparaison de sortie
    FUNCTION_CALL = "function_call" # Test d'appel de fonction
    PERFORMANCE = "performance"     # Tests de performance
    INTEGRATION = "integration"     # Tests d'intégration


@dataclass
class TestCase:
    """Cas de test structuré."""
    id: str
    description: str
    test_type: TestType
    input_data: Optional[Any] = None
    expected_output: Optional[str] = None
    expected_result: Optional[Any] = None
    function_name: Optional[str] = None
    function_args: Optional[List] = None
    function_kwargs: Optional[Dict] = None
    assertion_code: Optional[str] = None
    timeout: float = 5.0
    points: int = 1
    is_required: bool = True
    hints: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Résultat d'un test."""
    test_case: TestCase
    passed: bool
    actual_output: Optional[str] = None
    actual_result: Optional[Any] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    score: float = 0.0
    feedback: str = ""


@dataclass
class ExecutionReport:
    """Rapport complet d'exécution."""
    code: str
    syntax_valid: bool
    execution_successful: bool
    test_results: List[TestResult]
    total_score: float
    max_score: float
    pass_rate: float
    execution_time: float
    memory_used: int
    warnings: List[str]
    suggestions: List[str]
    security_issues: List[str]


class CodeAnalyzer:
    """Analyseur de code Python."""
    
    @staticmethod
    def extract_functions(code: str) -> List[Dict[str, Any]]:
        """Extrait les fonctions définies dans le code."""
        functions = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'lineno': node.lineno,
                        'has_docstring': ast.get_docstring(node) is not None,
                        'returns': node.returns is not None
                    }
                    functions.append(func_info)
                    
        except SyntaxError:
            pass
            
        return functions
    
    @staticmethod
    def extract_imports(code: str) -> List[str]:
        """Extrait les imports du code."""
        imports = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
                        
        except SyntaxError:
            pass
            
        return imports
    
    @staticmethod
    def count_complexity(code: str) -> Dict[str, int]:
        """Calcule des métriques de complexité."""
        metrics = {
            'lines': len(code.splitlines()),
            'functions': 0,
            'classes': 0,
            'loops': 0,
            'conditions': 0,
            'complexity': 0
        }
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics['functions'] += 1
                elif isinstance(node, ast.ClassDef):
                    metrics['classes'] += 1
                elif isinstance(node, (ast.For, ast.While)):
                    metrics['loops'] += 1
                    metrics['complexity'] += 1
                elif isinstance(node, ast.If):
                    metrics['conditions'] += 1
                    metrics['complexity'] += 1
                    
        except SyntaxError:
            pass
            
        return metrics


class TestExecutor:
    """Exécuteur de tests pour code Python."""
    
    def __init__(self, sandbox_mode: SandboxMode = SandboxMode.EDUCATIONAL):
        self.sandbox = PythonSandbox(mode=sandbox_mode)
        self.security_validator = SecurityValidator()
    
    def execute_test_case(self, code: str, test_case: TestCase) -> TestResult:
        """Exécute un cas de test spécifique."""
        result = TestResult(
            test_case=test_case,
            passed=False
        )
        
        start_time = time.time()
        
        try:
            if test_case.test_type == TestType.OUTPUT:
                result = self._execute_output_test(code, test_case, result)
            elif test_case.test_type == TestType.FUNCTION_CALL:
                result = self._execute_function_test(code, test_case, result)
            elif test_case.test_type == TestType.ASSERTION:
                result = self._execute_assertion_test(code, test_case, result)
            elif test_case.test_type == TestType.PERFORMANCE:
                result = self._execute_performance_test(code, test_case, result)
            else:
                result = self._execute_unit_test(code, test_case, result)
                
        except Exception as e:
            result.error_message = f"Erreur d'exécution du test: {str(e)}"
            
        result.execution_time = time.time() - start_time
        
        # Calculer le score
        if result.passed:
            result.score = test_case.points
        else:
            result.score = 0.0
            
        return result
    
    def _execute_output_test(self, code: str, test_case: TestCase, result: TestResult) -> TestResult:
        """Exécute un test de comparaison de sortie."""
        # Ajouter input si nécessaire
        test_code = code
        if test_case.input_data:
            # Simuler input() avec les données fournies
            input_lines = str(test_case.input_data).split('\n')
            input_mock = f"""
import sys
from io import StringIO

_test_input = {repr(input_lines)}
_test_input_iter = iter(_test_input)

def mock_input(prompt=''):
    try:
        return next(_test_input_iter)
    except StopIteration:
        return ''

# Remplacer input par notre mock
input = mock_input
"""
            test_code = input_mock + "\n" + code
        
        # Exécuter le code
        execution_result = self.sandbox.execute_code(test_code)
        
        result.actual_output = execution_result.output.strip()
        result.execution_time = execution_result.execution_time
        
        if execution_result.success:
            expected = test_case.expected_output.strip() if test_case.expected_output else ""
            result.passed = result.actual_output == expected
            
            if not result.passed:
                result.feedback = f"Sortie attendue: '{expected}', obtenue: '{result.actual_output}'"
        else:
            result.error_message = execution_result.error
            result.feedback = "Erreur d'exécution du code"
            
        return result
    
    def _execute_function_test(self, code: str, test_case: TestCase, result: TestResult) -> TestResult:
        """Exécute un test d'appel de fonction."""
        if not test_case.function_name:
            result.error_message = "Nom de fonction requis pour ce type de test"
            return result
        
        # Construire le code de test
        args_str = ", ".join(repr(arg) for arg in (test_case.function_args or []))
        kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in (test_case.function_kwargs or {}).items())
        
        call_args = []
        if args_str:
            call_args.append(args_str)
        if kwargs_str:
            call_args.append(kwargs_str)
        
        test_code = f"""
{code}

# Test d'appel de fonction
try:
    result = {test_case.function_name}({", ".join(call_args)})
    print(f"RESULT:{repr(result)}")
except Exception as e:
    print(f"ERROR:{str(e)}")
"""
        
        execution_result = self.sandbox.execute_code(test_code)
        
        if execution_result.success:
            output = execution_result.output.strip()
            
            if output.startswith("RESULT:"):
                result_str = output[7:]  # Enlever "RESULT:"
                try:
                    result.actual_result = eval(result_str)
                    result.actual_output = str(result.actual_result)
                    
                    # Comparer avec le résultat attendu
                    if test_case.expected_result is not None:
                        result.passed = result.actual_result == test_case.expected_result
                        if not result.passed:
                            result.feedback = f"Résultat attendu: {test_case.expected_result}, obtenu: {result.actual_result}"
                    else:
                        result.passed = True
                        
                except Exception as e:
                    result.error_message = f"Erreur d'évaluation du résultat: {e}"
                    
            elif output.startswith("ERROR:"):
                result.error_message = output[6:]  # Enlever "ERROR:"
                result.feedback = "Erreur lors de l'appel de la fonction"
        else:
            result.error_message = execution_result.error
            
        return result
    
    def _execute_assertion_test(self, code: str, test_case: TestCase, result: TestResult) -> TestResult:
        """Exécute un test avec assertions."""
        if not test_case.assertion_code:
            result.error_message = "Code d'assertion requis pour ce type de test"
            return result
        
        test_code = f"""
{code}

# Test avec assertion
try:
    {test_case.assertion_code}
    print("ASSERTION_PASSED")
except AssertionError as e:
    print(f"ASSERTION_FAILED:{str(e)}")
except Exception as e:
    print(f"ERROR:{str(e)}")
"""
        
        execution_result = self.sandbox.execute_code(test_code)
        
        if execution_result.success:
            output = execution_result.output.strip()
            
            if output == "ASSERTION_PASSED":
                result.passed = True
                result.feedback = "Assertion réussie"
            elif output.startswith("ASSERTION_FAILED:"):
                result.passed = False
                result.error_message = output[17:]  # Enlever "ASSERTION_FAILED:"
                result.feedback = f"Assertion échouée: {result.error_message}"
            elif output.startswith("ERROR:"):
                result.error_message = output[6:]  # Enlever "ERROR:"
        else:
            result.error_message = execution_result.error
            
        return result
    
    def _execute_performance_test(self, code: str, test_case: TestCase, result: TestResult) -> TestResult:
        """Exécute un test de performance."""
        test_code = f"""
import time

{code}

# Test de performance
start_time = time.time()
try:
    # Exécuter le code à tester
    {test_case.assertion_code or "pass"}
    execution_time = time.time() - start_time
    print(f"PERFORMANCE:{execution_time}")
except Exception as e:
    print(f"ERROR:{str(e)}")
"""
        
        execution_result = self.sandbox.execute_code(test_code)
        
        if execution_result.success:
            output = execution_result.output.strip()
            
            if output.startswith("PERFORMANCE:"):
                try:
                    perf_time = float(output[12:])  # Enlever "PERFORMANCE:"
                    result.execution_time = perf_time
                    
                    # Vérifier si dans les limites de temps
                    if perf_time <= test_case.timeout:
                        result.passed = True
                        result.feedback = f"Performance acceptable: {perf_time:.3f}s"
                    else:
                        result.passed = False
                        result.feedback = f"Performance insuffisante: {perf_time:.3f}s (limite: {test_case.timeout}s)"
                        
                except ValueError:
                    result.error_message = "Erreur de mesure de performance"
            elif output.startswith("ERROR:"):
                result.error_message = output[6:]
        else:
            result.error_message = execution_result.error
            
        return result
    
    def _execute_unit_test(self, code: str, test_case: TestCase, result: TestResult) -> TestResult:
        """Exécute un test unitaire générique."""
        execution_result = self.sandbox.execute_code(code)
        
        result.actual_output = execution_result.output.strip()
        result.execution_time = execution_result.execution_time
        
        if execution_result.success:
            result.passed = True
            result.feedback = "Code exécuté avec succès"
        else:
            result.passed = False
            result.error_message = execution_result.error
            result.feedback = "Erreur d'exécution"
            
        return result


class AdvancedCodeExecutor:
    """Exécuteur de code avancé avec rapport complet."""
    
    def __init__(self):
        self.test_executor = TestExecutor()
        self.analyzer = CodeAnalyzer()
        self.security_validator = SecurityValidator()
    
    def execute_and_evaluate(
        self,
        code: str,
        test_cases: List[TestCase],
        user_id: Optional[int] = None,
        quest_id: Optional[str] = None
    ) -> ExecutionReport:
        """
        Exécute et évalue du code avec rapport complet.
        
        Args:
            code: Code Python à exécuter
            test_cases: Liste des cas de test
            user_id: ID de l'utilisateur
            quest_id: ID de la quête
            
        Returns:
            ExecutionReport complet
        """
        start_time = time.time()
        
        # Initialiser le rapport
        report = ExecutionReport(
            code=code,
            syntax_valid=False,
            execution_successful=False,
            test_results=[],
            total_score=0.0,
            max_score=sum(tc.points for tc in test_cases),
            pass_rate=0.0,
            execution_time=0.0,
            memory_used=0,
            warnings=[],
            suggestions=[],
            security_issues=[]
        )
        
        # 1. Validation de la syntaxe
        try:
            ast.parse(code)
            report.syntax_valid = True
        except SyntaxError as e:
            report.warnings.append(f"Erreur de syntaxe ligne {e.lineno}: {e.msg}")
            return report
        
        # 2. Validation de sécurité
        security_result = self.security_validator.validate_code(code)
        report.security_issues = security_result.get('issues', [])
        
        if security_result.get('is_dangerous', False):
            report.warnings.append("Code potentiellement dangereux détecté")
            return report
        
        # 3. Analyse du code
        complexity = self.analyzer.count_complexity(code)
        functions = self.analyzer.extract_functions(code)
        imports = self.analyzer.extract_imports(code)
        
        # 4. Exécution des tests
        passed_tests = 0
        total_points = 0
        
        for test_case in test_cases:
            test_result = self.test_executor.execute_test_case(code, test_case)
            report.test_results.append(test_result)
            
            if test_result.passed:
                passed_tests += 1
                total_points += test_result.score
        
        # 5. Calcul des métriques
        report.total_score = total_points
        if len(test_cases) > 0:
            report.pass_rate = passed_tests / len(test_cases)
        
        report.execution_successful = report.pass_rate > 0
        report.execution_time = time.time() - start_time
        
        # 6. Génération de suggestions
        report.suggestions = self._generate_suggestions(code, complexity, functions, report.test_results)
        
        # 7. Logger les résultats
        self._log_execution_results(user_id, quest_id, report)
        
        return report
    
    def _generate_suggestions(
        self,
        code: str,
        complexity: Dict[str, int],
        functions: List[Dict],
        test_results: List[TestResult]
    ) -> List[str]:
        """Génère des suggestions d'amélioration."""
        suggestions = []
        
        # Suggestions basées sur la complexité
        if complexity['lines'] > 50:
            suggestions.append("Considérez diviser votre code en plusieurs fonctions plus petites")
        
        if complexity['complexity'] > 10:
            suggestions.append("La complexité cyclomatique est élevée, simplifiez la logique")
        
        # Suggestions basées sur les fonctions
        for func in functions:
            if not func['has_docstring']:
                suggestions.append(f"Ajoutez une docstring à la fonction '{func['name']}'")
            
            if len(func['args']) > 5:
                suggestions.append(f"La fonction '{func['name']}' a beaucoup de paramètres, considérez la refactorisation")
        
        # Suggestions basées sur les tests échoués
        failed_tests = [tr for tr in test_results if not tr.passed]
        if failed_tests:
            common_errors = {}
            for test_result in failed_tests:
                if test_result.error_message:
                    error_type = test_result.error_message.split(':')[0]
                    common_errors[error_type] = common_errors.get(error_type, 0) + 1
            
            for error_type, count in common_errors.items():
                if count > 1:
                    suggestions.append(f"Erreur récurrente: {error_type} - vérifiez votre logique")
        
        # Suggestions de style
        if 'import *' in code:
            suggestions.append("Évitez les imports avec '*', préférez les imports explicites")
        
        if re.search(r'print\s*\(.*\)', code) and not any('print' in tr.test_case.description.lower() for tr in test_results):
            suggestions.append("Retirez les print() de debug de votre code final")
        
        return suggestions
    
    def _log_execution_results(self, user_id: Optional[int], quest_id: Optional[str], report: ExecutionReport):
        """Log les résultats d'exécution."""
        logger.info(
            f"Exécution terminée - User: {user_id}, Quest: {quest_id}, "
            f"Score: {report.total_score}/{report.max_score}, "
            f"Taux réussite: {report.pass_rate:.1%}, "
            f"Temps: {report.execution_time:.2f}s"
        )
        
        if report.security_issues:
            logger.warning(f"Problèmes de sécurité détectés pour user {user_id}: {report.security_issues}")


class TestCaseBuilder:
    """Constructeur de cas de test."""
    
    @staticmethod
    def create_output_test(
        test_id: str,
        description: str,
        expected_output: str,
        input_data: Optional[str] = None,
        points: int = 1
    ) -> TestCase:
        """Crée un test de sortie."""
        return TestCase(
            id=test_id,
            description=description,
            test_type=TestType.OUTPUT,
            expected_output=expected_output,
            input_data=input_data,
            points=points
        )
    
    @staticmethod
    def create_function_test(
        test_id: str,
        description: str,
        function_name: str,
        function_args: Optional[List] = None,
        function_kwargs: Optional[Dict] = None,
        expected_result: Optional[Any] = None,
        points: int = 1
    ) -> TestCase:
        """Crée un test de fonction."""
        return TestCase(
            id=test_id,
            description=description,
            test_type=TestType.FUNCTION_CALL,
            function_name=function_name,
            function_args=function_args or [],
            function_kwargs=function_kwargs or {},
            expected_result=expected_result,
            points=points
        )
    
    @staticmethod
    def create_assertion_test(
        test_id: str,
        description: str,
        assertion_code: str,
        points: int = 1
    ) -> TestCase:
        """Crée un test avec assertion."""
        return TestCase(
            id=test_id,
            description=description,
            test_type=TestType.ASSERTION,
            assertion_code=assertion_code,
            points=points
        )
    
    @staticmethod
    def create_performance_test(
        test_id: str,
        description: str,
        assertion_code: str,
        timeout: float = 1.0,
        points: int = 1
    ) -> TestCase:
        """Crée un test de performance."""
        return TestCase(
            id=test_id,
            description=description,
            test_type=TestType.PERFORMANCE,
            assertion_code=assertion_code,
            timeout=timeout,
            points=points
        )


# Instance globale
code_executor = AdvancedCodeExecutor()


# Fonctions utilitaires
def execute_code_with_tests(
    code: str,
    test_cases: List[TestCase],
    user_id: Optional[int] = None,
    quest_id: Optional[str] = None
) -> ExecutionReport:
    """
    Fonction utilitaire pour exécuter du code avec tests.
    
    Args:
        code: Code Python à exécuter
        test_cases: Liste des cas de test
        user_id: ID de l'utilisateur
        quest_id: ID de la quête
        
    Returns:
        ExecutionReport complet
    """
    return code_executor.execute_and_evaluate(code, test_cases, user_id, quest_id)


def quick_test_function(
    code: str,
    function_name: str,
    test_inputs: List[Tuple],
    expected_outputs: List[Any]
) -> ExecutionReport:
    """
    Test rapide d'une fonction avec plusieurs entrées/sorties.
    
    Args:
        code: Code contenant la fonction
        function_name: Nom de la fonction à tester
        test_inputs: Liste des tuples d'arguments
        expected_outputs: Liste des résultats attendus
        
    Returns:
        ExecutionReport
    """
    test_cases = []
    
    for i, (inputs, expected) in enumerate(zip(test_inputs, expected_outputs)):
        if isinstance(inputs, tuple):
            args = list(inputs)
        else:
            args = [inputs]
        
        test_case = TestCaseBuilder.create_function_test(
            test_id=f"test_{i+1}",
            description=f"Test {i+1}: {function_name}({', '.join(map(str, args))})",
            function_name=function_name,
            function_args=args,
            expected_result=expected
        )
        test_cases.append(test_case)
    
    return execute_code_with_tests(code, test_cases)


if __name__ == "__main__":
    # Exemple d'utilisation
    sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
    
    # Créer des tests
    test_cases = [
        TestCaseBuilder.create_function_test(
            "test_fib_1", "Test fibonacci(5)", "fibonacci", [5], expected_result=5
        ),
        TestCaseBuilder.create_function_test(
            "test_fib_2", "Test fibonacci(10)", "fibonacci", [10], expected_result=55
        ),
        TestCaseBuilder.create_function_test(
            "test_fact_1", "Test factorial(5)", "factorial", [5], expected_result=120
        )
    ]
    
    # Exécuter
    report = execute_code_with_tests(sample_code, test_cases)
    
    print(f"Score: {report.total_score}/{report.max_score}")
    print(f"Taux de réussite: {report.pass_rate:.1%}")
    print(f"Suggestions: {report.suggestions}")
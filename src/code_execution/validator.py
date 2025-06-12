"""
Validateur de code Python avec analyse statique et vérifications de qualité.
"""

import ast
import re
import sys
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import keyword
import builtins

from src.core.logger import get_logger

logger = get_logger(__name__)


class SeverityLevel(Enum):
    """Niveaux de sévérité des problèmes."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Problème détecté lors de la validation."""
    line: Optional[int]
    column: Optional[int]
    severity: SeverityLevel
    category: str
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class ValidationResult:
    """Résultat de validation du code."""
    is_valid: bool
    is_safe: bool
    syntax_errors: List[ValidationIssue]
    style_issues: List[ValidationIssue]
    security_issues: List[ValidationIssue]
    performance_issues: List[ValidationIssue]
    all_issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 0.0
    max_score: float = 100.0


class SyntaxValidator:
    """Validateur de syntaxe Python."""
    
    @staticmethod
    def validate_syntax(code: str) -> List[ValidationIssue]:
        """Valide la syntaxe Python."""
        issues = []
        
        try:
            # Compilation AST
            tree = ast.parse(code)
            
            # Vérifications supplémentaires
            issues.extend(SyntaxValidator._check_indentation(code))
            issues.extend(SyntaxValidator._check_encoding(code))
            
        except SyntaxError as e:
            issue = ValidationIssue(
                line=e.lineno,
                column=e.offset,
                severity=SeverityLevel.ERROR,
                category="syntax",
                message=f"Erreur de syntaxe: {e.msg}",
                suggestion="Vérifiez la syntaxe Python à cette ligne"
            )
            issues.append(issue)
            
        except IndentationError as e:
            issue = ValidationIssue(
                line=e.lineno,
                column=e.offset,
                severity=SeverityLevel.ERROR,
                category="indentation",
                message=f"Erreur d'indentation: {e.msg}",
                suggestion="Utilisez 4 espaces pour l'indentation"
            )
            issues.append(issue)
            
        return issues
    
    @staticmethod
    def _check_indentation(code: str) -> List[ValidationIssue]:
        """Vérifie la cohérence de l'indentation."""
        issues = []
        lines = code.split('\n')
        
        tab_lines = []
        space_lines = []
        
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                continue
                
            indent = line[:len(line) - len(stripped)]
            
            if '\t' in indent:
                tab_lines.append(i)
            elif ' ' in indent:
                space_lines.append(i)
        
        # Mélange de tabs et espaces
        if tab_lines and space_lines:
            issue = ValidationIssue(
                line=tab_lines[0] if tab_lines else space_lines[0],
                column=1,
                severity=SeverityLevel.WARNING,
                category="style",
                message="Mélange de tabulations et d'espaces pour l'indentation",
                suggestion="Utilisez uniquement 4 espaces pour l'indentation"
            )
            issues.append(issue)
        
        return issues
    
    @staticmethod
    def _check_encoding(code: str) -> List[ValidationIssue]:
        """Vérifie l'encodage des caractères."""
        issues = []
        
        try:
            code.encode('ascii')
        except UnicodeEncodeError:
            # Contient des caractères non-ASCII
            if not re.search(r'#.*coding[:=]\s*([-\w.]+)', code[:200]):
                issue = ValidationIssue(
                    line=1,
                    column=1,
                    severity=SeverityLevel.INFO,
                    category="encoding",
                    message="Code contient des caractères non-ASCII sans déclaration d'encodage",
                    suggestion="Ajoutez '# -*- coding: utf-8 -*-' en début de fichier"
                )
                issues.append(issue)
        
        return issues


class StyleValidator:
    """Validateur de style Python (PEP 8)."""
    
    MAX_LINE_LENGTH = 79
    
    @staticmethod
    def validate_style(code: str) -> List[ValidationIssue]:
        """Valide le style du code selon PEP 8."""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            issues.extend(StyleValidator._check_line_length(line, i))
            issues.extend(StyleValidator._check_whitespace(line, i))
            issues.extend(StyleValidator._check_naming(line, i))
            issues.extend(StyleValidator._check_imports(line, i))
        
        # Vérifications AST
        try:
            tree = ast.parse(code)
            issues.extend(StyleValidator._check_ast_style(tree))
        except SyntaxError:
            pass  # Erreurs de syntaxe déjà gérées
        
        return issues
    
    @staticmethod
    def _check_line_length(line: str, line_num: int) -> List[ValidationIssue]:
        """Vérifie la longueur des lignes."""
        issues = []
        
        if len(line) > StyleValidator.MAX_LINE_LENGTH:
            issue = ValidationIssue(
                line=line_num,
                column=StyleValidator.MAX_LINE_LENGTH,
                severity=SeverityLevel.WARNING,
                category="style",
                message=f"Ligne trop longue ({len(line)} caractères, max {StyleValidator.MAX_LINE_LENGTH})",
                suggestion="Divisez la ligne ou utilisez des parenthèses pour la continuation"
            )
            issues.append(issue)
        
        return issues
    
    @staticmethod
    def _check_whitespace(line: str, line_num: int) -> List[ValidationIssue]:
        """Vérifie les espaces en fin de ligne."""
        issues = []
        
        if line.endswith(' ') or line.endswith('\t'):
            issue = ValidationIssue(
                line=line_num,
                column=len(line.rstrip()) + 1,
                severity=SeverityLevel.INFO,
                category="style",
                message="Espaces en fin de ligne",
                suggestion="Supprimez les espaces en fin de ligne"
            )
            issues.append(issue)
        
        return issues
    
    @staticmethod
    def _check_naming(line: str, line_num: int) -> List[ValidationIssue]:
        """Vérifie les conventions de nommage."""
        issues = []
        
        # Variables avec un seul caractère (sauf compteurs)
        var_pattern = r'\b([a-zA-Z])\s*='
        matches = re.finditer(var_pattern, line)
        
        for match in matches:
            var_name = match.group(1)
            if var_name not in ['i', 'j', 'k', 'x', 'y', 'z'] and not var_name.isupper():
                issue = ValidationIssue(
                    line=line_num,
                    column=match.start(),
                    severity=SeverityLevel.INFO,
                    category="naming",
                    message=f"Nom de variable trop court: '{var_name}'",
                    suggestion="Utilisez des noms de variables descriptifs"
                )
                issues.append(issue)
        
        # Noms en CamelCase pour les variables (devrait être snake_case)
        camel_pattern = r'\b([a-z]+[A-Z][a-zA-Z]*)\s*='
        matches = re.finditer(camel_pattern, line)
        
        for match in matches:
            var_name = match.group(1)
            snake_case = re.sub(r'([A-Z])', r'_\1', var_name).lower()
            issue = ValidationIssue(
                line=line_num,
                column=match.start(),
                severity=SeverityLevel.INFO,
                category="naming",
                message=f"Utilisez snake_case pour les variables: '{var_name}' → '{snake_case}'",
                suggestion=f"Renommez '{var_name}' en '{snake_case}'"
            )
            issues.append(issue)
        
        return issues
    
    @staticmethod
    def _check_imports(line: str, line_num: int) -> List[ValidationIssue]:
        """Vérifie les imports."""
        issues = []
        
        # Import avec *
        if re.search(r'from\s+\w+\s+import\s+\*', line):
            issue = ValidationIssue(
                line=line_num,
                column=1,
                severity=SeverityLevel.WARNING,
                category="imports",
                message="Évitez 'from module import *'",
                suggestion="Importez seulement les fonctions nécessaires explicitement"
            )
            issues.append(issue)
        
        # Imports multiples sur une ligne
        if line.strip().startswith('import ') and ',' in line:
            issue = ValidationIssue(
                line=line_num,
                column=1,
                severity=SeverityLevel.INFO,
                category="imports",
                message="Évitez les imports multiples sur une ligne",
                suggestion="Séparez chaque import sur sa propre ligne"
            )
            issues.append(issue)
        
        return issues
    
    @staticmethod
    def _check_ast_style(tree: ast.AST) -> List[ValidationIssue]:
        """Vérifie le style via l'AST."""
        issues = []
        
        for node in ast.walk(tree):
            # Fonctions sans docstring
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node):
                    issue = ValidationIssue(
                        line=node.lineno,
                        column=node.col_offset,
                        severity=SeverityLevel.INFO,
                        category="documentation",
                        message=f"Fonction '{node.name}' sans docstring",
                        suggestion="Ajoutez une docstring décrivant le but de la fonction"
                    )
                    issues.append(issue)
                
                # Trop de paramètres
                if len(node.args.args) > 5:
                    issue = ValidationIssue(
                        line=node.lineno,
                        column=node.col_offset,
                        severity=SeverityLevel.WARNING,
                        category="complexity",
                        message=f"Fonction '{node.name}' a trop de paramètres ({len(node.args.args)})",
                        suggestion="Considérez regrouper les paramètres ou diviser la fonction"
                    )
                    issues.append(issue)
            
            # Classes sans docstring
            elif isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    issue = ValidationIssue(
                        line=node.lineno,
                        column=node.col_offset,
                        severity=SeverityLevel.INFO,
                        category="documentation",
                        message=f"Classe '{node.name}' sans docstring",
                        suggestion="Ajoutez une docstring décrivant le but de la classe"
                    )
                    issues.append(issue)
        
        return issues


class SecurityValidator:
    """Validateur de sécurité du code."""
    
    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'reload', 'vars', 'globals', 'locals'
    }
    
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'glob', 'tempfile',
        'multiprocessing', 'threading', 'socket', 'urllib', 'requests',
        'pickle', 'shelve', 'ctypes', 'importlib'
    }
    
    @staticmethod
    def validate_security(code: str) -> List[ValidationIssue]:
        """Valide la sécurité du code."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                issues.extend(SecurityValidator._check_dangerous_calls(node))
                issues.extend(SecurityValidator._check_dangerous_imports(node))
                issues.extend(SecurityValidator._check_infinite_loops(node))
                
        except SyntaxError:
            pass  # Déjà géré par SyntaxValidator
        
        # Vérifications par pattern
        issues.extend(SecurityValidator._check_dangerous_patterns(code))
        
        return issues
    
    @staticmethod
    def _check_dangerous_calls(node: ast.AST) -> List[ValidationIssue]:
        """Vérifie les appels de fonctions dangereuses."""
        issues = []
        
        if isinstance(node, ast.Call):
            func_name = None
            
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            
            if func_name in SecurityValidator.DANGEROUS_FUNCTIONS:
                issue = ValidationIssue(
                    line=getattr(node, 'lineno', None),
                    column=getattr(node, 'col_offset', None),
                    severity=SeverityLevel.CRITICAL,
                    category="security",
                    message=f"Fonction dangereuse détectée: {func_name}()",
                    suggestion="Cette fonction peut présenter des risques de sécurité"
                )
                issues.append(issue)
        
        return issues
    
    @staticmethod
    def _check_dangerous_imports(node: ast.AST) -> List[ValidationIssue]:
        """Vérifie les imports dangereux."""
        issues = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in SecurityValidator.DANGEROUS_MODULES:
                    issue = ValidationIssue(
                        line=getattr(node, 'lineno', None),
                        column=getattr(node, 'col_offset', None),
                        severity=SeverityLevel.ERROR,
                        category="security",
                        message=f"Import dangereux: {alias.name}",
                        suggestion="Ce module n'est pas autorisé dans l'environnement d'apprentissage"
                    )
                    issues.append(issue)
        
        elif isinstance(node, ast.ImportFrom):
            if node.module in SecurityValidator.DANGEROUS_MODULES:
                issue = ValidationIssue(
                    line=getattr(node, 'lineno', None),
                    column=getattr(node, 'col_offset', None),
                    severity=SeverityLevel.ERROR,
                    category="security",
                    message=f"Import dangereux: from {node.module}",
                    suggestion="Ce module n'est pas autorisé dans l'environnement d'apprentissage"
                )
                issues.append(issue)
        
        return issues
    
    @staticmethod
    def _check_infinite_loops(node: ast.AST) -> List[ValidationIssue]:
        """Détecte les boucles infinies potentielles."""
        issues = []
        
        if isinstance(node, ast.While):
            # while True sans break
            if (isinstance(node.test, ast.Constant) and node.test.value is True) or \
               (isinstance(node.test, ast.NameConstant) and node.test.value is True):
                
                has_break = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Break):
                        has_break = True
                        break
                
                if not has_break:
                    issue = ValidationIssue(
                        line=getattr(node, 'lineno', None),
                        column=getattr(node, 'col_offset', None),
                        severity=SeverityLevel.WARNING,
                        category="security",
                        message="Boucle infinie potentielle détectée",
                        suggestion="Ajoutez une condition de sortie ou un break"
                    )
                    issues.append(issue)
        
        return issues
    
    @staticmethod
    def _check_dangerous_patterns(code: str) -> List[ValidationIssue]:
        """Vérifie les patterns dangereux dans le code."""
        issues = []
        lines = code.split('\n')
        
        dangerous_patterns = [
            (r'while\s+True\s*:', "Boucle infinie détectée"),
            (r'for\s+\w+\s+in\s+range\s*\(\s*\d{6,}\s*\)', "Boucle avec très grand nombre d'itérations"),
            (r'recursion|recursive', "Récursion détectée - attention aux stack overflow"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, message in dangerous_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issue = ValidationIssue(
                        line=i,
                        column=1,
                        severity=SeverityLevel.WARNING,
                        category="security",
                        message=message,
                        suggestion="Vérifiez que le code ne causera pas de problèmes de performance"
                    )
                    issues.append(issue)
        
        return issues


class PerformanceValidator:
    """Validateur de performance du code."""
    
    @staticmethod
    def validate_performance(code: str) -> List[ValidationIssue]:
        """Analyse les problèmes de performance potentiels."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                issues.extend(PerformanceValidator._check_nested_loops(node))
                issues.extend(PerformanceValidator._check_string_concatenation(node))
                issues.extend(PerformanceValidator._check_list_operations(node))
                
        except SyntaxError:
            pass
        
        issues.extend(PerformanceValidator._check_performance_patterns(code))
        
        return issues
    
    @staticmethod
    def _check_nested_loops(node: ast.AST) -> List[ValidationIssue]:
        """Détecte les boucles imbriquées."""
        issues = []
        
        def count_nested_loops(node, depth=0):
            if isinstance(node, (ast.For, ast.While)):
                depth += 1
                if depth > 2:
                    issue = ValidationIssue(
                        line=getattr(node, 'lineno', None),
                        column=getattr(node, 'col_offset', None),
                        severity=SeverityLevel.WARNING,
                        category="performance",
                        message=f"Boucles imbriquées détectées (profondeur: {depth})",
                        suggestion="Considérez optimiser l'algorithme ou utiliser des structures de données plus efficaces"
                    )
                    issues.append(issue)
            
            for child in ast.iter_child_nodes(node):
                count_nested_loops(child, depth)
        
        count_nested_loops(node)
        return issues
    
    @staticmethod
    def _check_string_concatenation(node: ast.AST) -> List[ValidationIssue]:
        """Détecte les concaténations de chaînes inefficaces."""
        issues = []
        
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            # Vérifier si c'est une addition de chaînes
            if isinstance(node.left, ast.Str) or isinstance(node.right, ast.Str):
                issue = ValidationIssue(
                    line=getattr(node, 'lineno', None),
                    column=getattr(node, 'col_offset', None),
                    severity=SeverityLevel.INFO,
                    category="performance",
                    message="Concaténation de chaînes avec +",
                    suggestion="Utilisez f-strings ou join() pour de meilleures performances"
                )
                issues.append(issue)
        
        return issues
    
    @staticmethod
    def _check_list_operations(node: ast.AST) -> List[ValidationIssue]:
        """Vérifie les opérations de liste inefficaces."""
        issues = []
        
        # Vérifier append() dans une boucle
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if child.func.attr == 'append':
                        issue = ValidationIssue(
                            line=getattr(child, 'lineno', None),
                            column=getattr(child, 'col_offset', None),
                            severity=SeverityLevel.INFO,
                            category="performance",
                            message="append() dans une boucle",
                            suggestion="Considérez utiliser une list comprehension"
                        )
                        issues.append(issue)
        
        return issues
    
    @staticmethod
    def _check_performance_patterns(code: str) -> List[ValidationIssue]:
        """Vérifie les patterns de performance dans le code."""
        issues = []
        lines = code.split('\n')
        
        patterns = [
            (r'range\s*\(\s*len\s*\(', "Utilisez enumerate() au lieu de range(len())"),
            (r'\.append\s*\([^)]*\)\s*, "Dans une boucle, considérez une list comprehension"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, suggestion in patterns:
                if re.search(pattern, line):
                    issue = ValidationIssue(
                        line=i,
                        column=1,
                        severity=SeverityLevel.INFO,
                        category="performance",
                        message="Pattern inefficace détecté",
                        suggestion=suggestion
                    )
                    issues.append(issue)
        
        return issues


class ComprehensiveValidator:
    """Validateur complet combinant tous les aspects."""
    
    def __init__(self):
        self.syntax_validator = SyntaxValidator()
        self.style_validator = StyleValidator()
        self.security_validator = SecurityValidator()
        self.performance_validator = PerformanceValidator()
    
    def validate_code(self, code: str) -> ValidationResult:
        """
        Validation complète du code.
        
        Args:
            code: Code Python à valider
            
        Returns:
            ValidationResult complet
        """
        # Validation syntaxe
        syntax_issues = self.syntax_validator.validate_syntax(code)
        
        # Si erreurs de syntaxe critiques, arrêter ici
        critical_syntax_errors = [i for i in syntax_issues if i.severity == SeverityLevel.ERROR]
        if critical_syntax_errors:
            return ValidationResult(
                is_valid=False,
                is_safe=False,
                syntax_errors=syntax_issues,
                style_issues=[],
                security_issues=[],
                performance_issues=[],
                all_issues=syntax_issues,
                score=0.0
            )
        
        # Validation style
        style_issues = self.style_validator.validate_style(code)
        
        # Validation sécurité
        security_issues = self.security_validator.validate_security(code)
        
        # Validation performance
        performance_issues = self.performance_validator.validate_performance(code)
        
        # Agréger tous les problèmes
        all_issues = syntax_issues + style_issues + security_issues + performance_issues
        
        # Évaluer la sécurité
        critical_security = [i for i in security_issues if i.severity in [SeverityLevel.ERROR, SeverityLevel.CRITICAL]]
        is_safe = len(critical_security) == 0
        
        # Évaluer la validité
        critical_errors = [i for i in all_issues if i.severity == SeverityLevel.ERROR]
        is_valid = len(critical_errors) == 0
        
        # Calculer le score
        score = self._calculate_score(all_issues)
        
        return ValidationResult(
            is_valid=is_valid,
            is_safe=is_safe,
            syntax_errors=syntax_issues,
            style_issues=style_issues,
            security_issues=security_issues,
            performance_issues=performance_issues,
            all_issues=all_issues,
            score=score
        )
    
    def _calculate_score(self, issues: List[ValidationIssue]) -> float:
        """Calcule un score de qualité du code."""
        base_score = 100.0
        
        for issue in issues:
            if issue.severity == SeverityLevel.CRITICAL:
                base_score -= 25
            elif issue.severity == SeverityLevel.ERROR:
                base_score -= 15
            elif issue.severity == SeverityLevel.WARNING:
                base_score -= 5
            elif issue.severity == SeverityLevel.INFO:
                base_score -= 2
        
        return max(0.0, base_score)
    
    def get_validation_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """Génère un résumé de validation."""
        return {
            'is_valid': result.is_valid,
            'is_safe': result.is_safe,
            'score': result.score,
            'total_issues': len(result.all_issues),
            'by_severity': {
                'critical': len([i for i in result.all_issues if i.severity == SeverityLevel.CRITICAL]),
                'error': len([i for i in result.all_issues if i.severity == SeverityLevel.ERROR]),
                'warning': len([i for i in result.all_issues if i.severity == SeverityLevel.WARNING]),
                'info': len([i for i in result.all_issues if i.severity == SeverityLevel.INFO])
            },
            'by_category': {
                'syntax': len(result.syntax_errors),
                'style': len(result.style_issues),
                'security': len(result.security_issues),
                'performance': len(result.performance_issues)
            }
        }


# Instance globale
code_validator = ComprehensiveValidator()


# Fonctions utilitaires
def validate_python_code(code: str) -> ValidationResult:
    """
    Valide du code Python de manière complète.
    
    Args:
        code: Code Python à valider
        
    Returns:
        ValidationResult avec tous les détails
    """
    return code_validator.validate_code(code)


def quick_syntax_check(code: str) -> Tuple[bool, Optional[str]]:
    """
    Vérification rapide de la syntaxe.
    
    Args:
        code: Code à vérifier
        
    Returns:
        Tuple (is_valid, error_message)
    """
    syntax_issues = SyntaxValidator.validate_syntax(code)
    
    if not syntax_issues:
        return True, None
    
    error_issues = [i for i in syntax_issues if i.severity == SeverityLevel.ERROR]
    if error_issues:
        return False, error_issues[0].message
    
    return True, None


def get_security_assessment(code: str) -> Dict[str, Any]:
    """
    Évaluation de sécurité rapide.
    
    Args:
        code: Code à évaluer
        
    Returns:
        Dictionnaire avec l'évaluation de sécurité
    """
    security_issues = SecurityValidator.validate_security(code)
    
    return {
        'is_safe': len([i for i in security_issues if i.severity in [SeverityLevel.ERROR, SeverityLevel.CRITICAL]]) == 0,
        'risk_level': 'high' if any(i.severity == SeverityLevel.CRITICAL for i in security_issues) else 
                     'medium' if any(i.severity == SeverityLevel.ERROR for i in security_issues) else 
                     'low',
        'issues': [{'message': i.message, 'severity': i.severity.value} for i in security_issues],
        'total_issues': len(security_issues)
    }


if __name__ == "__main__":
    # Test du validateur
    test_code = """
import os  # Import dangereux
def test():  # Pas de docstring
    x=1+2  # Espacement
    while True:  # Boucle infinie
        print("hello")
        if x > 10:
            break
    return x
"""
    
    result = validate_python_code(test_code)
    print(f"Valide: {result.is_valid}")
    print(f"Sûr: {result.is_safe}")
    print(f"Score: {result.score}/100")
    print(f"Problèmes: {len(result.all_issues)}")
    
    for issue in result.all_issues[:5]:  # Afficher les 5 premiers
        print(f"- Ligne {issue.line}: {issue.message} ({issue.severity.value})")
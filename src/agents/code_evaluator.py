# src/agents/code_evaluator.py
"""
Agent évaluateur de code - Analyse et évalue le code Python des utilisateurs
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import ast
import sys
import io
import contextlib
import logging
import re
from datetime import datetime, timezone

from src.agents.base_agent import BaseAgent, AgentType, AgentContext, AgentResponse
from src.code_execution.executor import CodeExecutor
from src.code_execution.validator import CodeValidator as ExecutionValidator
from src.utils import CodeValidator, ValidationResult
from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class CodeEvaluatorAgent(BaseAgent):
    """Agent pour l'évaluation complète du code Python"""
    
    def __init__(self, agent_type: AgentType = AgentType.CODE_EVALUATOR, name: str = None, **kwargs):
        super().__init__(agent_type, name or "code_evaluator")
        
        # Composants d'évaluation
        self.code_validator = CodeValidator()
        self.execution_validator = ExecutionValidator()
        self.code_executor = CodeExecutor()
        self.llm_client = OllamaClient()
        
        # Configuration d'évaluation
        self.strict_mode = kwargs.get('strict_mode', False)
        self.provide_hints = kwargs.get('provide_hints', True)
        self.auto_fix_suggestions = kwargs.get('auto_fix_suggestions', True)
        self.max_execution_time = kwargs.get('max_execution_time', 5)  # secondes
        self.enable_style_check = kwargs.get('enable_style_check', True)
        
        # Critères d'évaluation
        self.evaluation_criteria = {
            'syntax': 0.3,      # 30% - Syntaxe correcte
            'logic': 0.25,      # 25% - Logique algorithmique
            'style': 0.15,      # 15% - Style et bonnes pratiques
            'execution': 0.20,  # 20% - Exécution réussie
            'efficiency': 0.10  # 10% - Efficacité
        }
        
        logger.info(f"Évaluateur de code initialisé (strict_mode: {self.strict_mode})")
    
    async def process(self, input_data: Dict[str, Any], context: AgentContext) -> AgentResponse:
        """
        Évalue le code soumis selon plusieurs critères
        
        Args:
            input_data: Contient 'code', 'expected_output' (optionnel), 'test_cases' (optionnel)
            context: Contexte utilisateur
            
        Returns:
            Évaluation complète du code
        """
        code = input_data.get('code', '').strip()
        expected_output = input_data.get('expected_output')
        test_cases = input_data.get('test_cases', [])
        exercise_type = input_data.get('exercise_type', 'general')
        
        if not code:
            return AgentResponse(
                success=False,
                message="Code vide fourni",
                errors=["Le code à évaluer ne peut pas être vide"]
            )
        
        try:
            # 1. Évaluation syntaxique
            syntax_evaluation = await self._evaluate_syntax(code)
            
            # 2. Évaluation de la logique
            logic_evaluation = await self._evaluate_logic(code, context)
            
            # 3. Évaluation du style
            style_evaluation = await self._evaluate_style(code)
            
            # 4. Évaluation de l'exécution
            execution_evaluation = await self._evaluate_execution(
                code, expected_output, test_cases
            )
            
            # 5. Évaluation de l'efficacité
            efficiency_evaluation = await self._evaluate_efficiency(code)
            
            # 6. Calcul du score global
            overall_score = self._calculate_overall_score({
                'syntax': syntax_evaluation,
                'logic': logic_evaluation,
                'style': style_evaluation,
                'execution': execution_evaluation,
                'efficiency': efficiency_evaluation
            })
            
            # 7. Génération du feedback pédagogique
            feedback = await self._generate_feedback(
                code, overall_score, {
                    'syntax': syntax_evaluation,
                    'logic': logic_evaluation,
                    'style': style_evaluation,
                    'execution': execution_evaluation,
                    'efficiency': efficiency_evaluation
                }, context
            )
            
            # 8. Suggestions d'amélioration
            suggestions = await self._generate_suggestions(
                code, overall_score, context
            )
            
            return AgentResponse(
                success=True,
                message=feedback['main_message'],
                data={
                    'overall_score': overall_score,
                    'detailed_scores': {
                        'syntax': syntax_evaluation.get('score', 0),
                        'logic': logic_evaluation.get('score', 0),
                        'style': style_evaluation.get('score', 0),
                        'execution': execution_evaluation.get('score', 0),
                        'efficiency': efficiency_evaluation.get('score', 0)
                    },
                    'evaluation_details': {
                        'syntax': syntax_evaluation,
                        'logic': logic_evaluation,
                        'style': style_evaluation,
                        'execution': execution_evaluation,
                        'efficiency': efficiency_evaluation
                    },
                    'feedback': feedback,
                    'exercise_type': exercise_type,
                    'code_metrics': self._calculate_code_metrics(code)
                },
                suggestions=suggestions,
                confidence=overall_score / 100.0,
                reasoning=feedback.get('reasoning', '')
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation du code: {e}")
            return AgentResponse(
                success=False,
                message="Erreur lors de l'évaluation du code",
                errors=[str(e)]
            )
    
    async def _evaluate_syntax(self, code: str) -> Dict[str, Any]:
        """Évalue la syntaxe du code"""
        try:
            # Validation avec notre validateur
            validation = self.code_validator.validate_syntax(code)
            
            if validation.is_valid:
                return {
                    'score': 100,
                    'valid': True,
                    'errors': [],
                    'warnings': [],
                    'ast_tree': validation.data,
                    'details': 'Syntaxe correcte'
                }
            else:
                return {
                    'score': 0,
                    'valid': False,
                    'errors': validation.errors,
                    'warnings': validation.warnings,
                    'details': 'Erreurs de syntaxe détectées'
                }
                
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation syntaxique: {e}")
            return {
                'score': 0,
                'valid': False,
                'errors': [f"Erreur d'évaluation: {str(e)}"],
                'warnings': [],
                'details': 'Échec de l\'analyse syntaxique'
            }
    
    async def _evaluate_logic(self, code: str, context: AgentContext) -> Dict[str, Any]:
        """Évalue la logique du code"""
        try:
            # Validation logique de base
            logic_validation = self.code_validator.validate_logic(code)
            
            # Analyse AST pour détecter les patterns logiques
            logical_patterns = self._analyze_logical_patterns(code)
            
            # Score de base selon la validation
            base_score = 100 if logic_validation.is_valid else 70
            
            # Ajustements selon les patterns détectés
            score_adjustments = 0
            
            # Patterns positifs
            if logical_patterns['has_functions']:
                score_adjustments += 5
            if logical_patterns['has_error_handling']:
                score_adjustments += 10
            if logical_patterns['good_variable_names']:
                score_adjustments += 5
            if logical_patterns['has_comments']:
                score_adjustments += 5
            
            # Patterns négatifs
            if logical_patterns['infinite_loops']:
                score_adjustments -= 20
            if logical_patterns['unused_variables']:
                score_adjustments -= 5
            if logical_patterns['deep_nesting']:
                score_adjustments -= 10
            
            final_score = max(0, min(100, base_score + score_adjustments))
            
            return {
                'score': final_score,
                'valid': logic_validation.is_valid,
                'errors': logic_validation.errors,
                'warnings': logic_validation.warnings,
                'patterns': logical_patterns,
                'details': self._describe_logic_evaluation(logical_patterns, final_score)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation logique: {e}")
            return {
                'score': 50,
                'valid': False,
                'errors': [f"Erreur d'analyse logique: {str(e)}"],
                'warnings': [],
                'patterns': {},
                'details': 'Analyse logique incomplète'
            }
    
    def _analyze_logical_patterns(self, code: str) -> Dict[str, Any]:
        """Analyse les patterns logiques dans le code"""
        patterns = {
            'has_functions': False,
            'has_classes': False,
            'has_loops': False,
            'has_conditions': False,
            'has_error_handling': False,
            'has_comments': False,
            'good_variable_names': True,
            'infinite_loops': False,
            'unused_variables': False,
            'deep_nesting': False,
            'complexity_score': 0
        }
        
        try:
            tree = ast.parse(code)
            
            # Compteurs pour l'analyse
            function_count = 0
            variable_names = []
            
            for node in ast.walk(tree):
                # Fonctions
                if isinstance(node, ast.FunctionDef):
                    patterns['has_functions'] = True
                    function_count += 1
                
                # Classes
                elif isinstance(node, ast.ClassDef):
                    patterns['has_classes'] = True
                
                # Boucles
                elif isinstance(node, (ast.For, ast.While)):
                    patterns['has_loops'] = True
                    # Détecter les boucles infinies potentielles
                    if isinstance(node, ast.While):
                        if isinstance(node.test, ast.Constant) and node.test.value is True:
                            patterns['infinite_loops'] = True
                
                # Conditions
                elif isinstance(node, ast.If):
                    patterns['has_conditions'] = True
                
                # Gestion d'erreurs
                elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                    patterns['has_error_handling'] = True
                
                # Variables
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    variable_names.append(node.id)
            
            # Vérifier les noms de variables
            bad_names = [name for name in variable_names 
                        if len(name) <= 1 or name in ['a', 'b', 'c', 'x', 'y', 'z']]
            if len(bad_names) > len(variable_names) * 0.3:  # Plus de 30% de mauvais noms
                patterns['good_variable_names'] = False
            
            # Vérifier les commentaires
            patterns['has_comments'] = '#' in code or '"""' in code or "'''" in code
            
            # Calculer la complexité cyclomatique approximative
            complexity = 1  # Base
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            patterns['complexity_score'] = complexity
            
            # Détecter l'imbrication profonde
            if complexity > 10:
                patterns['deep_nesting'] = True
            
        except SyntaxError:
            # En cas d'erreur de syntaxe, on ne peut pas analyser
            pass
        except Exception as e:
            logger.warning(f"Erreur lors de l'analyse des patterns: {e}")
        
        return patterns
    
    def _describe_logic_evaluation(self, patterns: Dict[str, Any], score: int) -> str:
        """Génère une description de l'évaluation logique"""
        descriptions = []
        
        if patterns.get('has_functions'):
            descriptions.append("Code structuré avec des fonctions")
        if patterns.get('has_error_handling'):
            descriptions.append("Gestion d'erreurs présente")
        if patterns.get('good_variable_names'):
            descriptions.append("Noms de variables appropriés")
        else:
            descriptions.append("Noms de variables à améliorer")
        
        if patterns.get('infinite_loops'):
            descriptions.append("Attention aux boucles infinies potentielles")
        if patterns.get('deep_nesting'):
            descriptions.append("Complexité élevée détectée")
        
        complexity = patterns.get('complexity_score', 0)
        if complexity <= 5:
            descriptions.append("Complexité simple")
        elif complexity <= 10:
            descriptions.append("Complexité modérée")
        else:
            descriptions.append("Complexité élevée")
        
        return "; ".join(descriptions) if descriptions else "Analyse logique de base"
    
    async def _evaluate_style(self, code: str) -> Dict[str, Any]:
        """Évalue le style du code"""
        if not self.enable_style_check:
            return {'score': 100, 'details': 'Vérification de style désactivée'}
        
        try:
            style_validation = self.code_validator.validate_style(code)
            
            # Score de base
            base_score = 100
            
            # Déductions selon les problèmes de style
            style_issues = []
            deductions = 0
            
            for warning in style_validation.warnings:
                if 'ligne trop longue' in warning.lower():
                    deductions += 2
                    style_issues.append('Lignes trop longues')
                elif 'espaces' in warning.lower():
                    deductions += 1
                    style_issues.append('Problèmes d\'espacement')
                elif 'camelcase' in warning.lower():
                    deductions += 3
                    style_issues.append('Convention de nommage')
                else:
                    deductions += 1
                    style_issues.append('Autre problème de style')
            
            # Vérifications supplémentaires
            additional_checks = self._additional_style_checks(code)
            deductions += additional_checks['deductions']
            style_issues.extend(additional_checks['issues'])
            
            final_score = max(0, base_score - deductions)
            
            return {
                'score': final_score,
                'warnings': style_validation.warnings,
                'issues': list(set(style_issues)),
                'suggestions': additional_checks.get('suggestions', []),
                'details': f'Style évalué avec {len(style_issues)} problèmes détectés'
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation du style: {e}")
            return {
                'score': 80,
                'warnings': [],
                'issues': [],
                'suggestions': [],
                'details': 'Évaluation de style incomplète'
            }
    
    def _additional_style_checks(self, code: str) -> Dict[str, Any]:
        """Vérifications de style supplémentaires"""
        issues = []
        suggestions = []
        deductions = 0
        
        lines = code.split('\n')
        
        # Vérifier l'indentation
        inconsistent_indent = False
        indent_levels = []
        for line in lines:
            if line.strip():  # Ignorer les lignes vides
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 0:
                    indent_levels.append(leading_spaces)
        
        if indent_levels:
            # Vérifier la cohérence (doit être multiple de 4)
            inconsistent = any(level % 4 != 0 for level in indent_levels)
            if inconsistent:
                issues.append('Indentation incohérente')
                suggestions.append('Utilisez 4 espaces pour l\'indentation')
                deductions += 5
        
        # Vérifier les lignes vides
        consecutive_empty = 0
        max_consecutive_empty = 0
        for line in lines:
            if not line.strip():
                consecutive_empty += 1
                max_consecutive_empty = max(max_consecutive_empty, consecutive_empty)
            else:
                consecutive_empty = 0
        
        if max_consecutive_empty > 2:
            issues.append('Trop de lignes vides consécutives')
            suggestions.append('Limitez les lignes vides consécutives à 2')
            deductions += 2
        
        # Vérifier les imports
        import_lines = [line for line in lines if line.strip().startswith(('import ', 'from '))]
        if len(import_lines) > len(set(import_lines)):
            issues.append('Imports dupliqués')
            suggestions.append('Supprimez les imports en double')
            deductions += 3
        
        # Vérifier la longueur des fonctions
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    if func_lines > 50:
                        issues.append('Fonctions très longues')
                        suggestions.append('Divisez les fonctions longues en plus petites')
                        deductions += 5
                        break
        except:
            pass
        
        return {
            'deductions': deductions,
            'issues': issues,
            'suggestions': suggestions
        }
    
    async def _evaluate_execution(self, code: str, expected_output: Optional[str] = None, 
                                 test_cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Évalue l'exécution du code"""
        try:
            # Exécution de base
            execution_result = await self.code_executor.execute_code_safely(
                code, timeout=self.max_execution_time
            )
            
            base_score = 100 if execution_result['success'] else 0
            
            # Tests avec cas de test
            test_results = []
            if test_cases:
                for i, test_case in enumerate(test_cases):
                    test_result = await self._run_test_case(code, test_case)
                    test_results.append(test_result)
            
            # Tests avec sortie attendue
            output_match = False
            if expected_output and execution_result['success']:
                actual_output = execution_result.get('output', '').strip()
                expected_clean = expected_output.strip()
                output_match = actual_output == expected_clean
            
            # Calculer le score final
            final_score = base_score
            
            if test_cases:
                passed_tests = sum(1 for test in test_results if test['passed'])
                test_score = (passed_tests / len(test_cases)) * 100
                final_score = (final_score + test_score) / 2
            
            if expected_output:
                if output_match:
                    final_score = min(100, final_score + 10)
                else:
                    final_score = max(0, final_score - 20)
            
            return {
                'score': final_score,
                'success': execution_result['success'],
                'output': execution_result.get('output'),
                'error': execution_result.get('error'),
                'execution_time': execution_result.get('execution_time'),
                'test_results': test_results,
                'output_match': output_match,
                'expected_output': expected_output,
                'details': self._describe_execution_result(
                    execution_result, test_results, output_match
                )
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation d'exécution: {e}")
            return {
                'score': 0,
                'success': False,
                'output': None,
                'error': str(e),
                'execution_time': None,
                'test_results': [],
                'output_match': False,
                'details': 'Échec de l\'exécution'
            }
    
    async def _run_test_case(self, code: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute un cas de test spécifique"""
        try:
            # Préparer le code avec les entrées du test
            test_input = test_case.get('input', '')
            expected_output = test_case.get('expected_output', '')
            
            # Modifier le code pour inclure les entrées
            if test_input:
                # Remplacer input() par des valeurs prédéfinies
                modified_code = self._inject_test_input(code, test_input)
            else:
                modified_code = code
            
            # Exécuter le code modifié
            result = await self.code_executor.execute_code_safely(
                modified_code, timeout=self.max_execution_time
            )
            
            # Comparer la sortie
            if result['success']:
                actual_output = result.get('output', '').strip()
                expected_clean = expected_output.strip()
                passed = actual_output == expected_clean
            else:
                passed = False
            
            return {
                'passed': passed,
                'input': test_input,
                'expected_output': expected_output,
                'actual_output': result.get('output', ''),
                'error': result.get('error'),
                'execution_time': result.get('execution_time')
            }
            
        except Exception as e:
            return {
                'passed': False,
                'input': test_case.get('input', ''),
                'expected_output': test_case.get('expected_output', ''),
                'actual_output': '',
                'error': str(e),
                'execution_time': None
            }
    
    def _inject_test_input(self, code: str, test_input: str) -> str:
        """Injecte les entrées de test dans le code"""
        # Simple remplacement pour les cas basiques
        # Pour des cas plus complexes, il faudrait une analyse AST plus poussée
        if 'input()' in code:
            inputs = test_input.split('\n') if '\n' in test_input else [test_input]
            input_values = ', '.join([f"'{inp}'" for inp in inputs])
            
            # Remplacer input() par des valeurs prédéfinies
            modified = f"""
test_inputs = [{input_values}]
input_index = 0

def mock_input(prompt=''):
    global input_index, test_inputs
    if input_index < len(test_inputs):
        value = test_inputs[input_index]
        input_index += 1
        return value
    return ''

# Remplacer input par notre mock
input = mock_input

{code}
"""
            return modified
        
        return code
    
    def _describe_execution_result(self, execution_result: Dict[str, Any], 
                                 test_results: List[Dict[str, Any]], 
                                 output_match: bool) -> str:
        """Génère une description du résultat d'exécution"""
        if not execution_result['success']:
            return f"Échec d'exécution: {execution_result.get('error', 'Erreur inconnue')}"
        
        descriptions = ["Exécution réussie"]
        
        if test_results:
            passed_tests = sum(1 for test in test_results if test['passed'])
            descriptions.append(f"{passed_tests}/{len(test_results)} tests réussis")
        
        if output_match:
            descriptions.append("Sortie conforme à l'attendu")
        elif output_match is False:  # Explicitement False, pas None
            descriptions.append("Sortie différente de l'attendu")
        
        execution_time = execution_result.get('execution_time')
        if execution_time:
            descriptions.append(f"Temps d'exécution: {execution_time:.3f}s")
        
        return "; ".join(descriptions)
    
    async def _evaluate_efficiency(self, code: str) -> Dict[str, Any]:
        """Évalue l'efficacité du code"""
        try:
            # Analyse statique de l'efficacité
            efficiency_score = 100
            issues = []
            suggestions = []
            
            # Analyser la complexité algorithmique
            complexity_analysis = self._analyze_algorithmic_complexity(code)
            
            # Détecter les patterns inefficaces
            inefficient_patterns = self._detect_inefficient_patterns(code)
            
            # Calculer les déductions
            for pattern in inefficient_patterns:
                efficiency_score -= pattern['deduction']
                issues.append(pattern['issue'])
                suggestions.append(pattern['suggestion'])
            
            # Ajuster selon la complexité
            if complexity_analysis['estimated_complexity'] == 'O(n²)':
                efficiency_score -= 10
                suggestions.append('Considérez un algorithme plus efficace')
            elif complexity_analysis['estimated_complexity'] == 'O(n³)' or 'O(2^n)' in complexity_analysis['estimated_complexity']:
                efficiency_score -= 20
                suggestions.append('Algorithme très inefficace détecté')
            
            final_score = max(0, efficiency_score)
            
            return {
                'score': final_score,
                'complexity': complexity_analysis,
                'issues': issues,
                'suggestions': suggestions,
                'details': f'Efficacité évaluée: {complexity_analysis["estimated_complexity"]}'
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation d'efficacité: {e}")
            return {
                'score': 80,
                'complexity': {'estimated_complexity': 'Non déterminé'},
                'issues': [],
                'suggestions': [],
                'details': 'Évaluation d\'efficacité incomplète'
            }
    
    def _analyze_algorithmic_complexity(self, code: str) -> Dict[str, Any]:
        """Analyse la complexité algorithmique approximative"""
        try:
            tree = ast.parse(code)
            
            nested_loops = 0
            max_nesting = 0
            current_nesting = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                    nested_loops += 1
                # Approximation simple - dans un vrai analyseur il faudrait
                # une analyse plus sophistiquée du graphe de contrôle
            
            # Estimation basique
            if max_nesting == 0:
                complexity = 'O(1)'
            elif max_nesting == 1:
                complexity = 'O(n)'
            elif max_nesting == 2:
                complexity = 'O(n²)'
            elif max_nesting == 3:
                complexity = 'O(n³)'
            else:
                complexity = f'O(n^{max_nesting})'
            
            return {
                'estimated_complexity': complexity,
                'nested_loops': nested_loops,
                'max_nesting_depth': max_nesting
            }
            
        except:
            return {
                'estimated_complexity': 'Non déterminé',
                'nested_loops': 0,
                'max_nesting_depth': 0
            }
    
    def _detect_inefficient_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Détecte des patterns inefficaces dans le code"""
        patterns = []
        
        # Patterns à détecter
        inefficient_checks = [
            {
                'pattern': r'\.append\([^)]+\)\s*in\s+for\s+.*\s+in\s+range\(',
                'issue': 'Utilisation de append() dans une boucle range',
                'suggestion': 'Considérez la compréhension de liste',
                'deduction': 5
            },
            {
                'pattern': r'for\s+\w+\s+in\s+range\(len\(',
                'issue': 'Boucle avec range(len()) au lieu d\'itération directe',
                'suggestion': 'Itérez directement sur la liste ou utilisez enumerate()',
                'deduction': 3
            },
            {
                'pattern': r'list\(.*\)\s*\+\s*list\(',
                'issue': 'Concaténation inefficace de listes',
                'suggestion': 'Utilisez extend() ou une compréhension de liste',
                'deduction': 5
            }
        ]
        
        for check in inefficient_checks:
            if re.search(check['pattern'], code, re.IGNORECASE):
                patterns.append({
                    'issue': check['issue'],
                    'suggestion': check['suggestion'],
                    'deduction': check['deduction']
                })
        
        # Vérifications supplémentaires
        if 'time.sleep(' in code:
            patterns.append({
                'issue': 'Utilisation de time.sleep()',
                'suggestion': 'Évitez les pauses dans les algorithmes',
                'deduction': 10
            })
        
        return patterns
    
    def _calculate_overall_score(self, evaluations: Dict[str, Dict[str, Any]]) -> float:
        """Calcule le score global pondéré"""
        total_score = 0.0
        
        for criterion, weight in self.evaluation_criteria.items():
            evaluation = evaluations.get(criterion, {})
            score = evaluation.get('score', 0)
            total_score += score * weight
        
        return round(total_score, 1)
    
    async def _generate_feedback(self, code: str, overall_score: float, 
                               evaluations: Dict[str, Dict[str, Any]], 
                               context: AgentContext) -> Dict[str, Any]:
        """Génère un feedback pédagogique personnalisé"""
        
        # Déterminer le niveau de feedback selon le score
        if overall_score >= 90:
            level = "excellent"
        elif overall_score >= 80:
            level = "très bien"
        elif overall_score >= 70:
            level = "bien"
        elif overall_score >= 60:
            level = "correct"
        else:
            level = "à améliorer"
        
        # Message principal
        main_message = f"Votre code obtient un score de {overall_score}/100 ({level})"
        
        # Détails par critère
        feedback_details = []
        
        for criterion, evaluation in evaluations.items():
            score = evaluation.get('score', 0)
            if score >= 90:
                feedback_details.append(f"✅ {criterion.title()}: Excellent ({score}/100)")
            elif score >= 70:
                feedback_details.append(f"👍 {criterion.title()}: Bien ({score}/100)")
            else:
                feedback_details.append(f"⚠️ {criterion.title()}: À améliorer ({score}/100)")
        
        # Feedback spécifique selon le contexte utilisateur
        personalized_feedback = ""
        if context.user_progress:
            user_level = context.user_progress.get('level', 'beginner')
            if user_level == 'beginner' and overall_score >= 70:
                personalized_feedback = "Excellent travail pour un débutant ! Continuez ainsi."
            elif user_level == 'advanced' and overall_score < 80:
                personalized_feedback = "Avec votre niveau, vous pouvez faire encore mieux."
        
        # Générer un feedback LLM personnalisé
        llm_feedback = await self._generate_llm_feedback(
            code, overall_score, evaluations, context
        )
        
        return {
            'main_message': main_message,
            'level': level,
            'details': feedback_details,
            'personalized': personalized_feedback,
            'llm_feedback': llm_feedback,
            'reasoning': f"Score calculé selon {len(self.evaluation_criteria)} critères pondérés"
        }
    
    async def _generate_llm_feedback(self, code: str, overall_score: float,
                                   evaluations: Dict[str, Dict[str, Any]],
                                   context: AgentContext) -> str:
        """Génère un feedback personnalisé avec le LLM"""
        
        # Préparer le contexte pour le LLM
        user_context = ""
        if context.user_progress:
            user_context = f"Niveau utilisateur: {context.user_progress.get('level', 'débutant')}"
        
        # Résumé des évaluations
        eval_summary = []
        for criterion, evaluation in evaluations.items():
            score = evaluation.get('score', 0)
            eval_summary.append(f"{criterion}: {score}/100")
        
        prompt = f"""Génère un feedback pédagogique bienveillant et constructif pour ce code Python.

Code évalué:
```python
{code[:500]}{'...' if len(code) > 500 else ''}
```

Score global: {overall_score}/100
Évaluations détaillées: {'; '.join(eval_summary)}
{user_context}

Le feedback doit être:
- Encourageant et positif
- Spécifique aux points forts et à améliorer
- Adapté au niveau de l'utilisateur
- Avec des conseils concrets d'amélioration
- En français, maximum 200 mots

Concentre-toi sur 2-3 points principaux."""
        
        try:
            feedback = await self.llm_client.generate_async(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )
            return feedback.strip()
        except Exception as e:
            logger.error(f"Erreur génération feedback LLM: {e}")
            return "Continuez vos efforts, votre code montre de bonnes bases !"
    
    async def _generate_suggestions(self, code: str, overall_score: float, 
                                  context: AgentContext) -> List[str]:
        """Génère des suggestions d'amélioration"""
        suggestions = []
        
        # Suggestions basées sur le score
        if overall_score < 60:
            suggestions.append("Concentrez-vous d'abord sur la correction des erreurs de syntaxe")
            suggestions.append("Revoyez les concepts de base de Python")
        elif overall_score < 80:
            suggestions.append("Améliorez la lisibilité de votre code avec de meilleurs noms de variables")
            suggestions.append("Ajoutez des commentaires pour expliquer votre logique")
        else:
            suggestions.append("Explorez des techniques d'optimisation avancées")
            suggestions.append("Considérez l'utilisation de fonctions pour structurer votre code")
        
        # Suggestions selon le niveau utilisateur
        if context.user_progress:
            user_level = context.user_progress.get('level', 'beginner')
            if user_level == 'beginner':
                suggestions.append("Pratiquez avec des exercices simples pour solidifier vos bases")
            elif user_level == 'intermediate':
                suggestions.append("Explorez les structures de données plus avancées")
            else:
                suggestions.append("Étudiez les patterns de conception et l'architecture logicielle")
        
        return suggestions[:5]  # Limiter à 5 suggestions
    
    def _calculate_code_metrics(self, code: str) -> Dict[str, Any]:
        """Calcule des métriques sur le code"""
        lines = code.split('\n')
        
        metrics = {
            'total_lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'total_characters': len(code),
            'average_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
        }
        
        try:
            tree = ast.parse(code)
            
            # Compter les différents éléments
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            imports = len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])
            variables = len([node for node in ast.walk(tree) 
                           if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)])
            
            metrics.update({
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'variables': variables
            })
            
        except SyntaxError:
            # En cas d'erreur de syntaxe, on ne peut pas calculer ces métriques
            metrics.update({
                'functions': 0,
                'classes': 0,
                'imports': 0,
                'variables': 0
            })
        
        return metrics
    
    def get_required_fields(self) -> List[str]:
        """Champs requis pour l'évaluateur"""
        return ['code']
    
    def get_capabilities(self) -> List[str]:
        """Capacités de l'évaluateur de code"""
        return [
            'syntax_validation',
            'logic_analysis',
            'style_checking',
            'execution_testing',
            'efficiency_evaluation',
            'automated_feedback',
            'test_case_execution',
            'performance_analysis',
            'code_metrics_calculation',
            'pedagogical_suggestions'
        ]
    
    async def _health_check_test(self, test_input: Dict[str, Any], test_context: AgentContext) -> bool:
        """Test de santé spécifique à l'évaluateur"""
        try:
            # Test avec du code simple
            test_data = {'code': 'print("Hello, World!")'}
            response = await self.process(test_data, test_context)
            
            return (response.success and 
                    response.data and 
                    'overall_score' in response.data and
                    response.data['overall_score'] > 0)
        except Exception:
            return False
    
    def configure_evaluation_criteria(self, criteria: Dict[str, float]):
        """Configure les critères d'évaluation et leurs poids"""
        total_weight = sum(criteria.values())
        if abs(total_weight - 1.0) > 0.01:  # Tolérance pour les erreurs de flottant
            logger.warning(f"Les poids des critères ne totalisent pas 1.0: {total_weight}")
            return False
        
        self.evaluation_criteria = criteria.copy()
        logger.info(f"Critères d'évaluation mis à jour: {criteria}")
        return True
    
    async def evaluate_code_snippet(self, code: str, context: AgentContext = None) -> Dict[str, Any]:
        """
        Méthode utilitaire pour évaluer rapidement un snippet de code
        
        Args:
            code: Code à évaluer
            context: Contexte optionnel
            
        Returns:
            Résultat d'évaluation simplifié
        """
        if not context:
            context = AgentContext()
        
        input_data = {'code': code}
        response = await self.process(input_data, context)
        
        if response.success:
            return {
                'score': response.data['overall_score'],
                'feedback': response.message,
                'suggestions': response.suggestions,
                'valid_syntax': response.data['detailed_scores']['syntax'] > 0,
                'can_execute': response.data['detailed_scores']['execution'] > 0
            }
        else:
            return {
                'score': 0,
                'feedback': response.message,
                'suggestions': [],
                'valid_syntax': False,
                'can_execute': False,
                'errors': response.errors
            }
    
    async def compare_solutions(self, solutions: List[str], context: AgentContext = None) -> Dict[str, Any]:
        """
        Compare plusieurs solutions de code
        
        Args:
            solutions: Liste des codes à comparer
            context: Contexte utilisateur
            
        Returns:
            Comparaison détaillée des solutions
        """
        if not context:
            context = AgentContext()
        
        evaluations = []
        
        # Évaluer chaque solution
        for i, code in enumerate(solutions):
            input_data = {'code': code}
            response = await self.process(input_data, context)
            
            evaluations.append({
                'index': i,
                'code': code,
                'score': response.data['overall_score'] if response.success else 0,
                'detailed_scores': response.data.get('detailed_scores', {}) if response.success else {},
                'feedback': response.message,
                'success': response.success
            })
        
        # Classer par score
        evaluations.sort(key=lambda x: x['score'], reverse=True)
        
        # Générer la comparaison
        comparison = {
            'best_solution': evaluations[0] if evaluations else None,
            'worst_solution': evaluations[-1] if evaluations else None,
            'all_evaluations': evaluations,
            'average_score': sum(e['score'] for e in evaluations) / len(evaluations) if evaluations else 0,
            'comparison_summary': self._generate_comparison_summary(evaluations)
        }
        
        return comparison
    
    def _generate_comparison_summary(self, evaluations: List[Dict[str, Any]]) -> str:
        """Génère un résumé de comparaison"""
        if not evaluations:
            return "Aucune solution à comparer"
        
        if len(evaluations) == 1:
            return f"Solution unique avec un score de {evaluations[0]['score']}/100"
        
        best_score = evaluations[0]['score']
        worst_score = evaluations[-1]['score']
        
        summary = f"Comparaison de {len(evaluations)} solutions : "
        summary += f"meilleure solution {best_score}/100, "
        summary += f"moins bonne {worst_score}/100. "
        
        # Analyser les différences principales
        criteria_scores = {}
        for evaluation in evaluations:
            for criterion, score in evaluation.get('detailed_scores', {}).items():
                if criterion not in criteria_scores:
                    criteria_scores[criterion] = []
                criteria_scores[criterion].append(score)
        
        # Trouver le critère le plus discriminant
        max_variance = 0
        most_discriminant = None
        
        for criterion, scores in criteria_scores.items():
            if len(scores) > 1:
                variance = max(scores) - min(scores)
                if variance > max_variance:
                    max_variance = variance
                    most_discriminant = criterion
        
        if most_discriminant and max_variance > 20:
            summary += f"Principal facteur de différenciation: {most_discriminant}."
        
        return summary
    
    async def generate_improvement_plan(self, code: str, context: AgentContext = None) -> Dict[str, Any]:
        """
        Génère un plan d'amélioration personnalisé pour un code
        
        Args:
            code: Code à améliorer
            context: Contexte utilisateur
            
        Returns:
            Plan d'amélioration structuré
        """
        if not context:
            context = AgentContext()
        
        # Évaluer le code
        input_data = {'code': code}
        response = await self.process(input_data, context)
        
        if not response.success:
            return {
                'success': False,
                'message': 'Impossible d\'analyser le code pour créer un plan d\'amélioration',
                'errors': response.errors
            }
        
        detailed_scores = response.data.get('detailed_scores', {})
        overall_score = response.data.get('overall_score', 0)
        
        # Identifier les points faibles (score < 70)
        weak_areas = [(criterion, score) for criterion, score in detailed_scores.items() if score < 70]
        weak_areas.sort(key=lambda x: x[1])  # Trier par score croissant
        
        # Générer le plan d'amélioration
        improvement_plan = {
            'current_score': overall_score,
            'target_score': min(100, overall_score + 20),
            'priority_areas': [],
            'action_items': [],
            'learning_resources': [],
            'estimated_effort': 'Faible'
        }
        
        # Définir les priorités
        for criterion, score in weak_areas[:3]:  # Top 3 des points faibles
            priority = 'Haute' if score < 50 else 'Moyenne'
            improvement_plan['priority_areas'].append({
                'area': criterion,
                'current_score': score,
                'priority': priority,
                'improvement_potential': min(30, 100 - score)
            })
        
        # Générer des actions spécifiques
        improvement_plan['action_items'] = self._generate_action_items(weak_areas, context)
        
        # Ressources d'apprentissage
        improvement_plan['learning_resources'] = self._suggest_learning_resources(weak_areas, context)
        
        # Estimer l'effort
        if len(weak_areas) > 3 or any(score < 30 for _, score in weak_areas):
            improvement_plan['estimated_effort'] = 'Élevé'
        elif len(weak_areas) > 1:
            improvement_plan['estimated_effort'] = 'Moyen'
        
        return {
            'success': True,
            'improvement_plan': improvement_plan,
            'current_evaluation': response.data
        }
    
    def _generate_action_items(self, weak_areas: List[Tuple[str, float]], 
                              context: AgentContext) -> List[Dict[str, Any]]:
        """Génère des actions concrètes d'amélioration"""
        actions = []
        
        for criterion, score in weak_areas:
            if criterion == 'syntax':
                actions.append({
                    'action': 'Corriger les erreurs de syntaxe',
                    'description': 'Utiliser un éditeur avec coloration syntaxique et vérifier la syntaxe',
                    'estimated_time': '30 minutes',
                    'difficulty': 'Facile'
                })
            
            elif criterion == 'logic':
                actions.append({
                    'action': 'Améliorer la logique algorithmique',
                    'description': 'Revoir l\'algorithme, ajouter des commentaires, simplifier la logique',
                    'estimated_time': '2 heures',
                    'difficulty': 'Moyen'
                })
            
            elif criterion == 'style':
                actions.append({
                    'action': 'Améliorer le style de code',
                    'description': 'Suivre les conventions PEP 8, améliorer les noms de variables',
                    'estimated_time': '1 heure',
                    'difficulty': 'Facile'
                })
            
            elif criterion == 'execution':
                actions.append({
                    'action': 'Résoudre les problèmes d\'exécution',
                    'description': 'Déboguer les erreurs d\'exécution, tester avec différents cas',
                    'estimated_time': '1-3 heures',
                    'difficulty': 'Moyen'
                })
            
            elif criterion == 'efficiency':
                actions.append({
                    'action': 'Optimiser les performances',
                    'description': 'Revoir les algorithmes, éviter les boucles imbriquées inutiles',
                    'estimated_time': '2-4 heures',
                    'difficulty': 'Difficile'
                })
        
        return actions[:5]  # Limiter à 5 actions
    
    def _suggest_learning_resources(self, weak_areas: List[Tuple[str, float]], 
                                   context: AgentContext) -> List[Dict[str, str]]:
        """Suggère des ressources d'apprentissage"""
        resources = []
        
        for criterion, score in weak_areas:
            if criterion == 'syntax':
                resources.append({
                    'type': 'Documentation',
                    'title': 'Guide de syntaxe Python',
                    'description': 'Documentation officielle sur la syntaxe Python'
                })
            
            elif criterion == 'logic':
                resources.append({
                    'type': 'Cours',
                    'title': 'Algorithmes et structures de données',
                    'description': 'Cours sur les algorithmes fondamentaux'
                })
            
            elif criterion == 'style':
                resources.append({
                    'type': 'Guide',
                    'title': 'PEP 8 - Style Guide for Python Code',
                    'description': 'Guide officiel de style pour Python'
                })
            
            elif criterion == 'efficiency':
                resources.append({
                    'type': 'Livre',
                    'title': 'Effective Python',
                    'description': 'Techniques pour écrire du Python efficace'
                })
        
        return resources[:3]  # Limiter à 3 ressources
    
    async def batch_evaluate(self, code_samples: List[Dict[str, Any]], 
                           context: AgentContext = None) -> Dict[str, Any]:
        """
        Évalue plusieurs échantillons de code en lot
        
        Args:
            code_samples: Liste de dictionnaires avec 'code' et métadonnées optionnelles
            context: Contexte utilisateur
            
        Returns:
            Résultats d'évaluation groupés
        """
        if not context:
            context = AgentContext()
        
        results = []
        total_score = 0
        
        for i, sample in enumerate(code_samples):
            try:
                code = sample.get('code', '')
                if not code:
                    continue
                
                # Évaluer le code
                input_data = {'code': code}
                response = await self.process(input_data, context)
                
                result = {
                    'index': i,
                    'code': code,
                    'metadata': sample.get('metadata', {}),
                    'evaluation': response.data if response.success else None,
                    'score': response.data.get('overall_score', 0) if response.success else 0,
                    'success': response.success,
                    'errors': response.errors if not response.success else []
                }
                
                results.append(result)
                total_score += result['score']
                
            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation du code {i}: {e}")
                results.append({
                    'index': i,
                    'code': sample.get('code', ''),
                    'metadata': sample.get('metadata', {}),
                    'evaluation': None,
                    'score': 0,
                    'success': False,
                    'errors': [str(e)]
                })
        
        # Statistiques globales
        successful_evaluations = [r for r in results if r['success']]
        average_score = total_score / len(successful_evaluations) if successful_evaluations else 0
        
        return {
            'results': results,
            'statistics': {
                'total_samples': len(code_samples),
                'successful_evaluations': len(successful_evaluations),
                'failed_evaluations': len(results) - len(successful_evaluations),
                'average_score': round(average_score, 1),
                'best_score': max(r['score'] for r in results) if results else 0,
                'worst_score': min(r['score'] for r in results) if results else 0
            }
        }
    
    def get_evaluation_report(self, evaluation_data: Dict[str, Any]) -> str:
        """
        Génère un rapport d'évaluation formaté
        
        Args:
            evaluation_data: Données d'évaluation
            
        Returns:
            Rapport formaté en texte
        """
        overall_score = evaluation_data.get('overall_score', 0)
        detailed_scores = evaluation_data.get('detailed_scores', {})
        
        report = f"""
=== RAPPORT D'ÉVALUATION DE CODE ===

Score global: {overall_score}/100

Détail par critère:
"""
        
        for criterion, score in detailed_scores.items():
            status_icon = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            report += f"  {status_icon} {criterion.title()}: {score}/100\n"
        
        evaluation_details = evaluation_data.get('evaluation_details', {})
        
        # Ajouter les détails pour chaque critère
        for criterion, details in evaluation_details.items():
            if details.get('errors') or details.get('warnings') or details.get('suggestions'):
                report += f"\n--- {criterion.upper()} ---\n"
                
                if details.get('errors'):
                    report += "Erreurs:\n"
                    for error in details['errors']:
                        report += f"  • {error}\n"
                
                if details.get('warnings'):
                    report += "Avertissements:\n"
                    for warning in details['warnings']:
                        report += f"  • {warning}\n"
                
                if details.get('suggestions'):
                    report += "Suggestions:\n"
                    for suggestion in details['suggestions']:
                        report += f"  • {suggestion}\n"
        
        code_metrics = evaluation_data.get('code_metrics', {})
        if code_metrics:
            report += f"""
--- MÉTRIQUES DU CODE ---
Lignes totales: {code_metrics.get('total_lines', 0)}
Lignes non vides: {code_metrics.get('non_empty_lines', 0)}
Fonctions: {code_metrics.get('functions', 0)}
Classes: {code_metrics.get('classes', 0)}
Variables: {code_metrics.get('variables', 0)}
"""
        
        return report
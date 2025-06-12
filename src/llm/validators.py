import ast
import sys
from io import StringIO
from typing import Dict, Any, Tuple, Optional

from ..core.logger import logger


class CodeValidator:
    """Validateur et exécuteur de code Python sécurisé"""
    
    def __init__(self):
        self.safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
            'chr', 'complex', 'dict', 'divmod', 'enumerate', 'filter',
            'float', 'format', 'frozenset', 'hash', 'hex', 'int', 'len',
            'list', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'print',
            'range', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
            'sum', 'tuple', 'type', 'zip'
        }
        
        self.forbidden_imports = {
            'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
            'shutil', 'glob', 'pickle', 'eval', 'exec', 'compile'
        }
    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Valide la syntaxe du code Python"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Erreur de syntaxe: {e}"
        except Exception as e:
            return False, f"Erreur de validation: {e}"
    
    def check_security(self, code: str) -> Tuple[bool, Optional[str]]:
        """Vérifie la sécurité du code"""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Vérifier les imports dangereux
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.forbidden_imports:
                            return False, f"Import interdit: {alias.name}"
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.forbidden_imports:
                        return False, f"Import interdit: {node.module}"
                
                # Vérifier les appels de fonction dangereux
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile']:
                            return False, f"Fonction interdite: {node.func.id}"
            
            return True
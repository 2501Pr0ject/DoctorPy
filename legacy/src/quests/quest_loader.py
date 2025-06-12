"""
Chargeur de quêtes depuis fichiers JSON et autres sources.
Gère l'import, l'export et la synchronisation des quêtes.
"""

import json
import yaml
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import shutil
from datetime import datetime

from sqlalchemy.orm import Session

from src.core.database import get_session
from src.core.logger import get_logger
from src.core.config import get_settings
from src.models.quest import Quest, QuestStep
from src.models.schemas import QuestCreate, QuestStepCreate

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class LoadResult:
    """Résultat de chargement de quêtes."""
    success: bool
    loaded_count: int
    skipped_count: int
    error_count: int
    errors: List[str]
    loaded_quests: List[str]


@dataclass
class QuestTemplate:
    """Template de quête pour la génération."""
    category: str
    difficulty: str
    base_structure: Dict[str, Any]
    variable_parts: List[str]
    example_content: Dict[str, Any]


class QuestFileLoader:
    """Chargeur de fichiers de quêtes."""
    
    def __init__(self):
        self.data_dir = Path(settings.DATA_DIR) / "quests"
        self.supported_formats = ['.json', '.yaml', '.yml']
        self.templates_dir = self.data_dir / "templates"
        self.imported_dir = self.data_dir / "imported"
        
        # Créer les dossiers s'ils n'existent pas
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        self.imported_dir.mkdir(exist_ok=True)
    
    def load_quest_from_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Charge une quête depuis un fichier.
        
        Args:
            file_path: Chemin vers le fichier
            
        Returns:
            Données de la quête ou None si erreur
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Fichier introuvable: {file_path}")
            return None
        
        if file_path.suffix not in self.supported_formats:
            logger.error(f"Format non supporté: {file_path.suffix}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.json':
                    data = json.load(f)
                elif file_path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    return None
            
            # Valider la structure
            if self._validate_quest_structure(data):
                return data
            else:
                logger.error(f"Structure invalide dans {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {file_path}: {e}")
            return None
    
    def load_quests_from_directory(self, directory: Union[str, Path]) -> LoadResult:
        """
        Charge toutes les quêtes d'un répertoire.
        
        Args:
            directory: Répertoire à scanner
            
        Returns:
            Résultat du chargement
        """
        directory = Path(directory)
        result = LoadResult(
            success=True,
            loaded_count=0,
            skipped_count=0,
            error_count=0,
            errors=[],
            loaded_quests=[]
        )
        
        if not directory.exists():
            result.success = False
            result.errors.append(f"Répertoire introuvable: {directory}")
            return result
        
        # Scanner les fichiers
        quest_files = []
        for ext in self.supported_formats:
            quest_files.extend(directory.glob(f"**/*{ext}"))
        
        logger.info(f"Trouvé {len(quest_files)} fichiers de quêtes dans {directory}")
        
        for file_path in quest_files:
            try:
                quest_data = self.load_quest_from_file(file_path)
                
                if quest_data:
                    # Importer en base de données
                    quest_id = self._import_quest_to_database(quest_data)
                    
                    if quest_id:
                        result.loaded_count += 1
                        result.loaded_quests.append(quest_id)
                        logger.info(f"Quête chargée: {quest_id} depuis {file_path.name}")
                    else:
                        result.error_count += 1
                        result.errors.append(f"Échec import: {file_path.name}")
                else:
                    result.skipped_count += 1
                    result.errors.append(f"Données invalides: {file_path.name}")
                    
            except Exception as e:
                result.error_count += 1
                result.errors.append(f"Erreur {file_path.name}: {str(e)}")
                logger.error(f"Erreur lors du chargement de {file_path}: {e}")
        
        if result.error_count > 0:
            result.success = False
        
        return result
    
    def _validate_quest_structure(self, data: Dict[str, Any]) -> bool:
        """Valide la structure d'une quête."""
        required_fields = ['title', 'description', 'difficulty', 'steps']
        
        # Vérifier les champs requis
        for field in required_fields:
            if field not in data:
                logger.error(f"Champ requis manquant: {field}")
                return False
        
        # Vérifier la structure des étapes
        if not isinstance(data['steps'], list) or len(data['steps']) == 0:
            logger.error("Le champ 'steps' doit être une liste non vide")
            return False
        
        for i, step in enumerate(data['steps']):
            if not isinstance(step, dict):
                logger.error(f"Étape {i+1} doit être un objet")
                return False
            
            step_required = ['title', 'content', 'step_type']
            for field in step_required:
                if field not in step:
                    logger.error(f"Champ requis manquant dans l'étape {i+1}: {field}")
                    return False
        
        # Vérifier les valeurs des énumérations
        valid_difficulties = ['beginner', 'intermediate', 'advanced', 'expert']
        if data['difficulty'] not in valid_difficulties:
            logger.error(f"Difficulté invalide: {data['difficulty']}")
            return False
        
        valid_step_types = ['theory', 'practice', 'quiz', 'project']
        for i, step in enumerate(data['steps']):
            if step['step_type'] not in valid_step_types:
                logger.error(f"Type d'étape invalide dans l'étape {i+1}: {step['step_type']}")
                return False
        
        return True
    
    def _import_quest_to_database(self, quest_data: Dict[str, Any]) -> Optional[str]:
        """Importe une quête en base de données."""
        try:
            with get_session() as db:
                # Vérifier si la quête existe déjà
                existing_quest = None
                if 'id' in quest_data:
                    existing_quest = db.query(Quest).filter(Quest.id == quest_data['id']).first()
                
                # Générer un nouvel ID si nécessaire
                quest_id = quest_data.get('id', str(uuid.uuid4()))
                
                if existing_quest:
                    logger.info(f"Quête existante trouvée: {quest_id}, mise à jour...")
                    quest = existing_quest
                    
                    # Mettre à jour les champs
                    quest.title = quest_data['title']
                    quest.description = quest_data['description']
                    quest.difficulty = quest_data['difficulty']
                    quest.estimated_time = quest_data.get('estimated_time', 30)
                    quest.category = quest_data.get('category', 'general')
                    quest.tags = quest_data.get('tags', [])
                    quest.learning_objectives = quest_data.get('learning_objectives', [])
                    quest.prerequisites = quest_data.get('prerequisites', {})
                    quest.updated_at = datetime.utcnow()
                    
                    # Supprimer les anciennes étapes
                    db.query(QuestStep).filter(QuestStep.quest_id == quest.id).delete()
                else:
                    # Créer une nouvelle quête
                    quest = Quest(
                        id=quest_id,
                        title=quest_data['title'],
                        description=quest_data['description'],
                        difficulty=quest_data['difficulty'],
                        estimated_time=quest_data.get('estimated_time', 30),
                        category=quest_data.get('category', 'general'),
                        tags=quest_data.get('tags', []),
                        learning_objectives=quest_data.get('learning_objectives', []),
                        prerequisites=quest_data.get('prerequisites', {}),
                        creator_id=quest_data.get('creator_id', 1),  # Utilisateur système par défaut
                        is_active=True,
                        created_at=datetime.utcnow()
                    )
                    db.add(quest)
                
                db.flush()  # Pour obtenir l'ID
                
                # Ajouter les étapes
                total_score = 0
                for i, step_data in enumerate(quest_data['steps']):
                    step_score = step_data.get('max_score', 10)
                    total_score += step_score
                    
                    step = QuestStep(
                        quest_id=quest.id,
                        step_number=i + 1,
                        title=step_data['title'],
                        content=step_data['content'],
                        step_type=step_data['step_type'],
                        code_template=step_data.get('code_template', ''),
                        expected_output=step_data.get('expected_output', ''),
                        hints=step_data.get('hints', []),
                        resources=step_data.get('resources', []),
                        max_score=step_score,
                        metadata=json.dumps(step_data.get('metadata', {}))
                    )
                    db.add(step)
                
                # Mettre à jour le score maximum de la quête
                quest.max_score = total_score
                
                db.commit()
                return quest.id
                
        except Exception as e:
            logger.error(f"Erreur lors de l'import en base: {e}")
            return None
    
    def export_quest_to_file(self, quest_id: str, output_path: Union[str, Path]) -> bool:
        """
        Exporte une quête vers un fichier.
        
        Args:
            quest_id: ID de la quête
            output_path: Chemin de sortie
            
        Returns:
            True si succès
        """
        try:
            with get_session() as db:
                quest = db.query(Quest).filter(Quest.id == quest_id).first()
                if not quest:
                    logger.error(f"Quête introuvable: {quest_id}")
                    return False
                
                steps = db.query(QuestStep).filter(
                    QuestStep.quest_id == quest_id
                ).order_by(QuestStep.step_number).all()
                
                # Construire les données d'export
                quest_data = {
                    'id': quest.id,
                    'title': quest.title,
                    'description': quest.description,
                    'difficulty': quest.difficulty,
                    'estimated_time': quest.estimated_time,
                    'category': quest.category,
                    'tags': quest.tags,
                    'learning_objectives': quest.learning_objectives,
                    'prerequisites': quest.prerequisites,
                    'max_score': quest.max_score,
                    'created_at': quest.created_at.isoformat() if quest.created_at else None,
                    'steps': []
                }
                
                for step in steps:
                    step_data = {
                        'title': step.title,
                        'content': step.content,
                        'step_type': step.step_type,
                        'code_template': step.code_template,
                        'expected_output': step.expected_output,
                        'hints': step.hints,
                        'resources': step.resources,
                        'max_score': step.max_score,
                        'metadata': json.loads(step.metadata) if step.metadata else {}
                    }
                    quest_data['steps'].append(step_data)
                
                output_path = Path(output_path)
                
                # Déterminer le format selon l'extension
                if output_path.suffix == '.json':
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(quest_data, f, indent=2, ensure_ascii=False)
                elif output_path.suffix in ['.yaml', '.yml']:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        yaml.dump(quest_data, f, default_flow_style=False, allow_unicode=True)
                else:
                    # Par défaut, utiliser JSON
                    output_path = output_path.with_suffix('.json')
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(quest_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Quête exportée: {quest_id} vers {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Erreur lors de l'export: {e}")
            return False
    
    def backup_all_quests(self, backup_dir: Optional[Union[str, Path]] = None) -> bool:
        """
        Sauvegarde toutes les quêtes.
        
        Args:
            backup_dir: Répertoire de sauvegarde
            
        Returns:
            True si succès
        """
        if backup_dir is None:
            backup_dir = self.data_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with get_session() as db:
                quests = db.query(Quest).filter(Quest.is_active == True).all()
                
                success_count = 0
                for quest in quests:
                    output_file = backup_dir / f"{quest.id}.json"
                    if self.export_quest_to_file(quest.id, output_file):
                        success_count += 1
                
                logger.info(f"Sauvegarde terminée: {success_count}/{len(quests)} quêtes")
                return success_count == len(quests)
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def create_quest_template(self, category: str, difficulty: str) -> QuestTemplate:
        """Crée un template de quête."""
        base_structure = {
            "title": f"[TITRE] - {category.title()} {difficulty.title()}",
            "description": "[DESCRIPTION DE LA QUÊTE]",
            "difficulty": difficulty,
            "category": category,
            "estimated_time": 30,
            "tags": [category],
            "learning_objectives": [
                "[OBJECTIF 1]",
                "[OBJECTIF 2]"
            ],
            "prerequisites": {},
            "steps": [
                {
                    "title": "[TITRE ÉTAPE 1] - Théorie",
                    "content": "[CONTENU THÉORIQUE]",
                    "step_type": "theory",
                    "max_score": 10
                },
                {
                    "title": "[TITRE ÉTAPE 2] - Pratique",
                    "content": "[INSTRUCTIONS PRATIQUES]",
                    "step_type": "practice",
                    "code_template": "# Votre code ici\n",
                    "expected_output": "[SORTIE ATTENDUE]",
                    "hints": [
                        "[INDICE 1]",
                        "[INDICE 2]"
                    ],
                    "max_score": 20
                }
            ]
        }
        
        # Exemples spécifiques par catégorie
        examples = {
            "python_basics": {
                "title": "Les Variables en Python",
                "description": "Apprenez à déclarer et utiliser des variables en Python",
                "steps": [
                    {
                        "title": "Comprendre les variables",
                        "content": "Une variable est un nom qui fait référence à une valeur...",
                        "step_type": "theory"
                    },
                    {
                        "title": "Créer vos premières variables",
                        "content": "Créez trois variables : nom, age, et actif",
                        "step_type": "practice",
                        "code_template": "# Créez vos variables ici\nnom = \nage = \nactif = \n\nprint(nom, age, actif)",
                        "expected_output": "Alice 25 True"
                    }
                ]
            }
        }
        
        example_content = examples.get(category, {})
        
        return QuestTemplate(
            category=category,
            difficulty=difficulty,
            base_structure=base_structure,
            variable_parts=["title", "description", "steps"],
            example_content=example_content
        )
    
    def generate_quest_from_template(self, template: QuestTemplate, custom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une quête à partir d'un template."""
        quest_data = template.base_structure.copy()
        
        # Appliquer les données personnalisées
        for key, value in custom_data.items():
            if key in quest_data:
                quest_data[key] = value
        
        # Appliquer l'exemple si disponible
        if template.example_content:
            for key, value in template.example_content.items():
                if key not in custom_data:  # Ne pas écraser les données personnalisées
                    quest_data[key] = value
        
        return quest_data
    
    def import_from_external_source(self, source_url: str, source_type: str = "github") -> LoadResult:
        """Importe des quêtes depuis une source externe."""
        # Placeholder pour l'import depuis GitHub, GitLab, etc.
        logger.info(f"Import depuis {source_type}: {source_url}")
        
        # TODO: Implémenter l'import depuis des sources externes
        return LoadResult(
            success=False,
            loaded_count=0,
            skipped_count=0,
            error_count=1,
            errors=["Import externe non encore implémenté"],
            loaded_quests=[]
        )


class QuestSynchronizer:
    """Synchroniseur de quêtes entre différentes sources."""
    
    def __init__(self):
        self.loader = QuestFileLoader()
    
    def sync_with_directory(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """Synchronise les quêtes avec un répertoire."""
        directory = Path(directory)
        
        # Charger les quêtes depuis le répertoire
        load_result = self.loader.load_quests_from_directory(directory)
        
        # Analyser les différences
        sync_result = {
            'loaded': load_result.loaded_count,
            'errors': load_result.error_count,
            'skipped': load_result.skipped_count,
            'total_files': load_result.loaded_count + load_result.error_count + load_result.skipped_count,
            'success_rate': load_result.loaded_count / max(1, load_result.loaded_count + load_result.error_count),
            'details': load_result.errors
        }
        
        return sync_result
    
    def create_sync_report(self, sync_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Crée un rapport de synchronisation."""
        total_loaded = sum(r['loaded'] for r in sync_results)
        total_errors = sum(r['errors'] for r in sync_results)
        total_files = sum(r['total_files'] for r in sync_results)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_sources': len(sync_results),
                'total_files_processed': total_files,
                'total_loaded': total_loaded,
                'total_errors': total_errors,
                'overall_success_rate': total_loaded / max(1, total_files)
            },
            'details': sync_results
        }


# Instances globales
quest_loader = QuestFileLoader()
quest_synchronizer = QuestSynchronizer()


# Fonctions utilitaires
def load_quest_file(file_path: Union[str, Path]) -> Optional[str]:
    """
    Charge une quête depuis un fichier.
    
    Args:
        file_path: Chemin vers le fichier
        
    Returns:
        ID de la quête chargée ou None
    """
    quest_data = quest_loader.load_quest_from_file(file_path)
    if quest_data:
        return quest_loader._import_quest_to_database(quest_data)
    return None


def load_quests_directory(directory: Union[str, Path]) -> LoadResult:
    """
    Charge toutes les quêtes d'un répertoire.
    
    Args:
        directory: Répertoire à scanner
        
    Returns:
        Résultat du chargement
    """
    return quest_loader.load_quests_from_directory(directory)


def export_quest(quest_id: str, output_path: Union[str, Path]) -> bool:
    """
    Exporte une quête vers un fichier.
    
    Args:
        quest_id: ID de la quête
        output_path: Chemin de sortie
        
    Returns:
        True si succès
    """
    return quest_loader.export_quest_to_file(quest_id, output_path)


def backup_quests(backup_dir: Optional[Union[str, Path]] = None) -> bool:
    """
    Sauvegarde toutes les quêtes.
    
    Args:
        backup_dir: Répertoire de sauvegarde
        
    Returns:
        True si succès
    """
    return quest_loader.backup_all_quests(backup_dir)


if __name__ == "__main__":
    # Test du chargeur de quêtes
    print("=== Test du Chargeur de Quêtes ===")
    
    # Créer un exemple de quête
    example_quest = {
        "title": "Introduction aux Variables Python",
        "description": "Apprenez les bases des variables en Python",
        "difficulty": "beginner",
        "category": "python_basics",
        "estimated_time": 20,
        "tags": ["variables", "python", "basics"],
        "learning_objectives": [
            "Comprendre ce qu'est une variable",
            "Savoir déclarer des variables",
            "Utiliser différents types de données"
        ],
        "steps": [
            {
                "title": "Qu'est-ce qu'une variable ?",
                "content": "Une variable est un conteneur pour stocker des données...",
                "step_type": "theory",
                "max_score": 5
            },
            {
                "title": "Créer votre première variable",
                "content": "Créez une variable nommée 'message' avec la valeur 'Hello World'",
                "step_type": "practice",
                "code_template": "# Créez votre variable ici\nmessage = \n\nprint(message)",
                "expected_output": "Hello World",
                "hints": ["Utilisez des guillemets pour les chaînes de caractères"],
                "max_score": 10
            }
        ]
    }
    
    # Test de validation
    is_valid = quest_loader._validate_quest_structure(example_quest)
    print(f"Structure valide: {is_valid}")
    
    # Test de création de template
    template = quest_loader.create_quest_template("python_basics", "beginner")
    print(f"Template créé pour: {template.category} - {template.difficulty}")
    
    print("Chargeur de quêtes testé avec succès!")
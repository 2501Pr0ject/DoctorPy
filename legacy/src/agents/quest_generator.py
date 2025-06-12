# src/agents/quest_generator.py
"""
Agent générateur de quêtes - Crée des quêtes pédagogiques personnalisées
"""

from typing import Dict, Any, List, Optional
import asyncio
import json
import logging
from datetime import datetime, timezone

from src.agents.base_agent import BaseAgent, AgentType, AgentContext, AgentResponse
from src.llm.ollama_client import OllamaClient
from src.models import Quest, QuestStep, Question, QuestCategory, QuestDifficulty, QuestionType
from src.core.database import get_db_session
from src.utils import slugify, ValidationResult, CodeValidator, sanitize_input

logger = logging.getLogger(__name__)

class QuestGeneratorAgent(BaseAgent):
    """Agent pour la génération automatique de quêtes pédagogiques"""
    
    def __init__(self, agent_type: AgentType = AgentType.QUEST_GENERATOR, name: str = None, **kwargs):
        super().__init__(agent_type, name or "quest_generator")
        
        # Clients et outils
        self.llm_client = OllamaClient()
        self.code_validator = CodeValidator()
        
        # Configuration de génération
        self.creativity_level = kwargs.get('creativity_level', 0.8)
        self.difficulty_progression = kwargs.get('difficulty_progression', 'adaptive')
        self.max_steps_per_quest = kwargs.get('max_steps_per_quest', 10)
        self.include_code_exercises = kwargs.get('include_code_exercises', True)
        self.generate_variations = kwargs.get('generate_variations', False)
        
        # Templates et patterns
        self.quest_templates = self._load_quest_templates()
        self.learning_objectives_bank = self._load_learning_objectives()
        
        logger.info(f"Générateur de quêtes initialisé avec créativité: {self.creativity_level}")
    
    async def process(self, input_data: Dict[str, Any], context: AgentContext) -> AgentResponse:
        """
        Génère une quête complète selon les spécifications
        
        Args:
            input_data: Contient les paramètres de génération
            context: Contexte utilisateur
            
        Returns:
            Quête générée avec étapes et questions
        """
        try:
            # Validation des paramètres
            generation_params = self._validate_generation_params(input_data)
            if not generation_params['valid']:
                return AgentResponse(
                    success=False,
                    message="Paramètres de génération invalides",
                    errors=generation_params['errors']
                )
            
            # 1. Générer la structure de base de la quête
            quest_structure = await self._generate_quest_structure(
                generation_params['data'], context
            )
            
            # 2. Créer les étapes détaillées
            quest_steps = await self._generate_quest_steps(
                quest_structure, generation_params['data'], context
            )
            
            # 3. Générer les questions et exercices
            questions = await self._generate_questions_for_steps(
                quest_steps, generation_params['data']
            )
            
            # 4. Valider la cohérence de la quête
            validation_result = await self._validate_quest_coherence(
                quest_structure, quest_steps, questions
            )
            
            if not validation_result['valid']:
                # Essayer de corriger automatiquement
                quest_structure, quest_steps, questions = await self._auto_correct_quest(
                    quest_structure, quest_steps, questions, validation_result
                )
            
            # 5. Sauvegarder en base de données (optionnel)
            quest_id = None
            if generation_params['data'].get('save_to_db', False):
                quest_id = await self._save_quest_to_db(
                    quest_structure, quest_steps, questions, context
                )
            
            # 6. Préparer la réponse
            response_data = {
                'quest': quest_structure,
                'steps': quest_steps,
                'questions': questions,
                'quest_id': quest_id,
                'validation': validation_result,
                'generation_metadata': {
                    'creativity_level': self.creativity_level,
                    'difficulty_progression': self.difficulty_progression,
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }
            }
            
            return AgentResponse(
                success=True,
                message=f"Quête '{quest_structure['title']}' générée avec succès",
                data=response_data,
                confidence=validation_result.get('confidence', 0.8),
                reasoning=f"Quête de {len(quest_steps)} étapes sur {quest_structure['category']}"
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de quête: {e}")
            return AgentResponse(
                success=False,
                message="Erreur lors de la génération de la quête",
                errors=[str(e)]
            )
    
    def _validate_generation_params(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Valide les paramètres de génération"""
        errors = []
        
        # Paramètres requis
        required_fields = ['topic', 'target_level', 'category']
        for field in required_fields:
            if field not in input_data:
                errors.append(f"Paramètre requis manquant: {field}")
        
        # Validation des valeurs
        if 'target_level' in input_data:
            valid_levels = ['beginner', 'intermediate', 'advanced']
            if input_data['target_level'] not in valid_levels:
                errors.append(f"Niveau invalide. Valeurs acceptées: {valid_levels}")
        
        if 'category' in input_data:
            valid_categories = [cat.value for cat in QuestCategory]
            if input_data['category'] not in valid_categories:
                errors.append(f"Catégorie invalide. Valeurs acceptées: {valid_categories}")
        
        # Paramètres optionnels avec valeurs par défaut
        validated_data = input_data.copy()
        validated_data.setdefault('estimated_duration', 30)
        validated_data.setdefault('num_steps', 5)
        validated_data.setdefault('difficulty', 'medium')
        validated_data.setdefault('include_practical_exercises', True)
        validated_data.setdefault('save_to_db', False)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'data': validated_data
        }
    
    async def _generate_quest_structure(
        self, params: Dict[str, Any], context: AgentContext
    ) -> Dict[str, Any]:
        """Génère la structure de base de la quête"""
        
        # Adapter selon le contexte utilisateur
        user_context = ""
        if context.user_progress:
            user_context = f"""
Contexte apprenant :
- Niveau actuel : {context.user_progress.get('level', 'débutant')}
- Compétences fortes : {self._format_skills(context.user_progress.get('skill_scores', {}))}
- Préférences : {context.preferences.get('learning_style', 'visuel') if context.preferences else 'visuel'}
"""
        
        prompt = f"""Génère une structure de quête pédagogique pour l'apprentissage de Python.

Paramètres :
- Sujet : {params['topic']}
- Niveau cible : {params['target_level']}
- Catégorie : {params['category']}
- Durée estimée : {params['estimated_duration']} minutes
- Nombre d'étapes : {params['num_steps']}
- Difficulté : {params['difficulty']}

{user_context}

La quête doit être :
- Engaging et motivante
- Progressive dans la difficulté
- Pratique avec des exemples concrets
- Adaptée au niveau spécifié

Génère une structure JSON avec :
{{
    "title": "Titre accrocheur de la quête",
    "description": "Description détaillée et motivante",
    "short_description": "Résumé en une phrase",
    "category": "{params['category']}",
    "difficulty": "{params['difficulty']}",
    "level": "{params['target_level']}",
    "estimated_duration": {params['estimated_duration']},
    "learning_objectives": ["objectif1", "objectif2", "objectif3"],
    "prerequisites": ["prérequis1", "prérequis2"],
    "tags": ["tag1", "tag2", "tag3"],
    "xp_reward": 150,
    "passing_score": 0.7
}}

Assure-toi que le contenu soit en français et adapté au contexte d'apprentissage."""
        
        try:
            response = await self.llm_client.generate_async(
                prompt=prompt,
                max_tokens=1000,
                temperature=self.creativity_level
            )
            
            # Parser la réponse JSON
            quest_structure = self._parse_llm_json_response(response)
            
            # Ajouter des métadonnées
            quest_structure['slug'] = slugify(quest_structure['title'])
            quest_structure['uuid'] = None  # Sera généré en DB
            quest_structure['status'] = 'draft'
            quest_structure['total_steps'] = params['num_steps']
            
            return quest_structure
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de structure: {e}")
            # Fallback avec template
            return self._generate_fallback_quest_structure(params)
    
    async def _generate_quest_steps(
        self, quest_structure: Dict[str, Any], params: Dict[str, Any], context: AgentContext
    ) -> List[Dict[str, Any]]:
        """Génère les étapes détaillées de la quête"""
        
        steps = []
        objectives = quest_structure.get('learning_objectives', [])
        
        for i in range(params['num_steps']):
            step_prompt = f"""Génère l'étape {i+1} d'une quête sur {params['topic']}.

Contexte de la quête :
- Titre : {quest_structure['title']}
- Niveau : {quest_structure['level']}
- Objectifs : {', '.join(objectives)}

Cette étape doit :
- Être la {i+1}ème étape sur {params['num_steps']} au total
- Progresser logiquement vers les objectifs
- Inclure du contenu théorique et pratique
- Proposer des exercices adaptés

Génère un JSON avec :
{{
    "order": {i+1},
    "title": "Titre de l'étape",
    "description": "Description détaillée de ce qui sera appris",
    "step_type": "content",
    "content": "Contenu pédagogique détaillé avec exemples",
    "code_template": "# Code template si applicable",
    "expected_output": "Résultat attendu",
    "resources": [
        {{"type": "documentation", "title": "Titre", "url": "URL"}},
        {{"type": "example", "title": "Exemple", "content": "Code exemple"}}
    ],
    "hints": ["Indice 1", "Indice 2"],
    "validation_rules": {{"check_syntax": true, "check_logic": true}},
    "max_attempts": 3
}}

Le contenu doit être riche, pédagogique et en français."""
            
            try:
                response = await self.llm_client.generate_async(
                    prompt=step_prompt,
                    max_tokens=1500,
                    temperature=self.creativity_level
                )
                
                step_data = self._parse_llm_json_response(response)
                step_data['quest_id'] = None  # Sera rempli lors de la sauvegarde
                steps.append(step_data)
                
            except Exception as e:
                logger.error(f"Erreur lors de la génération de l'étape {i+1}: {e}")
                # Fallback avec étape basique
                steps.append(self._generate_fallback_step(i+1, quest_structure))
        
        return steps
    
    async def _generate_questions_for_steps(
        self, steps: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Génère les questions pour chaque étape"""
        
        all_questions = []
        
        for step_idx, step in enumerate(steps):
            # Déterminer le nombre de questions par étape
            num_questions = 2 if params.get('include_practical_exercises') else 1
            
            for q_idx in range(num_questions):
                question_type = self._determine_question_type(step, q_idx)
                
                question_prompt = f"""Génère une question pédagogique pour l'étape : {step['title']}

Contenu de l'étape : {step['description'][:200]}...

Type de question : {question_type}
Numéro de la question : {q_idx + 1} sur {num_questions}

Génère un JSON avec :
{{
    "question_text": "Énoncé clair de la question",
    "question_type": "{question_type}",
    "choices": ["choix1", "choix2", "choix3", "choix4"],
    "correct_answer": "bonne_réponse",
    "explanation": "Explication détaillée de la réponse",
    "points": 1,
    "time_limit": 120,
    "shuffle_choices": true,
    "case_sensitive": false
}}

Pour les questions de code :
- Propose des exercices pratiques
- Inclus des cas de test
- Donne des explications détaillées

La question doit tester la compréhension de l'étape."""
                
                try:
                    response = await self.llm_client.generate_async(
                        prompt=question_prompt,
                        max_tokens=800,
                        temperature=self.creativity_level * 0.8  # Moins de créativité pour les questions
                    )
                    
                    question_data = self._parse_llm_json_response(response)
                    question_data['step_id'] = None  # Sera rempli lors de la sauvegarde
                    question_data['step_order'] = step['order']
                    
                    # Validation spéciale pour les questions de code
                    if question_type in ['code_completion', 'code_writing']:
                        await self._validate_code_question(question_data)
                    
                    all_questions.append(question_data)
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la génération de question: {e}")
                    # Fallback
                    all_questions.append(self._generate_fallback_question(step, q_idx))
        
        return all_questions
    
    def _determine_question_type(self, step: Dict[str, Any], question_index: int) -> str:
        """Détermine le type de question selon l'étape"""
        step_type = step.get('step_type', 'content')
        
        if step_type == 'coding':
            return 'code_writing' if question_index == 0 else 'code_completion'
        elif 'code' in step.get('content', '').lower():
            return 'code_completion'
        elif question_index == 0:
            return 'multiple_choice'
        else:
            return 'short_answer'
    
    async def _validate_code_question(self, question_data: Dict[str, Any]):
        """Valide une question contenant du code"""
        correct_answer = question_data.get('correct_answer', '')
        
        if correct_answer and len(correct_answer) > 10:  # Probablement du code
            validation = self.code_validator.validate_syntax(correct_answer)
            if not validation.is_valid:
                logger.warning(f"Code invalide dans la question: {validation.errors}")
                # Corriger automatiquement si possible
                question_data['correct_answer'] = self._fix_code_syntax(correct_answer)
    
    def _fix_code_syntax(self, code: str) -> str:
        """Tente de corriger des erreurs de syntaxe basiques"""
        # Corrections basiques
        fixed_code = code.strip()
        
        # Supprimer les caractères non-ASCII problématiques
        fixed_code = ''.join(char for char in fixed_code if ord(char) < 128)
        
        # Corriger l'indentation de base
        lines = fixed_code.split('\n')
        if len(lines) > 1:
            # Réindenter basiquement
            fixed_lines = []
            for line in lines:
                cleaned_line = line.strip()
                if cleaned_line:
                    fixed_lines.append(cleaned_line)
            fixed_code = '\n'.join(fixed_lines)
        
        return fixed_code
    
    async def _validate_quest_coherence(
        self, quest_structure: Dict[str, Any], steps: List[Dict[str, Any]], questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Valide la cohérence globale de la quête"""
        
        validation_result = {
            'valid': True,
            'confidence': 1.0,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Vérifier la progression logique des étapes
        if not self._validate_step_progression(steps):
            validation_result['valid'] = False
            validation_result['errors'].append("Progression illogique des étapes")
            validation_result['confidence'] -= 0.3
        
        # Vérifier l'alignement avec les objectifs
        objectives = quest_structure.get('learning_objectives', [])
        if not self._validate_objectives_coverage(steps, objectives):
            validation_result['warnings'].append("Certains objectifs ne sont pas couverts")
            validation_result['confidence'] -= 0.1
        
        # Vérifier la qualité des questions
        question_quality = self._assess_question_quality(questions)
        if question_quality < 0.7:
            validation_result['warnings'].append("Qualité des questions à améliorer")
            validation_result['confidence'] -= 0.1
        
        # Vérifier la durée estimée
        estimated_duration = quest_structure.get('estimated_duration', 30)
        calculated_duration = len(steps) * 6 + len(questions) * 2  # Estimation
        if abs(estimated_duration - calculated_duration) > 15:
            validation_result['suggestions'].append(
                f"Ajuster la durée estimée à {calculated_duration} minutes"
            )
        
        # Vérifier la cohérence du niveau de difficulté
        if not self._validate_difficulty_consistency(quest_structure, steps, questions):
            validation_result['warnings'].append("Incohérence dans le niveau de difficulté")
            validation_result['confidence'] -= 0.1
        
        return validation_result
    
    def _validate_step_progression(self, steps: List[Dict[str, Any]]) -> bool:
        """Valide que les étapes progressent logiquement"""
        if len(steps) < 2:
            return True
        
        # Vérifier l'ordre des étapes
        orders = [step.get('order', 0) for step in steps]
        return orders == sorted(orders)
    
    def _validate_objectives_coverage(self, steps: List[Dict[str, Any]], objectives: List[str]) -> bool:
        """Valide que les objectifs sont couverts par les étapes"""
        if not objectives:
            return True
        
        # Analyse basique : chercher les mots-clés des objectifs dans les étapes
        step_content = ' '.join([
            step.get('title', '') + ' ' + step.get('description', '') 
            for step in steps
        ]).lower()
        
        covered_objectives = 0
        for objective in objectives:
            # Extraire les mots-clés de l'objectif
            objective_words = objective.lower().split()
            if any(word in step_content for word in objective_words if len(word) > 3):
                covered_objectives += 1
        
        return covered_objectives >= len(objectives) * 0.7  # 70% des objectifs couverts
    
    def _assess_question_quality(self, questions: List[Dict[str, Any]]) -> float:
        """Évalue la qualité des questions générées"""
        if not questions:
            return 0.0
        
        quality_score = 0.0
        
        for question in questions:
            question_score = 1.0
            
            # Vérifier la longueur de l'énoncé
            question_text = question.get('question_text', '')
            if len(question_text) < 10:
                question_score -= 0.3
            elif len(question_text) > 200:
                question_score -= 0.1
            
            # Vérifier la présence d'explication
            if not question.get('explanation'):
                question_score -= 0.2
            
            # Vérifier les choix pour les QCM
            if question.get('question_type') == 'multiple_choice':
                choices = question.get('choices', [])
                if len(choices) < 3:
                    question_score -= 0.3
                elif len(choices) > 5:
                    question_score -= 0.1
            
            # Vérifier la cohérence de la réponse correcte
            correct_answer = question.get('correct_answer', '')
            if not correct_answer:
                question_score -= 0.5
            
            quality_score += max(0.0, question_score)
        
        return quality_score / len(questions)
    
    def _validate_difficulty_consistency(
        self, quest_structure: Dict[str, Any], steps: List[Dict[str, Any]], questions: List[Dict[str, Any]]
    ) -> bool:
        """Valide la cohérence du niveau de difficulté"""
        quest_level = quest_structure.get('level', 'beginner')
        
        # Critères de difficulté par niveau
        level_criteria = {
            'beginner': {
                'max_steps': 8,
                'max_concepts_per_step': 2,
                'avoid_advanced_concepts': True
            },
            'intermediate': {
                'max_steps': 12,
                'max_concepts_per_step': 3,
                'avoid_advanced_concepts': False
            },
            'advanced': {
                'max_steps': 15,
                'max_concepts_per_step': 5,
                'avoid_advanced_concepts': False
            }
        }
        
        criteria = level_criteria.get(quest_level, level_criteria['beginner'])
        
        # Vérifier le nombre d'étapes
        if len(steps) > criteria['max_steps']:
            return False
        
        # Analyser la complexité du contenu
        advanced_concepts = [
            'metaclass', 'decorator', 'generator', 'async', 'threading',
            'multiprocessing', 'context manager', 'descriptor'
        ]
        
        if criteria['avoid_advanced_concepts']:
            content = ' '.join([step.get('content', '') for step in steps]).lower()
            if any(concept in content for concept in advanced_concepts):
                return False
        
        return True
    
    async def _auto_correct_quest(
        self, quest_structure: Dict[str, Any], steps: List[Dict[str, Any]], 
        questions: List[Dict[str, Any]], validation_result: Dict[str, Any]
    ) -> tuple:
        """Tente de corriger automatiquement les problèmes détectés"""
        
        corrected_structure = quest_structure.copy()
        corrected_steps = steps.copy()
        corrected_questions = questions.copy()
        
        # Corriger l'ordre des étapes si nécessaire
        if any("progression illogique" in error for error in validation_result.get('errors', [])):
            corrected_steps.sort(key=lambda x: x.get('order', 0))
            for i, step in enumerate(corrected_steps):
                step['order'] = i + 1
        
        # Ajuster la durée estimée
        for suggestion in validation_result.get('suggestions', []):
            if 'durée estimée' in suggestion:
                # Extraire la nouvelle durée suggérée
                import re
                match = re.search(r'(\d+) minutes', suggestion)
                if match:
                    new_duration = int(match.group(1))
                    corrected_structure['estimated_duration'] = new_duration
        
        # Améliorer les questions de faible qualité
        for question in corrected_questions:
            if len(question.get('question_text', '')) < 10:
                question['question_text'] = f"Question sur {quest_structure.get('title', 'le sujet')}: " + question.get('question_text', '')
            
            if not question.get('explanation'):
                question['explanation'] = "Explication à compléter"
        
        return corrected_structure, corrected_steps, corrected_questions
    
    async def _save_quest_to_db(
        self, quest_structure: Dict[str, Any], steps: List[Dict[str, Any]], 
        questions: List[Dict[str, Any]], context: AgentContext
    ) -> Optional[int]:
        """Sauvegarde la quête en base de données"""
        
        try:
            with get_db_session() as db:
                # Créer la quête principale
                quest = Quest(
                    title=quest_structure['title'],
                    description=quest_structure['description'],
                    short_description=quest_structure.get('short_description'),
                    category=quest_structure['category'],
                    difficulty=quest_structure['difficulty'],
                    level=quest_structure['level'],
                    slug=quest_structure['slug'],
                    estimated_duration=quest_structure['estimated_duration'],
                    passing_score=quest_structure.get('passing_score', 0.7),
                    xp_reward=quest_structure.get('xp_reward', 100),
                    created_by=context.user_id,
                    status='draft'
                )
                
                # Définir les objectifs et tags
                quest.set_learning_objectives(quest_structure.get('learning_objectives', []))
                quest.set_tags(quest_structure.get('tags', []))
                if quest_structure.get('prerequisites'):
                    quest.set_prerequisites(quest_structure['prerequisites'])
                
                db.add(quest)
                db.flush()  # Pour obtenir l'ID
                
                # Créer les étapes
                step_objects = []
                for step_data in steps:
                    step = QuestStep(
                        quest_id=quest.id,
                        order=step_data['order'],
                        title=step_data['title'],
                        description=step_data['description'],
                        step_type=step_data.get('step_type', 'content'),
                        content=step_data.get('content'),
                        code_template=step_data.get('code_template'),
                        expected_output=step_data.get('expected_output'),
                        max_attempts=step_data.get('max_attempts', 3)
                    )
                    
                    # Définir les ressources et indices
                    if step_data.get('resources'):
                        step.set_resources(step_data['resources'])
                    if step_data.get('hints'):
                        step.set_hints(step_data['hints'])
                    if step_data.get('validation_rules'):
                        step.set_validation_rules(step_data['validation_rules'])
                    
                    db.add(step)
                    step_objects.append(step)
                
                db.flush()  # Pour obtenir les IDs des étapes
                
                # Créer les questions
                for question_data in questions:
                    # Trouver l'étape correspondante
                    step_order = question_data.get('step_order', 1)
                    step_obj = next((s for s in step_objects if s.order == step_order), None)
                    
                    if step_obj:
                        question = Question(
                            step_id=step_obj.id,
                            question_text=question_data['question_text'],
                            question_type=question_data['question_type'],
                            correct_answer=question_data['correct_answer'],
                            explanation=question_data.get('explanation'),
                            points=question_data.get('points', 1),
                            time_limit=question_data.get('time_limit'),
                            shuffle_choices=question_data.get('shuffle_choices', True),
                            case_sensitive=question_data.get('case_sensitive', False)
                        )
                        
                        # Définir les choix pour les QCM
                        if question_data.get('choices'):
                            question.set_choices(question_data['choices'])
                        
                        db.add(question)
                
                # Mettre à jour les compteurs de la quête
                quest.total_steps = len(steps)
                quest.total_questions = len(questions)
                
                db.commit()
                
                logger.info(f"Quête sauvegardée en DB avec l'ID: {quest.id}")
                return quest.id
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde en DB: {e}")
            return None
    
    def _load_quest_templates(self) -> Dict[str, Any]:
        """Charge les templates de quêtes prédéfinis"""
        return {
            'basic_programming': {
                'structure': 'introduction -> concept -> practice -> application -> assessment',
                'step_types': ['content', 'content', 'coding', 'coding', 'question']
            },
            'problem_solving': {
                'structure': 'problem_definition -> analysis -> solution_design -> implementation -> testing',
                'step_types': ['content', 'content', 'content', 'coding', 'coding']
            },
            'concept_exploration': {
                'structure': 'overview -> deep_dive -> examples -> variations -> synthesis',
                'step_types': ['content', 'content', 'content', 'content', 'question']
            }
        }
    
    def _load_learning_objectives(self) -> Dict[str, List[str]]:
        """Charge une banque d'objectifs pédagogiques"""
        return {
            'python_basics': [
                "Comprendre la syntaxe de base de Python",
                "Maîtriser les types de données fondamentaux",
                "Utiliser les structures de contrôle",
                "Écrire des fonctions simples"
            ],
            'data_structures': [
                "Manipuler les listes et dictionnaires",
                "Comprendre les algorithmes de tri",
                "Optimiser l'accès aux données",
                "Choisir la structure appropriée"
            ],
            'object_oriented': [
                "Définir et utiliser des classes",
                "Implémenter l'héritage",
                "Comprendre l'encapsulation",
                "Appliquer le polymorphisme"
            ]
        }
    
    def _parse_llm_json_response(self, response: str) -> Dict[str, Any]:
        """Parse une réponse JSON du LLM avec fallback"""
        try:
            # Nettoyer la réponse
            cleaned_response = response.strip()
            
            # Extraire le JSON si entouré de texte
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Essayer de parser directement
                return json.loads(cleaned_response)
                
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON: {e}")
            logger.debug(f"Réponse reçue: {response[:200]}...")
            raise ValueError(f"Réponse JSON invalide du LLM: {e}")
    
    def _generate_fallback_quest_structure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une structure de quête de fallback"""
        return {
            'title': f"Introduction à {params['topic']}",
            'description': f"Apprenez les bases de {params['topic']} à travers des exercices pratiques",
            'short_description': f"Découverte de {params['topic']}",
            'category': params['category'],
            'difficulty': params['difficulty'],
            'level': params['target_level'],
            'estimated_duration': params['estimated_duration'],
            'learning_objectives': [
                f"Comprendre les concepts de base de {params['topic']}",
                f"Appliquer {params['topic']} dans des exercices pratiques",
                f"Maîtriser les bonnes pratiques de {params['topic']}"
            ],
            'prerequisites': [],
            'tags': [params['topic'], params['target_level'], 'programmation'],
            'xp_reward': 100,
            'passing_score': 0.7,
            'slug': slugify(f"introduction-{params['topic']}"),
            'status': 'draft',
            'total_steps': params['num_steps']
        }
    
    def _generate_fallback_step(self, order: int, quest_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une étape de fallback"""
        return {
            'order': order,
            'title': f"Étape {order} - {quest_structure['title']}",
            'description': f"Contenu pédagogique pour l'étape {order}",
            'step_type': 'content',
            'content': f"Contenu à développer pour l'étape {order}",
            'code_template': "# Code à compléter",
            'expected_output': "",
            'resources': [],
            'hints': [f"Indice pour l'étape {order}"],
            'validation_rules': {'check_syntax': True},
            'max_attempts': 3
        }
    
    def _generate_fallback_question(self, step: Dict[str, Any], question_index: int) -> Dict[str, Any]:
        """Génère une question de fallback"""
        return {
            'question_text': f"Question sur {step['title']}",
            'question_type': 'multiple_choice',
            'choices': ['Option A', 'Option B', 'Option C', 'Option D'],
            'correct_answer': 'Option A',
            'explanation': 'Explication à développer',
            'points': 1,
            'time_limit': 120,
            'shuffle_choices': True,
            'case_sensitive': False,
            'step_order': step['order']
        }
    
    def _format_skills(self, skill_scores: Dict[str, float]) -> str:
        """Formate les compétences pour affichage"""
        if not skill_scores:
            return "En évaluation"
        
        strong_skills = [skill for skill, score in skill_scores.items() if score >= 75.0]
        return ', '.join(strong_skills) if strong_skills else "En développement"
    
    def get_required_fields(self) -> List[str]:
        """Champs requis pour le générateur"""
        return ['topic', 'target_level', 'category']
    
    def get_capabilities(self) -> List[str]:
        """Capacités du générateur de quêtes"""
        return [
            'quest_structure_generation',
            'step_by_step_creation',
            'question_generation',
            'code_exercise_creation',
            'difficulty_adaptation',
            'learning_objective_alignment',
            'content_validation',
            'database_integration'
        ]
    
    async def generate_quest_variation(self, base_quest_id: int, variation_type: str) -> AgentResponse:
        """
        Génère une variation d'une quête existante
        
        Args:
            base_quest_id: ID de la quête de base
            variation_type: Type de variation (easier, harder, different_approach)
            
        Returns:
            Nouvelle quête variée
        """
        try:
            with get_db_session() as db:
                base_quest = db.query(Quest).filter(Quest.id == base_quest_id).first()
                if not base_quest:
                    return AgentResponse(
                        success=False,
                        message="Quête de base non trouvée",
                        errors=[f"Aucune quête avec l'ID {base_quest_id}"]
                    )
                
                # Créer les paramètres de variation
                variation_params = self._create_variation_params(base_quest, variation_type)
                
                # Générer la nouvelle quête
                context = AgentContext()
                return await self.process(variation_params, context)
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération de variation: {e}")
            return AgentResponse(
                success=False,
                message="Erreur lors de la génération de variation",
                errors=[str(e)]
            )
    
    def _create_variation_params(self, base_quest: Quest, variation_type: str) -> Dict[str, Any]:
        """Crée les paramètres pour une variation de quête"""
        base_params = {
            'topic': base_quest.title,
            'target_level': base_quest.level,
            'category': base_quest.category,
            'estimated_duration': base_quest.estimated_duration,
            'num_steps': base_quest.total_steps or 5,
            'difficulty': base_quest.difficulty
        }
        
        if variation_type == 'easier':
            if base_params['target_level'] == 'intermediate':
                base_params['target_level'] = 'beginner'
            base_params['difficulty'] = 'easy'
            base_params['num_steps'] = max(3, base_params['num_steps'] - 2)
            
        elif variation_type == 'harder':
            if base_params['target_level'] == 'beginner':
                base_params['target_level'] = 'intermediate'
            elif base_params['target_level'] == 'intermediate':
                base_params['target_level'] = 'advanced'
            base_params['difficulty'] = 'hard'
            base_params['num_steps'] = min(12, base_params['num_steps'] + 2)
            
        elif variation_type == 'different_approach':
            base_params['topic'] = f"{base_params['topic']} - Approche alternative"
            
        return base_params
# src/agents/tutor_agent.py
"""
Agent tuteur principal - Gère les interactions pédagogiques avec l'utilisateur
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime, timezone

from src.agents.base_agent import BaseAgent, AgentType, AgentContext, AgentResponse
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import PromptManager
from src.rag.retriever import DocumentRetriever
from src.models import User, UserProgress, Quest
from src.core.database import get_db_session
from src.utils import TextProcessor, clean_whitespace, extract_keywords

logger = logging.getLogger(__name__)

class TutorAgent(BaseAgent):
    """Agent tuteur pour l'assistance pédagogique personnalisée"""
    
    def __init__(self, agent_type: AgentType = AgentType.TUTOR, name: str = None, **kwargs):
        super().__init__(agent_type, name or "tutor_agent")
        
        # Clients et services
        self.llm_client = OllamaClient()
        self.prompt_manager = PromptManager()
        self.document_retriever = DocumentRetriever()
        self.text_processor = TextProcessor()
        
        # Configuration spécifique au tuteur
        self.max_context_length = kwargs.get('max_context_length', 4000)
        self.response_style = kwargs.get('response_style', 'encouraging')
        self.explanation_depth = kwargs.get('explanation_depth', 'detailed')
        self.use_rag = kwargs.get('use_rag', True)
        self.personalization_level = kwargs.get('personalization_level', 'high')
        
        # Cache pour les réponses fréquentes
        self.response_cache = {}
        self.cache_ttl_minutes = 30
        
        logger.info(f"Agent tuteur initialisé avec style: {self.response_style}")
    
    async def process(self, input_data: Dict[str, Any], context: AgentContext) -> AgentResponse:
        """
        Traite une requête d'assistance pédagogique
        
        Args:
            input_data: Contient 'question', 'subject' (optionnel), 'difficulty' (optionnel)
            context: Contexte utilisateur
            
        Returns:
            Réponse pédagogique personnalisée
        """
        question = input_data.get('question', '').strip()
        subject = input_data.get('subject', 'python')
        difficulty = input_data.get('difficulty', 'auto')
        
        if not question:
            return AgentResponse(
                success=False,
                message="Question vide",
                errors=["Une question est requise"]
            )
        
        try:
            # 1. Analyser la question
            question_analysis = await self._analyze_question(question, subject)
            
            # 2. Adapter le niveau selon l'utilisateur
            if difficulty == 'auto' and context.user_progress:
                difficulty = self._determine_difficulty(question_analysis, context.user_progress)
            
            # 3. Récupérer le contexte documentaire
            relevant_docs = []
            if self.use_rag:
                relevant_docs = await self._retrieve_relevant_context(
                    question, subject, question_analysis.get('keywords', [])
                )
            
            # 4. Générer la réponse pédagogique
            response_content = await self._generate_pedagogical_response(
                question=question,
                question_analysis=question_analysis,
                relevant_docs=relevant_docs,
                context=context,
                difficulty=difficulty
            )
            
            # 5. Post-traiter et enrichir la réponse
            enhanced_response = await self._enhance_response(
                response_content, question_analysis, context
            )
            
            # 6. Enregistrer l'interaction pour l'apprentissage
            await self._log_interaction(question, enhanced_response, context)
            
            return AgentResponse(
                success=True,
                message=enhanced_response['content'],
                data={
                    'explanation_type': enhanced_response.get('type', 'general'),
                    'confidence': enhanced_response.get('confidence', 0.8),
                    'subject': subject,
                    'difficulty': difficulty,
                    'keywords': question_analysis.get('keywords', []),
                    'concepts': question_analysis.get('concepts', [])
                },
                suggestions=enhanced_response.get('suggestions', []),
                next_actions=enhanced_response.get('next_actions', []),
                confidence=enhanced_response.get('confidence', 0.8),
                reasoning=enhanced_response.get('reasoning', '')
            )
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement tuteur: {e}")
            return AgentResponse(
                success=False,
                message="Désolé, je n'ai pas pu traiter votre question. Pouvez-vous la reformuler ?",
                errors=[str(e)]
            )
    
    async def _analyze_question(self, question: str, subject: str) -> Dict[str, Any]:
        """
        Analyse la question pour comprendre l'intention et extraire des informations
        
        Args:
            question: Question de l'utilisateur
            subject: Sujet principal
            
        Returns:
            Analyse de la question
        """
        # Extraction de mots-clés
        keywords = extract_keywords(question, max_keywords=10)
        
        # Analyse des concepts de programmation
        programming_concepts = self.text_processor.extract_programming_concepts(question)
        
        # Détection du type de question
        question_type = self._classify_question_type(question)
        
        # Analyse de complexité
        complexity = self.text_processor.analyze_text_complexity(question)
        
        return {
            'keywords': keywords,
            'concepts': programming_concepts,
            'type': question_type,
            'complexity': complexity,
            'subject': subject,
            'requires_code_example': self._requires_code_example(question),
            'difficulty_indicators': self._extract_difficulty_indicators(question)
        }
    
    def _classify_question_type(self, question: str) -> str:
        """Classifie le type de question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['comment', 'how', 'wie']):
            return 'how_to'
        elif any(word in question_lower for word in ['pourquoi', 'why', 'warum']):
            return 'explanation'
        elif any(word in question_lower for word in ['quoi', 'what', 'was', 'que']):
            return 'definition'
        elif any(word in question_lower for word in ['erreur', 'error', 'bug', 'problème']):
            return 'debugging'
        elif any(word in question_lower for word in ['exemple', 'example', 'beispiel']):
            return 'example_request'
        elif any(word in question_lower for word in ['différence', 'difference', 'vs', 'versus']):
            return 'comparison'
        else:
            return 'general'
    
    def _requires_code_example(self, question: str) -> bool:
        """Détermine si la question nécessite un exemple de code"""
        code_indicators = [
            'exemple', 'example', 'code', 'syntaxe', 'syntax',
            'comment faire', 'how to', 'implementation', 'implémentation'
        ]
        return any(indicator in question.lower() for indicator in code_indicators)
    
    def _extract_difficulty_indicators(self, question: str) -> List[str]:
        """Extrait les indicateurs de difficulté de la question"""
        question_lower = question.lower()
        indicators = []
        
        # Indicateurs de niveau débutant
        if any(word in question_lower for word in ['débutant', 'beginner', 'simple', 'basic']):
            indicators.append('beginner')
        
        # Indicateurs de niveau avancé
        if any(word in question_lower for word in ['avancé', 'advanced', 'complex', 'expert']):
            indicators.append('advanced')
        
        # Concepts avancés
        advanced_concepts = ['metaclass', 'decorator', 'generator', 'async', 'threading']
        if any(concept in question_lower for concept in advanced_concepts):
            indicators.append('advanced_concepts')
        
        return indicators
    
    def _determine_difficulty(self, question_analysis: Dict[str, Any], user_progress: Dict[str, Any]) -> str:
        """Détermine le niveau de difficulté approprié pour l'utilisateur"""
        user_level = user_progress.get('level', 'beginner')
        
        # Ajuster selon les indicateurs de la question
        difficulty_indicators = question_analysis.get('difficulty_indicators', [])
        
        if 'beginner' in difficulty_indicators:
            return 'beginner'
        elif 'advanced' in difficulty_indicators or 'advanced_concepts' in difficulty_indicators:
            return 'advanced'
        else:
            # Utiliser le niveau utilisateur par défaut
            return user_level
    
    async def _retrieve_relevant_context(
        self, question: str, subject: str, keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Récupère le contexte documentaire pertinent
        
        Args:
            question: Question originale
            subject: Sujet
            keywords: Mots-clés extraits
            
        Returns:
            Liste des documents pertinents
        """
        try:
            # Recherche dans la base documentaire
            search_query = f"{question} {' '.join(keywords[:5])}"
            docs = await self.document_retriever.retrieve_relevant_documents(
                query=search_query,
                subject=subject,
                max_results=3
            )
            
            return docs
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de contexte: {e}")
            return []
    
    async def _generate_pedagogical_response(
        self,
        question: str,
        question_analysis: Dict[str, Any],
        relevant_docs: List[Dict[str, Any]],
        context: AgentContext,
        difficulty: str
    ) -> str:
        """
        Génère une réponse pédagogique personnalisée
        
        Args:
            question: Question originale
            question_analysis: Analyse de la question
            relevant_docs: Documents pertinents
            context: Contexte utilisateur
            difficulty: Niveau de difficulté
            
        Returns:
            Réponse générée
        """
        # Construire le prompt personnalisé
        prompt = await self._build_tutor_prompt(
            question, question_analysis, relevant_docs, context, difficulty
        )
        
        # Générer la réponse avec le LLM
        response = await self.llm_client.generate_async(
            prompt=prompt,
            max_tokens=self.max_context_length,
            temperature=0.7
        )
        
        return response
    
    async def _build_tutor_prompt(
        self,
        question: str,
        question_analysis: Dict[str, Any],
        relevant_docs: List[Dict[str, Any]],
        context: AgentContext,
        difficulty: str
    ) -> str:
        """Construit le prompt pour le tuteur"""
        
    async def _build_tutor_prompt(
        self,
        question: str,
        question_analysis: Dict[str, Any],
        relevant_docs: List[Dict[str, Any]],
        context: AgentContext,
        difficulty: str
    ) -> str:
        """Construit le prompt pour le tuteur"""
        
        # Contexte utilisateur
        user_context = ""
        if context.user_progress:
            user_context = f"""
Informations sur l'apprenant :
- Niveau : {context.user_progress.get('level', 'débutant')}
- XP : {context.user_progress.get('xp_points', 0)} points
- Série actuelle : {context.user_progress.get('current_streak', 0)} jours
- Compétences fortes : {self._get_strong_skills(context.user_progress.get('skill_scores', {}))}
"""
        
        # Contexte documentaire
        doc_context = ""
        if relevant_docs:
            doc_context = "\nDocumentation pertinente :\n"
            for i, doc in enumerate(relevant_docs[:3], 1):
                doc_context += f"{i}. {doc.get('title', 'Document')}: {doc.get('content', '')[:200]}...\n"
        
        # Style de réponse adapté
        style_instruction = self._get_style_instruction()
        
        # Historique de conversation
        conversation_history = ""
        if context.conversation_history:
            recent_history = context.conversation_history[-3:]  # 3 derniers échanges
            conversation_history = "\nHistorique récent :\n"
            for entry in recent_history:
                role = entry.get('role', 'user')
                content = entry.get('content', '')[:100]
                conversation_history += f"- {role}: {content}...\n"
        
        # Prompt principal
        prompt = f"""Tu es un tuteur expert en programmation Python, spécialisé dans l'apprentissage personnalisé.

{user_context}

Question de l'apprenant : "{question}"

Type de question : {question_analysis.get('type', 'general')}
Concepts identifiés : {', '.join(question_analysis.get('concepts', {}).get('python_keywords', []))}
Niveau de difficulté : {difficulty}
Nécessite un exemple de code : {'Oui' if question_analysis.get('requires_code_example') else 'Non'}

{doc_context}

{conversation_history}

Instructions de réponse :
{style_instruction}

IMPORTANT : 
- Adapte ton niveau de langage au niveau de l'apprenant ({difficulty})
- Donne des exemples concrets et pratiques
- Encourage l'apprenant dans sa progression
- Si tu donnes du code, explique chaque partie importante
- Propose des exercices de pratique si pertinent
- Reste bienveillant et pédagogue

Réponds maintenant à la question de manière claire et structurée :"""

        return prompt
    
    def _get_strong_skills(self, skill_scores: Dict[str, float]) -> str:
        """Identifie les compétences fortes de l'utilisateur"""
        if not skill_scores:
            return "En cours d'évaluation"
        
        strong_skills = [skill for skill, score in skill_scores.items() if score >= 75.0]
        return ', '.join(strong_skills) if strong_skills else "En développement"
    
    def _get_style_instruction(self) -> str:
        """Retourne les instructions de style selon la configuration"""
        styles = {
            'encouraging': """
- Sois encourageant et positif dans tes réponses
- Félicite les bonnes questions et la curiosité
- Transforme les erreurs en opportunités d'apprentissage
- Utilise des phrases comme "Excellente question !", "Tu es sur la bonne voie"
""",
            'formal': """
- Adopte un ton professionnel et académique
- Utilise un vocabulaire technique précis
- Structure tes réponses de manière méthodique
- Cite des références et bonnes pratiques
""",
            'casual': """
- Utilise un ton décontracté et amical
- Emploie des exemples du quotidien
- Rends l'apprentissage ludique et accessible
- N'hésite pas à utiliser des analogies simples
""",
            'concise': """
- Sois direct et va à l'essentiel
- Privilégie les réponses courtes et précises
- Évite les explications trop longues
- Focus sur la solution pratique
"""
        }
        
        return styles.get(self.response_style, styles['encouraging'])
    
    async def _enhance_response(
        self, 
        response_content: str, 
        question_analysis: Dict[str, Any], 
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Enrichit la réponse avec des éléments pédagogiques supplémentaires
        
        Args:
            response_content: Contenu de la réponse
            question_analysis: Analyse de la question
            context: Contexte utilisateur
            
        Returns:
            Réponse enrichie
        """
        enhanced = {
            'content': clean_whitespace(response_content),
            'type': question_analysis.get('type', 'general'),
            'confidence': 0.8,
            'suggestions': [],
            'next_actions': [],
            'reasoning': ''
        }
        
        # Ajouter des suggestions selon le type de question
        if question_analysis.get('type') == 'how_to':
            enhanced['suggestions'] = [
                "Essayez de reproduire l'exemple dans votre environnement",
                "Modifiez les paramètres pour voir l'effet",
                "Consultez la documentation officielle pour plus de détails"
            ]
        elif question_analysis.get('type') == 'debugging':
            enhanced['suggestions'] = [
                "Utilisez print() pour déboguer étape par étape",
                "Vérifiez les types de vos variables",
                "Lisez attentivement les messages d'erreur"
            ]
        elif question_analysis.get('type') == 'explanation':
            enhanced['suggestions'] = [
                "Pratiquez avec des exemples simples",
                "Dessinez un schéma si c'est un concept complexe",
                "Reliez ce concept à ce que vous connaissez déjà"
            ]
        
        # Ajouter des actions suivantes
        if question_analysis.get('requires_code_example'):
            enhanced['next_actions'].append({
                'action': 'practice_code',
                'description': 'Pratiquer avec du code',
                'priority': 'high'
            })
        
        # Proposer des quêtes liées
        if context.user_progress:
            enhanced['next_actions'].append({
                'action': 'explore_quest',
                'description': 'Explorer des quêtes sur ce sujet',
                'priority': 'medium'
            })
        
        # Calculer la confiance selon la qualité de la réponse
        if len(response_content) > 100 and question_analysis.get('concepts'):
            enhanced['confidence'] = 0.9
        elif len(response_content) < 50:
            enhanced['confidence'] = 0.6
        
        enhanced['reasoning'] = f"Réponse de type {question_analysis.get('type')} adaptée au niveau utilisateur"
        
        return enhanced
    
    async def _log_interaction(
        self, 
        question: str, 
        response: Dict[str, Any], 
        context: AgentContext
    ):
        """
        Enregistre l'interaction pour l'amélioration continue
        
        Args:
            question: Question posée
            response: Réponse fournie
            context: Contexte de l'interaction
        """
        try:
            if context.user_id:
                interaction_data = {
                    'user_id': context.user_id,
                    'agent_type': self.agent_type.value,
                    'question': question,
                    'response_type': response.get('type'),
                    'confidence': response.get('confidence'),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Ajouter à l'historique du contexte
                context.conversation_history.append({
                    'role': 'user',
                    'content': question,
                    'timestamp': interaction_data['timestamp']
                })
                
                context.conversation_history.append({
                    'role': 'assistant',
                    'content': response['content'][:200] + '...',  # Version tronquée
                    'timestamp': interaction_data['timestamp']
                })
                
                # Limiter l'historique
                if len(context.conversation_history) > 20:
                    context.conversation_history = context.conversation_history[-20:]
                
                logger.debug(f"Interaction tuteur enregistrée pour l'utilisateur {context.user_id}")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de l'interaction: {e}")
    
    def get_required_fields(self) -> List[str]:
        """Champs requis pour le tuteur"""
        return ['question']
    
    def get_capabilities(self) -> List[str]:
        """Capacités du tuteur"""
        return [
            'explanation_generation',
            'code_examples',
            'personalized_learning',
            'concept_explanation',
            'debugging_help',
            'progress_tracking',
            'adaptive_difficulty',
            'multilingual_support'
        ]
    
    async def _health_check_test(self, test_input: Dict[str, Any], test_context: AgentContext) -> bool:
        """Test de santé spécifique au tuteur"""
        try:
            # Test avec une question simple
            test_data = {'question': 'Qu\'est-ce qu\'une variable en Python ?'}
            response = await self.process(test_data, test_context)
            
            return response.success and len(response.message) > 20
        except Exception:
            return False
    
    def configure_response_style(self, style: str):
        """Configure le style de réponse du tuteur"""
        valid_styles = ['encouraging', 'formal', 'casual', 'concise']
        if style in valid_styles:
            self.response_style = style
            logger.info(f"Style de réponse mis à jour: {style}")
        else:
            logger.warning(f"Style invalide: {style}. Styles valides: {valid_styles}")
    
    def configure_explanation_depth(self, depth: str):
        """Configure la profondeur des explications"""
        valid_depths = ['brief', 'detailed', 'comprehensive']
        if depth in valid_depths:
            self.explanation_depth = depth
            logger.info(f"Profondeur d'explication mise à jour: {depth}")
        else:
            logger.warning(f"Profondeur invalide: {depth}. Profondeurs valides: {valid_depths}")
    
    async def suggest_learning_path(self, context: AgentContext) -> List[Dict[str, Any]]:
        """
        Suggère un parcours d'apprentissage personnalisé
        
        Args:
            context: Contexte utilisateur
            
        Returns:
            Liste des suggestions de parcours
        """
        if not context.user_progress:
            return []
        
        suggestions = []
        skill_scores = context.user_progress.get('skill_scores', {})
        user_level = context.user_progress.get('level', 'beginner')
        
        # Identifier les compétences à améliorer
        weak_skills = [skill for skill, score in skill_scores.items() if score < 60.0]
        
        for skill in weak_skills:
            suggestions.append({
                'type': 'skill_improvement',
                'skill': skill,
                'current_score': skill_scores.get(skill, 0),
                'target_score': min(skill_scores.get(skill, 0) + 20, 100),
                'recommended_actions': [
                    f"Pratiquer des exercices sur {skill}",
                    f"Revoir les concepts de base de {skill}",
                    f"Faire des projets impliquant {skill}"
                ]
            })
        
        # Suggérer des quêtes selon le niveau
        if user_level == 'beginner':
            suggestions.append({
                'type': 'quest_recommendation',
                'category': 'python_basics',
                'description': 'Maîtriser les fondamentaux de Python',
                'estimated_duration': '2-3 semaines'
            })
        elif user_level == 'intermediate':
            suggestions.append({
                'type': 'quest_recommendation',
                'category': 'python_intermediate',
                'description': 'Approfondir les concepts avancés',
                'estimated_duration': '3-4 semaines'
            })
        
        return suggestions[:5]  # Limiter à 5 suggestions
    
    async def explain_concept(self, concept: str, context: AgentContext) -> AgentResponse:
        """
        Explique un concept spécifique de manière détaillée
        
        Args:
            concept: Concept à expliquer
            context: Contexte utilisateur
            
        Returns:
            Explication détaillée du concept
        """
        input_data = {
            'question': f"Peux-tu m'expliquer le concept de {concept} en Python ?",
            'subject': 'python',
            'difficulty': context.user_progress.get('level', 'beginner') if context.user_progress else 'beginner'
        }
        
        return await self.process(input_data, context)
    
    async def generate_practice_exercise(self, topic: str, difficulty: str) -> Dict[str, Any]:
        """
        Génère un exercice de pratique sur un sujet donné
        
        Args:
            topic: Sujet de l'exercice
            difficulty: Niveau de difficulté
            
        Returns:
            Exercice généré
        """
        prompt = f"""Génère un exercice de programmation Python sur le sujet : {topic}
        
Niveau de difficulté : {difficulty}

L'exercice doit contenir :
1. Une description claire du problème
2. Des exemples d'entrée et de sortie attendue
3. Des indices pour guider l'apprenant
4. Une solution commentée
5. Des variantes possibles pour approfondir

Format la réponse en JSON avec les clés : description, examples, hints, solution, variants"""
        
        try:
            response = await self.llm_client.generate_async(prompt, max_tokens=1500)
            # Ici on pourrait parser le JSON et valider la structure
            return {
                'success': True,
                'exercise': response,
                'topic': topic,
                'difficulty': difficulty
            }
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'exercice: {e}")
            return {
                'success': False,
                'error': str(e)
            }
from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage

from ..core.logger import logger


class PromptManager:
    """Gestionnaire des templates de prompts"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Charge tous les templates de prompts"""
        return {
            "tutor_system": """Tu es un assistant pédagogique spécialisé en Python, bienveillant et patient.

Ton rôle:
- Aider les apprenants à comprendre Python de manière progressive
- Expliquer les concepts avec des exemples concrets
- Encourager la pratique et l'expérimentation
- Adapter ton niveau de langage à celui de l'apprenant
- Donner des conseils pratiques et des bonnes pratiques

Contexte disponible:
{context}

Instructions:
- Utilise le contexte fourni pour donner des réponses précises
- Si le contexte ne suffit pas, dis-le clairement
- Propose toujours des exemples de code quand c'est pertinent
- Encourage l'apprenant et reste positif
- Pose des questions pour vérifier la compréhension""",

            "rag_qa": """Contexte de documentation Python:
{context}

Question de l'utilisateur: {question}

Instructions:
- Utilise UNIQUEMENT les informations du contexte fourni
- Si le contexte ne contient pas la réponse, dis "Je ne trouve pas cette information dans la documentation"
- Donne une réponse précise et structurée
- Inclus des exemples de code si pertinent
- Cite les sources quand possible""",

            "quest_generator": """Tu dois créer une quête pédagogique sur le sujet: {topic}

Niveau de difficulté: {difficulty}
Durée estimée: {duration} minutes

La quête doit contenir:
1. Un titre accrocheur
2. Une description claire des objectifs
3. Des étapes progressives (3-5 étapes)
4. Des exercices pratiques avec solution
5. Des conseils et astuces

Format de réponse en JSON:
{{
    "title": "...",
    "description": "...",
    "objectives": ["...", "..."],
    "steps": [
        {{
            "title": "...",
            "content": "...",
            "exercise": "...",
            "solution": "...",
            "tips": ["...", "..."]
        }}
    ]
}}""",

            "code_evaluator": """Évalue ce code Python:

Code à évaluer:
```python
{code}
```

Exercice attendu: {exercise_description}

Instructions:
- Vérifie si le code fonctionne
- Évalue si il répond aux exigences
- Donne des conseils d'amélioration
- Note sur 10 avec justification
- Propose une version optimisée si nécessaire

Format de réponse:
- ✅/❌ Fonctionnel
- ✅/❌ Répond aux exigences
- Note: X/10
- Commentaires: ...
- Améliorations suggérées: ...""",

            "beginner_explainer": """Explique ce concept Python de manière très simple pour un débutant:

Concept: {concept}

Instructions:
- Utilise un langage simple et accessible
- Donne des analogies de la vie quotidienne
- Fournis un exemple de code très basique
- Évite le jargon technique
- Encourage l'apprenant""",

            "debug_helper": """L'utilisateur a ce problème avec son code Python:

Code:
```python
{code}
```

Erreur:
{error}

Instructions:
- Identifie la cause de l'erreur
- Explique pourquoi ça ne marche pas
- Propose une solution étape par étape
- Donne des conseils pour éviter ce type d'erreur
- Reste pédagogique et encourageant"""
        }
    
    def get_prompt(self, template_name: str, **kwargs) -> str:
        """Récupère un prompt formaté"""
        try:
            template = self.templates.get(template_name)
            if not template:
                logger.error(f"Template '{template_name}' non trouvé")
                raise ValueError(f"Template '{template_name}' non trouvé")
            
            return template.format(**kwargs)
            
        except KeyError as e:
            logger.error(f"Variable manquante dans le template '{template_name}': {e}")
            raise ValueError(f"Variable manquante: {e}")
    
    def create_chat_messages(
        self, 
        system_prompt: str, 
        user_message: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Crée une liste de messages pour le chat"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Ajouter l'historique de conversation
        if conversation_history:
            messages.extend(conversation_history)
        
        # Ajouter le message utilisateur
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def format_context_for_rag(self, documents: List[Any]) -> str:
        """Formate les documents récupérés pour le contexte RAG"""
        if not documents:
            return "Aucune documentation pertinente trouvée."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = getattr(doc, 'metadata', {})
            source = metadata.get('source', 'Source inconnue')
            title = metadata.get('title', f'Document {i}')
            
            context_parts.append(f"## {title}\nSource: {source}\n\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
from typing import Dict, Any, List, Optional
from langchain.docstore.document import Document

from ..core.logger import logger
from ..core.exceptions import LLMError
from .ollama_client import OllamaClient
from .prompts import PromptManager
from ..rag.retriever import DocumentRetriever


class LLMChainManager:
    """Gestionnaire des chaînes LLM avec RAG"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.prompt_manager = PromptManager()
        self.retriever = DocumentRetriever()
    
    def answer_with_rag(
        self, 
        question: str, 
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Répond à une question en utilisant RAG"""
        try:
            logger.info(f"Question RAG: {question}")
            
            # Récupérer les documents pertinents
            relevant_docs = self.retriever.retrieve_relevant_documents(question)
            
            # Formater le contexte
            context = self.prompt_manager.format_context_for_rag(relevant_docs)
            
            # Créer le prompt RAG
            rag_prompt = self.prompt_manager.get_prompt(
                "rag_qa",
                context=context,
                question=question
            )
            
            # Générer la réponse
            response = self.ollama_client.generate(rag_prompt)
            
            return {
                "answer": response,
                "sources": [doc.metadata.get('source', '') for doc in relevant_docs],
                "num_sources": len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Erreur dans answer_with_rag: {e}")
            raise LLMError(f"Impossible de répondre avec RAG: {e}")
    
    def tutor_conversation(
        self, 
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        user_level: str = "beginner"
    ) -> str:
        """Conversation avec l'assistant tuteur"""
        try:
            # Récupérer du contexte si nécessaire
            relevant_docs = self.retriever.retrieve_relevant_documents(
                user_message, max_docs=3
            )
            context = self.prompt_manager.format_context_for_rag(relevant_docs)
            
            # Créer le prompt système
            system_prompt = self.prompt_manager.get_prompt(
                "tutor_system",
                context=context
            )
            
            # Créer les messages de chat
            messages = self.prompt_manager.create_chat_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                conversation_history=conversation_history
            )
            
            # Générer la réponse
            response = self.ollama_client.chat(messages)
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur dans tutor_conversation: {e}")
            raise LLMError(f"Erreur dans la conversation tuteur: {e}")
    
    def evaluate_code(self, code: str, exercise_description: str) -> Dict[str, Any]:
        """Évalue un code Python soumis par l'apprenant"""
        try:
            prompt = self.prompt_manager.get_prompt(
                "code_evaluator",
                code=code,
                exercise_description=exercise_description
            )
            
            response = self.ollama_client.generate(prompt)
            
            # Parser la réponse pour extraire les informations structurées
            evaluation = self._parse_code_evaluation(response)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Erreur dans evaluate_code: {e}")
            raise LLMError(f"Erreur lors de l'évaluation du code: {e}")
    
    def _parse_code_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse la réponse d'évaluation de code"""
        # Logique simple de parsing - à améliorer
        lines = response.split('\n')
        
        evaluation = {
            "functional": "✅" in response and "Fonctionnel" in response,
            "meets_requirements": "✅" in response and "exigences" in response,
            "score": 0,
            "comments": response,
            "suggestions": []
        }
        
        # Extraire la note
        for line in lines:
            if "Note:" in line:
                try:
                    score_part = line.split("Note:")[1].split("/")[0].strip()
                    evaluation["score"] = int(score_part)
                except:
                    pass
        
        return evaluation
    
    def generate_quest(
        self, 
        topic: str, 
        difficulty: str = "beginner", 
        duration: int = 30
    ) -> Dict[str, Any]:
        """Génère une nouvelle quête pédagogique"""
        try:
            prompt = self.prompt_manager.get_prompt(
                "quest_generator",
                topic=topic,
                difficulty=difficulty,
                duration=duration
            )
            
            response = self.ollama_client.generate(prompt)
            
            # Tenter de parser le JSON
            try:
                import json
                quest_data = json.loads(response)
                return quest_data
            except json.JSONDecodeError:
                logger.warning("Réponse non-JSON reçue pour la génération de quête")
                return {"error": "Format de réponse invalide", "raw_response": response}
                
        except Exception as e:
            logger.error(f"Erreur dans generate_quest: {e}")
            raise LLMError(f"Erreur lors de la génération de quête: {e}")


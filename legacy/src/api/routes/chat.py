# src/api/routes/chat.py
"""
Routes API pour les fonctionnalités de chat avec l'assistant pédagogique
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import logging
import json
import asyncio
from datetime import datetime

from src.agents.state_manager import StateManager
from src.core.exceptions import ValidationError, NotFoundError
from src.api.middleware.auth import get_current_user
from src.models.schemas import UserBase

logger = logging.getLogger(__name__)

router = APIRouter()


# Modèles Pydantic pour les requêtes/réponses
class ChatMessage(BaseModel):
    role: str = Field(..., description="Rôle du message (user/assistant)")
    content: str = Field(..., description="Contenu du message")
    timestamp: Optional[str] = Field(None, description="Horodatage")
    agent: Optional[str] = Field(None, description="Agent qui a généré le message")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Message de l'utilisateur")
    message_type: str = Field(default="text", description="Type de message (text, code, file)")
    session_id: Optional[str] = Field(None, description="ID de session existante")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contexte additionnel")


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="ID de la session")
    response: str = Field(..., description="Réponse de l'assistant")
    conversation_state: str = Field(..., description="État de la conversation")
    suggestions: List[Dict[str, Any]] = Field(default_factory=list, description="Suggestions d'actions")
    next_actions: List[str] = Field(default_factory=list, description="Actions suivantes possibles")
    quest_status: Optional[Dict[str, Any]] = Field(None, description="Statut de quête actuelle")
    user_progress: Dict[str, Any] = Field(default_factory=dict, description="Progrès utilisateur")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées de réponse")


class SessionStartRequest(BaseModel):
    initial_context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SessionStartResponse(BaseModel):
    session_id: str = Field(..., description="ID de la nouvelle session")
    welcome_message: str = Field(..., description="Message de bienvenue")
    user_context: Dict[str, Any] = Field(default_factory=dict)


class SessionInfoResponse(BaseModel):
    session_id: str
    user_id: Optional[int]
    conversation_state: str
    total_interactions: int
    session_duration: int
    current_quest: Optional[Dict[str, Any]]
    user_skills: Dict[str, float]
    last_activity: str


class StreamChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., description="ID de session")
    stream_options: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Dépendance pour obtenir le gestionnaire d'états
async def get_state_manager() -> StateManager:
    """Récupère le gestionnaire d'états depuis l'état de l'application"""
    from src.api.main import app
    if not hasattr(app.state, 'state_manager'):
        raise HTTPException(
            status_code=503,
            detail="Gestionnaire d'états non disponible"
        )
    return app.state.state_manager


@router.post("/sessions", response_model=SessionStartResponse)
async def start_chat_session(
    request: SessionStartRequest,
    current_user: Optional[UserBase] = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Démarre une nouvelle session de chat
    """
    try:
        logger.info(f"Démarrage d'une nouvelle session pour l'utilisateur {current_user.id if current_user else 'anonyme'}")
        
        # Démarrer la session
        user_id = current_user.id if current_user else None
        session_id = await state_manager.start_session(
            user_id=user_id,
            initial_context=request.initial_context
        )
        
        # Récupérer le message de bienvenue
        session_info = await state_manager.get_session_info(session_id)
        session_state = state_manager.active_sessions.get(session_id)
        
        welcome_message = session_state.get("last_response", "Bienvenue ! Comment puis-je vous aider ?")
        
        # Contexte utilisateur
        user_context = {
            "user_level": session_state.get("user_level", "beginner"),
            "user_skills": session_state.get("user_skills", {}),
            "learning_objectives": session_state.get("learning_objectives", [])
        }
        
        return SessionStartResponse(
            session_id=session_id,
            welcome_message=welcome_message,
            user_context=user_context
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du démarrage de session: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du démarrage de session")


@router.post("/sessions/{session_id}/messages", response_model=ChatResponse)
async def send_message(
    session_id: str,
    request: ChatRequest,
    current_user: Optional[UserBase] = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Envoie un message dans une session de chat
    """
    try:
        logger.info(f"Message reçu pour la session {session_id}: {request.message[:100]}...")
        
        # Vérifier que la session existe
        if session_id not in state_manager.active_sessions:
            raise NotFoundError(f"Session {session_id} non trouvée")
        
        # Vérifier que l'utilisateur a accès à cette session
        session_state = state_manager.active_sessions[session_id]
        if current_user and session_state.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Accès non autorisé à cette session")
        
        # Traiter le message
        result = await state_manager.process_user_input(
            session_id=session_id,
            user_input=request.message,
            input_type=request.message_type
        )
        
        # Vérifier s'il y a eu une erreur
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Construire la réponse
        response = ChatResponse(
            session_id=result["session_id"],
            response=result["response"],
            conversation_state=result["conversation_state"],
            suggestions=result.get("suggestions", []),
            next_actions=result.get("next_actions", []),
            quest_status=result.get("quest_status"),
            user_progress=result.get("user_progress", {}),
            metadata=result.get("metadata", {})
        )
        
        logger.info(f"Réponse générée pour la session {session_id}")
        return response
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors du traitement du message: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du traitement du message")


@router.post("/sessions/{session_id}/stream")
async def stream_chat(
    session_id: str,
    request: StreamChatRequest,
    current_user: Optional[UserBase] = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Chat en streaming pour les réponses longues
    """
    async def generate_stream():
        try:
            # Vérifier la session
            if session_id not in state_manager.active_sessions:
                yield f"data: {json.dumps({'error': 'Session non trouvée'})}\n\n"
                return
            
            # Simuler le streaming (à adapter selon votre implémentation)
            # Pour l'instant, on traite normalement et on stream la réponse
            result = await state_manager.process_user_input(
                session_id=session_id,
                user_input=request.message,
                input_type="text"
            )
            
            if "error" in result:
                yield f"data: {json.dumps({'error': result['error']})}\n\n"
                return
            
            # Découper la réponse en chunks pour le streaming
            response_text = result["response"]
            chunk_size = 50  # Caractères par chunk
            
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                data = {
                    "chunk": chunk,
                    "is_complete": i + chunk_size >= len(response_text),
                    "session_id": session_id
                }
                
                if data["is_complete"]:
                    # Ajouter les métadonnées finales
                    data.update({
                        "conversation_state": result["conversation_state"],
                        "suggestions": result.get("suggestions", []),
                        "quest_status": result.get("quest_status")
                    })
                
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.1)  # Délai pour simuler le streaming
            
        except Exception as e:
            logger.error(f"Erreur lors du streaming: {e}")
            yield f"data: {json.dumps({'error': 'Erreur lors du streaming'})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(
    session_id: str,
    current_user: Optional[UserBase] = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Récupère les informations d'une session
    """
    try:
        session_info = await state_manager.get_session_info(session_id)
        
        if not session_info:
            raise NotFoundError(f"Session {session_id} non trouvée")
        
        # Vérifier l'accès
        if current_user and session_info.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Accès non autorisé à cette session")
        
        return SessionInfoResponse(**session_info)
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de session: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de session")


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: Optional[UserBase] = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Récupère l'historique des messages d'une session
    """
    try:
        if session_id not in state_manager.active_sessions:
            raise NotFoundError(f"Session {session_id} non trouvée")
        
        session_state = state_manager.active_sessions[session_id]
        
        # Vérifier l'accès
        if current_user and session_state.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Accès non autorisé à cette session")
        
        # Récupérer les messages avec pagination
        messages = session_state.get("messages", [])
        total_messages = len(messages)
        
        # Appliquer la pagination
        start_idx = offset
        end_idx = min(offset + limit, total_messages)
        paginated_messages = messages[start_idx:end_idx]
        
        # Convertir en format API
        chat_messages = []
        for msg in paginated_messages:
            chat_messages.append(ChatMessage(
                role=msg.get("role", "unknown"),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp"),
                agent=msg.get("agent"),
                metadata=msg.get("metadata", {})
            ))
        
        return chat_messages
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des messages: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des messages")


@router.delete("/sessions/{session_id}")
async def end_session(
    session_id: str,
    current_user: Optional[UserBase] = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Termine une session de chat
    """
    try:
        if session_id not in state_manager.active_sessions:
            raise NotFoundError(f"Session {session_id} non trouvée")
        
        session_state = state_manager.active_sessions[session_id]
        
        # Vérifier l'accès
        if current_user and session_state.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Accès non autorisé à cette session")
        
        # Terminer la session
        result = await state_manager.end_session(session_id)
        
        logger.info(f"Session {session_id} terminée")
        return {
            "message": "Session terminée avec succès",
            "session_id": session_id,
            "final_stats": result.get("session_stats", {})
        }
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la fin de session: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la fin de session")


@router.get("/sessions")
async def list_user_sessions(
    current_user: UserBase = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Liste les sessions actives de l'utilisateur
    """
    try:
        all_sessions = await state_manager.get_active_sessions()
        
        # Filtrer par utilisateur
        user_sessions = [
            session for session in all_sessions
            if session.get("user_id") == current_user.id
        ]
        
        return {
            "sessions": user_sessions,
            "total": len(user_sessions)
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la liste des sessions: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des sessions")


@router.post("/sessions/{session_id}/context")
async def update_session_context(
    session_id: str,
    context_update: Dict[str, Any],
    current_user: Optional[UserBase] = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Met à jour le contexte d'une session
    """
    try:
        if session_id not in state_manager.active_sessions:
            raise NotFoundError(f"Session {session_id} non trouvée")
        
        session_state = state_manager.active_sessions[session_id]
        
        # Vérifier l'accès
        if current_user and session_state.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Accès non autorisé à cette session")
        
        # Mettre à jour le contexte
        current_context = session_state.get("temp_data", {})
        current_context.update(context_update)
        session_state["temp_data"] = current_context
        
        return {
            "message": "Contexte mis à jour avec succès",
            "session_id": session_id,
            "updated_context": current_context
        }
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du contexte: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour du contexte")
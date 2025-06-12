#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoctorPy - Interface Streamlit
Assistant Pédagogique IA avec Dark Mode
"""

import streamlit as st
import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Ajouter le répertoire src au path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Imports DoctorPy
try:
    from src.agents.state_manager_simple import SimpleStateManager, ConversationMode
    from src.core.exceptions import ValidationError, NotFoundError
except ImportError as e:
    st.error(f"Erreur d'import: {e}")
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="DoctorPy - Assistant Pédagogique IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/votre-repo/doctorpy',
        'Report a bug': 'https://github.com/votre-repo/doctorpy/issues',
        'About': "DoctorPy - Assistant d'apprentissage Python avec IA"
    }
)

# Chargement des styles CSS
def load_css():
    """Charge les styles CSS personnalisés"""
    css_files = [
        project_root / "ui" / "styles" / "main.css",
        project_root / "ui" / "styles" / "components.css"
    ]
    
    css_content = ""
    for css_file in css_files:
        if css_file.exists():
            with open(css_file, 'r', encoding='utf-8') as f:
                css_content += f.read() + "\n"
    
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Initialisation de l'état de session
def init_session_state():
    """Initialise les variables de session"""
    if 'state_manager' not in st.session_state:
        st.session_state.state_manager = SimpleStateManager()
    
    if 'user_session_id' not in st.session_state:
        st.session_state.user_session_id = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Chat"
    
    if 'user_stats' not in st.session_state:
        st.session_state.user_stats = {
            'total_xp': 1250,
            'level': 'Intermédiaire',
            'quests_completed': 8,
            'streak_days': 5,
            'messages_today': len(st.session_state.chat_history)
        }
    
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'

# Chargement des quêtes
@st.cache_data
def load_quests() -> List[Dict[str, Any]]:
    """Charge les quêtes depuis les fichiers JSON"""
    quests = []
    quest_dir = project_root / "data" / "quests"
    
    if quest_dir.exists():
        for quest_file in quest_dir.rglob("*.json"):
            try:
                with open(quest_file, 'r', encoding='utf-8') as f:
                    quest_data = json.load(f)
                    quests.append(quest_data)
            except Exception as e:
                st.error(f"Erreur lors du chargement de {quest_file}: {e}")
    
    return quests

# Interface de chat
def render_chat_interface():
    """Rendu de l'interface de chat"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<h1>🤖 DoctorPy Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p>Votre assistant IA pour apprendre Python de manière interactive</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Conteneur de chat
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Affichage de l'historique des messages
        for i, message in enumerate(st.session_state.chat_history):
            message_class = "user" if message["role"] == "user" else "assistant"
            
            st.markdown(f'''
            <div class="chat-message {message_class}">
                <div class="message-header">
                    {"👤 Vous" if message["role"] == "user" else "🤖 DoctorPy"} • {message["timestamp"]}
                </div>
                <div class="message-content">
                    {message["content"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Zone de saisie
    with st.container():
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_area(
                "",
                placeholder="Posez votre question sur Python, demandez une quête, ou discutez avec DoctorPy...",
                key="chat_input",
                height=68,
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Envoyer", key="send_button", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Traitement du message
    if send_button and user_input.strip():
        process_chat_message(user_input.strip())

async def process_chat_message_async(message: str):
    """Traite un message de chat de manière asynchrone"""
    try:
        # Créer une session si nécessaire
        if st.session_state.user_session_id is None:
            st.session_state.user_session_id = await st.session_state.state_manager.create_session(
                user_id=1,  # ID utilisateur par défaut
                mode=ConversationMode.FREE_CHAT
            )
        
        # Ajouter le message de l'utilisateur
        user_message = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.chat_history.append(user_message)
        
        # Traiter le message avec le state manager
        response = await st.session_state.state_manager.process_message(
            st.session_state.user_session_id,
            message
        )
        
        # Ajouter la réponse de l'assistant
        assistant_message = {
            "role": "assistant",
            "content": response.get("content", "Désolé, je n'ai pas pu traiter votre message."),
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.chat_history.append(assistant_message)
        
        # Mettre à jour les stats
        st.session_state.user_stats['messages_today'] = len([
            msg for msg in st.session_state.chat_history 
            if msg["role"] == "user"
        ])
        
    except Exception as e:
        st.error(f"Erreur lors du traitement du message: {e}")

def process_chat_message(message: str):
    """Wrapper synchrone pour le traitement du message"""
    with st.spinner("🤖 DoctorPy réfléchit..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(process_chat_message_async(message))
        finally:
            loop.close()
    
    # Rerun pour afficher les nouveaux messages
    st.rerun()

# Interface des quêtes
def render_quests_interface():
    """Rendu de l'interface des quêtes"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<h1>🎯 Quêtes d\'Apprentissage</h1>', unsafe_allow_html=True)
    st.markdown('<p>Progressez dans votre apprentissage Python avec des défis structurés</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    quests = load_quests()
    
    if not quests:
        st.warning("Aucune quête trouvée. Vérifiez le dossier data/quests/")
        return
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        difficulty_filter = st.selectbox(
            "Difficulté",
            ["Toutes", "beginner", "intermediate", "advanced"],
            key="difficulty_filter"
        )
    
    with col2:
        category_filter = st.selectbox(
            "Catégorie",
            ["Toutes"] + list(set(quest.get("category", "python_basics") for quest in quests)),
            key="category_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Trier par",
            ["Difficulté", "Temps estimé", "Titre"],
            key="sort_by"
        )
    
    # Filtrage des quêtes
    filtered_quests = quests
    
    if difficulty_filter != "Toutes":
        filtered_quests = [q for q in filtered_quests if q.get("difficulty") == difficulty_filter]
    
    if category_filter != "Toutes":
        filtered_quests = [q for q in filtered_quests if q.get("category") == category_filter]
    
    # Affichage des quêtes
    for quest in filtered_quests:
        render_quest_card(quest)

def render_quest_card(quest: Dict[str, Any]):
    """Rendu d'une carte de quête"""
    difficulty = quest.get("difficulty", "beginner")
    difficulty_colors = {
        "beginner": "success",
        "intermediate": "warning", 
        "advanced": "error"
    }
    
    with st.container():
        st.markdown(f'''
        <div class="quest-card">
            <div class="quest-title">{quest.get("title", "Quête sans titre")}</div>
            <div class="quest-difficulty {difficulty}">{difficulty.title()}</div>
            <p>{quest.get("description", "")}</p>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <span style="color: var(--text-secondary); font-size: 0.9rem;">
                    ⏱️ {quest.get("estimated_time", 0)} min • 
                    🎯 {len(quest.get("steps", []))} étapes • 
                    🏆 {quest.get("xp_reward", 0)} XP
                </span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button(f"Commencer", key=f"start_quest_{quest.get('id', '')}"):
                st.session_state.current_quest = quest
                st.session_state.current_page = "Quest Detail"
                st.rerun()

# Interface des statistiques
def render_stats_interface():
    """Rendu de l'interface des statistiques"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<h1>📊 Vos Statistiques</h1>', unsafe_allow_html=True)
    st.markdown('<p>Suivez votre progression et vos performances</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Grille de statistiques
    col1, col2, col3, col4 = st.columns(4)
    
    stats = st.session_state.user_stats
    
    with col1:
        st.markdown(f'''
        <div class="metric-container">
            <div class="stat-icon">🏆</div>
            <div class="metric-value">{stats["total_xp"]}</div>
            <div class="metric-label">Total XP</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-container">
            <div class="stat-icon">📈</div>
            <div class="metric-value">{stats["level"]}</div>
            <div class="metric-label">Niveau</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-container">
            <div class="stat-icon">✅</div>
            <div class="metric-value">{stats["quests_completed"]}</div>
            <div class="metric-label">Quêtes terminées</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-container">
            <div class="stat-icon">🔥</div>
            <div class="metric-value">{stats["streak_days"]}</div>
            <div class="metric-label">Jours consécutifs</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Graphiques de progression
    st.markdown("### 📈 Progression cette semaine")
    
    # Données de démonstration
    chart_data = {
        "Jour": ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
        "XP Gagné": [120, 80, 200, 150, 90, 180, 100]
    }
    
    st.bar_chart(chart_data, x="Jour", y="XP Gagné")

# Sidebar navigation
def render_sidebar():
    """Rendu de la sidebar de navigation"""
    with st.sidebar:
        # Logo et titre
        st.markdown('''
        <div class="nav-header">
            <div class="nav-logo">🤖 DoctorPy</div>
            <div class="nav-subtitle">Assistant Pédagogique IA</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Statut Ollama
        st.markdown("### 🔗 Statut du Système")
        
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                st.markdown('''
                <div style="display: flex; align-items: center; color: var(--success-color);">
                    <span class="status-indicator online"></span>
                    Ollama connecté
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div style="display: flex; align-items: center; color: var(--error-color);">
                    <span class="status-indicator offline"></span>
                    Ollama déconnecté
                </div>
                ''', unsafe_allow_html=True)
        except:
            st.markdown('''
            <div style="display: flex; align-items: center; color: var(--error-color);">
                <span class="status-indicator offline"></span>
                Ollama indisponible
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        
        pages = {
            "💬 Chat": "Chat",
            "🎯 Quêtes": "Quests", 
            "📊 Statistiques": "Stats",
            "⚙️ Paramètres": "Settings"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # Stats rapides
        st.markdown("### 📈 Stats Rapides")
        st.metric("Messages aujourd'hui", st.session_state.user_stats['messages_today'])
        st.metric("XP Total", st.session_state.user_stats['total_xp'])
        
        st.markdown("---")
        
        # Actions rapides
        st.markdown("### ⚡ Actions Rapides")
        
        if st.button("🔄 Nouvelle Session", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.user_session_id = None
            st.rerun()
        
        if st.button("💾 Sauvegarder", use_container_width=True):
            st.success("Session sauvegardée!")

# Interface des paramètres
def render_settings_interface():
    """Rendu de l'interface des paramètres"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<h1>⚙️ Paramètres</h1>', unsafe_allow_html=True)
    st.markdown('<p>Personnalisez votre expérience DoctorPy</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Paramètres généraux
    st.markdown("### 🎨 Apparence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox(
            "Thème",
            ["Dark Mode", "Light Mode"],
            index=0 if st.session_state.theme == 'dark' else 1
        )
        st.session_state.theme = 'dark' if theme == "Dark Mode" else 'light'
    
    with col2:
        language = st.selectbox(
            "Langue",
            ["Français", "English"],
            index=0
        )
    
    st.markdown("### 🤖 Assistant IA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model = st.selectbox(
            "Modèle IA",
            ["llama3.1:8b", "codellama:7b"],
            index=0
        )
    
    with col2:
        response_style = st.selectbox(
            "Style de réponse",
            ["Détaillé", "Concis", "Interactif"],
            index=0
        )
    
    st.markdown("### 📚 Apprentissage")
    
    difficulty_preference = st.selectbox(
        "Difficulté préférée",
        ["Débutant", "Intermédiaire", "Avancé"],
        index=1
    )
    
    show_hints = st.checkbox("Afficher les indices automatiquement", value=True)
    auto_progression = st.checkbox("Progression automatique des quêtes", value=True)
    
    # Bouton de sauvegarde
    if st.button("💾 Sauvegarder les paramètres", type="primary"):
        st.success("Paramètres sauvegardés avec succès!")

# Application principale
def main():
    """Fonction principale de l'application"""
    # Chargement des styles
    load_css()
    
    # Initialisation de l'état
    init_session_state()
    
    # Sidebar
    render_sidebar()
    
    # Contenu principal basé sur la page sélectionnée
    page = st.session_state.current_page
    
    if page == "Chat":
        render_chat_interface()
    elif page == "Quests":
        render_quests_interface()
    elif page == "Stats":
        render_stats_interface()
    elif page == "Settings":
        render_settings_interface()
    else:
        render_chat_interface()  # Page par défaut

if __name__ == "__main__":
    main()
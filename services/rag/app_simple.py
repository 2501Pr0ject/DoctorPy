"""
Application FastAPI simplifiée pour le service RAG
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

def create_app() -> FastAPI:
    """Créer l'application FastAPI pour le service RAG"""
    
    app = FastAPI(
        title="DoctorPy RAG Service",
        description="Service de récupération et génération assistée",
        version="1.0.0",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Routes simples
    @app.get("/")
    async def root():
        return {
            "service": "DoctorPy RAG Service",
            "version": "1.0.0",
            "status": "running",
            "mode": "demo"
        }
    
    @app.get("/health")
    async def health_check():
        """Endpoint de vérification de santé"""
        return {
            "status": "healthy",
            "service": "rag",
            "timestamp": "now"
        }
    
    @app.post("/api/v1/rag/query")
    async def query_rag(query_data: dict):
        """Requête RAG intelligente"""
        query = query_data.get("query", "").lower()
        query_type = query_data.get("query_type", "general")
        
        # Réponses intelligentes basées sur le contenu
        if any(word in query for word in ["variable", "variables"]):
            response = """En Python, une variable est un nom qui fait référence à une valeur stockée en mémoire.

Syntaxe de base:
```python
nom_variable = valeur
```

Exemples:
```python
age = 25
nom = "Alice"
prix = 19.99
est_etudiant = True
```

Les variables en Python sont dynamiquement typées - pas besoin de déclarer le type."""
            sources = ["python_basics.md", "variables_guide.py"]
            
        elif any(word in query for word in ["boucle", "loop", "for", "while"]):
            response = """Les boucles en Python permettent de répéter du code.

Boucle for (itération sur une séquence):
```python
for i in range(5):
    print(i)

for element in [1, 2, 3]:
    print(element)
```

Boucle while (tant que condition vraie):
```python
compteur = 0
while compteur < 5:
    print(compteur)
    compteur += 1
```"""
            sources = ["python_loops.md", "control_structures.py"]
            
        elif any(word in query for word in ["fonction", "function", "def"]):
            response = """Les fonctions en Python se définissent avec le mot-clé 'def'.

Syntaxe de base:
```python
def nom_fonction(parametre1, parametre2):
    # Corps de la fonction
    return resultat
```

Exemple:
```python
def calculer_aire(longueur, largeur):
    aire = longueur * largeur
    return aire

# Utilisation
resultat = calculer_aire(5, 3)
print(resultat)  # Affiche: 15
```"""
            sources = ["python_functions.md", "function_examples.py"]
            
        else:
            response = f"""Voici une réponse adaptée à votre question sur: {query}

Il s'agit d'un service RAG (Retrieval-Augmented Generation) qui combine:
- Recherche dans une base de connaissances Python
- Génération de réponses contextuelles
- Exemples de code pratiques

Pour une aide plus spécifique, essayez des mots-clés comme 'variables', 'boucles', 'fonctions'."""
            sources = ["general_python_guide.md", "doctorpy_docs.md"]
        
        return {
            "query": query_data.get("query", ""),
            "query_type": query_type,
            "response": response,
            "sources": sources,
            "confidence": 0.92,
            "timestamp": "2025-12-06T20:00:00Z"
        }
    
    @app.get("/api/v1/rag/templates")
    async def get_templates():
        """Templates disponibles (mode démo)"""
        return {
            "templates": {
                "code_help": "Template pour aide code Python",
                "debugging": "Template pour debugging",
                "general": "Template général"
            }
        }
    
    return app

# Point d'entrée pour développement
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8002)
# Documentation Python - Exemples pour Tests

## Variables en Python

Les variables en Python sont des conteneurs qui permettent de stocker des données. Contrairement à d'autres langages de programmation, Python n'exige pas de déclarer le type d'une variable à l'avance.

### Création de variables

```python
# Variables de différents types
nom = "Alice"           # Chaîne de caractères
age = 25               # Nombre entier
taille = 1.75          # Nombre décimal
est_etudiant = True    # Booléen
```

### Règles de nommage

- Les noms de variables doivent commencer par une lettre ou un underscore
- Ils peuvent contenir des lettres, des chiffres et des underscores
- Python est sensible à la casse (age et Age sont différents)
- Évitez les mots-clés Python (if, for, class, etc.)

### Bonnes pratiques

- Utilisez des noms descriptifs : `nom_utilisateur` plutôt que `n`
- Utilisez la convention snake_case : `mon_variable`
- Évitez les noms trop courts ou trop longs

## Fonctions en Python

Les fonctions permettent d'organiser le code en blocs réutilisables.

### Définition de base

```python
def ma_fonction():
    """Docstring décrivant la fonction."""
    print("Hello, World!")
    
# Appel de la fonction
ma_fonction()
```

### Fonctions avec paramètres

```python
def saluer(nom, age=None):
    """Fonction qui salue une personne.
    
    Args:
        nom (str): Le nom de la personne
        age (int, optional): L'âge de la personne
    """
    if age:
        print(f"Bonjour {nom}, vous avez {age} ans!")
    else:
        print(f"Bonjour {nom}!")

# Exemples d'utilisation
saluer("Alice")
saluer("Bob", 30)
```

### Valeurs de retour

```python
def additionner(a, b):
    """Additionne deux nombres."""
    return a + b

def calculer_stats(nombres):
    """Calcule la moyenne et la somme d'une liste."""
    somme = sum(nombres)
    moyenne = somme / len(nombres)
    return somme, moyenne  # Retour multiple

# Utilisation
resultat = additionner(5, 3)
total, moy = calculer_stats([1, 2, 3, 4, 5])
```

## Structures de données

### Listes

Les listes sont des collections ordonnées et modifiables.

```python
# Création
fruits = ["pomme", "banane", "orange"]
nombres = [1, 2, 3, 4, 5]
mixte = ["texte", 42, True, 3.14]

# Accès aux éléments
premier_fruit = fruits[0]        # "pomme"
dernier_fruit = fruits[-1]       # "orange"

# Modification
fruits.append("kiwi")            # Ajouter à la fin
fruits.insert(1, "mangue")       # Insérer à l'index 1
fruits.remove("banane")          # Supprimer un élément
```

### Dictionnaires

Les dictionnaires stockent des paires clé-valeur.

```python
# Création
personne = {
    "nom": "Alice",
    "age": 25,
    "ville": "Paris"
}

# Accès et modification
nom = personne["nom"]
personne["profession"] = "Développeuse"
personne.update({"age": 26, "email": "alice@example.com"})

# Méthodes utiles
cles = personne.keys()
valeurs = personne.values()
items = personne.items()
```

## Boucles et conditions

### Conditions

```python
age = 18

if age >= 18:
    print("Vous êtes majeur")
elif age >= 16:
    print("Vous pouvez conduire")
else:
    print("Vous êtes mineur")

# Opérateurs de comparaison
# == (égal), != (différent), < (inférieur), > (supérieur)
# <= (inférieur ou égal), >= (supérieur ou égal)
```

### Boucle for

```python
# Parcourir une liste
fruits = ["pomme", "banane", "orange"]
for fruit in fruits:
    print(f"J'aime les {fruit}s")

# Parcourir avec index
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

# Range
for i in range(5):          # 0 à 4
    print(i)

for i in range(2, 8):       # 2 à 7
    print(i)

for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
    print(i)
```

### Boucle while

```python
compteur = 0
while compteur < 5:
    print(f"Compteur: {compteur}")
    compteur += 1

# Boucle infinie avec break
while True:
    reponse = input("Continuer? (o/n): ")
    if reponse.lower() == 'n':
        break
    print("On continue...")
```

## Gestion des erreurs

### Try/Except

```python
try:
    nombre = int(input("Entrez un nombre: "))
    resultat = 10 / nombre
    print(f"Résultat: {resultat}")
except ValueError:
    print("Ce n'est pas un nombre valide")
except ZeroDivisionError:
    print("Division par zéro impossible")
except Exception as e:
    print(f"Erreur inattendue: {e}")
finally:
    print("Nettoyage final")
```

### Erreurs courantes

- **SyntaxError** : Erreur de syntaxe dans le code
- **NameError** : Variable non définie
- **TypeError** : Type de données incorrect
- **ValueError** : Valeur incorrecte pour le type
- **IndexError** : Index hors limites pour une liste
- **KeyError** : Clé inexistante dans un dictionnaire

## Classes et Programmation Orientée Objet

### Définition de classe

```python
class Personne:
    """Classe représentant une personne."""
    
    # Variable de classe (partagée par toutes les instances)
    espece = "Homo sapiens"
    
    def __init__(self, nom, age):
        """Constructeur de la classe."""
        self.nom = nom          # Attribut d'instance
        self.age = age          # Attribut d'instance
    
    def se_presenter(self):
        """Méthode d'instance."""
        return f"Je suis {self.nom} et j'ai {self.age} ans"
    
    def avoir_anniversaire(self):
        """Augmente l'âge de 1."""
        self.age += 1
    
    @classmethod
    def creer_bebe(cls, nom):
        """Méthode de classe pour créer un bébé."""
        return cls(nom, 0)
    
    @staticmethod
    def est_majeur(age):
        """Méthode statique."""
        return age >= 18

# Utilisation
alice = Personne("Alice", 25)
print(alice.se_presenter())
alice.avoir_anniversaire()

bebe = Personne.creer_bebe("Tom")
print(Personne.est_majeur(17))  # False
```

### Héritage

```python
class Etudiant(Personne):
    """Classe héritant de Personne."""
    
    def __init__(self, nom, age, ecole):
        super().__init__(nom, age)  # Appel du constructeur parent
        self.ecole = ecole
    
    def se_presenter(self):
        """Surcharge de la méthode parent."""
        base = super().se_presenter()
        return f"{base} et j'étudie à {self.ecole}"
    
    def etudier(self, matiere):
        """Nouvelle méthode spécifique."""
        return f"{self.nom} étudie {matiere}"

# Utilisation
etudiant = Etudiant("Bob", 20, "Université Paris")
print(etudiant.se_presenter())
print(etudiant.etudier("Python"))
```

## Modules et Packages

### Import de modules

```python
# Import complet
import math
print(math.pi)
print(math.sqrt(16))

# Import spécifique
from math import pi, sqrt
print(pi)
print(sqrt(16))

# Import avec alias
import numpy as np
import matplotlib.pyplot as plt

# Import relatif (dans un package)
from .utils import helper_function
from ..config import settings
```

### Création d'un module

```python
# fichier: mon_module.py
"""Module d'exemple avec des fonctions utiles."""

PI = 3.14159

def aire_cercle(rayon):
    """Calcule l'aire d'un cercle."""
    return PI * rayon ** 2

def perimetre_cercle(rayon):
    """Calcule le périmètre d'un cercle."""
    return 2 * PI * rayon

if __name__ == "__main__":
    # Code exécuté seulement si le module est lancé directement
    print("Test du module:")
    print(f"Aire d'un cercle de rayon 5: {aire_cercle(5)}")
```

## Conseils de débogage

### Techniques de base

1. **Utilisez print()** pour afficher les valeurs des variables
2. **Lisez attentivement les messages d'erreur** - ils indiquent souvent exactement le problème
3. **Vérifiez l'indentation** - Python est strict sur l'indentation
4. **Testez petit à petit** - ajoutez du code progressivement

### Erreurs communes

```python
# Erreur d'indentation
def ma_fonction():
print("Hello")  # IndentationError

# Oubli de deux-points
if x > 5        # SyntaxError: missing ':'
    print("Grand")

# Confusion entre = et ==
if x = 5:       # SyntaxError: invalid syntax
    print("Test")

# Index hors limites
liste = [1, 2, 3]
print(liste[5])  # IndexError

# Division par zéro
print(10 / 0)    # ZeroDivisionError
```

### Outils de débogage

- **Debugger intégré** : utilisez `pdb` ou l'IDE
- **Assertions** : `assert condition, "message d'erreur"`
- **Logging** : plus professionnel que print() pour les gros projets
- **Tests unitaires** : vérifiez que votre code fonctionne comme attendu
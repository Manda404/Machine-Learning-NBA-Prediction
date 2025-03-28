
# Prédiction de Carrière des Joueurs NBA

## Description du Projet
Ce projet utilise **FastAPI**, **Streamlit** et **PostgreSQL** pour créer une application de prédiction de carrière des joueurs NBA. L'application permet aux utilisateurs de prédire si un joueur débutant en NBA durera plus de 5 ans en fonction de ses statistiques initiales.

L'API FastAPI reçoit les données des joueurs et renvoie les prédictions, tandis que l'application Streamlit sert d'interface utilisateur pour saisir et afficher les données. **PostgreSQL** est utilisé pour sauvegarder les requêtes reçues et les résultats des prédictions, permettant un suivi et des analyses approfondies des interactions utilisateur.

## Structure du Projet
```
nba-prediction-project/
│
├── api/                         # Code de l'API FastAPI
│   ├── __init__.py              # Fichier d'initialisation du module API
│   ├── api.py                   # Script principal de l'application FastAPI
│
├── app/                         # Interface utilisateur Streamlit
│   ├── __init__.py              # Fichier d'initialisation du module de l'application
│   ├── app.py                   # Code de l'application Streamlit
│   └── picture.png              # Image utilisée dans l'interface utilisateur
│
├── data/                        # Dossier des fichiers de données
│   ├── nba_logreg.csv           # Fichier de données des statistiques des joueurs
│   ├── test_data_clean.csv      # Données de test nettoyées
│   └── train_data_clean.csv     # Données d'entraînement nettoyées
│
├── dev/                         # Dossier de développement pour les essais et analyses
│   ├── catboost_info/           # Dossier contenant les fichiers de logs CatBoost
│   ├── images/                  # Images et graphiques utilisés pour l'analyse
│   └── nba-prediction-ensamble-classification.ipynb # Notebook Jupyter pour l'analyse et le développement du modèle
│
├── metrics/                     # Dossier contenant les fichiers de métriques
│   └── metrics.csv              # Fichier CSV des résultats des métriques de modèle
│
├── model/                       # Dossier des modèles pré-entraînés
│   ├── Extra_Trees_optimized_model.pkl         # Modèle optimisé Extra Trees
│   ├── Gradient_Boosting_optimized_model.pkl   # Modèle optimisé Gradient Boosting
│   ├── Logistic_Regression_optimized_model.pkl # Modèle optimisé de régression logistique
│   ├── Random_Forest_optimized_model.pkl       # Modèle optimisé Random Forest
│   ├── Ridge_Classifier_optimized_model.pkl    # Modèle optimisé Ridge Classifier
│   └── model.pkl                               # Modèle par défaut utilisé par l'API
│
├── tables_postgresql/           # Scripts pour la gestion des bases de données PostgreSQL
│   └── create_request_log_table.sh # Script Bash pour créer la table de journalisation des requêtes
│
├── train/                       # Dossier pour le code d'entraînement du modèle
│   └── train.py                 # Script d'entraînement du modèle
│
└── README.md                    # Documentation du projet
```

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/votre-utilisateur/nba-prediction-project.git
   cd nba-prediction-project
   ```

2. **Créer et activer un environnement virtuel** :
   ```bash
   python -m venv env
   source env/bin/activate  # Pour Linux/Mac
   .\env\Scripts\activate   # Pour Windows
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### Lancer l'API FastAPI
Dans un terminal, exécutez la commande suivante :
```bash
uvicorn api.api:app --reload
```
L'API sera accessible à `http://localhost:8000`.

### Lancer l'application Streamlit
Ouvrez un autre terminal, activez l'environnement virtuel et exécutez :
```bash
streamlit run app/app.py
```
L'application s'ouvrira dans votre navigateur par défaut à l'adresse `http://localhost:8501`.

## Fonctionnement de l'application
- **Saisie manuelle** : Permet à l'utilisateur de saisir manuellement les statistiques d'un joueur et d'obtenir une prédiction.
- **Chargement de fichier CSV** : Permet de charger un fichier CSV contenant les statistiques de plusieurs joueurs et d'afficher les prédictions pour chaque joueur.

### Exemple de Requête à l'API
Vous pouvez envoyer une requête à l'API FastAPI avec les données suivantes :
```json
{
    "GP": 82,
    "MIN": 2400,
    "PTS": 15.2,
    "FGM": 500,
    "FGA": 1000,
    "FG_perc": 50.0,
    "ThreeP_Made": 100,
    "ThreePA": 250,
    "ThreeP_perc": 40.0,
    "FTM": 200,
    "FTA": 250,
    "FT_perc": 80.0,
    "OREB": 100,
    "DREB": 300,
    "REB": 400,
    "AST": 150,
    "STL": 50,
    "BLK": 30,
    "TOV": 100
}
```

### Accéder à la Documentation de l'API
Accédez à `http://localhost:8000/docs` pour visualiser la documentation interactive de l'API générée par FastAPI.

## Technologies Utilisées
- **FastAPI** pour la création de l'API
- **Streamlit** pour l'interface utilisateur
- **scikit-learn** pour le modèle de prédiction
- **pandas**, **numpy** pour la manipulation des données
- **matplotlib**, **seaborn** pour la visualisation des données

## Contribution
Les contributions sont les bienvenues. Veuillez créer une *issue* ou soumettre une *pull request* pour toute amélioration ou suggestion.

## Auteurs
- **Rostand Surel**

## Licence
Ce projet est sous licence MIT. Consultez le fichier `LICENSE` pour plus d'informations.
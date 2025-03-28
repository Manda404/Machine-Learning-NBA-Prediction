# Auteur : Rostand Surel
# Project Name : NBA Prediction API
# Description : Cette application FastAPI permet de faire des prédictions sur les performances des joueurs de NBA
#               en utilisant un modèle de machine learning préalablement entraîné. Elle enregistre également chaque 
#               requête et la réponse dans une base de données PostgreSQL pour un suivi ultérieur.

import pickle
import numpy as np
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
from datetime import datetime

# Charger le modèle
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model_path = "/Users/surelmanda/Downloads/AirGUARD/nba-prediction-project/model/model.pkl"  # Changez le chemin si nécessaire
model = load_model(model_path)

# Initialiser l'application FastAPI
app = FastAPI()

# Paramètres de connexion à la base de données
DB_HOST = "localhost"
DB_NAME = "NBAPredictionLogs"
DB_USER = "surelmanda"
DB_PASSWORD = "surelmanda"

# Fonction pour vérifier la connexion à la base de données
def check_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.close()
        print("Connexion à la base de données réussie.")
    except Exception as e:
        print(f"Erreur : Impossible de se connecter à la base de données '{DB_NAME}'. {e}")
        raise

# Vérifier la connexion à la base de données au démarrage de l'API
@app.on_event("startup")
def startup_event():
    try:
        check_db_connection()
    except Exception as e:
        raise RuntimeError(f"Erreur au démarrage de l'API : {e}")


class PlayerData(BaseModel):
    GP: int = Field(..., ge=0, description="Games Played, doit être >= 0")
    MIN: float = Field(..., ge=0.0, description="Minutes Played, doit être >= 0", format="%.3f")
    PTS: float = Field(..., ge=0.0, description="Points Per Game, doit être >= 0", format="%.3f")
    FGM: float = Field(..., ge=0.0, description="Field Goals Made, doit être >= 0", format="%.3f")
    FGA: float = Field(..., ge=0.0, description="Field Goal Attempts, doit être >= 0", format="%.3f")
    FG_perc: float = Field(..., ge=0.0, le=100.0, description="Field Goal Percent, doit être entre 0 et 100", format="%.3f")
    ThreeP_Made: float = Field(..., ge=0.0, description="3 Points Made, doit être >= 0", format="%.3f")
    ThreePA: float = Field(..., ge=0.0, description="3 Point Attempts, doit être >= 0", format="%.3f")
    ThreeP_perc: float = Field(..., ge=0.0, le=100.0, description="3 Point Percent, doit être entre 0 et 100", format="%.3f")
    FTM: float = Field(..., ge=0.0, description="Free Throws Made, doit être >= 0", format="%.3f")
    FTA: float = Field(..., ge=0.0, description="Free Throw Attempts, doit être >= 0", format="%.3f")
    FT_perc: float = Field(..., ge=0.0, le=100.0, description="Free Throw Percent, doit être entre 0 et 100", format="%.3f")
    OREB: float = Field(..., ge=0.0, description="Offensive Rebounds, doit être >= 0", format="%.3f")
    DREB: float = Field(..., ge=0.0, description="Defensive Rebounds, doit être >= 0", format="%.3f")
    REB: float = Field(..., ge=0.0, description="Total Rebounds, doit être >= 0", format="%.3f")
    AST: float = Field(..., ge=0.0, description="Assists, doit être >= 0", format="%.3f")
    STL: float = Field(..., ge=0.0, description="Steals, doit être >= 0", format="%.3f")
    BLK: float = Field(..., ge=0.0, description="Blocks, doit être >= 0", format="%.3f")
    TOV: float = Field(..., ge=0.0, description="Turnovers, doit être >= 0", format="%.3f")

# Nom du concepteur de l'application
nom_concepteur = "Rostand Surel"  #

# Endpoint racine pour vérifier que l'API fonctionne
@app.get("/")
def root():
    return {
        "message": f"Cette API a été conçue par {nom_concepteur}. Elle est actuellement en cours d'exécution et fonctionne parfaitement."
    }


def process_prediction(probabilities, prediction, threshold=0.5):
    """
    Traite les résultats de la prédiction en arrondissant les probabilités et en retournant
    le texte de la classe prédite.

    Parameters:
    - probabilities (list or np.array): Les probabilités des classes.
    - prediction (int): L'indice de la classe prédite.
    - threshold (float, optional): Le seuil de décision (par défaut 0.5).

    Returns:
    - str: Le texte de la classe prédite.
    """
    # Arrondir les probabilités à 3 chiffres après la virgule
    round_probabilities = np.round(probabilities, 3)

    # Définir les classes
    classes = np.array(["Ne durera pas 5 ans", "Durera 5 ans ou plus"])

    # Texte de la classe prédite
    prediction_text = classes[prediction]

    return prediction_text

def save_prediction_to_db(player_data, probabilities, prediction):
    """
    Enregistre la requête et la réponse de la prédiction dans la table `request_log` de la base de données.

    Parameters:
    - player_data (PlayerData): Les données du joueur.
    - probabilities (list): Liste contenant les probabilités des deux classes.
    - prediction (int): Indice de la classe prédite.
    """
    # Utiliser la fonction process_prediction pour obtenir le texte de la prédiction
    prediction_text = process_prediction(probabilities, prediction)

    try:
        # Connexion à la base de données
        conn = psycopg2.connect(
            host="localhost",
            database="NBAPredictionLogs",
            user="surelmanda",
            password="surelmanda"
        )
        cursor = conn.cursor()

        # Insertion des données dans la table
        cursor.execute(
            """
            INSERT INTO request_log (
                gp, min_played, pts, fgm, fga, fg_perc, threep_made, threepa, threep_perc, 
                ftm, fta, ft_perc, oreb, dreb, reb, ast, stl, blk, tov, 
                probability_class_0, probability_class_1, prediction, prediction_text, timestamp
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                player_data.GP, player_data.MIN, player_data.PTS, player_data.FGM, player_data.FGA,
                player_data.FG_perc, player_data.ThreeP_Made, player_data.ThreePA, player_data.ThreeP_perc,
                player_data.FTM, player_data.FTA, player_data.FT_perc, player_data.OREB, player_data.DREB,
                player_data.REB, player_data.AST, player_data.STL, player_data.BLK, player_data.TOV,
                probabilities[0], probabilities[1], prediction, prediction_text, datetime.now()
            )
        )
        conn.commit()
        cursor.close()
        conn.close()
        print("Données sauvegardées dans la table 'request_log'.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des données : {e}")


# Endpoint pour faire des prédictions avec vérification des données reçues
@app.post("/predict/")
async def predict(player_data: PlayerData, request: Request):
    try:
        # Journaliser les données reçues
        body = await request.json()
        print("Données reçues :", body)

        # Convertir les données en DataFrame
        input_data = pd.DataFrame([player_data.dict()])
        print("Données transformées en DataFrame :", input_data)

        # Faire des prédictions
        probabilities = model.predict_proba(input_data)[0]
        prediction = int(np.argmax(probabilities))

        # Appel de la fonction pour sauvegarder la prédiction dans la base de données
        save_prediction_to_db(player_data, probabilities, prediction)

        # Retourner les résultats
        return {
            "probabilities": probabilities.tolist(),
            "prediction": prediction
        }

    except AttributeError as e:
        # Gestion des erreurs liées au modèle
        print("Erreur liée au modèle :", e)
        raise HTTPException(status_code=500, detail="Erreur du modèle : méthode non supportée")
    
    except psycopg2.DatabaseError as e:
        # Gestion des erreurs de la base de données
        print("Erreur de la base de données :", e)
        raise HTTPException(status_code=500, detail="Erreur de connexion ou d'écriture dans la base de données")

    except Exception as e:
        # Gestion d'autres erreurs inattendues
        print("Erreur inconnue :", e)
        raise HTTPException(status_code=500, detail="Une erreur inconnue est survenue")

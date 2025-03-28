# Fichier: app.py
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Chemin vers le modèle et le fichier de données
model_path = '/Users/surelmanda/Downloads/AirGUARD/nba-prediction-project/model/model.pkl'
file_path = '/Users/surelmanda/Downloads/AirGUARD/nba-prediction-project/data/nba_logreg.csv'

def load_dataset(file_path):
    """
    Charge le dataset à partir d'un fichier CSV et renomme certaines colonnes.
    :param file_path: Chemin vers le fichier CSV.
    :return: DataFrame Pandas contenant les données chargées.
    """
    try:
        # Charger le dataset
        data = pd.read_csv(file_path)
        print("Dataset chargé avec succès.")
        print("Aperçu des premières lignes du dataset :")
        print(data.head())

        # Vérifier les colonnes existantes
        print("Colonnes originales :", data.columns.tolist())

        # Dictionnaire de renommage des colonnes
        rename_columns = {
            'FG%': 'FG_perc',
            '3P Made': 'ThreeP_Made',
            '3PA': 'ThreePA',
            '3P%': 'ThreeP_perc',
            'FT%': 'FT_perc'
        }

        # Renommer les colonnes
        data.rename(columns=rename_columns, inplace=True)
        print("Renommage des colonnes effectué.")

        # Vérifier les colonnes renommées
        print("Colonnes renommées :", data.columns.tolist())

        return data

    except FileNotFoundError:
        print(f"Erreur : le fichier spécifié à '{file_path}' est introuvable.")
        return None
    except pd.errors.EmptyDataError:
        print("Erreur : le fichier est vide.")
        return None
    except pd.errors.ParserError:
        print("Erreur : problème de parsing lors de la lecture du fichier.")
        return None
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")
        return None

def load_model():
    """
    Charge un modèle pré-entraîné à partir d'un fichier pickle.
    :return: Modèle chargé.
    """
    try:
        with open(model_path, 'rb') as f:
            print("Modèle chargé avec succès.")
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Erreur : le fichier du modèle '{model_path}' est introuvable.")
        return None
    except Exception as e:
        print(f"Une erreur inattendue lors du chargement du modèle : {e}")
        return None

def impute_missing_values_with_mean(data, column_name):
    """
    Remplit les valeurs manquantes dans une colonne spécifique d'un DataFrame
    en utilisant la moyenne de cette colonne.
    :param data: DataFrame Pandas contenant les données.
    :param column_name: Nom de la colonne sur laquelle appliquer l'imputation.
    :return: DataFrame modifié avec les valeurs manquantes imputées.
    """
    if column_name in data.columns:
        data[column_name] = data[column_name].fillna(data[column_name].mean())
        print(f"Imputation effectuée sur la colonne '{column_name}'.")
    else:
        print(f"Erreur : la colonne '{column_name}' n'existe pas dans le DataFrame.")
    
    return data

def train_model():
    """
    Charge le dataset, entraîne un modèle RandomForestClassifier, et sauvegarde le modèle.
    """
    # Charger le dataset
    data = load_dataset(file_path)

    if data is not None:
        print("Chargement et renommage du dataset réussis.")

        # Imputation des valeurs manquantes
        data = impute_missing_values_with_mean(data, 'ThreeP_perc')
        print("Imputation des valeurs manquantes terminée.")

        # Séparer les features et la cible
        print("Séparation des features et de la cible.")
        X = data.drop(['Name', 'TARGET_5Yrs'], axis=1)  # Enlever la colonne 'Name' et la cible
        y = data['TARGET_5Yrs']
        print("Séparation effectuée avec succès.")

        # Diviser les données en ensembles d'entraînement et de test
        print("Division des données en ensembles d'entraînement et de test.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        print("Division effectuée.")

        # Entraîner le modèle
        print("Entraînement du modèle RandomForestClassifier en cours...")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("Entraînement terminé.")

        # Évaluer le modèle
        print("Évaluation du modèle.")
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"F1 Score du modèle: {f1:.2f}")

        # Sauvegarder le modèle
        print("Sauvegarde du modèle en cours...")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print("Modèle sauvegardé avec succès.")
    else:
        print("Le chargement du dataset a échoué.")

if __name__ == "__main__":
    train_model()
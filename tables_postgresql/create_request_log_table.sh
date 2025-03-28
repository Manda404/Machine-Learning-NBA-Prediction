#!/bin/bash

# Auteur : Rostand Surel
# Project Name : NBA
# Description : Script pour vérifier la connexion, créer la base de données et créer la table 'request_log' dans la base de données PostgreSQL 'NBAPredictionLogs'

# Paramètres de connexion à la base de données
DB_HOST="localhost"
DB_NAME="NBAPredictionLogs"
DB_USER="surelmanda"
DB_PASSWORD="surelmanda"

# Vérifier la connexion à PostgreSQL
echo "Vérification de la connexion à PostgreSQL..."
if ! PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "\conninfo" 2>/dev/null; then
    echo "Erreur : Impossible de se connecter à PostgreSQL. Veuillez vérifier vos informations de connexion."
    exit 1
fi
echo "Connexion réussie."

# Vérifier si la base de données existe
DB_EXIST=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -lqt | cut -d \| -f 1 | grep -w $DB_NAME | wc -l)

if [ "$DB_EXIST" -eq 0 ]; then
    echo "La base de données '$DB_NAME' n'existe pas. Création en cours..."
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -c "CREATE DATABASE $DB_NAME;"
    
    # Vérifier si la création a réussi
    DB_CREATED=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -lqt | cut -d \| -f 1 | grep -w $DB_NAME | wc -l)
    if [ "$DB_CREATED" -eq 1 ]; then
        echo "La base de données '$DB_NAME' a été créée avec succès."
    else
        echo "Erreur : La base de données '$DB_NAME' n'a pas pu être créée. Veuillez vérifier les logs pour plus d'informations."
        exit 1
    fi
else
    echo "La base de données '$DB_NAME' existe déjà."
fi

# Supprimer la table si elle existe déjà
echo "Vérification de l'existence de la table 'request_log'..."
TABLE_EXIST=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -d $DB_NAME -U $DB_USER -c "\dt" | grep -w "request_log" | wc -l)

if [ "$TABLE_EXIST" -eq 1 ]; then
    echo "La table 'request_log' existe déjà. Suppression en cours..."
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -d $DB_NAME -U $DB_USER -c "DROP TABLE request_log;"
    echo "Table 'request_log' supprimée."
fi

# Script SQL pour créer la table avec des descriptions
echo "Création de la table 'request_log'..."
SQL_QUERY="
CREATE TABLE request_log (
    id SERIAL PRIMARY KEY,  -- Identifiant unique de la requête
    gp INTEGER NOT NULL,  -- Nombre de matchs joués
    min_played FLOAT NOT NULL,  -- Minutes jouées
    pts FLOAT NOT NULL,  -- Points marqués par match
    fgm FLOAT NOT NULL,  -- Tirs réussis
    fga FLOAT NOT NULL,  -- Tentatives de tir
    fg_perc FLOAT NOT NULL,  -- Pourcentage de réussite des tirs
    threep_made FLOAT NOT NULL,  -- 3 points réussis
    threepa FLOAT NOT NULL,  -- Tentatives de 3 points
    threep_perc FLOAT NOT NULL,  -- Pourcentage de réussite des 3 points
    ftm FLOAT NOT NULL,  -- Lancers francs réussis
    fta FLOAT NOT NULL,  -- Tentatives de lancer franc
    ft_perc FLOAT NOT NULL,  -- Pourcentage de réussite des lancers francs
    oreb FLOAT NOT NULL,  -- Rebonds offensifs
    dreb FLOAT NOT NULL,  -- Rebonds défensifs
    reb FLOAT NOT NULL,  -- Total des rebonds
    ast FLOAT NOT NULL,  -- Passes décisives
    stl FLOAT NOT NULL,  -- Interceptions
    blk FLOAT NOT NULL,  -- Contres
    tov FLOAT NOT NULL,  -- Balles perdues
    probability_class_0 FLOAT NOT NULL,  -- Probabilité que l'événement soit 'Ne durera pas 5 ans'
    probability_class_1 FLOAT NOT NULL,  -- Probabilité que l'événement soit 'Durera 5 ans ou plus'
    prediction INTEGER NOT NULL,  -- Classe prédite (0 ou 1)
    prediction_text TEXT NOT NULL,  -- Texte décrivant la classe prédite
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Date et heure de la requête
);
COMMENT ON COLUMN request_log.gp IS 'Nombre de matchs joués';
COMMENT ON COLUMN request_log.min_played IS 'Minutes jouées';
COMMENT ON COLUMN request_log.pts IS 'Points marqués par match';
COMMENT ON COLUMN request_log.fgm IS 'Tirs réussis';
COMMENT ON COLUMN request_log.fga IS 'Tentatives de tir';
COMMENT ON COLUMN request_log.fg_perc IS 'Pourcentage de réussite des tirs';
COMMENT ON COLUMN request_log.threep_made IS '3 points réussis';
COMMENT ON COLUMN request_log.threepa IS 'Tentatives de 3 points';
COMMENT ON COLUMN request_log.threep_perc IS 'Pourcentage de réussite des 3 points';
COMMENT ON COLUMN request_log.ftm IS 'Lancers francs réussis';
COMMENT ON COLUMN request_log.fta IS 'Tentatives de lancer franc';
COMMENT ON COLUMN request_log.ft_perc IS 'Pourcentage de réussite des lancers francs';
COMMENT ON COLUMN request_log.oreb IS 'Rebonds offensifs';
COMMENT ON COLUMN request_log.dreb IS 'Rebonds défensifs';
COMMENT ON COLUMN request_log.reb IS 'Total des rebonds';
COMMENT ON COLUMN request_log.ast IS 'Passes décisives';
COMMENT ON COLUMN request_log.stl IS 'Interceptions';
COMMENT ON COLUMN request_log.blk IS 'Contres';
COMMENT ON COLUMN request_log.tov IS 'Balles perdues';
COMMENT ON COLUMN request_log.probability_class_0 IS 'Probabilité que l''événement soit ''Ne durera pas 5 ans''';
COMMENT ON COLUMN request_log.probability_class_1 IS 'Probabilité que l''événement soit ''Durera 5 ans ou plus''';
COMMENT ON COLUMN request_log.prediction IS 'Classe prédite (0 ou 1)';
COMMENT ON COLUMN request_log.prediction_text IS 'Texte décrivant la classe prédite';
COMMENT ON COLUMN request_log.timestamp IS 'Date et heure de la requête';
"

# Exécuter la commande pour créer la table
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -d $DB_NAME -U $DB_USER -c "$SQL_QUERY"

# Vérifier si la table a été créée avec succès
TABLE_CREATED=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -d $DB_NAME -U $DB_USER -c "\dt" | grep -w "request_log" | wc -l)

if [ "$TABLE_CREATED" -eq 1 ]; then
    echo "La table 'request_log' a été créée avec succès dans la base de données '$DB_NAME'."
else
    echo "Erreur : La table 'request_log' n'a pas pu être créée. Veuillez vérifier les logs pour plus d'informations."
    exit 1
fi

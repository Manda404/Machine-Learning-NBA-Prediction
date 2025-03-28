import os
import time
import pickle
import joblib
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import norm
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# URL de l'API FastAPI
API_URL = "http://localhost:8000/predict/"

# Initialisation de st.session_state pour stocker les résultats de la comparaison
if 'meilleur_rapport' not in st.session_state:
    st.session_state.meilleur_rapport = False

# Initialiser `model` dans st.session_state si ce n'est pas déjà fait
if 'model' not in st.session_state:
    st.session_state['model'] = None  # Valeur par défaut

# Initialisation de la variable de rafraîchissement
if 'refresh' not in st.session_state:
    st.session_state['refresh'] = False


# Initialiser `report_nouveau` dans st.session_state si ce n'est pas déjà fait
if 'report_nouveau' not in st.session_state:
    st.session_state['report_nouveau'] = {
        'Accuracy': 0.0,
        'Precision': 0.0,
        'Recall': 0.0,
        'F1 Score': 0.0
    }

# Appeler set_page_config avant tout autre appel à Streamlit
st.set_page_config(page_title="🛠️ Prédiction de carrière des joueurs NBA", page_icon="🏀", layout="wide")

# Fonction pour charger le modèle
def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        raise FileNotFoundError("Le fichier modèle est introuvable. Vérifiez le chemin et réessayez.")

# Charger le dataset avec la nouvelle méthode de mise en cache
@st.cache_data
def load_dataset(file_path):
    """
    Charge le dataset à partir d'un fichier CSV et renomme certaines colonnes.
    :param file_path: Chemin vers le fichier CSV.
    :return: DataFrame Pandas contenant les données chargées.
    """
    try:
        # Charger le dataset
        data = pd.read_csv(file_path)

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

# Charger le dataset pour l'évaluation
@st.cache_data
def split_data(data):
    # Imputation des valeurs manquantes
    data['ThreeP_perc'] = data['ThreeP_perc'].fillna(data['ThreeP_perc'].mean())
    X = data.drop(['TARGET_5Yrs','Name'], axis=1)
    y = data['TARGET_5Yrs']
    return X, y

# Fonction modifiée pour entraîner et évaluer le modèle, et retourner TP, TN, FP, FN
def evaluate_model(model, X_train, X_test, y_train, y_test, threshold=0.5):
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculer la matrice de confusion pour obtenir TP, TN, FP, FN
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return model, accuracy, precision, recall, f1, tp, tn, fp, fn

# Fonction pour afficher la barre latérale
def display_sidebar():
    st.sidebar.header("🏀 Prédiction de carrière des joueurs NBA")
    st.sidebar.image('picture.png')  # Remplacez 'picture.png' par l'image appropriée si nécessaire

    # Informations sur la version de l'application
    st.sidebar.markdown(
        """
        <div style="text-align: center; font-size: 18px; margin-top: 10px;">
            📅 Version de l'application : 1.0.0
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Informations sur le projet
    st.sidebar.markdown(
        """
        <hr style="margin-top: 10px; margin-bottom: 10px;" />
        <h3 style="text-align: center; font-size: 20px; margin-top: 10px;">
            📝 Informations sur le projet :
        </h3>
        <div style="font-size: 16px; margin-top: 20px;">
            <b>👨‍💻 Auteur :</b> Rostand Surel <br/>
            <b>🗓️ Date de création :</b> {creation_date}<br/>
            <b>🔄 Dernière mise à jour :</b> {last_update}
        </div>
        """.format(
            creation_date="2024-11-06",
            last_update=datetime.now().strftime("%Y-%m-%d")
        ), 
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

# Fonction pour afficher l'onglet de présentation
def display_presentation():
    st.title("Présentation")
    st.write("""
    **Objectif du projet** : Cette application vise à prédire si un joueur débutant en NBA durera plus de 5 ans 
    en utilisant ses statistiques initiales. L'application est divisée en plusieurs sections :
    
    - **Exploratory Data Analysis (EDA)** : Visualisation des statistiques des joueurs.
    - **Global Performance** : Analyse des performances du modèle à l'aide de métriques.
    - **Prediction** : Saisie manuelle ou chargement de fichier pour tester la prédiction.
    """)
    st.subheader("Structure de l'interface")
    st.markdown("""
    1. **🏠 Présentation** : Introduction au projet et explication de l'objectif.
    2. **📈 Exploratory Data Analysis** : Exploration visuelle des données du dataset.
    3. **🏋️‍♂️ Global Performance** : Présentation des performances du modèle.
    4. **🏃‍♂️ Prediction** : Tester la prédiction sur des données saisies manuellement ou via un fichier CSV.
    """)

# Fonction pour afficher l'onglet EDA
def display_eda(data):
    st.title("Exploratory Data Analysis (EDA)")

    # Sélectionner les colonnes numériques pour l'analyse
    numerical_features = data.select_dtypes(include=['number']).columns.tolist()

    # Colonne cible à exclure
    target_column = 'TARGET_5Yrs'

    # Supprimer la colonne cible de la liste des colonnes numériques si elle est présente
    numerical_features.remove(target_column)

    #st.header("Analyse des Données", divider='rainbow')
    #st.header("🏀📊 Analyse des Données 🔍", divider='rainbow')
    #st.header("_Analyse des Données de Basket_ :orange_basketball: :blue[📊] :sunglasses:")
    st.header("_Analyse des Données de Basket_ :basketball: :blue[📊] :orange[🔥]",divider='rainbow')




    with st.expander("Afficher le Dataset NBA", expanded=True):
        # Colonnes par défaut à afficher
        colonnes_defaut = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_perc', 'ThreeP_Made', 'ThreePA', 'ThreeP_perc','TARGET_5Yrs']
        # ,'FTM', 'FTA', 'FT_perc', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'TARGET_5Yrs']

        # Sélection des colonnes à afficher avec multiselect
        colonnes_filtrees = st.multiselect('Sélectionner les colonnes à afficher :', data.columns, default=colonnes_defaut)

        # Nombre de lignes à afficher
        nb_lignes = st.number_input('Nombre de lignes à afficher', min_value=1, max_value=data.shape[0], value=5)

        # Afficher les données filtrées
        st.dataframe(data[colonnes_filtrees].head(nb_lignes), use_container_width=True)

        # Calcul du nombre de colonnes catégorielles et numériques
        nb_colonnes_categorielles = data.select_dtypes(include=['object', 'category']).shape[1]
        nb_colonnes_numeriques = data.select_dtypes(include=['number']).shape[1]

        # Calcul des métriques de base
        metriques = {
            'Nombre de lignes': data.shape[0],
            'Nombre de colonnes': data.shape[1],
            'Valeurs manquantes': data.isnull().sum().sum(),
            'Colonnes catégorielles': nb_colonnes_categorielles,
            'Colonnes numériques': nb_colonnes_numeriques
        }

        # Affichage des métriques
        colonnes = st.columns(5, gap="medium")
        for col, (etiquette, valeur) in zip(colonnes, metriques.items()):
            with col:
                st.metric(label=f"🔍 {etiquette}", value=f"{valeur:,.0f}")

    with st.expander("Exploration de la colonne Cible", expanded=True):
        st.markdown("""
            #### Informations sur les cas et les variables disponibles dans le dataset NBA
            - **TARGET_5Yrs** : Notre variable cible. Elle indique si un joueur a eu une carrière de plus de 5 ans (1) ou non (0).
            - L'équilibre de la classe est un facteur important à prendre en compte, car il peut influencer les performances du modèle de prédiction.
            """)

        # Calculer le nombre de valeurs dans la colonne "TARGET_5Yrs"
        target_counts = data['TARGET_5Yrs'].value_counts().reset_index()
        target_counts.columns = ['Cible', 'Nombre']

        # Créer la mise en page Streamlit avec deux colonnes
        col1, col2 = st.columns(2)

        with col1:
            # Couleurs pour les classes
            colors = ['#636EFA', '#EF553B']

            # Création du graphique en barres
            fig_bar = px.bar(
                target_counts, x='Cible', y='Nombre', text='Nombre',
                labels={'Cible': 'Cible', 'Nombre': 'Nombre'}, template='seaborn',
                color='Cible', color_discrete_sequence=colors
            )
            fig_bar.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Création d'un graphique en secteurs
            fig_pie = px.pie(
                target_counts, values='Nombre', names='Cible', hole=0.5,
                labels={'Nombre': 'Nombre'}, template='seaborn',
                color_discrete_sequence=colors
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(title='Proportions de la Cible')
            st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("Explorer le nombre d'apparitions des joueurs", expanded=True):
        st.markdown("### Analyse des joueurs par nombre d'apparitions")
        
        # Sélection du nombre de joueurs à afficher
        n = st.slider("Sélectionnez le nombre de joueurs à afficher", min_value=1, max_value=50, value=10)
        
        # Afficher le graphique si le dataset est chargé
        if 'Name' in data.columns:
            fig = plot_player_appearance_count(data, n)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("La colonne 'Name' n'est pas présente dans le dataset.")

    with st.expander("Visualiser les valeurs manquantes par colonne", expanded=True):
        st.markdown("### Analyse des valeurs manquantes par colonne")
        
        # Afficher le graphique si le dataset est chargé et non vide
        if not data.empty:
            fig = plot_missing_values(data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Le dataset est vide ou non chargé.")

    with st.expander("Visualiser les valeurs aberrantes par colonne", expanded=True):
        st.markdown("### Analyse des valeurs aberrantes par colonne")
        
        # Définir les percentiles pour les quartiles
        first_quartile, third_quartile = 0.25, 0.75
        
        # Calcul des informations sur les valeurs aberrantes
        df_outliers_info = detect_and_count_outliers(data, numerical_features, first_quartile, third_quartile)
        
        # Afficher le DataFrame des valeurs aberrantes
        #st.dataframe(df_outliers_info)
        
        # Visualiser les valeurs aberrantes
        fig = visualize_outliers(df_outliers_info)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Analyse et Distribution des Données Numériques", expanded=True):
        # Choisir une colonne pour l'analyse
        selected_col = st.selectbox("Sélectionnez une colonne numérique à analyser :", numerical_features, index=None, key=1)
        if selected_col:
            st.markdown(f"Analyse de la colonne {selected_col}")
            data, mean, std_dev = compute_statistics(data.copy(), selected_col)
            div1, div2 = st.columns(2)
            # Calculer le nombre d'outliers
            outlier_count = count_outliers_Specifying_quartile(data, selected_col, 0.25, 0.75)
            #st.markdown(f"<span style='color:blue;'>**Nombre de valeurs aberrantes** : {outlier_count}</span>", unsafe_allow_html=True)
            # Calculer le nombre d'outliers et le pourcentage
            outlier_count = count_outliers_Specifying_quartile(data, selected_col, 0.25, 0.75)
            total_count = data[selected_col].shape[0]
            outlier_percentage = (outlier_count / total_count) * 100
            
            # Afficher le nombre d'outliers et le pourcentage
            st.metric(label="Nombre de valeurs aberrantes", value=outlier_count, delta=f"{outlier_percentage:.2f}%")


            with div1:
                st.subheader("Courbe Normale")
                fig_normal = plot_normal_distribution(data, selected_col, mean, std_dev)
                st.plotly_chart(fig_normal, use_container_width=True)
            
            with div2:
                try:
                    # Afficher le box plot
                    fig_box, quartiles_table, min_value, max_value, quartiles = plot_box_and_quartiles(data, selected_col)
                    st.subheader("Box Plot")
                    st.plotly_chart(fig_box, use_container_width=True)
                except Exception as e:
                    st.info(f"Erreur : {str(e)}")

    with st.expander("Distribution et Répartition de la colonne cible", expanded=True):
        selected_col = st.selectbox("Sélectionnez une colonne numérique à analyser :", numerical_features, index=None, key=124)
        
        # Première ligne avec 2 colonnes
        div3, div4 = st.columns(2)

        # Deuxième ligne avec 2 colonnes
        div5, div6 = st.columns(2)

        if selected_col:
            with div3:
                st.subheader("Distribution simple")
                fig_distribution_col = px.histogram(data, x=selected_col, nbins=30, title=f"Distribution de {selected_col}",
                                                    labels={selected_col: f"{selected_col}"},
                                                    template='seaborn')
                st.plotly_chart(fig_distribution_col, use_container_width=True)
                    
            with div4:
                st.subheader("Distribution par classe cible")
                fig_distribution_col_target = px.histogram(data, x=selected_col, color=target_column, nbins=30,
                                                        title=f"Répartition de {selected_col} selon {target_column}",
                                                        labels={selected_col: f"{selected_col}", target_column: "Classe cible"},
                                                        template='seaborn',
                                                        color_discrete_sequence=px.colors.qualitative.Set1)
                st.plotly_chart(fig_distribution_col_target, use_container_width=True)

            with div5:
                st.subheader("Graphique en violon")
                fig_distribution_col = px.violin(data, y=selected_col, box=True, points='all',
                                                title=f"Distribution de {selected_col}",
                                                labels={selected_col: f"{selected_col}"},
                                                template='seaborn')
                st.plotly_chart(fig_distribution_col, use_container_width=True)
                    
            with div6:
                st.subheader("Distribution superposée")
                fig_density = px.histogram(data, x=selected_col, color=target_column, 
                                        histnorm='density', barmode='overlay',
                                        marginal='violin', opacity=0.6,
                                        title=f"Distribution de {selected_col} par {target_column}",
                                        labels={selected_col: f"{selected_col}", target_column: "Classe cible"},
                                        template='seaborn')
                st.plotly_chart(fig_density, use_container_width=True)

# Fonction pour calculer la moyenne et l'écart-type
def compute_statistics(dataset, col_name):
    mean = dataset[col_name].mean()
    std_dev = dataset[col_name].std()
    dataset['Z score'] = (dataset[col_name] - mean) / std_dev
    return dataset, mean, std_dev

# Fonction pour tracer la distribution normale
def plot_normal_distribution(dataset, col_name, mean, std_dev):
    x_values = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
    y_values = norm.pdf(x_values, mean, std_dev)
    trace_curve = go.Scatter(x=x_values, y=y_values, mode='lines', name='Normal Distribution Curve')
    trace_threshold_neg = go.Scatter(x=[mean - 3 * std_dev, mean - 3 * std_dev], y=[0, max(y_values)], mode='lines', name='-3 SD Threshold', line=dict(color='red', dash='dash'))
    trace_threshold_pos = go.Scatter(x=[mean + 3 * std_dev, mean + 3 * std_dev], y=[0, max(y_values)], mode='lines', name='+3 SD Threshold', line=dict(color='red', dash='dash'))
    outliers = np.abs(dataset['Z score']) > 3
    trace_outliers = go.Scatter(x=dataset[col_name][outliers], y=norm.pdf(dataset[col_name][outliers], mean, std_dev), mode='markers', name='Outliers', marker=dict(color='red'))
    layout = go.Layout(xaxis=dict(title=col_name), yaxis=dict(title='Probability Density'))
    fig = go.Figure(data=[trace_curve, trace_threshold_neg, trace_threshold_pos, trace_outliers], layout=layout)
    return fig

# Fonction pour tracer le box plot et les informations des quartiles
def plot_box_and_quartiles(dataset, col_name):
    quartiles = dataset[col_name].quantile([0.25, 0.5, 0.75])
    min_value = dataset[col_name].min()
    max_value = dataset[col_name].max()
    quartiles_table = pd.DataFrame({
        'Quartiles': ['First Quartile (Q1)', 'Second Quartile (Q2)', 'Third Quartile (Q3)', 'Min', 'Max'],
        'Value': [quartiles[0.25], quartiles[0.5], quartiles[0.75], min_value, max_value]
    })
    fig = px.box(dataset, y=col_name, points='all', title=f"Box Plot de {col_name}")
    return fig, quartiles_table, min_value, max_value, quartiles

# Fonction pour obtenir les valeurs IQR et autres statistiques
def get_iqr_values_Specifying_quartile(df_in, col_name, first_quartile, third_quartile):
    median = df_in[col_name].median()
    q1 = df_in[col_name].quantile(first_quartile)  # xth percentile / 1st quartile
    q3 = df_in[col_name].quantile(third_quartile)  # yth percentile / 3rd quartile
    iqr = q3 - q1  # Interquartile range
    minimum = q1 - (1.5 * iqr)  # The minimum value or the |- marker in the box plot
    maximum = q3 + (1.5 * iqr)  # The maximum value or the -| marker in the box plot
    return median, q1, q3, iqr, minimum, maximum

# Fonction pour compter le nombre de valeurs aberrantes dans une colonne
def count_outliers_Specifying_quartile(df_in, col_name, first_quartile, third_quartile):
    _, _, _, _, minimum, maximum = get_iqr_values_Specifying_quartile(df_in, col_name, first_quartile, third_quartile)
    df_outliers = df_in.loc[(df_in[col_name] <= minimum) | (df_in[col_name] >= maximum)]
    return df_outliers.shape[0]

# Fonction pour détecter et compter les valeurs aberrantes dans plusieurs colonnes
def detect_and_count_outliers(df_in, list_des_colonnes, first_quartile, third_quartile):
    result_data = []
    for col_name in list_des_colonnes:
        outlier_count = count_outliers_Specifying_quartile(df_in, col_name, first_quartile, third_quartile)
        total_count = len(df_in)
        outlier_percentage = round((outlier_count / total_count) * 100, 2)
        result_data.append([col_name, outlier_count, outlier_percentage])
    
    # Création du DataFrame et tri par nombre de valeurs aberrantes en ordre décroissant
    result_df = pd.DataFrame(result_data, columns=['Nom de la colonne', 'Nombre de valeurs aberrantes', 'Pourcentage de valeurs aberrantes (%)'])
    result_df = result_df.sort_values(by='Nombre de valeurs aberrantes', ascending=False)
    return result_df

# Fonction pour visualiser les valeurs aberrantes
def visualize_outliers(df):
    fig = px.bar(df, x='Nom de la colonne', y='Nombre de valeurs aberrantes',
                 labels={'Nom de la colonne': 'Nom de la colonne', 'Nombre de valeurs aberrantes': 'Nombre de valeurs aberrantes'},
                 title='Nombre de valeurs aberrantes par colonne (ordre décroissant)',
                 text='Pourcentage de valeurs aberrantes (%)')
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    return fig

# Fonction pour calculer et visualiser le nombre de valeurs manquantes par colonne
def plot_missing_values(data):
    # Calculer le nombre de valeurs manquantes par colonne
    missing_counts = data.isnull().sum().reset_index()
    missing_counts.columns = ['Colonne', 'Nombre de valeurs manquantes']
    
    # Calculer le pourcentage de valeurs manquantes par colonne
    missing_counts['Pourcentage'] = (missing_counts['Nombre de valeurs manquantes'] / len(data)) * 100
    
    # Trier par nombre de valeurs manquantes en ordre décroissant
    missing_counts = missing_counts.sort_values(by='Nombre de valeurs manquantes', ascending=False)
    
    # Créer le graphique avec Plotly Express pour toutes les colonnes
    fig = px.bar(missing_counts, x='Colonne', y='Nombre de valeurs manquantes', 
                 title="Nombre de valeurs manquantes par colonne (ordre décroissant)",
                 labels={'Colonne': 'Nom de la colonne', 'Nombre de valeurs manquantes': 'Nombre de valeurs manquantes'})
    
    # Ajouter du texte sur les barres pour afficher le nombre et le pourcentage
    fig.update_traces(text=missing_counts.apply(lambda row: f"{row['Nombre de valeurs manquantes']} ({row['Pourcentage']:.1f}%)", axis=1), 
                      textposition='outside')
    
    return fig

# Fonction pour tracer le nombre d'apparitions des joueurs
def plot_player_appearance_count(data, n):
    # Grouper par la colonne 'Name' et compter les occurrences pour chaque joueur
    player_counts = data['Name'].value_counts().reset_index()
    player_counts.columns = ['Name', 'Count']
    
    # Calculer le total pour la conversion en pourcentage
    total_count = player_counts['Count'].sum()
    
    # Trier les joueurs par nombre d'apparitions de manière décroissante et sélectionner les 'n' premiers
    top_players = player_counts.sort_values(by='Count', ascending=False).head(n)
    
    # Calculer le pourcentage pour chaque joueur
    top_players['Percentage'] = (top_players['Count'] / total_count) * 100
    
    # Créer le graphique avec Plotly Express
    fig = px.bar(top_players, x='Name', y='Count', title=f"Top {n} joueurs par nombre d'apparitions",
                 labels={'Name': 'Nom du joueur', 'Count': 'Nombre d\'apparitions'})
    
    # Ajouter du texte sur les barres pour afficher le nombre et le pourcentage
    fig.update_traces(text=top_players.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1), 
                      textposition='outside')
    
    return fig

# Fonction pour créer une jauge circulaire
def create_gauge(name, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,  # Conversion en pourcentage
        number={'suffix': "%"},
        title={'text': name, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value * 100  # Conversion en pourcentage
            }
        },
        delta={'reference': 50, 'position': "top"}
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=250,
        width=250
    )
    return fig

# Fonction pour lire le fichier CSV et retourner les données sous forme de dictionnaire
def lire_metrics(path, nom_fichier='metrics.csv'):
    """
    Lit le fichier CSV contenant les métriques et retourne les données sous forme de dictionnaire.
    
    Args:
        path (str): Le chemin du fichier CSV.
        nom_fichier (str): Le nom du fichier CSV (par défaut 'metrics.csv').
    
    Returns:
        dict: Les données du fichier CSV sous forme de dictionnaire.
    """
    # Créer le chemin complet du fichier
    chemin_complet = os.path.join(path, nom_fichier)
    
    # Lire le fichier CSV
    df = pd.read_csv(chemin_complet)
    
    # Convertir le DataFrame en dictionnaire
    report_dict = df.to_dict(orient='records')[0]  # On prend le premier enregistrement
    return report_dict

# Fonction pour sauvegarder le rapport des métriques dans un fichier CSV
def sauvegarder_metrics(report, path, nom_fichier='metrics.csv'):
    """
    Sauvegarde le rapport des métriques dans un fichier CSV.
    
    Args:
        report (dict): Le rapport des métriques à sauvegarder.
        path (str): Le chemin où le fichier CSV sera sauvegardé.
        nom_fichier (str): Le nom du fichier CSV (par défaut 'metrics.csv').
    """
    # Convertir le dictionnaire en DataFrame
    df = pd.DataFrame([report])

    
    # Créer le chemin complet du fichier
    chemin_complet = os.path.join(path, nom_fichier)
    
    # Sauvegarder le DataFrame en CSV
    df.to_csv(chemin_complet, index=False)
    #print(f"Rapport des métriques sauvegardé à : {chemin_complet}")

# Fonction pour comparer les rapports de métriques et vérifier si le nouveau rapport est meilleur
def comparer_rapports(report_actuelle, report_nouveau):
    """
    Compare les rapports actuels et nouveaux et vérifie si les nouvelles métriques sont meilleures.
    
    Args:
        report_actuelle (dict): Rapport des métriques actuel.
        report_nouveau (dict): Nouveau rapport des métriques.

    Returns:
        bool: Retourne True si toutes les valeurs de report_nouveau sont meilleures, sinon False.
    """
    for key in report_nouveau:
        if report_nouveau[key] <= report_actuelle[key]:
            return False
    return True

# Fonction pour comparer deux rapports de métriques
def comparer_rapports(report_actuelle, report_nouveau):
    """
    Compare deux rapports de métriques pour vérifier si le nouveau rapport est meilleur.
    """
    return all(report_nouveau[metric] > report_actuelle.get(metric, 0) for metric in report_nouveau)

# Fonction pour comparer le F1 Score entre deux rapports de métriques
def comparer_f1_score(report_actuelle, report_nouveau):
    """
    Compare le F1 Score entre deux rapports de métriques pour vérifier si le nouveau rapport est meilleur.
    """
    return report_nouveau.get("F1 Score", 0) > report_actuelle.get("F1 Score", 0)

# Fonction pour sauvegarder uniquement le modèle
def sauvegarder_nouveau_modele(model, path_model):
    """
    Sauvegarde uniquement le modèle au chemin spécifié.
    
    Args:
        model (object): Le modèle à sauvegarder.
        path_model (str): Dossier où sauvegarder le modèle.
    """
    # Créer le chemin complet du fichier modèle
    chemin_complet_model = os.path.join(path_model, 'model.pkl')
    
    # Sauvegarder le modèle
    #joblib.dump(model, chemin_complet_model)

    with open(chemin_complet_model, 'wb') as f:
        pickle.dump(model, f)

    
    # Message de succès avec des emojis
    # st.success(f"✅ Le nouveau modèle a été sauvegardé avec succès à l'emplacement : {chemin_complet_model} 🚀💾")

# Fonction principale pour l'affichage et la comparaison des performances
def display_global_performance(data, path_rapport, path_model):
    st.title("Performance Globale")
    st.subheader("Métriques de performance")

    # Initialisation ou réinitialisation de la variable de session
    # st.session_state.meilleur_rapport = False

    # Initialisation du meilleur model de classification
    trained_model = None

    # Chargement des métriques actuelles
    report_actuelle = lire_metrics(path_rapport)

    # Création des graphiques pour chaque métrique
    figures = [create_gauge(name, value) for name, value in report_actuelle.items()]

    with st.expander("Détails et explications des métriques", expanded=True):
        st.markdown("""
        **Accuracy (Exactitude)** — Cette métrique mesure la proportion de prédictions correctes parmi toutes les prédictions effectuées.
        """, unsafe_allow_html=True)
        st.markdown("""
        **Precision (Précision)** — Cette métrique mesure la proportion de prédictions positives correctes parmi toutes les prédictions positives faites par le modèle.
        """, unsafe_allow_html=True)
        st.markdown("""
        **Recall (Rappel)** — Aussi appelé sensibilité, le rappel mesure la proportion d'instances positives réelles qui ont été correctement identifiées par le modèle.
        """, unsafe_allow_html=True)
        st.markdown("""
        **F1 Score** — La moyenne harmonique de la précision et du rappel, qui équilibre les deux métriques. Il est utile lorsqu'il est important de trouver un équilibre entre la précision et le rappel.
        """, unsafe_allow_html=True)

        # Affichage des graphiques dans les colonnes
        col1, col2, col3, col4 = st.columns(4)
        for col, fig in zip([col1, col2, col3, col4], figures):
            col.plotly_chart(fig, use_container_width=True)

    with st.expander("Ajuster les hyperparamètres des modèles de classification", expanded=True):
        # Sélectionner le modèle de classification
        model_choice = st.selectbox("Choisissez un modèle de classification", ["Random Forest", "Gradient Boosting", "Logistic Regression"],index=None)

        if model_choice == "Random Forest":
            n_estimators = st.slider("Nombre d'estimateurs", 10, 500, 100, step=10)
            max_depth = st.slider("Profondeur maximale", 1, 50, 10)
            min_samples_split = st.slider("Nombre minimal d'échantillons pour diviser un nœud", 2, 10, 2)
            min_samples_leaf = st.slider("Nombre minimal d'échantillons par feuille", 1, 10, 1)
            max_features = st.selectbox("Nombre maximal de caractéristiques à considérer", [None, 'sqrt', 'log2'])
            bootstrap = st.checkbox("Utiliser le bootstrap", value=True)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=42
            )

        elif model_choice == "Gradient Boosting":
            learning_rate = st.slider("Taux d'apprentissage", 0.01, 1.0, 0.1, step=0.01)
            n_estimators = st.slider("Nombre d'estimateurs", 10, 500, 100, step=10)
            max_depth = st.slider("Profondeur maximale", 1, 50, 3)
            min_samples_split = st.slider("Nombre minimal d'échantillons pour diviser un nœud", 2, 10, 2)
            min_samples_leaf = st.slider("Nombre minimal d'échantillons par feuille", 1, 10, 1)
            subsample = st.slider("Sous-échantillonnage des échantillons", 0.1, 1.0, 1.0, step=0.1)
            max_features = st.selectbox("Nombre maximal de caractéristiques à considérer", [None, 'sqrt', 'log2', None])

            model = GradientBoostingClassifier(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                max_features=max_features,
                random_state=42
            )

        elif model_choice == "Logistic Regression":
            C = st.slider("Paramètre de régularisation (C)", 0.01, 10.0, 1.0, step=0.01)
            max_iter = st.slider("Nombre maximal d'itérations", 100, 2000, 300)
            solver = st.selectbox("Algorithme de résolution", ['lbfgs', 'liblinear', 'sag', 'saga'])
            penalty = st.selectbox("Type de pénalité", ['l2', 'l1', 'elasticnet', 'none'])

            model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver=solver,
                penalty=penalty,
                random_state=42
            )

        if model_choice:
            # Ajuster le threshold de prédiction
            threshold = st.slider("Threshold de décision", 0.0, 1.0, 0.5, step=0.01)

            if st.button("Train", key="train_button"):
                with st.spinner("Entraînement en cours..."):
                    X, y = split_data(data)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
                    trained_model, accuracy, precision, recall, f1, tp, tn, fp, fn = evaluate_model(model, X_train, X_test, y_train, y_test, threshold)

                    # Afficher les résultats en temps réel avec st.metric()
                    st.write("### Évaluation du modèle")

                    # Afficher les métriques de performance
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.2f}")
                    col2.metric("Precision", f"{precision:.2f}")
                    col3.metric("Recall", f"{recall:.2f}")
                    col4.metric("F1 Score", f"{f1:.2f}")

                    # Afficher TP, TN, FP, FN
                    st.write("### Valeurs de la matrice de confusion")
                    col5, col6, col7, col8 = st.columns(4)
                    col5.metric("TP (Vrai Positifs)", tp)
                    col6.metric("TN (Vrai Négatifs)", tn)
                    col7.metric("FP (Faux Positifs)", fp)
                    col8.metric("FN (Faux Négatifs)", fn)

                report_nouveau = {'Accuracy': accuracy,'Precision': precision,'Recall': recall,'F1 Score': f1}
                #st.write(report_nouveau)
                #st.write(report_actuelle)

                # Comparer les rapports et mettre à jour l'état de session
                if comparer_f1_score(report_actuelle, report_nouveau):
                    st.success("✨✅ Le nouveau modèle a de meilleures performances.")
                    # Mettre à jour la variable `flag` dans `st.session_state`
                    st.session_state.meilleur_rapport = True
                    # Mettre à jour `model` dans `st.session_state`
                    st.session_state['model'] = model
                    # Mettre à jour `report_nouveau` dans `st.session_state`
                    st.session_state['report_nouveau'] = report_nouveau
                else:
                    st.session_state.meilleur_rapport = False
                    st.warning("⚠️ Le nouveau modèle n'a pas des performances meilleures que le modèle actuel.")

    # Afficher le bouton de sauvegarde si le nouveau modèle est meilleur
    if st.session_state.meilleur_rapport:
        # Créer une section extensible pour les options de sauvegarde
        with st.expander("Options de sauvegarde"):
            # Afficher un bouton pour permettre la sauvegarde du modèle et des métriques
            if st.button("Sauvegarder le modèle & les métriques", key="save_button"):
                # Accéder au rapport des nouvelles métriques stocké dans `st.session_state`
                report_nouveau = st.session_state['report_nouveau']
                # Accéder au modèle entraîné stocké dans `st.session_state`
                trained_model = st.session_state['model']

                # Appeler la fonction pour sauvegarder les métriques dans un fichier CSV
                sauvegarder_metrics(report_nouveau, path_rapport, nom_fichier='metrics.csv')

                # Appeler la fonction pour sauvegarder le modèle au chemin spécifié
                sauvegarder_nouveau_modele(trained_model, path_model)      

                # Afficher un message de succès avec des emojis pour informer l'utilisateur
                st.success("✅ Le modèle et les métriques ont été sauvegardés avec succès ! 🚀💾")
                
                # Réinitialiser la variable de session `meilleur_rapport` pour indiquer que l'opération est terminée
                st.session_state.meilleur_rapport = False

                # Changer la valeur de 'refresh' pour provoquer une mise à jour
                st.session_state['refresh'] = not st.session_state['refresh']

# Fonction pour afficher l'onglet de prédiction
def display_prediction(model):
    st.title("Prédiction")
    option = st.selectbox("Choisissez la méthode de saisie des données :", ["Saisie manuelle", "Charger un fichier CSV"], placeholder="Sélectionnez la méthode de saisie...")
    if option == "Saisie manuelle":
        st.subheader("Saisie manuelle des données")
        input_data = collect_input_data()
        if st.button("Prédire"):
            response = requests.post(API_URL, json=input_data)
            if response.status_code == 200:
                result = response.json()
                display_prediction_results(result['probabilities'], result['prediction'])
            else:
                st.error("Erreur lors de la prédiction. Veuillez réessayer.")
    elif option == "Charger un fichier CSV":
        st.subheader("Chargement de données à partir d'un fichier CSV")
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Aperçu des données chargées :")
            st.dataframe(data.head())
            
            if st.button("Prédire pour tous les joueurs"):
                results = []
                for _, row in data.iterrows():
                    response = requests.post(API_URL, json=row.to_dict())
                    if response.status_code == 200:
                        results.append(response.json())
                # Ajouter les prédictions au DataFrame et afficher les résultats
                data['Prédiction'] = [res['prediction'] for res in results]
                data['probabilities'] = [res['probabilities'] for res in results]
                data['label'] = data['Prédiction'].apply(lambda x: "Durera 5 ans ou plus" if x == 1 else "Ne durera pas 5 ans")
                st.write("Prédictions pour les joueurs :")
                st.dataframe(data)
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(label="Télécharger les prédictions", data=csv, file_name='predictions_nba.csv', mime='text/csv')

# Fonction pour collecter les données d'entrée
def collect_input_data():
    input_data = {}
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data['GP'] = int(st.number_input("Games Played (GP)", min_value=0, step=1))
        input_data['MIN'] = float(st.number_input("Minutes Played (MIN)", min_value=0.0, format="%.3f"))
        input_data['PTS'] = float(st.number_input("Points Per Game (PTS)", min_value=0.0, format="%.3f"))
        input_data['FGM'] = float(st.number_input("Field Goals Made (FGM)", min_value=0.0, format="%.3f"))
        input_data['FGA'] = float(st.number_input("Field Goal Attempts (FGA)", min_value=0.0, format="%.3f"))
        input_data['FG_perc'] = float(st.number_input("Field Goal Percent (FG%)", min_value=0.0, max_value=100.0, format="%.3f"))
    with col2:
        input_data['ThreeP_Made'] = float(st.number_input("3P Made", min_value=0.0, format="%.3f"))
        input_data['ThreePA'] = float(st.number_input("3PA", min_value=0.0, format="%.3f"))
        input_data['ThreeP_perc'] = float(st.number_input("3P%", min_value=0.00, max_value=100.0, format="%.3f"))
        input_data['FTM'] = float(st.number_input("Free Throws Made (FTM)", min_value=0.0, format="%.3f"))
        input_data['FTA'] = float(st.number_input("Free Throw Attempts (FTA)", min_value=0.0, format="%.3f"))
        input_data['FT_perc'] = float(st.number_input("Free Throw Percent (FT%)", min_value=0.0, max_value=100.0, format="%.3f"))
    with col3:
        input_data['OREB'] = float(st.number_input("Offensive Rebounds (OREB)", min_value=0.0, format="%.3f"))
        input_data['DREB'] = float(st.number_input("Defensive Rebounds (DREB)", min_value=0.0, format="%.3f"))
        input_data['REB'] = float(st.number_input("Rebounds (REB)", min_value=0.0, format="%.3f"))
        input_data['AST'] = float(st.number_input("Assists (AST)", min_value=0.0, format="%.3f"))
        input_data['STL'] = float(st.number_input("Steals (STL)", min_value=0.0, format="%.3f"))
        input_data['BLK'] = float(st.number_input("Blocks (BLK)", min_value=0.0, format="%.3f"))
        input_data['TOV'] = float(st.number_input("Turnovers (TOV)", min_value=0.0, format="%.3f"))
    return input_data

# Fonction pour afficher les résultats de prédiction
def display_prediction_results(probabilities, prediction):
    # Arrondir les probabilités à 3 chiffres après la virgule
    probabilities = np.round(probabilities, 3)
    
    # Créer un DataFrame pour les probabilités arrondies
    df_prediction_proba = pd.DataFrame([probabilities], columns=["Ne durera pas 5 ans", "Durera 5 ans ou plus"])
    
    # Afficher les résultats dans l'interface Streamlit
    st.subheader('Probabilités des prédictions')
    st.dataframe(df_prediction_proba, column_config={
        'Ne durera pas 5 ans': st.column_config.ProgressColumn('Ne durera pas 5 ans', format='%0.3f', width='medium', min_value=0, max_value=1),
        'Durera 5 ans ou plus': st.column_config.ProgressColumn('Durera 5 ans ou plus', format='%0.3f', width='medium', min_value=0, max_value=1)
    }, hide_index=True)
    
    # Définir les classes avec emojis
    classes = np.array(["Ne durera pas 5 ans ⚠️", "Durera 5 ans ou plus ✅"])
    
    # Afficher le résultat avec un emoji
    st.success(f"Prédiction : {classes[prediction]}")

# Interface principale
def main():
    model_path = "/Users/surelmanda/Downloads/AirGUARD/nba-prediction-project/model/model.pkl"
    dataset_path = "/Users/surelmanda/Downloads/AirGUARD/nba-prediction-project/data/nba_logreg.csv"
    path_rapport = '/Users/surelmanda/Downloads/AirGUARD/nba-prediction-project/metrics'
    path_model = '/Users/surelmanda/Downloads/AirGUARD/nba-prediction-project/model'

    model = load_model(model_path)
    data = load_dataset(dataset_path)
    display_sidebar()
    tab1, tab2, tab3, tab4 = st.tabs(["1 - 🏠 Présentation", "2 - 📈 Exploratory Data Analysis", "3 - 🏋️‍♂️ Global Performance", "4 - 🏃‍♂️ Prediction"])
    with tab1:
        display_presentation()
    with tab2:
        display_eda(data)
    with tab3:
        display_global_performance(data,path_rapport,path_model)
    with tab4:
        display_prediction(model)

if __name__ == "__main__":
    main()

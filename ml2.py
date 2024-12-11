import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Modèles avec hyperparamètres pour optimisation
MODELS_WITH_PARAMS = {
    "Linear Regression": (LinearRegression(), {}),
    "Ridge Regression": (Ridge(), {"model__alpha": [0.1, 1.0, 10]}),
    "Lasso Regression": (Lasso(), {"model__alpha": [0.1, 1.0, 10]}),
    "Elastic Net": (ElasticNet(), {"model__alpha": [0.1, 1.0, 10], "model__l1_ratio": [0.2, 0.5, 0.8]}),
    "Random Forest": (RandomForestRegressor(), {"model__n_estimators": [100, 200], "model__max_depth": [5, 10]}),
    "Gradient Boosting": (GradientBoostingRegressor(), {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1]}),
    "XGBoost": (XGBRegressor(), {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1]}),
}


# Variable globale pour stocker la base prétraitée
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None


# Page de preprocessing

def preprocessing_page():
    st.title("Présentation des données")
    uploaded_file = st.file_uploader("Chargez un fichier CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.original_data = df  # Sauvegarde des données brutes
        st.markdown("**1. Aperçu des données**")
        n_bins = st.number_input(label = "Choisissez le nombre de lignes à afficher",min_value = 5, value = 5)
        st.write(df.head(n_bins))
        st.write("La base de données contient :", df.shape[0],"lignes et", df.shape[1], "colonnes (variables).")
        st.markdown("**2. Type de variables**")
        # Préparer un DataFrame pour afficher les noms et types de variables
        df_types = pd.DataFrame({"Variables": df.dtypes.index, "Type": df.dtypes.values}).reset_index(drop=True)
        n_bins2 = st.number_input(label = "Choisissez le nombre de variables",min_value = 5, value = 5)
        st.table(df_types.head(n_bins2))
        st.markdown("**3. Valeurs manquantes et doublons**")
        df_types2 = pd.DataFrame({"Variables": df.columns, "Nombre de valeurs manquantes": df.isnull().sum().values}).reset_index(drop=True)  
        n_bins3   = st.number_input(label = "Nombre de variables à afficher", min_value = 5, value = 5)
        st.table(df_types2.head(n_bins3))
        st.write("Nombre de doublons :", df.duplicated().sum())
        st.markdown("**4. Statistiques descriptives**")
        st.write("Analyses descriptives des variables quantitatives:", df.describe())


# Page pour traitement des valeurs manquantes et outliers
def missing_values_page():
    st.title("Valeurs manquantes et outliers")
    if "original_data" in st.session_state:
        df = st.session_state.original_data.copy()
        st.markdown(
            """<div style="text-align: justify;">
            La présente section donne un état des lieux des valeurs manquantes et des valeurs extrêmes et propose des traitements adéquats.
            Pour les variables qualitatives, on impute par le mode si le pourcentage de valeurs manquantes est inférieur au seuil de 10%,
            et pour les variables quantitatives, on impute par la médiane suivant le même seuil. Au dela de 10%,
            on supprime la variable car imputer peu biaiser l'analyse. Les valeurs extrêmes sont traitées par la méthode du Z-scores.
            </div>""", unsafe_allow_html=True)
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_percent = missing_percent.reset_index()  # Transformer en DataFrame pour Plotly
        missing_percent.columns = ["Variables", "Pourcentage"]
        fig = px.bar(
            missing_percent,
            x="Variables",
            y="Pourcentage",
            title="1. Pourcentage de valeurs manquantes par variable",
            labels={"Pourcentage": "Pourcentage", "Variables": "Variables"},
            color="Pourcentage",
            color_continuous_scale="Viridis")

        # Mise à jour de la mise en page du graphique
        fig.update_layout(
            xaxis_tickangle=45,
            height=600)
        st.plotly_chart(fig)

        # Imputation
        qual_vars = df.select_dtypes(include="object").columns
        quant_vars = df.select_dtypes(exclude="object").columns
        if st.button("Traiter les valeurs manquantes"):
            for col in qual_vars:
                df[col].fillna(df[col].mode()[0], inplace=True)
            for col in quant_vars:
                df[col].fillna(df[col].median(), inplace=True)
            st.markdown("**2. Nombre de valeurs manquantes par variable après traitement**")
            df_types3 = pd.DataFrame({"Variables": df.columns, "Nombre de valeurs manquantes": df.isnull().sum().values}).reset_index(drop=True) 
            st.table(df_types3)


        # Traitement des outliers
        if st.button("Outliers (Z-score)"):
            # Filtrer les variables quantitatives
            quantitative_vars = df.select_dtypes(include=["number"])
            # Transformation du DataFrame en format long pour Plotly
            df_long = quantitative_vars.melt(var_name="Variables", value_name="Valeurs")
            # Création du boxplot avec Plotly
            fig = px.box(
                df_long,
                x="Variables",
                y="Valeurs",
                title="3. Box-plots",
                color="Variables",
                color_discrete_sequence=px.colors.qualitative.Vivid
                )
            # Mise à jour de la mise en page pour plus de lisibilité
            fig.update_layout(
                xaxis_title="Variables",
                yaxis_title="Valeurs",
                height=600,
                showlegend=False)
            

            # Supposons que df soit votre DataFrame
            from scipy.stats import zscore
            df_no_outliers = df[(np.abs(zscore(df[quant_vars])) < 3).all(axis=1)]
            # Reshape du DataFrame
            df_melted = df_no_outliers.melt(var_name='Variables', value_name='Valeurs', value_vars=quant_vars)

            # Affichage dans Streamlit
            st.plotly_chart(fig)
            st.session_state.processed_data = df_no_outliers  # Mise à jour de la base prétraitée
            st.write("Après traitement des outliers le jeu de données contient :", df_no_outliers.shape[0], "lignes et", df_no_outliers.shape[1], "colonnes.")

            # Afficher les box-plots pour chaque variable quantitative après le traitement des outliers
            st.markdown("**4. Box-plots après traitement des outliers**")
            # Création du boxplot avec Plotly
            fig = px.box(
                df_melted,
                x="Variables",  # Nom de la variable (qui sera maintenant dans la colonne 'Variables')
                y="Valeurs",    # Valeurs des variables (qui seront maintenant dans la colonne 'Valeurs')
                title="Box-plots des variables quantitatives après traitement des outliers",
                color="Variables",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )

            # Mise à jour de la mise en page pour plus de lisibilité
            fig.update_layout(
                xaxis_title="Variables",
                yaxis_title="Valeurs",
                height=600,
                showlegend=False
            )

            # Affichage du graphique
            st.plotly_chart(fig)
    else:
        st.warning("Veuillez d'abord importer les données.")


# Page de visualisation
# Page de visualisation
def visualization_page():
    st.title("Visualisation des données")
    if st.session_state.get("processed_data") is not None:
        df = st.session_state.processed_data

        st.markdown("**1. Aperçu des données traitées**")
        st.write("Après le traitement des données, le jeu de données contient :", df.shape[0], "lignes et", df.shape[1], "colonnes.")
        st.write(df.head())

        # Séparation des variables quantitatives et qualitatives
        quant_vars = df.select_dtypes(include=["int64", "float64"]).columns
        qual_vars = df.select_dtypes(include=["object", "category"]).columns
        # Multiselect pour choisir les variables quantitatives
        st.markdown("**2. Graphique de corrélations**")
        selected_quant_vars = st.multiselect("Sélectionnez les variables quantitatives pour la heatmap :", quant_vars)

        if selected_quant_vars:
            # Calcul de la matrice de corrélation de Spearman
            corr = df[selected_quant_vars].corr(method="spearman")
            # Masque pour la partie supérieure de la matrice
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
            # Création de la heatmap
            plt.figure(figsize=(10, 7))
            sns.heatmap(corr, cmap='Blues', annot=True, square=True, fmt='.3f',
                        mask=mask, cbar=True, vmin=-1, vmax=1)
            # Affichage dans Streamlit
            st.pyplot(plt)

        # Choix de la variable à visualiser
        st.markdown("**3. Représentation graphique de toutes les variables**")
        var_type = st.radio("Choisissez le type de variable à visualiser :", ["Quantitative", "Qualitative"])
        if var_type == "Quantitative":
            selected_quant_var = st.selectbox("Sélectionnez une variable quantitative :", quant_vars)
            if selected_quant_var:
                plot_type = st.radio("Choisissez un type de graphique :", ["Box Plot", "Histogramme"])
                if plot_type == "Histogramme":
                    bins = st.slider("Nombre de bins pour l'histogramme :", 5, 50, 10)
                    fig = px.histogram(df, x=selected_quant_var, nbins=bins, title=f"Histogramme de {selected_quant_var}")
                    fig.update_traces(marker=dict(line=dict(width=1, color="black")))  # Légère bordure des barres
                    fig.update_layout(bargap=0.2)  # Espacement entre les barres
                    st.plotly_chart(fig)
                elif plot_type == "Box Plot":
                    fig = px.box(df, y=selected_quant_var, title=f"Box Plot de {selected_quant_var}")
                    st.plotly_chart(fig)

        elif var_type == "Qualitative":
            selected_qual_var = st.selectbox("Sélectionnez une variable qualitative :", qual_vars)
            if selected_qual_var:
                modality_count = df[selected_qual_var].nunique()
                if modality_count == 2:
                    plot_type = st.radio("Choisissez un type de graphique :", ["Camembert", "Graphique en Anneau"])
                    counts = df[selected_qual_var].value_counts(normalize=True) * 100  # Proportions en pourcentages
                    if plot_type == "Camembert":
                        fig = px.pie(df, names=counts.index, values=counts.values, title=f"Répartition de {selected_qual_var}")
                        st.plotly_chart(fig)
                    elif plot_type == "Graphique en Anneau":
                        fig = px.pie(df, names=counts.index, values=counts.values, title=f"Répartition de {selected_qual_var}", hole=0.4)
                        st.plotly_chart(fig)
                else:
                    # Barplot pour les variables à plusieurs modalités
                    counts = df[selected_qual_var].value_counts(normalize=True) * 100  # Proportions en pourcentages
                    fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                                 labels={"x": selected_qual_var, "y": "Pourcentage"},
                                 title=f"Barplot de {selected_qual_var}")
                    st.plotly_chart(fig)

        # Après les autres graphiques (ajout en bas de la page de visualisation)
        st.markdown("**4. Nuage de points**")
        # Sélection des variables quantitatives à comparer
        quant_vars = df.select_dtypes(include=["int64", "float64"]).columns
        var_x = st.selectbox("Sélectionnez la variable sur l'axe X :", quant_vars)
        var_y = st.selectbox("Sélectionnez la variable sur l'axe Y :", quant_vars)

        # Si les deux variables sont sélectionnées, afficher le nuage de points avec la droite de régression
        if var_x and var_y:
            # Option pour colorier en fonction d'une variable catégorielle
            qual_vars = df.select_dtypes(include=["object", "category"]).columns
            color_var = st.selectbox("Choisissez une variable catégorielle pour colorier le nuage de points (optionnel) :", qual_vars.tolist() + ["Aucune"])

            # Créer le nuage de points avec Plotly
            if color_var == "Aucune":
                fig = px.scatter(df, x=var_x, y=var_y, title=f"Nuage de points de {var_x} vs {var_y}",
                                trendline="ols", labels={var_x: var_x, var_y: var_y})
            else:
                fig = px.scatter(df, x=var_x, y=var_y, color=color_var, title=f"Nuage de points de {var_x} vs {var_y} par {color_var}",
                                 labels={var_x: var_x, var_y: var_y, color_var: color_var})
            
            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig)

    else:
        st.warning("Veuillez passer par l'étape de traitement des données avant de les visualiser.")


import joblib  # Pour sauvegarder et charger les modèles
import os

best_model_pipeline = None
best_model_name = None
best_model_params = None

def modeling_page():
    global best_model_pipeline, best_model_name, best_model_params

    st.title("Modélisation")
    st.subheader("1. Choix des variables, partition des données et résultats")
    
    if st.session_state.get("processed_data") is not None:
        df = st.session_state.processed_data
        # Sélection des variables
        target = st.selectbox("Sélectionnez la variable cible :", df.columns)
        features = st.multiselect("Sélectionnez les variables explicatives :", df.columns)
        test_size = st.slider("Pourcentage des données de test :", 10, 50, 20) / 100

        if st.button("Exécuter les modèles"):
            X = df[features]
            y = df[target]

            # Traitement des variables catégorielles
            cat_vars = X.select_dtypes(include="object").columns
            num_vars = X.select_dtypes(exclude="object").columns

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_vars),
                    ("cat", OneHotEncoder(), cat_vars),
                ]
            )

            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            results = []
            pipelines = {}

            # Modélisation avec tous les modèles
            for name, (model, params) in MODELS_WITH_PARAMS.items():
                pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(features) - 1)
                results.append({"Model": name, "MSE": mse, "R2": r2, "Adj R2": adj_r2})
                pipelines[name] = pipeline

            results_df = pd.DataFrame(results).sort_values(by=["R2", "MSE", "Adj R2"], ascending=[False, True, False])
            st.markdown("Résultats des modèles :")
            st.table(results_df)

            # Sélection du meilleur modèle
            best_model_name = results_df.iloc[0]["Model"]
            best_model_pipeline = pipelines[best_model_name]
            st.write(f"Meilleur modèle : {best_model_name}")

            # Validation croisée
            st.subheader("2. Validation croisée")
            cross_val_scores = cross_val_score(best_model_pipeline, X, y, cv=5, scoring="r2")
            
            # Créer un DataFrame avec les numéros des scores et les valeurs correspondantes
            scores_df = pd.DataFrame({
                "Score": [f"Score {i+1}" for i in range(len(cross_val_scores))],
                "R2 Value": cross_val_scores
            })

            # Créer un histogramme avec Plotly
            fig = px.bar(
                scores_df,
                x="Score",  # Les labels des scores (Score 1, Score 2, etc.)
                y="R2 Value",  # Les valeurs des scores R2
                title="Scores R2 par Validation Croisée",
                labels={"Score": "Score", "R2 Value": "Valeur du Score R2"},
            )

            # Mettre à jour la mise en page
            fig.update_layout(
                xaxis_title="Scores",
                yaxis_title="Valeur R2",
                showlegend=False
            )

            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig)
            st.write("Moyenne des scores R2 :", np.mean(cross_val_scores))

            # Optimisation des hyperparamètres
            st.subheader("3. Optimisation des hyperparamètres")
            model, params = MODELS_WITH_PARAMS[best_model_name]
            grid_search = GridSearchCV(
                estimator=Pipeline(steps=[("preprocessor", preprocessor), ("model", model)]),
                param_grid=params,
                cv=5,
                scoring="r2",
            )
            grid_search.fit(X, y)
            best_model_params = grid_search.best_params_
            st.write("Meilleurs paramètres :", best_model_params)

            # Sauvegarder le meilleur modèle et ses paramètres
            st.write("Sauvegarde du meilleur modèle...")
            model_filename = "best_model_pipeline.pkl"
            params_filename = "best_model_params.pkl"

            # Mise à jour des variables globales
            best_model_pipeline = best_model_pipeline
            best_model_name     = best_model_name
            best_model_params   = grid_search.best_params_
            
            # Sauvegarde du pipeline et des paramètres
            joblib.dump(best_model_pipeline, model_filename)
            joblib.dump(best_model_params, params_filename)

            
            st.write(f"Modèle sauvegardé dans '{model_filename}' et paramètres dans '{params_filename}'.")

            # Courbes d'apprentissage
            st.subheader("4. Courbes d'apprentissage")
            train_sizes, train_scores, test_scores = learning_curve(
                best_model_pipeline, X, y, cv=5, scoring="r2", train_sizes=np.linspace(0.1, 1.0, 10)
            )
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)  # Écart type des scores d'entraînement
            test_scores_std = np.std(test_scores, axis=1)  # Écart type des scores de validation

            # Création du graphique
            fig, ax = plt.subplots(figsize=(10, 6))

            # Courbes pour les scores d'entraînement et de validation
            ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
            ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation")

            # Zones d'intervalle de confiance (écart type)
            ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                            alpha=0.1, color="r")
            ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                            alpha=0.1, color="g")

            # Titres et labels
            ax.set_title("Courbe d'apprentissage")
            ax.set_xlabel("Taille des données d'entraînement")
            ax.set_ylabel("Score R2")
            ax.legend(loc="best")

            # Affichage dans Streamlit
            st.pyplot(fig)

# Page de prédiction mise à jour
import pandas as pd
import streamlit as st

# Page de prédiction
def prediction_page(shared_data):
    st.title("Prédiction")

    # Récupération des données partagées
    best_model_pipeline = shared_data.get("best_model_pipeline")
    best_model_name = shared_data.get("best_model_name")
    best_model_params = shared_data.get("best_model_params")
    df = shared_data.get("df")

    # Vérification de l'état de la modélisation
    if best_model_pipeline is None:
        st.warning("Veuillez d'abord exécuter la modélisation pour choisir un modèle.")
        return

    # Affichage des informations du modèle
    st.write(f"**Meilleur modèle** : {best_model_name}")
    st.write("**Meilleurs paramètres** :", best_model_params)

    # Identification des types de variables
    if df is not None:
        categorical_features = df.select_dtypes(include="object").columns.tolist()
        numerical_features = df.select_dtypes(exclude="object").columns.tolist()
    else:
        st.error("Les données ne sont pas disponibles. Veuillez vérifier la page de prétraitement.")
        return

    # Initialisation du dictionnaire pour les entrées utilisateur
    inputs = {}

    # Création des widgets pour les variables catégorielles
    st.subheader("Entrez les valeurs pour les variables")
    st.write("### Variables catégorielles")
    for col in categorical_features:
        unique_values = df[col].dropna().unique()
        inputs[col] = st.selectbox(f"{col} :", options=unique_values)

    # Création des widgets pour les variables numériques
    st.write("### Variables numériques")
    for col in numerical_features:
        default_value = round(df[col].mean(), 2) if df[col].dtype in ['float64', 'int64'] else 0.0
        inputs[col] = st.number_input(f"{col} :", value=default_value)

    # Bouton pour effectuer la prédiction
    if st.button("Prédire"):
        try:
            # Conversion des entrées en DataFrame
            input_data = pd.DataFrame([inputs])

            # Assurer la cohérence des types de données
            input_data[numerical_features] = input_data[numerical_features].apply(pd.to_numeric, errors='coerce')
            input_data[categorical_features] = input_data[categorical_features].astype(str)

            # Vérification des valeurs invalides
            if input_data.isnull().any().any():
                st.error("Certaines entrées contiennent des valeurs invalides. Veuillez corriger les données.")
                return

            # Faire la prédiction
            try:
                # Exécution de la prédiction
                prediction = best_model_pipeline.predict(input_data)
                st.success(f"Prédiction : {prediction[0]}")
            except ValueError as ve:
                st.error(f"Erreur de validation des données : {ve}")
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")

            # Affichage des probabilités si applicable
            model = best_model_pipeline.named_steps.get("model")
            if model and hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)
                st.write("### Probabilités par classe")
                st.write(probabilities)

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la prédiction : {e}")
# Structure de l'application
PAGES = {
    "Présentation de données": preprocessing_page,
    "Traitement de données": missing_values_page,
    "Data visualisation": visualization_page,
    "Modélisation": modeling_page,
    "Prédiction": lambda: prediction_page(shared_data),
}

def main():
    st.sidebar.title("Machine Learning App")
    page = st.sidebar.radio("Navigation", list(PAGES.keys()))
    PAGES[page]()

if __name__ == "__main__":
    # Dictionnaire partagé pour centraliser les données
    shared_data = {
        "df": None,  # DataFrame principal
        "best_model_pipeline": None,
        "best_model_name": None,
        "best_model_params": None,
    }
    main()

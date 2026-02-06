# Projet de Prédiction des Risques de Crédit

Ce projet a pour objectif d'analyser les données de crédit allemandes ("German Credit Data") afin de prédire la solvabilité des clients et d'évaluer le risque de crédit (Bon ou Mauvais).

Le projet se décompose en deux parties principales :
1.  **Analyse et Modélisation (`sample.ipynb`)** : Exploration des données, nettoyage, ingénierie des fonctionnalités et entraînement de plusieurs modèles de Machine Learning.
2.  **Application Web (`app.py`)** : Une interface utilisateur interactive créée avec Streamlit pour effectuer des prédictions sur de nouveaux profils clients.

## 📂 Structure du Projet

*   `sample.ipynb` : Notebook Jupyter contenant l'analyse exploratoire (EDA), la visualisation, le pré-traitement des données et l'entraînement des modèles (Decision Tree, Random Forest, Extra Trees, XGBoost).
*   `app.py` : Script de l'application Streamlit qui charge le modèle entraîné et permet à l'utilisateur de saisir les informations d'un client pour obtenir une prédiction.
*   `uv.lock` : Fichier de verrouillage des dépendances géré par `uv`, assurant la reproductibilité de l'environnement.
*   `models/` : Dossier (généré par le notebook) contenant les fichiers sérialisés (`.pkl`) du modèle final (`extra_tree_credit_model.pkl`) et des encodeurs nécessaires au fonctionnement de l'application.

## 🛠️ Outils et Technologies Utilisés

Voici un résumé des outils et bibliothèques utilisés dans ce projet :

*   **Python** : Langage de programmation principal du projet.
*   **Pandas** : Utilisé pour la manipulation, le nettoyage et l'analyse des données tabulaires (DataFrames).
*   **NumPy** : Bibliothèque fondamentale pour le calcul scientifique et la manipulation de tableaux multidimensionnels.
*   **Matplotlib & Seaborn** : Bibliothèques de visualisation de données utilisées pour créer des graphiques (histogrammes, boîtes à moustaches, matrices de corrélation) afin de comprendre les distributions et les relations entre les variables.
*   **Scikit-learn (sklearn)** : Bibliothèque de Machine Learning fournissant des outils pour le pré-traitement (LabelEncoder), la séparation des données (train_test_split), les métriques d'évaluation et les algorithmes de classification (Decision Tree, Random Forest, Extra Trees).
*   **XGBoost** : Bibliothèque optimisée de boosting de gradient, utilisée comme l'un des modèles de classification performants.
*   **Streamlit** : Framework open-source permettant de créer rapidement des applications web interactives pour la Data Science et le Machine Learning.
*   **Joblib** : Outil utilisé pour sauvegarder (sérialiser) et charger le modèle entraîné et les encodeurs, permettant leur réutilisation dans l'application web.
*   **Jupyter** : Environnement de développement interactif utilisé pour l'exploration des données et le prototypage.

## 🚀 Installation et Exécution

1.  **Installation des dépendances** :
    Assurez-vous d'avoir Python installé. Ce projet utilise `uv` pour la gestion des dépendances, mais vous pouvez installer les bibliothèques nécessaires via pip :
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib
    ```

2.  **Lancer l'application** :
    Pour démarrer l'interface web, exécutez la commande suivante dans votre terminal à la racine du projet :
    ```bash
    streamlit run app.py
    ```

## 📊 Modélisation

Le notebook explore plusieurs algorithmes pour déterminer le meilleur modèle :
*   Arbre de décision (Decision Tree)
*   Forêt aléatoire (Random Forest)
*   Extra Trees (Modèle retenu pour l'application)
*   XGBoost

Les hyperparamètres ont été optimisés via `GridSearchCV` pour maximiser la précision (accuracy).

# Modèle de Scoring pour la Prédiction de Faillite Client

Ce projet s'inscrit dans le cadre d'une **formation OpenClassrooms** et vise à développer, déployer et gérer un modèle de scoring prédictif basé sur les données de la compétition **Kaggle - Home Credit Default Risk**.  
L'objectif principal est de prédire la probabilité de défaut de paiement d'un client, tout en assurant une transparence et une gestion optimale du cycle de vie du modèle grâce aux pratiques MLOps.

## Objectifs du Projet

1. **Développement du Modèle de Scoring**  
   Construire un modèle de machine learning pour estimer automatiquement la probabilité de défaut de paiement d'un client, en exploitant les données riches et variées issues de la compétition Kaggle.

2. **Analyse des Features**  
   - Identifier les **features importantes** qui influencent le modèle à un niveau global (feature importance globale).  
   - Fournir des interprétations locales des scores pour expliquer les prédictions au niveau individuel (feature importance locale).  
   Ces analyses favorisent la **transparence** et permettent aux analystes d'expliquer les scores aux parties prenantes.

3. **Déploiement et API**  
   - Mettre en production le modèle via une **API RESTful**.  
   - Concevoir une **interface utilisateur de test** pour valider et démontrer l’utilisation opérationnelle des prédictions.

4. **Approche MLOps de bout en bout**  
   - Suivi des expérimentations avec un outil comme **MLFlow** pour centraliser les modèles et leurs métriques.  
   - Gestion du déploiement et analyse en production des dérives de données (**data drift**) pour garantir une performance optimale dans le temps.

## Contexte : Compétition Kaggle - Home Credit Default Risk

Les données utilisées dans ce projet proviennent de la compétition Kaggle **[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)**. Elles incluent des informations variées sur les clients, telles que :  
- Les antécédents financiers.  
- Les informations socio-démographiques.  
- Les données comportementales et contractuelles.

## Structure du Projet

- **`notebooks/`** : Contient les notebooks pour le développement initial, les analyses exploratoires et les expérimentations.  
- **`src/`** : Code source pour les étapes de prétraitement, d'entraînement, d'analyse des features et de déploiement.  
- **`models/`** : Répertoire pour stocker les modèles entraînés et sauvegardés.  
- **`api/`** : Scripts pour créer et gérer l'API RESTful du modèle.  
- **`mlops/`** : Scripts et configurations pour le suivi des expérimentations et la gestion des dérives en production.  

## Prérequis

- **Python 3.11**  
- Bibliothèques nécessaires :  
  - `pandas`, `numpy`, `scikit-learn`, `lightgbm`  
  - `shap` (pour l'analyse des features)  
  - `MLFlow` (pour le tracking des expérimentations)  
  - `Flask` ou `FastAPI` (pour le déploiement)  
- **Docker** : pour contenir et déployer le modèle.  

## Instructions

1. **Préparation et Entraînement du Modèle** :  
   - Préparez les données avec les scripts disponibles dans `src/data_preprocessing`, la préparation des données et le feature engineering est issus du [kernels Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)
   - Entraînez et évaluez le modèle avec les notebooks dans `notebooks/`.

2. **Analyse des Features** :  
   - Utilisez les outils dans `src/feature_analysis` pour générer les interprétations globales et locales.  
   - Visualisez les résultats à l’aide des bibliothèques comme `shap`.

3. **Déploiement** :  
   - Lancez l’API RESTful en utilisant les scripts dans `api/`.  
   - Testez l’API avec l’interface utilisateur incluse ou via des outils comme **Postman**.

4. **Approche MLOps** :  
   - Configurez MLFlow pour suivre les expérimentations et versionner les modèles.

## Formation OpenClassrooms

Ce projet est réalisé dans le cadre de la formation **[Openclassrooms - Implémentez un modèle de scoring](https://openclassrooms.com/fr/paths/164-data-scientist)**, qui inclut l'application des meilleures pratiques en science des données et en déploiement de modèles.

## Auteur

Projet réalisé par Mustapha OUMAHMOUD

# -*- coding: utf-8 -*-
import gc
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler

from src.utils.build_pipeline import build_pipeline
from src.utils.common_functions import normalize_column_names, custom_business_scorer, \
    optimal_threshold_scorer

# URI du serveur MLflow distant
remote_server_ui = "http://localhost:5000/"
mlflow.set_tracking_uri(remote_server_ui)

# Fonction pour entraîner le modèle
def kfold_dummyclassifier_with_gridsearch(df, num_vars, cat_vars, num_folds):
    """
    Entraîne un modèle DummyClassifier avec validation k-fold,
    choisit dynamiquement le meilleur paramètre 'strategy' via GridSearchCV,
    et enregistre les résultats dans MLflow.
    """
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    print(f"Starting DummyClassifier with GridSearchCV. Train shape: {train_df.shape}, test shape: {test_df.shape}")

    del df
    gc.collect()


    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)


    # Définition des caractéristiques
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Exemple d'entrée pour MLflow
    input_example = train_df[feats].iloc[:5].copy()

    # Signature pour MLflow
    from mlflow.models.signature import infer_signature
    model_signature = infer_signature(input_example, pd.Series(np.zeros(5)))

    # Définir la date et l'heure pour personnaliser les noms des runs
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name=f'DummyClassifier_GridSearch_{current_time}'):
        # Paramètres à tester via GridSearchCV
        param_grid = {'regressor__strategy': ["most_frequent", "prior", "stratified", "uniform"]}

        # Créer le scorer avec `make_scorer`
        business_scorer = make_scorer(
            custom_business_scorer,
            greater_is_better=False,  # Un coût plus faible est meilleur
            needs_proba=True
        )

        # Scorer pour GridSearchCV
        threshold_scorer = make_scorer(
            optimal_threshold_scorer,
            greater_is_better=False,  # On cherche à minimiser le coût métier
            needs_proba=True
        )

        # Générer le pipeline modèle
        model = build_pipeline(algo_ml= DummyClassifier(),
                               impute_num=SimpleImputer(strategy="mean"),
                               impute_var=SimpleImputer(strategy="most_frequent"), scaler=RobustScaler(),
                               num_vars = num_vars, cat_vars = cat_vars)

        # GridSearchCV
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=business_scorer,
            cv=folds,
            n_jobs=-1,
            verbose=1
        )

        # Entraînement
        X = train_df[feats]
        y = train_df['TARGET']
        grid.fit(X, y)

        # Meilleur modèle et paramètres
        best_params = grid.best_params_
        best_score = grid.best_score_
        print(f"Best strategy: {best_params}, Best score: {best_score:.2f}")
        mlflow.log_params(best_params)
        mlflow.log_metric("Best_Business_Score", best_score)

        # Courbe ROC pour le meilleur modèle
        best_model = grid.best_estimator_
        oof_preds = best_model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, oof_preds)
        full_auc = roc_auc_score(y, oof_preds)
        print(f'Full AUC score: {full_auc:.6f}')

        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % full_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")

        # Journalisation dans MLflow
        mlflow.sklearn.log_model(best_model)


if __name__ == "__main__":
    data = pd.read_csv("../../data/preprocessed_data.csv")
    data.columns = (
        data.columns
        .str.replace(r'[^A-Za-z0-9_]', '_', regex=True)
        .str.replace(r'__+', '_', regex=True)
        .str.strip('_')
    )
    num_vars = data.drop(['TARGET', 'SK_ID_CURR'], axis=1).select_dtypes(exclude=['object']).columns
    cat_vars = data.drop(['TARGET', 'SK_ID_CURR'], axis=1).drop(num_vars, axis=1).columns
    # Correction des noms des colonnes et le type des données
    data = normalize_column_names(data)
    # Entraînement du modèle avec recherche de grille
    kfold_dummyclassifier_with_gridsearch(data, num_vars, cat_vars, num_folds=10)

# -*- coding: utf-8 -*-
import gc
import os
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

from src.utils.common_functions import normalize_column_names, business_cost

# URI du serveur MLflow distant
remote_server_ui = "http://localhost:5000/"
mlflow.set_tracking_uri(remote_server_ui)


# Fonction pour entraîner le modèle
def train_xgboost(df, num_folds, use_smote=False):
    """
    Entraîne un modèle XGBoost avec validation k-fold, GridSearchCV, et une option pour gérer le déséquilibre des classes.

    :param df: Pandas DataFrame contenant les données.
    :param num_folds: Nombre de folds pour la validation croisée.
    :param use_smote: Si True, utilise SMOTE pour équilibrer les classes ; sinon, utilise StratifiedKFold.
    """
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    print(f"Starting XGBoost. Train shape: {train_df.shape}, test shape: {test_df.shape}")

    del df
    gc.collect()

    # Initialisation
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Définir la date et l'heure pour personnaliser les noms des runs
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name=f'XGBoost_{"SMOTE" if use_smote else "StratifiedKFold"}_{current_time}'):
        # Créer le répertoire pour les artefacts
        metrics_dir = "artifacts/metrics"
        os.makedirs(metrics_dir, exist_ok=True)

        # Enregistrer les datasets comme artefacts
        train_path = os.path.join(metrics_dir, "train_data.csv")
        test_path = os.path.join(metrics_dir, "test_data.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)

        # Définir le modèle et les hyperparamètres pour GridSearchCV
        base_model = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='auc',
            n_jobs=-1
        )
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [4, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Fonction de scoring pour GridSearchCV
        scoring = make_scorer(business_cost, needs_proba=True, greater_is_better=False)

        if use_smote:
            # KFold classique (pas stratifié) pour SMOTE
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
        else:
            # StratifiedKFold pour gérer le déséquilibre des classes
            folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)

        for fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            print(f"Fold {fold + 1} started.")

            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

            if use_smote:
                # Appliquer SMOTE pour équilibrer les classes
                smote = SMOTE(random_state=1001)
                train_x, train_y = smote.fit_resample(train_x, train_y)

            # Appliquer GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=scoring,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=1001),
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(train_x, train_y)

            best_model = grid_search.best_estimator_
            oof_preds[valid_idx] = best_model.predict_proba(valid_x)[:, 1]

            print(f"Fold {fold + 1} completed.")

        # Évaluer les performances globales
        full_auc = roc_auc_score(train_df['TARGET'], oof_preds)
        full_business_cost = business_cost(train_df['TARGET'], oof_preds)

        print(f'Full AUC score: {full_auc:.6f}, Full Business Cost: {full_business_cost:.2f}')
        mlflow.log_metric("Full_AUC", full_auc)
        mlflow.log_metric("Full_Business_Cost", full_business_cost)

        # Courbe ROC
        fpr, tpr, _ = roc_curve(train_df['TARGET'], oof_preds)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % full_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        roc_curve_path = os.path.join(metrics_dir, "roc_curve.png")
        plt.savefig(roc_curve_path)
        mlflow.log_artifact(roc_curve_path)
        plt.close()

        # Enregistrer le modèle final
        mlflow.xgboost.log_model(best_model, "model")

    return feature_importance_df


if __name__ == "__main__":
    data = pd.read_csv("../../data/preprocessed_data.csv")
    # Correction des noms des colonnes
    data = normalize_column_names(data)
    # Entraînement du modèle avec SMOTE ou StratifiedKFold
    use_smote = True  # Passer à False pour utiliser StratifiedKFold
    feature_importance = train_xgboost(data, num_folds=10, use_smote=use_smote)

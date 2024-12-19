# -*- coding: utf-8 -*-
import gc
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from src.utils.common_functions import normalize_column_names, business_cost, custom_business_scorer, optimal_threshold_scorer

# URI du serveur MLflow distant
remote_server_ui = "http://localhost:5000/"
mlflow.set_tracking_uri(remote_server_ui)

# Fonction pour entraîner le modèle
def kfold_lightgbm(df, num_folds=10, stratified=True):
    """
    Entraîne un modèle LightGBM avec validation k-fold et enregistre les résultats dans MLflow.
    """
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    print(f"Starting LightGBM. Train shape: {train_df.shape}, test shape: {test_df.shape}")

    del df
    gc.collect()

    # Utilisation de StratifiedKFold
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)

    # Initialisation
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Exemple d'entrée pour MLflow
    input_example = train_df[feats].iloc[:5].copy()

    # Signature pour MLflow
    from mlflow.models.signature import infer_signature
    model_signature = infer_signature(input_example, pd.Series(np.zeros(5)))

    # Définir la date et l'heure pour personnaliser les noms des runs
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name=f'LightGBM_{current_time}'):
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

        # Paramètres du modèle
        model_params = {
            'nthread': 4,
            'n_estimators': 10000,
            'learning_rate': 0.02,
            'num_leaves': 34,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 39.3259775,
            'silent': -1,
            'verbose': -1
        }
        mlflow.log_params(model_params)

        thresholds = np.arange(0.1, 1.0, 0.1)
        optimal_thresholds = []

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

            # Initialisation du modèle
            clf = LGBMClassifier(**model_params)

            start_time = time.time()
            clf.fit(
                train_x,
                train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc',
                callbacks=[
                    early_stopping(stopping_rounds=200),
                    log_evaluation(200)
                ]
            )
            fit_time = time.time() - start_time

            # Prédictions
            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
            predict_time = time.time() - start_time

            # Calcul des métriques
            auc = roc_auc_score(valid_y, oof_preds[valid_idx])
            business_cost_fold = business_cost(valid_y, oof_preds[valid_idx], threshold=0.5)
            optimal_threshold = optimal_threshold_scorer(clf, valid_x, valid_y, thresholds)
            optimal_thresholds.append(optimal_threshold)

            print(f'Fold {n_fold + 1} AUC: {auc:.6f}, Business Cost: {business_cost_fold:.2f}, Optimal Threshold: {optimal_threshold:.2f}')

            mlflow.log_metric(f"AUC_Fold_{n_fold + 1}", auc)
            mlflow.log_metric(f"Business_Cost_Fold_{n_fold + 1}", business_cost_fold)
            mlflow.log_metric(f"Fit_Time_Fold_{n_fold + 1}", fit_time)
            mlflow.log_metric(f"Predict_Time_Fold_{n_fold + 1}", predict_time)
            mlflow.log_metric(f"Optimal_Threshold_Fold_{n_fold + 1}", optimal_threshold)

            del train_x, train_y, valid_x, valid_y
            gc.collect()

        # Calcul des métriques globales
        full_auc = roc_auc_score(train_df['TARGET'], oof_preds)
        full_business_cost = business_cost(train_df['TARGET'], oof_preds, threshold=0.5)
        avg_optimal_threshold = np.mean(optimal_thresholds)
        print(f'Full AUC score: {full_auc:.6f}, Full Business Cost: {full_business_cost:.2f}, Avg Optimal Threshold: {avg_optimal_threshold:.2f}')

        mlflow.log_metric("Full_AUC", full_auc)
        mlflow.log_metric("Full_Business Cost", full_business_cost)
        mlflow.log_metric("Avg Optimal Threshold", avg_optimal_threshold)

        # Enregistrer les artefacts et courbes
        feature_importance_df.to_csv(os.path.join(metrics_dir, "feature_importance.csv"), index=False)
        mlflow.log_artifact(os.path.join(metrics_dir, "feature_importance.csv"))

        # Courbe ROC
        fpr, tpr, _ = roc_curve(train_df['TARGET'], oof_preds)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % full_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # Random prediction line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        roc_curve_path = os.path.join(metrics_dir, "roc_curve.png")
        plt.savefig(roc_curve_path)
        mlflow.log_artifact(roc_curve_path)

        # Enregistrer le modèle final
        mlflow.lightgbm.log_model(
            clf,
            artifact_path="model",
            signature=model_signature,
            input_example=input_example
        )

if __name__ == "__main__":
    data = pd.read_csv("../../data/preprocessed_data.csv")
    data = normalize_column_names(data)
    kfold_lightgbm(data, num_folds=10, stratified=True)

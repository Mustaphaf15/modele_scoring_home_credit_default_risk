# -*- coding: utf-8 -*-
import gc
from datetime import datetime
import time
import re
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

from src.utils.common_functions import normalize_column_names, business_cost

# URI du serveur MLflow distant
remote_server_ui = "http://localhost:5000/"
mlflow.set_tracking_uri(remote_server_ui)


# Fonction pour entraîner le modèle
def kfold_lightgbm(df, num_folds, stratified=False):
    """
    Entraîne un modèle LightGBM avec validation k-fold et enregistre les résultats dans MLflow.
    """
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    print(f"Starting LightGBM. Train shape: {train_df.shape}, test shape: {test_df.shape}")

    del df
    gc.collect()

    # Choisir le type de KFold
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    # Initialisation
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Définir la date et l'heure pour personnaliser les noms des runs
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name=f'LightGBM_{current_time}'):
        # Enregistrer les datasets comme artefacts
        train_df.to_csv("../../train_data.csv", index=False)
        test_df.to_csv("../../test_data.csv", index=False)
        mlflow.log_artifact("../../train_data.csv")
        mlflow.log_artifact("../../test_data.csv")

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

            # Initialisation du modèle
            clf = LGBMClassifier(
                nthread=4,
                n_estimators=10000,
                learning_rate=0.02,
                num_leaves=34,
                colsample_bytree=0.9497036,
                subsample=0.8715623,
                max_depth=8,
                reg_alpha=0.041545473,
                reg_lambda=0.0735294,
                min_split_gain=0.0222415,
                min_child_weight=39.3259775,
                silent=-1,
                verbose=-1
            )

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
            start_time = time.time()
            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            predict_time = time.time() - start_time

            # Importance des features
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            # Calcul des métriques
            auc = roc_auc_score(valid_y, oof_preds[valid_idx])
            business_cost_fold = business_cost(valid_y, oof_preds[valid_idx])
            print(f'Fold {n_fold + 1} AUC: {auc:.6f}, Business Cost: {business_cost_fold:.2f}')

            mlflow.log_metric(f"AUC_Fold_{n_fold + 1}", auc)
            mlflow.log_metric(f"Business_Cost_Fold_{n_fold + 1}", business_cost_fold)
            mlflow.log_metric(f"Fit_Time_Fold_{n_fold + 1}", fit_time)
            mlflow.log_metric(f"Predict_Time_Fold_{n_fold + 1}", predict_time)

            #del clf, train_x, train_y, valid_x, valid_y
            #gc.collect()

        # Enregistrer les métriques globales
        full_auc = roc_auc_score(train_df['TARGET'], oof_preds)
        full_business_cost = business_cost(train_df['TARGET'], oof_preds)
        print(f'Full AUC score: {full_auc:.6f}, Full Business Cost: {full_business_cost:.2f}')

        mlflow.log_metric("Full_AUC", full_auc)
        mlflow.log_metric("Full_Business_Cost", full_business_cost)

        # Courbe ROC
        fpr, tpr, _ = roc_curve(train_df['TARGET'], oof_preds)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % full_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # Random prediction line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")

        # Enregistrer le modèle final
        mlflow.lightgbm.log_model(clf, "model")

    return feature_importance_df


if __name__ == "__main__":
    data = pd.read_csv("../../data/preprocessed_data.csv")
    # Correction des noms des columns et le type des données
    data = normalize_column_names(data)
    # Entraînement du modèle
    feature_importance = kfold_lightgbm(data, num_folds=10, stratified=True)

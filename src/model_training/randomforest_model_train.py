# -*- coding: utf-8 -*-
import gc
import os
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer  # Importer SimpleImputer pour l'imputation des valeurs manquantes
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV

from src.utils.common_functions import normalize_column_names, business_cost

# URI du serveur MLflow distant
remote_server_ui = "http://localhost:5000/"
mlflow.set_tracking_uri(remote_server_ui)

# Fonction pour entraîner le modèle
def train_random_forest(df, num_folds, use_smote=False):
    """
    Entraîne un modèle RandomForestClassifier avec validation k-fold, RandomizedSearchCV, et une option pour gérer le déséquilibre des classes.

    :param df: Pandas DataFrame contenant les données.
    :param num_folds: Nombre de folds pour la validation croisée.
    :param use_smote: Si True, utilise SMOTE pour équilibrer les classes ; sinon, utilise StratifiedKFold.
    """
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    print(f"Starting Random Forest. Train shape: {train_df.shape}, test shape: {test_df.shape}")

    del df
    gc.collect()

    # Initialisation
    oof_preds = np.zeros(train_df.shape[0])
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Imputer pour les valeurs manquantes
    imputer = SimpleImputer(strategy='mean')  # Utilisation de la moyenne pour imputer les valeurs manquantes

    # Définir la date et l'heure pour personnaliser les noms des runs
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name=f'RandomForest_{"SMOTE" if use_smote else "StratifiedKFold"}_{current_time}'):
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

        del test_df
        gc.collect()

        # Définir le modèle et les hyperparamètres pour RandomizedSearchCV
        base_model = RandomForestClassifier(
            random_state=1001,
            n_jobs=-1
        )
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Fonction de scoring pour RandomizedSearchCV
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

            # Imputer les valeurs manquantes
            train_x = imputer.fit_transform(train_x)
            valid_x = imputer.transform(valid_x)

            if use_smote:
                # Appliquer SMOTE pour équilibrer les classes
                smote = SMOTE(random_state=1001)
                train_x, train_y = smote.fit_resample(train_x, train_y)

            # Appliquer RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_dist,
                scoring=scoring,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=1001),
                n_jobs=-1,
                n_iter=50,  # Nombre d'itérations de la recherche aléatoire
                verbose=1,
                random_state=1001
            )
            random_search.fit(train_x, train_y)

            best_model = random_search.best_estimator_
            oof_preds[valid_idx] = best_model.predict_proba(valid_x)[:, 1]

            print(f"Fold {fold + 1} completed.")
            del train_x, train_y, valid_x, valid_y
            gc.collect()

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
        mlflow.sklearn.log_model(best_model, "model")


if __name__ == "__main__":
    data = pd.read_csv("../../data/preprocessed_data.csv")
    # Correction des noms des colonnes
    data = normalize_column_names(data)
    # Entraînement du modèle avec SMOTE ou StratifiedKFold
    apply_smote = True  # Passer à False pour utiliser StratifiedKFold
    train_random_forest(data, num_folds=10, use_smote=apply_smote)

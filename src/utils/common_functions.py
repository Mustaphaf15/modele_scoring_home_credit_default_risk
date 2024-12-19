import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def normalize_column_names(df):
    """
    Normalise les noms de colonnes en supprimant les caractères spéciaux et en unifiant le format.
    """
    df.columns = (
        df.columns
        .str.replace(r'[^A-Za-z0-9_]', '_', regex=True)
        .str.replace(r'__+', '_', regex=True)
        .str.strip('_')
    )
    # Liste des colonnes avec les données True, False et Nan
    column_bool = df.select_dtypes(include='object').columns
    df[column_bool] = df[column_bool].apply(
        lambda col: col.map({True: 1, False: 0}) if col.isin([True, False]).any() else col)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Sélectionner les colonnes de type float64
    # numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.96 GiB for an array with shape (339321, 774) and data type float64
    float_cols = df.select_dtypes(include=['float64']).columns

    # Convertir ces colonnes en float32
    df[float_cols] = df[float_cols].astype(np.float32)
    return df

def business_cost(y_true, y_pred_probs, threshold=0.5, cost_fn=10, cost_fp=1):
    # Convertir les probabilités en classes avec le seuil donné
    y_pred = (y_pred_probs >= threshold).astype(int)
    # Calculer la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Coût total
    total_cost = cost_fn * fn + cost_fp * fp
    return total_cost

def custom_business_scorer(estimator, X, y, cost_fn=10, cost_fp=1):
    # Obtenir les probabilités prédites pour la classe positive
    y_pred_probs = estimator.predict_proba(X)[:, 1]

    # Calculer le coût métier
    return business_cost(y, y_pred_probs, threshold=0.5, cost_fn=cost_fn, cost_fp=cost_fp)

def optimal_threshold_scorer(estimator, X, y, thresholds=np.arange(0.1, 1.0, 0.1), cost_fn=10, cost_fp=1):
    y_pred_probs = estimator.predict_proba(X)[:, 1]
    costs = [business_cost(y, y_pred_probs, threshold=t, cost_fn=cost_fn, cost_fp=cost_fp) for t in thresholds]
    optimal_threshold = thresholds[np.argmin(costs)]
    return optimal_threshold
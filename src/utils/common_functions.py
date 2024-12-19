import pandas as pd
import numpy as np

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

def business_cost(y_true, y_proba, threshold=0.5):
    """
    Calcule le coût métier basé sur les FN et FP.
    """
    y_pred = (y_proba >= threshold).astype(int)
    confusion = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    FN = confusion.loc[1, 0] if (1, 0) in confusion else 0
    FP = confusion.loc[0, 1] if (0, 1) in confusion else 0
    cost = 10 * FN + FP
    return cost
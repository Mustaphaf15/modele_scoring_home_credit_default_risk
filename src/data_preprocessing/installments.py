# -*- coding: utf-8 -*-
import gc
import pandas as pd
from .utils import one_hot_encoder

def installments(num_rows=None, nan_as_category=False):
    # Charger les données du fichier installments_payments
    installments = pd.read_csv('./data/installments_payments.csv', nrows=num_rows)
    print(f"Nombre d'échantillons dans installments_payments : {len(installments)}")

    # Encodage des variables binaires
    for bin_feature in ['SK_ID_CURR', 'SK_ID_PREV']:
        installments[bin_feature], uniques = pd.factorize(installments[bin_feature])

    # Appliquer l'encodage One-Hot
    installments, cat_cols = one_hot_encoder(installments, nan_as_category)

    del installments
    gc.collect()
    return installments

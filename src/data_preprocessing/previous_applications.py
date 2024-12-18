# -*- coding: utf-8 -*-
import gc
import pandas as pd
from .utils import one_hot_encoder

def previous_applications(num_rows=None, nan_as_category=False):
    # Charger les données du fichier previous_application
    prev_app = pd.read_csv('./data/previous_application.csv', nrows=num_rows)
    print(f"Nombre d'échantillons dans previous_application : {len(prev_app)}")

    # Encodage des variables binaires
    for bin_feature in ['NAME_CONTRACT_TYPE', 'CODE_GENDER']:
        prev_app[bin_feature], uniques = pd.factorize(prev_app[bin_feature])

    # Appliquer l'encodage One-Hot
    prev_app, cat_cols = one_hot_encoder(prev_app, nan_as_category)

    # Ajouter de nouvelles fonctionnalités simples
    prev_app['DAYS_DECISION_PERC'] = prev_app['DAYS_DECISION'] / prev_app['DAYS_BIRTH']
    prev_app['CREDIT_TERM'] = prev_app['AMT_CREDIT'] / prev_app['AMT_ANNUITY']

    # Supprimer les variables inutiles
    prev_app.drop(columns=['SK_ID_CURR'], inplace=True)

    del prev_app
    gc.collect()
    return prev_app

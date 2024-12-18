# -*- coding: utf-8 -*-
import gc
import pandas as pd
from .utils import one_hot_encoder


def pos_cash(num_rows=None, nan_as_category=False):
    # Charger les données du fichier POS_CASH_balance
    pos_cash = pd.read_csv('./data/POS_CASH_balance.csv', nrows=num_rows)
    print(f"Nombre d'échantillons dans POS_CASH_balance : {len(pos_cash)}")

    # Encodage des variables binaires
    for bin_feature in ['NAME_CONTRACT_STATUS', 'SK_ID_PREV']:
        pos_cash[bin_feature], uniques = pd.factorize(pos_cash[bin_feature])

    # Appliquer l'encodage One-Hot
    pos_cash, cat_cols = one_hot_encoder(pos_cash, nan_as_category)

    # Supprimer les variables inutiles
    pos_cash.drop(columns=['SK_ID_PREV'], inplace=True)

    del pos_cash
    gc.collect()
    return pos_cash

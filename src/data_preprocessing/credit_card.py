# -*- coding: utf-8 -*-
import gc
import pandas as pd
from .utils import one_hot_encoder


def credit_card(num_rows=None, nan_as_category=False):
    # Charger les données du fichier credit_card_balance
    credit_card = pd.read_csv('./data/credit_card_balance.csv', nrows=num_rows)
    print(f"Nombre d'échantillons dans credit_card_balance : {len(credit_card)}")

    # Encodage des variables binaires
    for bin_feature in ['SK_ID_CURR', 'SK_ID_PREV']:
        credit_card[bin_feature], uniques = pd.factorize(credit_card[bin_feature])

    # Appliquer l'encodage One-Hot
    credit_card, cat_cols = one_hot_encoder(credit_card, nan_as_category)

    del credit_card
    gc.collect()
    return credit_card

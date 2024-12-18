# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gc
from src.data_preprocessing.utils import one_hot_encoder


def application_train_test(num_rows=None, nan_as_category=False):
    # Charger les données des fichiers application_train et application_test
    df = pd.read_csv('../../data/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('../../data/application_test.csv', nrows=num_rows)
    df = pd.concat([df, test_df], ignore_index=True)

    # Supprimer les anomalies dans la colonne CODE_GENDER
    df = df[df['CODE_GENDER'] != 'XNA']

    # Encodage des variables binaires
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    # Encodage des variables catégoriques
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # Remplacement des valeurs aberrantes
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # Création de nouvelles caractéristiques
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # Nettoyage de la mémoire
    del test_df
    gc.collect()
    return df
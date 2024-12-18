# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gc
from .utils import one_hot_encoder

def preprocess_application(num_rows=None, nan_as_category=False):
    """Prétraitement des fichiers application_train.csv et application_test.csv."""
    df = pd.read_csv('./data/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('./data/application_test.csv', nrows=num_rows)
    print(f"Train samples: {len(df)}, test samples: {len(test_df)}")

    df = pd.concat([df, test_df], ignore_index=True)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Encodage binaire
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    # Encodage one-hot
    df, _ = one_hot_encoder(df, nan_as_category)

    # Remplacements et nouvelles features
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    del test_df
    gc.collect()
    return df

# -*- coding: utf-8 -*-
import pandas as pd
import gc
from .utils import one_hot_encoder


def preprocess_bureau(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('./data/bureau.csv', nrows=num_rows)
    bb = pd.read_csv('./data/bureau_balance.csv', nrows=num_rows)

    # Encodage one-hot
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Agrégations bureau_balance
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = ['_'.join(col).upper() for col in bb_agg.columns]
    bureau = bureau.merge(bb_agg, how='left', on='SK_ID_BUREAU')

    # Agrégations bureau
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    }
    cat_aggregations = {cat: ['mean'] for cat in bureau_cat}

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = ['BURO_' + '_'.join(col).upper() for col in bureau_agg.columns]

    del bureau, bb, bb_agg
    gc.collect()
    return bureau_agg

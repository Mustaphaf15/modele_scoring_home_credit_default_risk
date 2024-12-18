# -*- coding: utf-8 -*-
import pandas as pd
import gc
from src.data_preprocessing.utils import one_hot_encoder

def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('../../data/bureau.csv', nrows=num_rows)
    bb = pd.read_csv('../../data/bureau_balance.csv', nrows=num_rows)

    # One-hot encode
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = ['_'.join(col).upper() for col in bb_agg.columns]

    # Replace `join` with `merge`
    bureau = bureau.merge(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau numeric features aggregation
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum']
    }

    cat_aggregations = {cat: ['mean'] for cat in bureau_cat}

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = ['BURO_' + '_'.join(col).upper() for col in bureau_agg.columns]

    # Bureau: Active credits
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = ['ACTIVE_' + '_'.join(col).upper() for col in active_agg.columns]
    bureau_agg = bureau_agg.merge(active_agg, how='left', on='SK_ID_CURR')

    # Bureau: Closed credits
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = ['CLOSED_' + '_'.join(col).upper() for col in closed_agg.columns]
    bureau_agg = bureau_agg.merge(closed_agg, how='left', on='SK_ID_CURR')

    del closed, closed_agg, active, active_agg, bureau
    gc.collect()
    return bureau_agg
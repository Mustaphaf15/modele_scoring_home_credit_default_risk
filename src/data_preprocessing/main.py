# __init__.py
# Ce fichier initialise le module data_preprocessing.

# main.py
# -*- coding: utf-8 -*-
"""
Script principal pour exécuter les modules de prétraitement des données.
"""
import gc
from src.data_preprocessing.application_train_test import application_train_test
from src.data_preprocessing.bureau_and_balance import bureau_and_balance
from src.data_preprocessing.credit_card import credit_card_balance
from src.data_preprocessing.pos_cash import pos_cash
from src.data_preprocessing.previous_applications import previous_applications
from src.data_preprocessing.installments_payments import installments_payments
from src.data_preprocessing.utils import timer

def main():
    debug = False
    num_rows = 10000 if debug else None

    # Charger les données principales
    df = application_train_test(num_rows)

    # Traitement des données "bureau" et "bureau_balance"
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()

    # Traitement des applications précédentes
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()

    # Traitement des données POS-CASH
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()

    # Traitement des paiements échelonnés
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()

    # Traitement des soldes de carte de crédit
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()

    # Sauvegarde du fichier final
    path_output_file = '../../data/preprocessed_data.csv'
    df.to_csv(path_output_file, index=False)
    print(f'Le fichier {path_output_file} est sauvegardé')
    print('Fin du traitement')

if __name__ == "__main__":
    main()
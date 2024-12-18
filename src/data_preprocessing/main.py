# -*- coding: utf-8 -*-
import gc
from application import application
from bureau import bureau
from previous_applications import previous_applications
from pos_cash import pos_cash
from installments import installments
from credit_card import credit_card


def main():
    # Définir le nombre de lignes à lire (None pour tout lire)
    num_rows = None
    nan_as_category = False

    # Traiter les différents fichiers
    print("Traitement du fichier application...")
    app_data = application(num_rows=num_rows, nan_as_category=nan_as_category)

    print("Traitement du fichier bureau...")
    bureau_data = bureau(num_rows=num_rows, nan_as_category=nan_as_category)

    print("Traitement du fichier previous_application...")
    prev_app_data = previous_applications(num_rows=num_rows, nan_as_category=nan_as_category)

    print("Traitement du fichier POS_CASH_balance...")
    pos_cash_data = pos_cash(num_rows=num_rows, nan_as_category=nan_as_category)

    print("Traitement du fichier installments_payments...")
    installments_data = installments(num_rows=num_rows, nan_as_category=nan_as_category)

    print("Traitement du fichier credit_card_balance...")
    credit_card_data = credit_card(num_rows=num_rows, nan_as_category=nan_as_category)

    # Fusionner toutes les données traitées sur la clé SK_ID_CURR
    print("Fusion des données traitées...")
    data = app_data.merge(bureau_data, on='SK_ID_CURR', how='left')
    data = data.merge(prev_app_data, on='SK_ID_CURR', how='left')
    data = data.merge(pos_cash_data, on='SK_ID_CURR', how='left')
    data = data.merge(installments_data, on='SK_ID_CURR', how='left')
    data = data.merge(credit_card_data, on='SK_ID_CURR', how='left')

    # Sauvegarder le fichier final dans le dossier data
    data.to_csv('./data/data.csv', index=False)
    print("Le fichier final a été enregistré sous ./data/data.csv.")

    # Libérer la mémoire
    del app_data, bureau_data, prev_app_data, pos_cash_data, installments_data, credit_card_data, data
    gc.collect()

if __name__ == "__main__":
    main()

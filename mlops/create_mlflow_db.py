import os
from sqlalchemy import create_engine

# Définir les chemins des dossiers
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data')
db_path = os.path.join(data_dir, 'mlflow.db')

# Créer le dossier data s'il n'existe pas déjà
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Dossier {data_dir} créé.")

# Définir la base de données SQLAlchemy
DATABASE_URL = f"sqlite:///{db_path}"
engine = create_engine(DATABASE_URL, echo=True)

# il faudra lancer la commande pour lancer le serveur mlflow  mlflow ui --backend-store-uri sqlite:///data/mlflow.db --default-artifact-root ./artifacts --host localhost
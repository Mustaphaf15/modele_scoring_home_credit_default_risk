import subprocess
import os


def start_mlflow_server():
    """
    Lance le serveur MLflow avec les paramètres appropriés.
    """
    # Définir les chemins pour le backend-store-uri et les artefacts
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    backend_store_uri = f"sqlite:///{os.path.join(project_root, 'data', 'mlflow.db')}"
    artifact_root = os.path.join(project_root, 'artifacts')

    # Créer le dossier des artefacts s'il n'existe pas
    if not os.path.exists(artifact_root):
        os.makedirs(artifact_root)

    # Commande pour lancer le serveur MLflow
    command = [
        "mlflow", "ui",
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", artifact_root,
        "--host", "localhost"
    ]

    # Lancer le serveur
    print("Lancement du serveur MLflow...")
    print(f"Backend store URI : {backend_store_uri}")
    print(f"Default artifact root : {artifact_root}")
    subprocess.run(command)


if __name__ == "__main__":
    start_mlflow_server()

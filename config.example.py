from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEBUG = False
SECRET_KEY = "replace-this-with-a-private-secret-key"

USERS_FILE = BASE_DIR / "users.csv"
HISTORY_FILE = BASE_DIR / "history.csv"
DATASET_FILE = BASE_DIR / "news_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
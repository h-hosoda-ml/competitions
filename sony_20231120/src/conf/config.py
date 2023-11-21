import os

from dotenv import load_dotenv

load_dotenv()

# srcまでのパスを格納
BASE_PATH: str = os.getenv("BASE_PATH", "")
YOLO_BASE: str = os.getenv("YOLO_BASE", "")
MODEL_PATH: str = os.getenv("MODEL_PATH", "")
DATASET_PATH: str = os.path.join(BASE_PATH, "datasets")
YOLO_DATASET_PATH: str = os.path.join(YOLO_BASE, "dataset")

import os

from dotenv import load_dotenv

load_dotenv()

# srcまでのパスを格納
BASE_PATH: str = os.getenv("BASE_PATH", "")

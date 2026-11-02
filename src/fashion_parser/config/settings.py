"""Environment-driven settings (see ``.env.example``)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RANDOM_STATE = int(os.getenv("RANDOM_STATE", 41))
NUM_CATS = int(os.getenv("NUM_CATS", 46))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 512))

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SET_ROOT = Path(os.getenv("FGVC6_DATA_SET_ROOT_PATH", PROJECT_ROOT / "data"))

TRAIN_CSV_PATH = DATA_SET_ROOT / "train.csv"
SUBMISSION_CSV_PATH = DATA_SET_ROOT / "submission.csv"
SAMPLE_SUBMISSION_CSV_PATH = DATA_SET_ROOT / "sample_submission.csv"
LABEL_DESCRIPTIONS_PATH = DATA_SET_ROOT / "label_descriptions.json"
TRAIN_IMAGES_FOLDER_PATH = DATA_SET_ROOT / "train" / "train"
TEST_IMAGES_FOLDER_PATH = DATA_SET_ROOT / "test" / "test"

_weights = os.getenv("PRE_TRAINED_FASHION_WEIGHTS")
PRE_TRAINED_FASHION_WEIGHTS = Path(_weights) if _weights else None

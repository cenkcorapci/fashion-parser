import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)

# Experiments
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 41))
NUM_CATS = int(os.getenv('NUM_CATS', 46))
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', 512))

# Data Paths
# Use project root as base if not specified
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

DATA_SET_ROOT = Path(os.getenv('FGVC6_DATA_SET_ROOT_PATH', str(DEFAULT_DATA_ROOT)))
DL_MODELS_PATH = DATA_SET_ROOT / 'models' / 'dl'

# Dataset specific paths
TRAIN_CSV_PATH = DATA_SET_ROOT / 'train.csv'
SUBMISSION_CSV_PATH = DATA_SET_ROOT / 'submission.csv'
SAMPLE_SUBMISSION_CSV_PATH = DATA_SET_ROOT / 'sample_submission.csv'
LABEL_DESCRIPTIONS_PATH = DATA_SET_ROOT / 'label_descriptions.json'
TRAIN_IMAGES_FOLDER_PATH = DATA_SET_ROOT / 'train' / 'train'
TEST_IMAGES_FOLDER_PATH = DATA_SET_ROOT / 'test' / 'test'

# Aliases for backward compatibility
FGVC6_TRAIN_CSV_PATH = TRAIN_CSV_PATH
FGVC6_SUBMISSION_CSV_PATH = SUBMISSION_CSV_PATH
FGVC6_SAMPLE_SUBMISSION_CSV_PATH = SAMPLE_SUBMISSION_CSV_PATH
FGVC6_LABEL_DESCRIPTIONS_PATH = LABEL_DESCRIPTIONS_PATH
FGVC6_TRAIN_IMAGES_FOLDER_PATH = TRAIN_IMAGES_FOLDER_PATH
FGVC6_TEST_IMAGES_FOLDER_PATH = TEST_IMAGES_FOLDER_PATH
FGVC6_DATA_SET_ROOT_PATH = DATA_SET_ROOT

# Prepare directories
DL_MODELS_PATH.mkdir(parents=True, exist_ok=True)

PRE_TRAINED_FASHION_WEIGHTS = os.getenv('PRE_TRAINED_FASHION_WEIGHTS')
if PRE_TRAINED_FASHION_WEIGHTS:
    PRE_TRAINED_FASHION_WEIGHTS = Path(PRE_TRAINED_FASHION_WEIGHTS)

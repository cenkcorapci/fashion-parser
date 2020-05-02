# -*- coding: utf-8 -*-
import logging
import os
import pathlib
from os.path import expanduser

# Logs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Experiments
RANDOM_STATE = 41
EXPERIMENT_NAME = 'debug'
# Local files and folders
HOME = expanduser("~")
DATA_TRAIN_CSV = '/run/media/twoaday/data-storag/data-sets/fgcv7-fashion/imaterialist-fashion-2020-fgvc7/train.csv'
DATA_LABEL_DESCRIPTIONS = '/run/media/twoaday/data-storag/data-sets/fgcv7-fashion/imaterialist-fashion-2020-fgvc7/label_descriptions.json'
DATA_TRAIN_FOLDER = '/run/media/twoaday/data-storag/data-sets/fgcv7-fashion/imaterialist-fashion-2020-fgvc7/train'
DATA_TEST_FOLDER = '/run/media/twoaday/data-storag/data-sets/fgcv7-fashion/imaterialist-fashion-2020-fgvc7/test'
DL_MODELS_PATH = HOME + '/research/pre-trained-models/nlp/product-similarity/multisim-metric/' + EXPERIMENT_NAME
TB_LOGS_PATH = HOME + '/research/tb-logs/nlp/product-similarity/multisim-metric/'

TB_LOGS_PATH = os.getenv('TB_LOGS_PATH', TB_LOGS_PATH)
DL_MODELS_PATH = os.getenv('DL_MODELS_PATH', DL_MODELS_PATH)

# create directories
logging.info("Checking/creating directories...")
pathlib.Path(DL_MODELS_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(TB_LOGS_PATH).mkdir(parents=True, exist_ok=True)
logging.info("Directories are set.")

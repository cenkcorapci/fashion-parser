# -*- coding: utf-8 -*-
"""Model configs.
"""

import logging
import pathlib

# Logs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
LOGS_PATH = '/tmp/tb_logs/'

# Experiments
RANDOM_STATE = 41

# Local files
TEMP_DATA_PATH = '/tmp/'
TB_LOGS_PATH = '/tmp/tb_logs/'
DL_MODELS_PATH = TEMP_DATA_PATH + 'models/dl/'

FGVC6_TRAIN_CSV_PATH = '/Volumes/data-storag/data-sets/fgvc6-fashion/train.csv'
FGVC6_LABEL_DESCRIPTIONS_PATH = '/Volumes/data-storag/data-sets/fgvc6-fashion/label_descriptions.json'
FGVC6_TRAIN_CSV_PATH = '/Volumes/data-storag/data-sets/fgvc6-fashion/train.csv'
FGVC6_TRAIN_IMAGES_FOLDER_PATH = '/Volumes/data-storag/data-sets/fgvc6-fashion/train_images'
FGVC6_TEST_IMAGES_FOLDER_PATH = '/Volumes/data-storag/data-sets/fgvc6-fashion/test_images'

# create directories
logging.info("Checking directories...")
pathlib.Path(DL_MODELS_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(TB_LOGS_PATH).mkdir(parents=True, exist_ok=True)
logging.info("Directories are set.")

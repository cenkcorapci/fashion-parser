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

FGVC6_DATA_SET_ROOT_PATH = '/run/media/twoaday/data-storag/data-sets/fgvc6-fashion/'
FGVC6_TRAIN_CSV_PATH = '{0}train.csv'.format(FGVC6_DATA_SET_ROOT_PATH)
FGVC6_SUBMISSION_CSV_PATH = '{0}submission.csv'.format(FGVC6_DATA_SET_ROOT_PATH)
FGVC6_SAMPLE_SUBMISSION_CSV_PATH = '{0}sample_submission.csv'.format(FGVC6_DATA_SET_ROOT_PATH)
FGVC6_LABEL_DESCRIPTIONS_PATH = '{0}label_descriptions.json'.format(FGVC6_DATA_SET_ROOT_PATH)
FGVC6_TRAIN_CSV_PATH = '{0}train.csv'.format(FGVC6_DATA_SET_ROOT_PATH)
FGVC6_TRAIN_IMAGES_FOLDER_PATH = '{0}train_images/'.format(FGVC6_DATA_SET_ROOT_PATH)
FGVC6_TEST_IMAGES_FOLDER_PATH = '{0}test_images/'.format(FGVC6_DATA_SET_ROOT_PATH)

# create directories
logging.info("Checking directories...")
pathlib.Path(DL_MODELS_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(TB_LOGS_PATH).mkdir(parents=True, exist_ok=True)
logging.info("Directories are set.")

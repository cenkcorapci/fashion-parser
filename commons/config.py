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

MODANET_TRAIN_PATH = '/Users/cenkcorapci/Downloads/modanet2018_instances_train.json'
MODANET_TEST_PATH = '/Users/cenkcorapci/Downloads/modanet2018_instances_val.json'

CHICKTOPIA_IMAGE_DATA_SET_DOWNLOAD_URL = 'https://s3-ap-northeast-1.amazonaws.com/kyamagu-public/chictopia2/photos.lmdb.tar'  # 40 gb
MODANET_TRAIN_DOWNLOAD_URL = 'https://github.com/eBay/modanet/raw/master/annotations/modanet2018_instances_train.json'  # 101 mb
MODANET_VAL_DOWNLOAD_URL = 'https://github.com/eBay/modanet/raw/master/annotations/modanet2018_instances_val.json'  # 210 kb

# create directories
logging.info("Checking directories...")
pathlib.Path(DL_MODELS_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(TB_LOGS_PATH).mkdir(parents=True, exist_ok=True)
logging.info("Directories are set.")

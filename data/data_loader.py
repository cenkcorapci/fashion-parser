import json

import pandas as pd

from commons.config import *


class DataLoader:
    def __init__(self):
        with open(FGVC6_LABEL_DESCRIPTIONS_PATH) as json_file:
            data = json.load(json_file)
            logging.info('Loading label descriptions from: \n {0}'.format(data['info']))

            self.label_names = [x['name'] for x in data['categories']]
            self._segment_df = pd.read_csv(FGVC6_TRAIN_CSV_PATH)
            self._segment_df['CategoryId'] = self._segment_df['ClassId'].str.split('_').str[0]
            self.image_df = self._segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
            self.size_df = self._segment_df.groupby('ImageId')['Height', 'Width'].mean()
            self.image_df = self.image_df.join(self.size_df, on='ImageId')

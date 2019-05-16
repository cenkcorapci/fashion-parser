import numpy as np


class OneHotVectorization:
    def __init__(self, category_num):
        self._category_num = category_num

    def make_one_hot_vec(self, x):
        vec = np.zeros(self._category_num)
        vec[x] = 1
        return vec

# _*_ coding:utf-8 _*_
import sys
import os
import re
import numpy as np


def get_data(opt):
    if os.path.exists(opt.pickle_path):
        data = np.load(opt.pickle_path)
        data, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
        return data, word2ix, ix2word
    else:
        print('no data')

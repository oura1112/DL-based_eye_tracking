# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import cv2
import numpy as np

exe_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_path = exe_dir + '/faces_'

#ファイルのディレクトリ(ファイル名まで含む)の名前を渡す
def make_dataset_from_file():

    def numericalSort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
        
    file_list = sorted(glob.glob(data_dir_path+'/*jpg'), key = numericalSort)
    batch_size = 0
    sheets_num = 0
    a = 1
    image_name = []
    imgs = []
    roots = []
    labels_ = []
    
    num_path = exe_dir + '/faces.txt'
    
    with open(num_path, 'w') as f:
        for file_name in file_list:
            #画像に対する処理
            file_name = os.path.basename(file_name)
            root, ext = os.path.splitext(file_name)
            roots.append(root)
            f.write(root + '\n')
    print(roots)
make_dataset_from_file()

# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import cv2
import numpy as np

exe_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_path = exe_dir + '/left_eyes'

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
    save_num = []
    
    correct_num_path = exe_dir + '/faces.txt'
    img_save_dir = exe_dir + '/l_eyes'
    
    with open(correct_num_path, 'r') as f:
        for f_num in f.readlines():
            save_num.append(f_num.replace('\n',''))
    #print(save_num)
    
    i = 0
    #画像群
    for file_name in file_list:
        #画像に対する処理
        file_name = os.path.basename(file_name)
        root, ext = os.path.splitext(file_name)
        #print(type(root))
        
        #保存する番号と画像の番号が一致したら保存し次の画像・番号へ。一致しなければそのままの番号で次の画像と比較。
        if root == save_num[i]:
            abs_name = data_dir_path + '/' + file_name
            save_img = cv2.imread(abs_name)
            cv2.imwrite(os.path.join(img_save_dir, file_name), save_img)
            i += 1
        else:
            continue
        
make_dataset_from_file()

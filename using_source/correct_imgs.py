# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import cv2
import numpy as np

exe_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_path = exe_dir + '/faces'

#ファイルのディレクトリ(ファイル名まで含む)の名前を渡す
def make_dataset_from_file():
        
    #画像群があるディレクトリから全ての画像を読み込み、numpy配列として返す
    #ラベルファイルを展開し、np.arrayの型でlabelsに読み込む。
    
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
    
    r_eyes_path = exe_dir + '/l_eyes.txt'
    l_eyes_dir = exe_dir + '/faces_'
    
    with open(r_eyes_path, 'r') as f:
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
            l_eye = cv2.imread(abs_name)
            cv2.imwrite(os.path.join(l_eyes_dir, file_name), l_eye)
            i += 1
        else:
            continue
        
make_dataset_from_file()

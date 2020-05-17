# -*- coding: utf-8 -*-
#face_eyes_cnn2.pyの中に直接記述するかも

import os
import sys
import re
import glob
import cv2
import numpy as np
import keras

exe_dir = os.path.dirname(os.path.abspath(__file__))

r_eyes_x_path_tr = exe_dir + '/train_data/r_x_tr.txt'
r_eyes_y_path_tr = exe_dir + '/train_data/r_y_tr.txt'
l_eyes_x_path_tr = exe_dir + '/train_data/l_x_tr.txt'
l_eyes_y_path_tr = exe_dir + '/train_data/l_y_tr.txt'
faces_x_path_tr = exe_dir + '/train_data/faces_x_tr.txt'
faces_y_path_tr = exe_dir + '/train_data/faces_y_tr.txt'
faces_w_path_tr = exe_dir + '/train_data/faces_w_tr.txt'
label_path_tr = exe_dir + '/train_data/datalabels2_tr.txt'
r_eyes_dir_tr = exe_dir + '/train_data/r_eyes_tr'
l_eyes_dir_tr = exe_dir + '/train_data/l_eyes_tr'
faces_dir_tr = exe_dir + '/train_data/faces_tr'
"""
r_eyes_x_path_val = exe_dir + '/val_data/r_x_val2.txt'
r_eyes_y_path_val = exe_dir + '/val_data/r_y_val2.txt'
l_eyes_x_path_val = exe_dir + '/val_data/l_x_val2.txt'
l_eyes_y_path_val = exe_dir + '/val_data/l_y_val2.txt'
faces_x_path_val = exe_dir + '/val_data/faces_x_val2.txt'
faces_y_path_val = exe_dir + '/val_data/faces_y_val2.txt'
faces_w_path_val = exe_dir + '/val_data/faces_w_val2.txt'
label_path_val = exe_dir + '/val_data/datalabels_val2.txt'
r_eyes_dir_val = exe_dir + '/val_data/r_eyes_val'
l_eyes_dir_val = exe_dir + '/val_data/l_eyes_val'
faces_dir_val = exe_dir + '/val_data/faces_val'
"""
#１周の学習に用いる画像の数
batch_size = 160
#訓練用データサイズ
tr_data_size = 18880
#評価用データサイズ
val_data_size = 1600
#分類の個数(処理上0も含む)
num_classes = 2

def batch_iter(data_size, batch_size, label_path, r_eyes_x_path, r_eyes_y_path, l_eyes_x_path, l_eyes_y_path, faces_x_path, faces_y_path, faces_w_path, r_eyes_dir, l_eyes_dir, faces_dir):
    print("generating data")
    
    #データの生成
    def generate_arrays_from_file():
        
        def numericalSort(value):
            numbers = re.compile(r'(\d+)')
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts
            
        faces_list = sorted(glob.glob(faces_dir+'/*jpg'), key = numericalSort)
        r_eyes_list = sorted(glob.glob(r_eyes_dir+'/*jpg'), key = numericalSort)
        l_eyes_list = sorted(glob.glob(l_eyes_dir+'/*jpg'), key = numericalSort)
        batch_size_ = 0
        sheets_num = 0
        
        faces = []
        right_eyes = []
        left_eyes = []
        r_eyes_x = []
        r_eyes_y = []
        l_eyes_x = []
        l_eyes_y = []
        faces_x = []
        faces_y = []
        faces_w = []
        roots = []
        labels = []

        print("Converting data to NumPy Array ...")
        print("Converting labels to NumPy Array ...")

        with open(label_path, 'r') as label_f, open(r_eyes_x_path, 'r') as rex_f, open(r_eyes_y_path, 'r') as rey_f, open(l_eyes_x_path, 'r') as lex_f, open(l_eyes_y_path, 'r') as ley_f, open(faces_x_path, 'r') as fx_f, open(faces_y_path, 'r') as fy_f, open(faces_w_path, 'r') as fw_f:
                         
            for (face_path, r_eye_path, l_eye_path, rex, rey, lex, ley, fx, fy, fw, label) in zip(faces_list, r_eyes_list, l_eyes_list, rex_f.readlines(), rey_f.readlines(), lex_f.readlines(), ley_f.readlines(), fx_f.readlines(), fy_f.readlines(), fw_f.readlines(), label_f.readlines()):
                #print(3)
                #各画像の一時保持
                faces.append(cv2.imread(face_path))
                right_eyes.append(cv2.imread(r_eye_path))
                left_eyes.append(cv2.imread(l_eye_path))
                #各種データの一時保持
                r_eyes_x.append(rex)
                r_eyes_y.append(rey)
                l_eyes_x.append(lex)
                l_eyes_y.append(ley)
                faces_x.append(fx)
                faces_y.append(fy)
                faces_w.append(fw)
                #ラベルの一時保持
                labels.append(int(label))
                
                batch_size_ += 1
                
                #if batch_size_ == batch_size:
            
            #値の正規化
            faces = np.array(faces)
            right_eyes = np.array(right_eyes)
            left_eyes = np.array(left_eyes)
            r_eyes_x = np.array(r_eyes_x)
            r_eyes_y = np.array(r_eyes_y)
            l_eyes_x = np.array(l_eyes_x)
            l_eyes_y = np.array(l_eyes_y)
            faces_x = np.array(faces_x)
            faces_y = np.array(faces_y)
            faces_w = np.array(faces_w)
            
            faces = faces.astype("float32")
            right_eyes = right_eyes.astype("float32")
            left_eyes = left_eyes.astype("float32")
            r_eyes_x = r_eyes_x.astype("float32")
            r_eyes_y = r_eyes_y.astype("float32")
            l_eyes_x = l_eyes_x.astype("float32")
            l_eyes_y = l_eyes_y.astype("float32")
            faces_x = faces_x.astype("float32")
            faces_y = faces_y.astype("float32")
            faces_w = faces_w.astype("float32")
            
            right_eyes /= 255
            left_eyes /= 255
            faces /= 255
            r_eyes_x /= 1280
            r_eyes_y /= 800
            l_eyes_x /= 1280
            l_eyes_y /= 800
            faces_x /= 1280
            faces_y /= 800
            faces_w /= 800
            print(1)
            #ラベルをone-hot表現にする(ex:2→[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
            #"/home/oura/anaconda3/lib/python3.6/site-packages/keras/utils/np_utils.py", line 27, in to_categoricalを書き換えている
            labels = keras.utils.np_utils.to_categorical(labels, num_classes)
            """
            #yield 1 #[right_eyes, left_eyes, faces, r_eyes_x, r_eyes_y, l_eyes_x, l_eyes_y, faces_x, faces_y, faces_w], labels
            print(2)
            print(faces.shape)
            #保持用リストの初期化    
            faces = []
            right_eyes = []
            left_eyes = []
            r_eyes_x = []
            r_eyes_y = []
            l_eyes_x = []
            l_eyes_y = []
            faces_x = []
            faces_y = []
            faces_w = []
            roots = []
            labels = []
            batch_size_ = 0
            """        
    return generate_arrays_from_file()

a = 1
for i in batch_iter(tr_data_size, batch_size, label_path_tr, r_eyes_x_path_tr, r_eyes_y_path_tr, l_eyes_x_path_tr, l_eyes_y_path_tr, faces_x_path_tr, faces_y_path_tr, faces_w_path_tr, r_eyes_dir_tr, l_eyes_dir_tr, faces_dir_tr):
    a += 1
    
print(a)

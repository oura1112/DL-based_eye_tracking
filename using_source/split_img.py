# -*- coding: utf-8 -*-

import os
import re
import glob
import cv2
import numpy as np

exe_dir = os.path.dirname(os.path.abspath(__file__))
faces_dir = exe_dir + '/faces_'
r_eyes_dir = exe_dir + '/r_eyes_'
l_eyes_dir = exe_dir + '/l_eyes'

#ファイルのディレクトリ(ファイル名まで含む)の名前を渡す
def make_dataset_from_file():
        
    #画像群があるディレクトリから全ての画像を読み込み、numpy配列として返す
    #ラベルファイルを展開し、np.arrayの型でlabelsに読み込む。
    
    def numericalSort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
        
    faces_list = sorted(glob.glob(faces_dir+'/*jpg'), key = numericalSort)
    r_eyes_list = sorted(glob.glob(r_eyes_dir+'/*jpg'), key = numericalSort)
    l_eyes_list = sorted(glob.glob(l_eyes_dir+'/*jpg'), key = numericalSort)
    batch_size = 0
    sheets_num = 0
    a = 1
    image_name = []
    imgs = []
    roots = []
    labels_ = []
    save_num = []
    
    save_num_path = exe_dir + '/number_val.txt'
    save_faces_dir = exe_dir + '/val_data/faces_val'
    save_r_eyes_dir = exe_dir + '/val_data/r_eyes_val'
    save_l_eyes_dir = exe_dir + '/val_data/l_eyes_val'
    
    
    #保存する番号の保持
    with open(save_num_path, 'r') as f:
        for f_num in f.readlines():
            save_num.append(f_num.replace('\n',''))
    
    i = 0
    #画像群
    for (face_path, r_eye_path, l_eye_path) in zip(faces_list, r_eyes_list, l_eyes_list):
        #画像に対する処理
        face_path = os.path.basename(face_path)
        r_eye_path = os.path.basename(r_eye_path)
        l_eye_path = os.path.basename(l_eye_path)
        #各画像の番号取得(root)
        root, ext = os.path.splitext(face_path)
        
        #保存する番号と画像の番号が一致したら保存し次の画像・番号へ。一致しなければそのままの番号で次の画像と比較。
        if root == save_num[i]:
            abs_face_path = faces_dir + '/' + face_path
            abs_r_eye_path = r_eyes_dir + '/' + r_eye_path
            abs_l_eye_path = l_eyes_dir + '/' + l_eye_path
            face = cv2.imread(abs_face_path)
            r_eye = cv2.imread(abs_r_eye_path)
            l_eye = cv2.imread(abs_l_eye_path)
            cv2.imwrite(os.path.join(save_faces_dir, face_path), face)
            cv2.imwrite(os.path.join(save_r_eyes_dir, r_eye_path), r_eye)
            cv2.imwrite(os.path.join(save_l_eyes_dir, l_eye_path), l_eye)
            i += 1
        else:
            continue
        
make_dataset_from_file()

# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import cv2
import face_eyes_detect_class2
import numpy as np

exe_dir = os.path.dirname(os.path.abspath(__file__))

#分類数
num_class = 15

#face_eyes_detect_classのインスタンス生成
fedc = face_eyes_detect_class2.face_eyes_detect_class()

def batch_iter(data_dir_path, file_path_label2, file_path_label4, file_path_label6, file_path_label9):
    print(exe_dir)
    #ファイルのディレクトリ(ファイル名まで含む)の名前を渡す
    def generate_arrays_from_file():
            
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
        labels_2 = []
        labels_4 = []
        labels_6 = []
        labels_9 = []
        
        f2_path = exe_dir + '/r_eyes_x.txt'
        f3_path = exe_dir + '/r_eyes_y.txt'
        f4_path = exe_dir + '/l_eyes_x.txt'
        f5_path = exe_dir + '/l_eyes_y.txt'
        f6_path = exe_dir + '/faces_x.txt'
        f7_path = exe_dir + '/faces_y.txt'
        f8_path = exe_dir + '/faces_w.txt'
        f9_path = exe_dir + '/datalabels2.txt'
        f10_path = exe_dir + '/datalabels4.txt'
        f11_path = exe_dir + '/datalabels6.txt'
        f12_path = exe_dir + '/datalabels9.txt'
        r_eyes_path = 'right_eyes'
        l_eyes_path = 'left_eyes'
        faces_path = 'faces' 
        print("Converting data to NumPy Array ...")
        print("Converting labels to NumPy Array ...")
        """
        for file_name in file_list:
            file_name = os.path.basename(file_name)
            root, ext = os.path.splitext(file_name)
            image_name.append(root)
        print(image_name)
        """
        with open(file_path_label2, 'r') as f_label2, open(file_path_label4, 'r') as f_label4, open(file_path_label6, 'r') as f_label6, open(file_path_label9, 'r') as f_label9:
                         
            for (file_name, label2, label4, label6, label9) in zip(file_list, f_label2.readlines(), f_label4.readlines(), f_label6.readlines(), f_label9.readlines()):
                #画像に対する処理
                file_name = os.path.basename(file_name)
                root, ext = os.path.splitext(file_name)
                if ext == u'.png' or u'.jpeg' or u'.jpg':
                    abs_name = data_dir_path + '/' + file_name
                    imgs.append(np.array(cv2.imread(abs_name))) #ここで画像が順番に読み込めていない
                    roots.append(root)
                    
                #ラベルに対する処理
                labels_2.append(int(label2))
                labels_4.append(int(label4))
                labels_6.append(int(label6))
                labels_9.append(int(label9))
                
                batch_size += 1
                
                if batch_size == 100:

                    #print(roots)
                    #画像、ラベルの前処理実行
                    right_eyes, left_eyes, faces, r_eyes_x, r_eyes_y, l_eyes_x, l_eyes_y, faces_x, faces_y, faces_w, sheets_count, labels2, labels4, labels6, labels9 = fedc.face_eyes_detect(imgs, labels_2, labels_4, labels_6, labels_9)

                    imgs = []
                    labels_2 = []
                    labels_4 = []
                    labels_6 = []
                    labels_9 = []
                    roots = []
                    batch_size = 0
                    sheets_num += sheets_count
                    print(sheets_num)
                    
                    with open(f2_path, 'a') as f2, open(f3_path, 'a') as f3, open(f4_path, 'a') as f4, open(f5_path, 'a') as f5, open(f6_path, 'a') as f6, open(f7_path, 'a') as f7, open(f8_path, 'a') as f8, open(f9_path, 'a') as f9, open(f10_path, 'a') as f10, open(f11_path, 'a') as f11, open(f12_path, 'a') as f12:
                    
                        for (r_eye_x, r_eye_y, l_eye_x, l_eye_y, face_x, face_y, face_w, label2, label4, label6, label9) in zip(r_eyes_x, r_eyes_y, l_eyes_x, l_eyes_y, faces_x, faces_y, faces_w, labels2, labels4, labels6, labels9):
                            
                            f2.write(str(r_eye_x) + "\n")
                            f3.write(str(r_eye_y) + "\n")
                            f4.write(str(l_eye_x) + "\n")
                            f5.write(str(l_eye_y) + "\n")
                            f6.write(str(face_x) + "\n")
                            f7.write(str(face_y) + "\n")
                            f8.write(str(face_w) + "\n")
                            f9.write(str(label2) + "\n")
                            f10.write(str(label4) + "\n")
                            f11.write(str(label6) + "\n")
                            f12.write(str(label9) + "\n")
                   
                    for (right_eye, left_eye, face) in zip(right_eyes, left_eyes, faces):
                        cv2.imwrite(os.path.join(r_eyes_path, '%d.jpg' %(a)), right_eye)
                        cv2.imwrite(os.path.join(l_eyes_path, '%d.jpg' %(a)), left_eye)
                        cv2.imwrite(os.path.join(faces_path, '%d.jpg' %(a)), face)
                        a += 1
                    
                    print("write completed")
                    yield(1)

    return generate_arrays_from_file()
    print("Done")
    print(sheets_num)

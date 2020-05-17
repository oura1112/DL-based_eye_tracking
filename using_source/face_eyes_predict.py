import os
import re
import glob
import cv2
import numpy as np

import keras
from keras.models import model_from_json

exe_dir = os.path.dirname(os.path.abspath(__file__))

r_eyes_x_path_test = exe_dir + '/test_data/r_x_test.txt'
r_eyes_y_path_test = exe_dir + '/test_data/r_y_test.txt'
l_eyes_x_path_test = exe_dir + '/test_data/l_x_test.txt'
l_eyes_y_path_test = exe_dir + '/test_data/l_y_test.txt'
faces_x_path_test = exe_dir + '/test_data/faces_x_test.txt'
faces_y_path_test = exe_dir + '/test_data/faces_y_test.txt'
faces_w_path_test = exe_dir + '/test_data/faces_w_test.txt'

r_eyes_dir_test = exe_dir + '/test_data/r_eyes_test'
l_eyes_dir_test = exe_dir + '/test_data/l_eyes_test'
faces_dir_test = exe_dir + '/test_data/faces_test'

#正解データ
label2_test = exe_dir + '/test_data/datalabels4_test.txt'
#誤検出データの保存（1or-1）
failure_recog = exe_dir + "/failure_recog4.txt"
#推論する枚数サイズ
predict_size = 1200

def batch_iter_eva(batch_size, r_eyes_x_path, r_eyes_y_path, l_eyes_x_path, l_eyes_y_path, faces_x_path, faces_y_path, faces_w_path, r_eyes_dir, l_eyes_dir, faces_dir):
    print("generating test data")
    
    #データの生成
    def generate_arrays_from_file_eva():
        
        def numericalSort(value):
            numbers = re.compile(r'(\d+)')
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts
            
        faces_list = sorted(glob.glob(faces_dir+'/*jpg'), key = numericalSort)
        r_eyes_list = sorted(glob.glob(r_eyes_dir+'/*jpg'), key = numericalSort)
        l_eyes_list = sorted(glob.glob(l_eyes_dir+'/*jpg'), key = numericalSort)
        batch_size_ = 0

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
        
        print("Converting data to NumPy Array ...")
        print("Converting labels to NumPy Array ...")
        
        while True:
            with open(r_eyes_x_path, 'r') as rex_f, open(r_eyes_y_path, 'r') as rey_f, open(l_eyes_x_path, 'r') as lex_f, open(l_eyes_y_path, 'r') as ley_f, open(faces_x_path, 'r') as fx_f, open(faces_y_path, 'r') as fy_f, open(faces_w_path, 'r') as fw_f:
                             
                for (face_path, r_eye_path, l_eye_path, rex, rey, lex, ley, fx, fy, fw) in zip(faces_list, r_eyes_list, l_eyes_list, rex_f.readlines(), rey_f.readlines(), lex_f.readlines(), ley_f.readlines(), fx_f.readlines(), fy_f.readlines(), fw_f.readlines()):
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
                    
                    batch_size_ += 1
                    
                    if batch_size_ == batch_size:
                    
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

                        return [right_eyes, left_eyes, faces, r_eyes_x, r_eyes_y, l_eyes_x, l_eyes_y, faces_x, faces_y, faces_w]
                    
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
                        batch_size_ = 0

    return generate_arrays_from_file_eva()
    

model_cons = "/home/oura/デスクトップ/source/myprogram/data2/face_eyes_model_v2/label4/face_eyes_model_4_2.json"
model_weights = "/home/oura/デスクトップ/source/myprogram/data2/face_eyes_model_v2/label4/face_eyes_model_weights_4_2.hdf5"

model = None
with open(model_cons) as f:
    model = model_from_json(f.read())
    
model.load_weights(model_weights)

out_ = scores = model.predict(batch_iter_eva(predict_size, r_eyes_x_path_test, r_eyes_y_path_test, l_eyes_x_path_test, l_eyes_y_path_test, faces_x_path_test, faces_y_path_test, faces_w_path_test, r_eyes_dir_test, l_eyes_dir_test, faces_dir_test))

out = []

for i in range(predict_size):
    if out_[i][0]>out_[i][1]:
        out.append(1)
    else:
        out.append(2)

labels_ = []
labels  = []

with open(label2_test, 'r') as label2_f, open(failure_recog, 'w') as failrec_f:
    labels_ = label2_f.read().split()
    labels_ = [[int(elm) for elm in v] for v in labels_]
    labels_ = labels_[:predict_size]

    print(out[0])
    print(labels_[0])
    
    for i,j in zip(labels_,out):
        labels.append(i[0] - j)
        
    #print(labels)
    
    for x in labels:
        failrec_f.write(str(x) + "\n")
    

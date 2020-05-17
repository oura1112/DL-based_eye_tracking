# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import re
import glob
import pickle
import time
import cv2
import numpy as np

import h5py
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten
from keras.layers import concatenate
from keras.optimizers import Adam, rmsprop, SGD
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt

exe_dir = os.path.dirname(os.path.abspath(__file__))

r_eyes_x_path_tr = exe_dir + '/train_data/r_x_tr.txt'
r_eyes_y_path_tr = exe_dir + '/train_data/r_y_tr.txt'
l_eyes_x_path_tr = exe_dir + '/train_data/l_x_tr.txt'
l_eyes_y_path_tr = exe_dir + '/train_data/l_y_tr.txt'
faces_x_path_tr = exe_dir + '/train_data/faces_x_tr.txt'
faces_y_path_tr = exe_dir + '/train_data/faces_y_tr.txt'
faces_w_path_tr = exe_dir + '/train_data/faces_w_tr.txt'
label_path_tr = exe_dir + '/train_data/datalabels4_tr.txt'
r_eyes_dir_tr = exe_dir + '/train_data/r_eyes_tr'
l_eyes_dir_tr = exe_dir + '/train_data/l_eyes_tr'
faces_dir_tr = exe_dir + '/train_data/faces_tr'

r_eyes_x_path_val = exe_dir + '/val_data/r_x_val.txt'
r_eyes_y_path_val = exe_dir + '/val_data/r_y_val.txt'
l_eyes_x_path_val = exe_dir + '/val_data/l_x_val.txt'
l_eyes_y_path_val = exe_dir + '/val_data/l_y_val.txt'
faces_x_path_val = exe_dir + '/val_data/faces_x_val.txt'
faces_y_path_val = exe_dir + '/val_data/faces_y_val.txt'
faces_w_path_val = exe_dir + '/val_data/faces_w_val.txt'
label_path_val = exe_dir + '/val_data/datalabels4_val.txt'
r_eyes_dir_val = exe_dir + '/val_data/r_eyes_val'
l_eyes_dir_val = exe_dir + '/val_data/l_eyes_val'
faces_dir_val = exe_dir + '/val_data/faces_val'

#１周の学習に用いる画像の数
batch_size = 20
#訓練用データサイズ
tr_data_size = 18000
#検証用データサイズ
val_data_size = 2860
#分類の個数
num_classes = 4
#コールバック用
callbacks = []

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
        labels = []

        print("Converting data to NumPy Array ...")
        print("Converting labels to NumPy Array ...")
        
        while True:
            with open(label_path, 'r') as label_f, open(r_eyes_x_path, 'r') as rex_f, open(r_eyes_y_path, 'r') as rey_f, open(l_eyes_x_path, 'r') as lex_f, open(l_eyes_y_path, 'r') as ley_f, open(faces_x_path, 'r') as fx_f, open(faces_y_path, 'r') as fy_f, open(faces_w_path, 'r') as fw_f:
                             
                for (face_path, r_eye_path, l_eye_path, rex, rey, lex, ley, fx, fy, fw, label) in zip(faces_list, r_eyes_list, l_eyes_list, rex_f.readlines(), rey_f.readlines(), lex_f.readlines(), ley_f.readlines(), fx_f.readlines(), fy_f.readlines(), fw_f.readlines(), label_f.readlines()):
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
                        
                        #ラベルをone-hot表現にする(ex:2→[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
                        #"/home/oura/anaconda3/lib/python3.6/site-packages/keras/utils/np_utils.py", line 27, in to_categoricalを書き換えている
                        labels = keras.utils.np_utils.to_categorical(labels, num_classes)

                        yield [right_eyes, left_eyes, faces, r_eyes_x, r_eyes_y, l_eyes_x, l_eyes_y, faces_x, faces_y, faces_w], labels
                    
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

    return generate_arrays_from_file()
    
#ジェネレータからデータ（画像、各種データ、ラベル）を受け取りモデルを学習
def face_eyes_cnn():

    #回すエポック数
    epochs = 3
    #保存するモデルのパスを実行時ディレクトリの下に作る
    f_model = exe_dir + '/face_eyes_model_v2'
    
    
    #1epochあたりのミニバッチ数
    train_steps = int((tr_data_size - 1) / batch_size) + 1
    valid_steps = int((val_data_size - 1) / batch_size) + 1
    
    #入力データの形状
    input_shape_e = (25, 25, 3)
    input_shape_f = (224, 224, 3)

    #CNNの構築・学習・推論・保存 改良必要
    """
    畳み込みの出力サイズ。割り切れるように設定。

    （入力サイズ ー フィルタサイズ ＋ ２×フィルタパディング） ÷ フィルタストライド + 1
    """
    #右目領域のCNN演算
    input_r_eye = Input(shape=input_shape_e)
    #畳み込み
    r_eye = Conv2D(96, (11,11), padding='same', activation='relu', use_bias=True, kernel_initializer='he_normal')(input_r_eye) #カーネル数が次層のチャンネル数
    r_eye = Conv2D(256, (5,5), padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal')(r_eye)
    #最大値プーリング
    #r_eye = MaxPooling2D((3, 3), strides=2, padding='same')(r_eye)
    #畳み込み・空間的ドロップアウト
    r_eye = Conv2D(384, (3,3), padding='valid', use_bias=True, kernel_initializer='he_normal')(r_eye)
    r_eye = Conv2D(64, (1,1), padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal')(r_eye)
    #最大値プーリング
    r_eye = MaxPooling2D((2, 2), strides=2, padding='valid')(r_eye)
    #空間的ドロップアウト
    r_eye = Dropout(0.15)(r_eye)

    #全結合
    r_eye = Flatten()(r_eye)
    r_eye = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.001), activity_regularizer = regularizers.l1(0.001))(r_eye)
    
    #左目領域のCNN演算
    input_l_eye = Input(shape=input_shape_e)
    #畳み込み
    l_eye = Conv2D(96, (11,11), padding='same', activation='relu', use_bias=True, kernel_initializer='he_normal')(input_l_eye) #カーネル数が次層のチャンネル数
    l_eye = Conv2D(256, (5,5), padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal')(l_eye)
    #最大値プーリング
    #l_eye = MaxPooling2D((3, 3), strides=2, padding='same')(l_eye)
    #畳み込み
    l_eye = Conv2D(384, (3,3), padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal')(l_eye)
    l_eye = Conv2D(64, (1,1), padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal')(l_eye)
    #最大値プーリング
    l_eye = MaxPooling2D((2, 2), strides=2, padding='valid')(l_eye)
    #空間的ドロップアウト
    l_eye = Dropout(0.15)(l_eye)

    #全結合
    l_eye = Flatten()(l_eye)
    l_eye = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.001), activity_regularizer = regularizers.l1(0.001))(l_eye)

    #顔領域のCNN演算
    input_face = Input(shape=input_shape_f)
    #畳み込み
    face = Conv2D(96, (11,11), padding='same', activation='relu', use_bias=True, kernel_initializer='he_normal')(input_face) #カーネル数が次層のチャンネル数
    face = Conv2D(256, (5,5), padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal')(face)
    #最大値プーリング
    face = MaxPooling2D((2, 2), strides=2, padding='same')(face)
    #畳み込み
    face = Conv2D(384, (3,3), padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal')(face)
    face = Conv2D(256, (3,3), padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal')(face)
    #最大値プーリング
    face = MaxPooling2D((2, 2), strides=2, padding='valid')(face)
    #畳み込み
    face = Conv2D(256, (3,3), padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal')(face)
    face = Conv2D(64, (1,1), padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal')(face)
    #最大値プーリング
    face = MaxPooling2D((2, 2), strides=2, padding='valid')(face)
    #空間的ドロップアウト
    face = Dropout(0.15)(face)

    #全結合
    face = Flatten()(face)
    face = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.001), activity_regularizer = regularizers.l1(0.001))(face)
    face = Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.001), activity_regularizer = regularizers.l1(0.001))(face)    

    #r_eye, l_eye, face, r_eye_info, l_eye_info, face_infoの結合
    r_eye_x = Input(shape=(1,))
    r_eye_y = Input(shape=(1,))
    l_eye_x = Input(shape=(1,))
    l_eye_y = Input(shape=(1,))
    face_x = Input(shape=(1,))
    face_y = Input(shape=(1,))
    face_w = Input(shape=(1,))
    
    eyes_face = concatenate([r_eye, l_eye, face, r_eye_x, r_eye_y, l_eye_x, l_eye_y, face_x, face_y, face_w])

    #全結合・出力
    eyes_face = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01))(eyes_face)
    eyes_face = Dropout(0.25)(eyes_face)
    output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(eyes_face)

    #モデルの設定
    model = Model(inputs=[input_r_eye, input_l_eye, input_face, r_eye_x, r_eye_y, l_eye_x, l_eye_y, face_x, face_y, face_w], outputs=output)

    #モデルの要約の表示
    model.summary()
    
    #コールバックの設定
    callbacks.append(EarlyStopping)
    
    #オプティマイザの設定
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    #学習過程の設定
    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer = sgd,
                  metrics = ['accuracy'])

    #学習の実行
    #学習時間の計測
    start = time.time()
    
    history = model.fit_generator(generator = batch_iter(tr_data_size, batch_size, label_path_tr, r_eyes_x_path_tr, r_eyes_y_path_tr, l_eyes_x_path_tr, l_eyes_y_path_tr, faces_x_path_tr, faces_y_path_tr, faces_w_path_tr, r_eyes_dir_tr, l_eyes_dir_tr, faces_dir_tr),
                steps_per_epoch = train_steps,
                epochs=epochs,
                callbacks = None,
                validation_data = batch_iter(val_data_size, batch_size, label_path_val, r_eyes_x_path_val, r_eyes_y_path_val, l_eyes_x_path_val, l_eyes_y_path_val, faces_x_path_val, faces_y_path_val, faces_w_path_val, r_eyes_dir_val, l_eyes_dir_val, faces_dir_val),
                validation_steps = valid_steps)
    
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    
    
    #history = model.fit([train_r_eyes, train_l_eyes, train_faces, ], train_label, batch_size=batch_size, epochs=epochs,
              #verbose=1, validation_data=([val_eyes, val_faces], val_label))

    #モデルと重み及び重みのみの保存
    print('save the architecture of a model')
    json_string = model.to_json()
    open(os.path.join(f_model,'face_eyes_model.json'), 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(os.path.join(f_model,'face_eyes_model.yaml'), 'w').write(yaml_string)
    print('save weights')
    model.save_weights(os.path.join(f_model,'face_eyes_model_weights.hdf5'))
    
    #学習履歴の保存
    with open(os.path.join(f_model,'history.history'), 'wb') as f:
        pickle.dump(history.history, f)

    #モデルの構造を画像データに保存
    #plot_model(model, to_file='model2.png')
    
    #学習過程をグラフで出力
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    
    nb_epoch = len(loss)
    plt.plot(range(nb_epoch), loss, label='loss')
    plt.plot(range(nb_epoch), val_loss, label='val_loss')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
    plt.plot(range(nb_epoch), acc, label='acc')
    plt.plot(range(nb_epoch), val_acc, label='val_acc')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    
face_eyes_cnn()

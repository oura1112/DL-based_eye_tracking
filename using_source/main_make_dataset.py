# -*- coding: UTF-8 -*- 

import make_dataset

if __name__ == '__main__':

    #データセットのファイルを格納
    key_file = {
    'img':'/home/oura/デスクトップ/source/myprogram/dataset',  #←パス(画像ファイル名含まない)
    'label2':'/home/oura/デスクトップ/source/myprogram/datalabel2.txt', #←パス（ファイル名含む）
    'label4':'/home/oura/デスクトップ/source/myprogram/datalabel4.txt', #←パス（ファイル名含む）
    'label6':'/home/oura/デスクトップ/source/myprogram/datalabel6.txt', #←パス（ファイル名含む）
    'label9':'/home/oura/デスクトップ/source/myprogram/datalabel9.txt', #←パス（ファイル名含む）
}
    a = 0
    #データセットの作成
    for i in make_dataset.batch_iter(key_file['img'], key_file['label2'], key_file['label4'], key_file['label6'], key_file['label9']):
        a += 1
        print(1)
    print(a)

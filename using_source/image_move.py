# -*- coding: UTF-8 -*- 
import os
import shutil

folders = sorted(os.listdir("/home/oura/デスクトップ/source/myprogram/MPIIFaceGaze/p14")) #p00の数字部分を変えつつ実行
print(folders)

for folder in folders:
    folder = "/home/oura/デスクトップ/source/myprogram/MPIIFaceGaze/p14/" + folder #p00の数字部分を変えつつ実行
    try:
        images = sorted(os.listdir(folder))
        for image in images:
            root, ext = os.path.splitext(image)
            if ext == u'.jpg':
                shutil.move(folder+'/'+image, '/home/oura/デスクトップ/source/myprogram/dataset')
    except:
        print('no directory')

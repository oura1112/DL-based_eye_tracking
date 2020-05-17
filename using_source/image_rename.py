# coding: UTF-8
import sys
import os
import re
import glob

a = 36168 #前の人の最後の数字+1

for i in range(70):
    if i<9:
        data_dir_path = ('/home/oura/デスクトップ/source/myprogram/MPIIFaceGaze/p14/day0%d' %(i+1)) #p00の数字部分を変えつつ実行
        print(data_dir_path)
    else:
        data_dir_path = ('/home/oura/デスクトップ/source/myprogram/MPIIFaceGaze/p14/day%d' %(i+1)) #p00の数字部分を変えつつ実行
        print(data_dir_path)

    #昇順にソートする関数
    def numericalSort(value):
            numbers = re.compile(r'(\d+)')
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts
            
    #画像を昇順に取得        
    files = sorted(glob.glob(data_dir_path+'/*jpg'), key = numericalSort)
    
    #jpgファイルをリネームして保存
    for file in files:
        jpg = re.compile("jpg")
        if jpg.search(file):
            os.rename(file, os.path.join(data_dir_path, "%d.jpg" %(a)))
            a+=1
        else:
            pass

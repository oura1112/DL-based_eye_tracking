# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import cv2
import numpy as np
import linecache

save_num = []

exe_dir = os.path.dirname(os.path.abspath(__file__))
num_path = exe_dir + '/faces.txt'

read_data_path1 = exe_dir + '/datalabels2.txt'
save_data_path1 = exe_dir + '/datalabels2_.txt'

read_data_path2 = exe_dir + '/faces_x.txt'
save_data_path2 = exe_dir + '/f_x.txt'
read_data_path3 = exe_dir + '/faces_y.txt'
save_data_path3 = exe_dir + '/f_y.txt'
read_data_path4 = exe_dir + '/faces_w.txt'
save_data_path4 = exe_dir + '/f_w.txt'

read_data_path5 = exe_dir + '/r_eyes_x.txt'
save_data_path5 = exe_dir + '/r_x.txt'
read_data_path6 = exe_dir + '/r_eyes_y.txt'
save_data_path6 = exe_dir + '/r_y.txt'

read_data_path7 = exe_dir + '/l_eyes_x.txt'
save_data_path7 = exe_dir + '/l_x.txt'
read_data_path8 = exe_dir + '/l_eyes_y.txt'
save_data_path8 = exe_dir + '/l_y.txt'

read_data_path9 = exe_dir + '/datalabels4.txt'
save_data_path9 = exe_dir + '/datalabels4_.txt'
read_data_path10 = exe_dir + '/datalabels6.txt'
save_data_path10 = exe_dir + '/datalabels6_.txt'
read_data_path11 = exe_dir + '/datalabels9.txt'
save_data_path11 = exe_dir + '/datalabels9_.txt'

#保存する番号を保持
with open(num_path, 'r') as f:
        for f_num in f.readlines():
            save_num.append(f_num.replace('\n',''))
#print(save_num)

#保存対象の行番号のデータのみ保存
with open(save_data_path1, 'w') as f1, open(save_data_path2, 'w') as f2, open(save_data_path3, 'w') as f3, open(save_data_path4, 'w') as f4, open(save_data_path5, 'w') as f5, open(save_data_path6, 'w') as f6, open(save_data_path7, 'w') as f7, open(save_data_path8, 'w') as f8, open(save_data_path9, 'w') as f9, open(save_data_path10, 'w') as f10, open(save_data_path11, 'w') as f11:
    for i in save_num:
        save_number1 = linecache.getline(read_data_path1, int(i))
        f1.write(save_number1)
        
        save_number2 = linecache.getline(read_data_path2, int(i))
        f2.write(save_number2)
        
        save_number3 = linecache.getline(read_data_path3, int(i))
        f3.write(save_number3)
        
        save_number4 = linecache.getline(read_data_path4, int(i))
        f4.write(save_number4)
        
        save_number5 = linecache.getline(read_data_path5, int(i))
        f5.write(save_number5)
        
        save_number6 = linecache.getline(read_data_path6, int(i))
        f6.write(save_number6)
        
        save_number7 = linecache.getline(read_data_path7, int(i))
        f7.write(save_number7)
        
        save_number8 = linecache.getline(read_data_path8, int(i))
        f8.write(save_number8)
        
        save_number9 = linecache.getline(read_data_path9, int(i))
        f9.write(save_number9)
        
        save_number10 = linecache.getline(read_data_path10, int(i))
        f10.write(save_number10)
        
        save_number11 = linecache.getline(read_data_path11, int(i))
        f11.write(save_number11)


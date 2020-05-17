# -*- coding: utf-8 -*-

import os
import re
import glob
import cv2
import numpy as np
import linecache

exe_dir = os.path.dirname(os.path.abspath(__file__))
datanumber_path = exe_dir + '/faces.txt'
r_eyes_x_path = exe_dir + '/r_x.txt'
r_eyes_y_path = exe_dir + '/r_y.txt'
l_eyes_x_path = exe_dir + '/l_x.txt'
l_eyes_y_path = exe_dir + '/l_y.txt'
faces_x_path = exe_dir + '/f_x.txt'
faces_y_path = exe_dir + '/f_y.txt'
faces_w_path = exe_dir + '/f_w.txt'
label_path = exe_dir + '/datalabels9_.txt'

save_num_path2 = exe_dir + '/number_val.txt'
r_eyes_x_path2 = exe_dir + '/r_x_val.txt'
r_eyes_y_path2 = exe_dir + '/r_y_val.txt'
l_eyes_x_path2 = exe_dir + '/l_x_val.txt'
l_eyes_y_path2 = exe_dir + '/l_y_val.txt'
faces_x_path2 = exe_dir + '/faces_x_val.txt'
faces_y_path2 = exe_dir + '/faces_y_val.txt'
faces_w_path2 = exe_dir + '/faces_w_val.txt'
label2_path2 = exe_dir + '/datalabels2_val.txt'
label4_path2 = exe_dir + '/datalabels4_val.txt'
label6_path2 = exe_dir + '/datalabels6_val.txt'
label9_path2 = exe_dir + '/datalabels9_val.txt'

save_num_path3 = exe_dir + '/number_tr.txt'
r_eyes_x_path3 = exe_dir + '/r_x_tr.txt'
r_eyes_y_path3 = exe_dir + '/r_y_tr.txt'
l_eyes_x_path3 = exe_dir + '/l_x_tr.txt'
l_eyes_y_path3 = exe_dir + '/l_y_tr.txt'
faces_x_path3 = exe_dir + '/faces_x_tr.txt'
faces_y_path3 = exe_dir + '/faces_y_tr.txt'
faces_w_path3 = exe_dir + '/faces_w_tr.txt'
label2_path3 = exe_dir + '/datalabels2_tr.txt'
label4_path3 = exe_dir + '/datalabels4_tr.txt'
label6_path3 = exe_dir + '/datalabels6_tr.txt'
label9_path3 = exe_dir + '/datalabels9_tr.txt'

save_num_path4 = exe_dir + '/number_test.txt'
r_eyes_x_path4 = exe_dir + '/r_x_test.txt'
r_eyes_y_path4 = exe_dir + '/r_y_test.txt'
l_eyes_x_path4 = exe_dir + '/l_x_test.txt'
l_eyes_y_path4 = exe_dir + '/l_y_test.txt'
faces_x_path4 = exe_dir + '/faces_x_test.txt'
faces_y_path4 = exe_dir + '/faces_y_test.txt'
faces_w_path4 = exe_dir + '/faces_w_test.txt'
label2_path4 = exe_dir + '/datalabels2_test.txt'
label4_path4 = exe_dir + '/datalabels4_test.txt'
label6_path4 = exe_dir + '/datalabels6_test.txt'
label9_path4 = exe_dir + '/datalabels9_test.txt'

#3200個のデータをテストデータにする
with open(save_num_path2, 'w') as num_f2, open(label9_path2, 'w') as label_f2,open(r_eyes_x_path2, 'w') as rex_f2, open(r_eyes_y_path2, 'w') as rey_f2, open(l_eyes_x_path2, 'w') as lex_f2, open(l_eyes_y_path2, 'w') as ley_f2, open(faces_x_path2, 'w') as fx_f2, open(faces_y_path2, 'w') as fy_f2, open(faces_w_path2, 'w') as fw_f2, open(save_num_path3, 'w') as num_f3, open(label9_path3, 'w') as label_f3, open(r_eyes_x_path3, 'w') as rex_f3, open(r_eyes_y_path3, 'w') as rey_f3, open(l_eyes_x_path3, 'w') as lex_f3, open(l_eyes_y_path3, 'w') as ley_f3, open(faces_x_path3, 'w') as fx_f3, open(faces_y_path3, 'w') as fy_f3, open(faces_w_path3, 'w') as fw_f3:
    
    for i in range(1,287): 
        num_f2.write(linecache.getline(datanumber_path, int(i)))
        label_f2.write(linecache.getline(label_path, int(i)))
        rex_f2.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f2.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f2.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f2.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f2.write(linecache.getline(faces_x_path, int(i)))
        fy_f2.write(linecache.getline(faces_y_path, int(i)))
        fw_f2.write(linecache.getline(faces_w_path, int(i)))
   
    for i in range(287,2087): 
        num_f3.write(linecache.getline(datanumber_path, int(i)))
        label_f3.write(linecache.getline(label_path, int(i)))
        rex_f3.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f3.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f3.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f3.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f3.write(linecache.getline(faces_x_path, int(i)))
        fy_f3.write(linecache.getline(faces_y_path, int(i)))
        fw_f3.write(linecache.getline(faces_w_path, int(i)))
   
    for i in range(2287,2573):
        num_f2.write(linecache.getline(datanumber_path, int(i)))
        label_f2.write(linecache.getline(label_path, int(i)))
        rex_f2.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f2.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f2.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f2.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f2.write(linecache.getline(faces_x_path, int(i)))
        fy_f2.write(linecache.getline(faces_y_path, int(i)))
        fw_f2.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(2573,4373): 
        num_f3.write(linecache.getline(datanumber_path, int(i)))
        label_f3.write(linecache.getline(label_path, int(i)))
        rex_f3.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f3.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f3.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f3.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f3.write(linecache.getline(faces_x_path, int(i)))
        fy_f3.write(linecache.getline(faces_y_path, int(i)))
        fw_f3.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(4573,4859):
        num_f2.write(linecache.getline(datanumber_path, int(i)))
        label_f2.write(linecache.getline(label_path, int(i)))
        rex_f2.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f2.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f2.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f2.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f2.write(linecache.getline(faces_x_path, int(i)))
        fy_f2.write(linecache.getline(faces_y_path, int(i)))
        fw_f2.write(linecache.getline(faces_w_path, int(i)))
  
    for i in range(4859,6659): 
        num_f3.write(linecache.getline(datanumber_path, int(i)))
        label_f3.write(linecache.getline(label_path, int(i)))
        rex_f3.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f3.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f3.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f3.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f3.write(linecache.getline(faces_x_path, int(i)))
        fy_f3.write(linecache.getline(faces_y_path, int(i)))
        fw_f3.write(linecache.getline(faces_w_path, int(i)))            
    
    for i in range(6859,7145): 
        num_f2.write(linecache.getline(datanumber_path, int(i)))
        label_f2.write(linecache.getline(label_path, int(i)))
        rex_f2.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f2.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f2.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f2.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f2.write(linecache.getline(faces_x_path, int(i)))
        fy_f2.write(linecache.getline(faces_y_path, int(i)))
        fw_f2.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(7145,8945): 
        num_f3.write(linecache.getline(datanumber_path, int(i)))
        label_f3.write(linecache.getline(label_path, int(i)))
        rex_f3.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f3.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f3.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f3.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f3.write(linecache.getline(faces_x_path, int(i)))
        fy_f3.write(linecache.getline(faces_y_path, int(i)))
        fw_f3.write(linecache.getline(faces_w_path, int(i)))
   
    for i in range(9145,9431):
        num_f2.write(linecache.getline(datanumber_path, int(i)))
        label_f2.write(linecache.getline(label_path, int(i)))
        rex_f2.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f2.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f2.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f2.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f2.write(linecache.getline(faces_x_path, int(i)))
        fy_f2.write(linecache.getline(faces_y_path, int(i)))
        fw_f2.write(linecache.getline(faces_w_path, int(i)))
   
    for i in range(9431,11231): 
        num_f3.write(linecache.getline(datanumber_path, int(i)))
        label_f3.write(linecache.getline(label_path, int(i)))
        rex_f3.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f3.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f3.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f3.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f3.write(linecache.getline(faces_x_path, int(i)))
        fy_f3.write(linecache.getline(faces_y_path, int(i)))
        fw_f3.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(11431,11717):
        num_f2.write(linecache.getline(datanumber_path, int(i)))
        label_f2.write(linecache.getline(label_path, int(i)))
        rex_f2.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f2.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f2.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f2.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f2.write(linecache.getline(faces_x_path, int(i)))
        fy_f2.write(linecache.getline(faces_y_path, int(i)))
        fw_f2.write(linecache.getline(faces_w_path, int(i)))
   
    for i in range(11717,13517): 
        num_f3.write(linecache.getline(datanumber_path, int(i)))
        label_f3.write(linecache.getline(label_path, int(i)))
        rex_f3.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f3.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f3.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f3.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f3.write(linecache.getline(faces_x_path, int(i)))
        fy_f3.write(linecache.getline(faces_y_path, int(i)))
        fw_f3.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(13717,14003):
        num_f2.write(linecache.getline(datanumber_path, int(i)))
        label_f2.write(linecache.getline(label_path, int(i)))
        rex_f2.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f2.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f2.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f2.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f2.write(linecache.getline(faces_x_path, int(i)))
        fy_f2.write(linecache.getline(faces_y_path, int(i)))
        fw_f2.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(14003,15803): 
        num_f3.write(linecache.getline(datanumber_path, int(i)))
        label_f3.write(linecache.getline(label_path, int(i)))
        rex_f3.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f3.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f3.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f3.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f3.write(linecache.getline(faces_x_path, int(i)))
        fy_f3.write(linecache.getline(faces_y_path, int(i)))
        fw_f3.write(linecache.getline(faces_w_path, int(i)))
   
    for i in range(16003,16289):
        num_f2.write(linecache.getline(datanumber_path, int(i)))
        label_f2.write(linecache.getline(label_path, int(i)))
        rex_f2.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f2.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f2.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f2.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f2.write(linecache.getline(faces_x_path, int(i)))
        fy_f2.write(linecache.getline(faces_y_path, int(i)))
        fw_f2.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(16289,18089): 
        num_f3.write(linecache.getline(datanumber_path, int(i)))
        label_f3.write(linecache.getline(label_path, int(i)))
        rex_f3.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f3.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f3.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f3.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f3.write(linecache.getline(faces_x_path, int(i)))
        fy_f3.write(linecache.getline(faces_y_path, int(i)))
        fw_f3.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(18289,18575):
        num_f2.write(linecache.getline(datanumber_path, int(i)))
        label_f2.write(linecache.getline(label_path, int(i)))
        rex_f2.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f2.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f2.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f2.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f2.write(linecache.getline(faces_x_path, int(i)))
        fy_f2.write(linecache.getline(faces_y_path, int(i)))
        fw_f2.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(18575,20375): 
        num_f3.write(linecache.getline(datanumber_path, int(i)))
        label_f3.write(linecache.getline(label_path, int(i)))
        rex_f3.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f3.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f3.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f3.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f3.write(linecache.getline(faces_x_path, int(i)))
        fy_f3.write(linecache.getline(faces_y_path, int(i)))
        fw_f3.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(20575,20861):
        num_f2.write(linecache.getline(datanumber_path, int(i)))
        label_f2.write(linecache.getline(label_path, int(i)))
        rex_f2.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f2.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f2.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f2.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f2.write(linecache.getline(faces_x_path, int(i)))
        fy_f2.write(linecache.getline(faces_y_path, int(i)))
        fw_f2.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(20861,22661): 
        num_f3.write(linecache.getline(datanumber_path, int(i)))
        label_f3.write(linecache.getline(label_path, int(i)))
        rex_f3.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f3.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f3.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f3.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f3.write(linecache.getline(faces_x_path, int(i)))
        fy_f3.write(linecache.getline(faces_y_path, int(i)))
        fw_f3.write(linecache.getline(faces_w_path, int(i)))
        
with open(save_num_path4, 'w') as num_f4, open(label9_path4, 'w') as label_f4, open(r_eyes_x_path4, 'w') as rex_f4, open(r_eyes_y_path4, 'w') as rey_f4, open(l_eyes_x_path4, 'w') as lex_f4, open(l_eyes_y_path4, 'w') as ley_f4, open(faces_x_path4, 'w') as fx_f4, open(faces_y_path4, 'w') as fy_f4, open(faces_w_path4, 'w') as fw_f4:

    for i in range(2087,2287): 
        num_f4.write(linecache.getline(datanumber_path, int(i)))
        label_f4.write(linecache.getline(label_path, int(i)))
        rex_f4.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f4.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f4.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f4.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f4.write(linecache.getline(faces_x_path, int(i)))
        fy_f4.write(linecache.getline(faces_y_path, int(i)))
        fw_f4.write(linecache.getline(faces_w_path, int(i)))
        
    for i in range(4373,4573): 
        num_f4.write(linecache.getline(datanumber_path, int(i)))
        label_f4.write(linecache.getline(label_path, int(i)))
        rex_f4.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f4.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f4.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f4.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f4.write(linecache.getline(faces_x_path, int(i)))
        fy_f4.write(linecache.getline(faces_y_path, int(i)))
        fw_f4.write(linecache.getline(faces_w_path, int(i)))
    
    for i in range(6659,6859): 
        num_f4.write(linecache.getline(datanumber_path, int(i)))
        label_f4.write(linecache.getline(label_path, int(i)))
        rex_f4.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f4.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f4.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f4.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f4.write(linecache.getline(faces_x_path, int(i)))
        fy_f4.write(linecache.getline(faces_y_path, int(i)))
        fw_f4.write(linecache.getline(faces_w_path, int(i)))
        
    for i in range(8945,9145): 
        num_f4.write(linecache.getline(datanumber_path, int(i)))
        label_f4.write(linecache.getline(label_path, int(i)))
        rex_f4.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f4.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f4.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f4.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f4.write(linecache.getline(faces_x_path, int(i)))
        fy_f4.write(linecache.getline(faces_y_path, int(i)))
        fw_f4.write(linecache.getline(faces_w_path, int(i)))
        
    for i in range(11231,11431): 
        num_f4.write(linecache.getline(datanumber_path, int(i)))
        label_f4.write(linecache.getline(label_path, int(i)))
        rex_f4.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f4.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f4.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f4.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f4.write(linecache.getline(faces_x_path, int(i)))
        fy_f4.write(linecache.getline(faces_y_path, int(i)))
        fw_f4.write(linecache.getline(faces_w_path, int(i)))
        
    for i in range(13517,13717): 
        num_f4.write(linecache.getline(datanumber_path, int(i)))
        label_f4.write(linecache.getline(label_path, int(i)))
        rex_f4.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f4.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f4.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f4.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f4.write(linecache.getline(faces_x_path, int(i)))
        fy_f4.write(linecache.getline(faces_y_path, int(i)))
        fw_f4.write(linecache.getline(faces_w_path, int(i)))
        
    for i in range(15803,16003): 
        num_f4.write(linecache.getline(datanumber_path, int(i)))
        label_f4.write(linecache.getline(label_path, int(i)))
        rex_f4.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f4.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f4.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f4.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f4.write(linecache.getline(faces_x_path, int(i)))
        fy_f4.write(linecache.getline(faces_y_path, int(i)))
        fw_f4.write(linecache.getline(faces_w_path, int(i)))
        
    for i in range(18089,18289): 
        num_f4.write(linecache.getline(datanumber_path, int(i)))
        label_f4.write(linecache.getline(label_path, int(i)))
        rex_f4.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f4.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f4.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f4.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f4.write(linecache.getline(faces_x_path, int(i)))
        fy_f4.write(linecache.getline(faces_y_path, int(i)))
        fw_f4.write(linecache.getline(faces_w_path, int(i)))
        
    for i in range(20375,20575): 
        num_f4.write(linecache.getline(datanumber_path, int(i)))
        label_f4.write(linecache.getline(label_path, int(i)))
        rex_f4.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f4.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f4.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f4.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f4.write(linecache.getline(faces_x_path, int(i)))
        fy_f4.write(linecache.getline(faces_y_path, int(i)))
        fw_f4.write(linecache.getline(faces_w_path, int(i)))
        
    for i in range(22661,22869): 
        num_f4.write(linecache.getline(datanumber_path, int(i)))
        label_f4.write(linecache.getline(label_path, int(i)))
        rex_f4.write(linecache.getline(r_eyes_x_path, int(i)))
        rey_f4.write(linecache.getline(r_eyes_y_path, int(i)))
        lex_f4.write(linecache.getline(l_eyes_x_path, int(i)))
        ley_f4.write(linecache.getline(l_eyes_y_path, int(i)))
        fx_f4.write(linecache.getline(faces_x_path, int(i)))
        fy_f4.write(linecache.getline(faces_y_path, int(i)))
        fw_f4.write(linecache.getline(faces_w_path, int(i)))


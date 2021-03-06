# -*- coding: utf-8 -*-
import os
import sys
import re

ver = 900/3
side = 1440/3

side1 = side
side2 = side*2

ver1 = ver
ver2 = ver*2

label = []
exception = 0

exedir = os.path.dirname(os.path.abspath(__file__))

#注視座標が書かれているテキストファイルまでのディレクトリ.実行ディレクトリ配下に置く.
file_input = exedir + "/MPIIFaceGaze/p14/p14.txt"
#datalebel9.txtのパス
file_output = exedir + "/datalabel9.txt"

#座標値から見ている領域を分類
with open(file_input) as f:
    for l in f:
        line = re.split(" +", l)
        #print(line[1])
        #print(line[2])

        if (float(line[1])<=side1 and float(line[2])<=ver1):
            label.append('1')
        elif (float(line[1])>side1 and float(line[1])<=side2 and float(line[2])<=ver1):
            label.append('2')
        elif (float(line[1])>side2 and float(line[2])<=ver1):
            label.append('3')
        elif (float(line[1])<=side1 and float(line[2])>ver1 and float(line[2])<=ver2):
            label.append('4')
        elif (float(line[1])>side1 and float(line[1])<=side2 and float(line[2])>ver1 and float(line[2])<=ver2):
            label.append('5')
        elif (float(line[1])>side2 and float(line[2])>ver1 and float(line[2])<=ver2):
            label.append('6')
        elif (float(line[1])<=side1 and float(line[2])>ver2):
            label.append('7')
        elif (float(line[1])>side1 and float(line[1])<=side2 and float(line[2])>ver2):
            label.append('8')
        elif (float(line[1])>side2 and float(line[2])>ver2):
            label.append('9')
        else:
            exception += 1
f.close()
print(exception)
with open(file_output, 'a') as f:
    for x in label:
        f.write(x + "\n")
f.close()

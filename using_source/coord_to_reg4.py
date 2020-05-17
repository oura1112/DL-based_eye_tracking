# -*- coding: utf-8 -*-
import os
import sys
import re

ver = 900/2
side = 1440/2

side1 = side

ver1 = ver

label = []
exception = 0

exedir = os.path.dirname(os.path.abspath(__file__))

#注視座標が書かれているテキストファイルまでのディレクトリ.実行ディレクトリ配下に置く.
file_input = exedir + "/MPIIFaceGaze/p14/p14.txt"
#datalebel.txtのパス
file_output = exedir + "/datalabel4.txt"

#座標値から見ている領域を分類
with open(file_input) as f:
    for l in f:
        line = re.split(" +", l)
        #print(line[1])
        #print(line[2])

        if (float(line[1])<=side1 and float(line[2])<=ver1):
            label.append('1')
        elif (float(line[1])>side1 and float(line[2])<=ver1):
            label.append('2')
        elif (float(line[1])<=side1 and float(line[2])>ver1):
            label.append('3')
        elif (float(line[1])>side1 and float(line[2])>ver1):
            label.append('4')
        else:
            exception += 1
f.close()
print(exception)
with open(file_output, 'a') as f:
    for x in label:
        f.write(x + "\n")
f.close()

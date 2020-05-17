# -*- coding: utf-8 -*-

import cv2
import numpy as np 
import os

class face_eyes_detect_class:
    def __init__(self):
        print("Initialized!")
        
    def eye_img_resize(self, eye_imgs):
        #画像の形状を取得しリサイズ(24*24*3)
        width, height = 25, 25
        size = (width, height)
        eyes_resize = []
        for img in eye_imgs:
            eyes_resize.append(cv2.resize(img, size))

        return eyes_resize
        
    def face_img_resize(self, face_imgs):
        #画像の形状を取得しリサイズ(448*448*3)
        width, height = 224, 224
        size = (width, height)
        faces_resize = []
        for img in face_imgs:
            faces_resize.append(cv2.resize(img, size))

        return faces_resize
        
    def face_eyes_detect(self, imgs_src, labels2, labels4, labels6, labels9): 
        # 顔判定で使うxmlファイルを指定する。複数のパターンのカスケードを作っておく。
        face_cascade_path1 =  os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/haarcascade_frontalface_default.xml"
        face_cascade_path2 =  os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/haarcascade_frontalface_alt.xml"
        face_cascade_path3 =  os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/haarcascade_frontalface_alt2.xml"
        face_cascade_path4 =  os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/haarcascade_frontalface_alt_tree.xml"
        face_cascade_path5 =  os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/haarcascade_profileface.xml"
        eye_cascade_path1 =  os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/haarcascade_eye.xml"
        eye_cascade_path2 =  os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
        r_eye_cascade_path =  os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/haarcascade_righteye_2splits.xml"
        l_eye_cascade_path =  os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/haarcascade_lefteye_2splits.xml"
        
        face_cascade1 = cv2.CascadeClassifier(face_cascade_path1)
        face_cascade2 = cv2.CascadeClassifier(face_cascade_path2)
        face_cascade3 = cv2.CascadeClassifier(face_cascade_path3)
        face_cascade4 = cv2.CascadeClassifier(face_cascade_path4)
        face_cascade5 = cv2.CascadeClassifier(face_cascade_path5)
        eye_cascade1 = cv2.CascadeClassifier(eye_cascade_path1)
        eye_cascade2 = cv2.CascadeClassifier(eye_cascade_path2)
        r_eye_cascade = cv2.CascadeClassifier(r_eye_cascade_path)
        l_eye_cascade = cv2.CascadeClassifier(l_eye_cascade_path)

        minsize_f = (200, 200)
        minsize_e = (45, 45)

        # グレースケールに変換
        imgs_gray = []
        for img in imgs_src:
            imgs_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        print(np.array(imgs_gray).shape)

        #顔判定。検出失敗時は他のカスケードで検出
        """ 
        minSize で顔判定する際の最小の四角の大きさを指定できる。
        (小さい値を指定し過ぎると顔っぽい小さなシミのような部分も判定されてしまう。)
        """
        faces = []
        labels2_1 = []
        labels4_1 = []
        labels6_1 = []
        labels9_1 = []
        imgs_gray_ = []
        imgs_src_ = []
        
        #print(imgs_gray.shape)
        for (img_src, img_gray, label2, label4, label6, label9) in zip(imgs_src, imgs_gray, labels2, labels4, labels6, labels9):          
            detection_f = face_cascade1.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2, minSize=minsize_f)
            if len(detection_f) == 0:
               detection_f = face_cascade2.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2, minSize=minsize_f) 
               if len(detection_f) == 0:
                   detection_f = face_cascade3.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2, minSize=minsize_f)
                   if len(detection_f) == 0:
                       detection_f = face_cascade4.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2, minSize=minsize_f)
                       if len(detection_f) == 0:
                           detection_f = face_cascade5.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2, minSize=minsize_f)
                           if len(detection_f) == 0:
                               continue
            faces.append(detection_f)    #切り出した顔画像を保持
            labels2_1.append(label2)       #入力画像の内、顔を切り出せた画像のラベルを保持
            labels4_1.append(label4)
            labels6_1.append(label6)
            labels9_1.append(label9)
            imgs_src_.append(img_src)    #入力画像の内、顔を切り出せた画像の元画像を保持
            imgs_gray_.append(img_gray)  #入力画像の内、顔を切り出せた画像の元画像(グレースケール)を保持
        
        #入力の両目,顔画像保持用の変数
        right_eye_color = []
        left_eye_color = []
        faces_color = []
        #顔、目の座標等の保持用の変数
        right_eye_x = []
        right_eye_y = []
        left_eye_x = []
        left_eye_y = []
        faces_x = []
        faces_y = []
        faces_w = []
        #最終的なラベルの保持
        labels2_2 = []
        labels4_2 = []
        labels6_2 = []
        labels9_2 = []
        roots_2 = []
        #最終的な入力画像の組数
        sheets_count = 0
        r_eye = 0
        l_eye = 0
        
        #複数の画像を１枚ずつ処理
        for (face, img_gray_, img_src_, label2_1, label4_1, label6_1, label9_1) in zip(faces, imgs_gray_, imgs_src_, labels2_1, labels4_1, labels6_1, labels9_1):
            #print(3)
            #1枚の顔に対しての処理
            for (fx, fy, fw, fh) in face:
                #顔カラー領域の保持
                face_color = img_src_[fy:fy+fh, fx:fx+fw]
                #顔領域のグレースケール化の保持
                face_gray = img_gray_[fy:fy+fh, fx:fx+fw]

                #顔領域から目領域の抽出。検出失敗時は別のカスケードで検出。検出できなかった場合は次の顔画像へ移る。
                detection_e  =  eye_cascade1.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=2, minSize=minsize_e)
                if not len(detection_e) == 2:
                    detection_e = eye_cascade2.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=2, minSize=minsize_e)
                    if not len(detection_e) == 2:
                    """
                    誤検出が多いため使用しない
                        detection_r_e = r_eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=2, minSize=minsize_e)
                        detection_l_e = r_eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=2, minSize=minsize_e)
                    if not (len(detection_r_e) == 1 and len(detection_l_e) == 1):
                        print('Could not detect!')
                    """
                        break
                            
                #print(detection_e)        
                faces_color.append(img_src_[fy:fy+fh, fx:fx+fw])
                labels2_2.append(label2_1)
                labels4_2.append(label4_1)
                labels6_2.append(label6_1)
                labels9_2.append(label9_1)
                sheets_count += 1
                
                if len(detection_e) == 2:
                    eyes = detection_e
                    #保持した目領域の左右判定(x座標の比較)
                    if eyes[0][0] < eyes[1][0]:
                        r_eye = eyes[0]
                        l_eye = eyes[1]
                    else:
                        r_eye = eyes[1]
                        l_eye = eyes[0]
                """
                elif len(detection_e) != 2:
                    r_eye = detection_r_e[0]
                    l_eye = detection_l_e[0]             
                    print(r_eye)
                """

                #右目カラー領域の保持
                right_eye_color.append(face_color[r_eye[1]:r_eye[1]+r_eye[3], r_eye[0]:r_eye[0]+r_eye[2]])
                #右目の左上(x,y)座標の保持
                right_eye_x.append(fx + r_eye[0])
                right_eye_y.append(fy + r_eye[1])
                
                #左目カラー領域の保持
                left_eye_color.append(face_color[l_eye[1]:l_eye[1]+l_eye[3], l_eye[0]:l_eye[0]+l_eye[2]])
                #左目の左上(x,y)座標の保持
                left_eye_x.append(fx + l_eye[0])
                left_eye_y.append(fy + l_eye[1])
                    
                #顔の(x,y)座標及び検出領域の横幅の保持
                faces_x.append(fx)
                faces_y.append(fy)
                faces_w.append(fw)
        
        right_eye_color = self.eye_img_resize(right_eye_color)
        left_eye_color = self.eye_img_resize(left_eye_color)
        faces_color = self.face_img_resize(faces_color)
                
        print(sheets_count)

        return [right_eye_color, left_eye_color, faces_color, right_eye_x, right_eye_y, left_eye_x, left_eye_y, faces_x, faces_y, faces_w, sheets_count, labels2_2, labels4_2, labels6_2, labels9_2]

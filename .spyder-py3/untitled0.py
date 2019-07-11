# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 09:33:17 2018

@author: Yajat
"""
import cv2,glob,os

im_path = ""
s_path = ""

images = glob.glob(os.path.join(im_path,"*.jpg"))

i = 1
for image in images:
    img = cv2.imread(image,1)

    re = cv2.resize(img,(50,50))

    cv2.imshow("Checking",re)

    cv2.waitKey(500)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(s_path,"resized_{}.jpg".format(i)),re)
    i = i+1


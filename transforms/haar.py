import os
from cv2 import imread
import pywt
import transforms.fourier as fourier
import numpy as np
def haar_transform():
    X = []
    Y = []
    print(os.getcwd())
    for i in os.listdir(os.getcwd()+'/data'):
        for j in os.listdir(os.getcwd()+'/data/'+i):
            image = imread(os.getcwd()+'/data/'+i+'/'+j)
            image = fourier.preprocess(image)
            image = pywt.wavedec2(image, 'haar', level=5)
            coeffs_H=list(image)  
            coeffs_H[0] *= 0;  
            # reconstruction
            imArray_H=pywt.waverec2(coeffs_H, 'haar');
            imArray_H *= 255;
            imArray_H =  np.uint8(imArray_H)
            X.append(imArray_H)
            Y.append(i)
    X = np.asarray(X)
    Y = np.asarray(Y)
    # print(X)
    return X, Y
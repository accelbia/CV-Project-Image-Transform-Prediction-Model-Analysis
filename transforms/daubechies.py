import os
from cv2 import imread
import numpy as np
from transforms.fourier import preprocess
import mahotas


def daubechies_transform():
    X = []
    Y = []
    print(os.getcwd())
    for i in os.listdir(os.getcwd()+'/data'):
        for j in os.listdir(os.getcwd()+'/data/'+i):
            image = imread(os.getcwd()+'/data/'+i+'/'+j)
            image = preprocess(image)
            image = mahotas.daubechies(image, 'D8')
            X.append(image)
            Y.append(i)
    # X = np.asarray(X)
    # Y = np.asarray(Y)
    # print(X)
    return X, Y

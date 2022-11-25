import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import transform
from skimage.exposure import equalize_hist
import os

def fourier_transform():
    X = []
    Y = []
    print(os.getcwd())
    for i in os.listdir(os.getcwd()+'/data'):
        for j in os.listdir(os.getcwd()+'/data/'+i):
            image = imread(os.getcwd()+'/data/'+i+'/'+j)
            image = preprocess(image)
            image = np.fft.fftshift(np.fft.fft2(image))
            X.append(image)
            Y.append(i)
    X = np.asarray(X)
    Y = np.asarray(Y)
    #print(X)
    return X, Y

def preprocess(img):
    img = transform.resize(img, (128, 128))
    img = rgb2gray(img)
    img = equalize_hist(img)
    return img
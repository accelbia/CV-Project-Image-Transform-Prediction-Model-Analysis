import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage import transform
from skimage.exposure import equalize_hist
import os
import shutil


def fourier_transform(path):
    # os.chdir('../')
    X = []
    Y = []
    print(os.getcwd())
    for i in os.listdir(os.getcwd()+'/data'):
        for j in os.listdir(os.getcwd()+'/data/'+i):
            image = imread(os.getcwd()+'/data/'+i+'/'+j)
            image = rgb2gray(image)
            image = transform.resize(image, (128, 128))
            image = equalize_hist(image)
            image = np.fft.fftshift(np.fft.fft2(image))
            X.append(image)
            Y.append(i)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(X)
    return X, Y
fourier_transform('Hi')
from cv2 import imread, transform
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from zipfile import ZipFile
from PIL import Image
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
import visualkeras
from PIL import Image
import pywt
from skimage.color import rgb2gray
from skimage import transform
from skimage.exposure import equalize_hist
import mahotas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def asImage(arr):
    arr = arr.astype(np.uint8)
    val =  cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return val

def canny_transform():
    X = []
    Y = []
    # print(os.getcwd())
    for i in os.listdir(os.getcwd()+'/data'):
        for j in os.listdir(os.getcwd()+'/data/'+i):
            image = imread(os.getcwd()+'/data/'+i+'/'+j)
            image = preprocess(image)
            image = image * 255
            image = image.astype(np.uint8)
            edges = cv2.Canny(image, 200, 100)
            X.append(asImage(edges))
            Y.append(i)
    X = np.asarray(X)
    Y = np.asarray(Y)
    # print(X)
    return X, Y

def fourier_transform():
    X = []
    Y = []
    # print(os.getcwd())
    for i in os.listdir(os.getcwd()+'/data'):
        for j in os.listdir(os.getcwd()+'/data/'+i):
            image = imread(os.getcwd()+'/data/'+i+'/'+j)
            image = preprocess(image)
            image = np.fft.fftshift(np.fft.fft2(image))
            # print(asImage(image).shape)
            X.append(asImage(image))
            Y.append(i)
    X = np.asarray(X)
    Y = np.asarray(Y)
    # print(X)
    return X, Y

def preprocess(img):
    img = transform.resize(img, (128, 128))
    img = rgb2gray(img)
    img = equalize_hist(img)
    return img

def haar_transform():
    X = []
    Y = []
    # print(os.getcwd())
    for i in os.listdir(os.getcwd()+'/data'):
        for j in os.listdir(os.getcwd()+'/data/'+i):
            image = imread(os.getcwd()+'/data/'+i+'/'+j)
            image = preprocess(image)
            image = pywt.wavedec2(image, 'haar', level=5)
            coeffs_H=list(image)  
            coeffs_H[0] *= 0;  
            # reconstruction
            imArray_H=pywt.waverec2(coeffs_H, 'haar');
            imArray_H *= 255;
            # imArray_H =  np.uint8(imArray_H)
            # print(asImage(image).shape)
            X.append(asImage(imArray_H))
            Y.append(i)
    X = np.asarray(X)
    Y = np.asarray(Y)
    # print(X)
    return X, Y

def daubechies_transform():
    X = []
    Y = []
    # print(os.getcwd())
    for i in os.listdir(os.getcwd()+'/data'):
        for j in os.listdir(os.getcwd()+'/data/'+i):
            image = imread(os.getcwd()+'/data/'+i+'/'+j)
            image = preprocess(image)
            image = mahotas.daubechies(image, 'D8')
            # print(asImage(image).shape)
            X.append(asImage(image))
            Y.append(i)
    X = np.asarray(X)
    Y = np.asarray(Y)
    # print(X)
    return X, Y

def show_plots():
    col1, col2 = st.columns(2)
    
    with col1:
        st.title('Daubauchies')
        image = Image.open('cache/daubechies_acc.png')
        st.image(image, caption='Daubauchies Accuracy')
        image2 = Image.open('cache/daubechies_loss.png')
        st.image(image, caption='Daubauchies Loss')
        
    with col2:
        st.title('Haar')
        image = Image.open('cache/haar_acc.png')
        st.image(image, caption='Haar Accuracy')
        image = Image.open('cache/haar_loss.png')
        st.image(image, caption='Haar Loss')
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.title('Fourier')
        image = Image.open('cache/fourier_acc.png')
        st.image(image, caption='Fourier Accuracy')
        image = Image.open('cache/fourier_loss.png')
        st.image(image, caption='Fourier Loss')
        
    with col4:
        st.title('Canny Edge')
        image = Image.open('cache/canny_acc.png')
        st.image(image, caption='Canny Edge Accuracy')
        image = Image.open('cache/canny_loss.png')
        st.image(image, caption='Canny Edge Loss')


def start_training(classes, final_model, test_ratio, val_ratio, optimizer, loss, metrics):
    
    # try:
    #     shutil.rmtree('cache/dataset')
    # except:
    #     pass
    # finally:
    #     os.mkdir('cache/dataset')
    # splitfolders.ratio('data', 'cache/dataset', ratio=(1-test_ratio-val_ratio, val_ratio, test_ratio))
    
    X_daub, Y_daub = daubechies_transform()
    X_four, Y_four = fourier_transform()
    X_haar, Y_haar = haar_transform()
    X_can, Y_can = canny_transform()
    
    le = LabelEncoder()
    le.fit(classes)
    Y_daub = le.transform(Y_daub)
    Y_four = le.transform(Y_four)
    Y_haar = le.transform(Y_haar)
    Y_can = le.transform(Y_can)
    
    Y_daub = to_categorical(Y_daub)
    Y_four = to_categorical(Y_four)
    Y_haar = to_categorical(Y_haar)
    Y_can = to_categorical(Y_can)
    # st.write('X_daub', X_daub.shape)
    # st.write(Y_daub)
    # st.write('X_four', X_four.shape)
    # st.write(Y_four)
    # st.write('Y_haar', X_haar.shape)
    # st.write(Y_haar)
    
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_daub, Y_daub, test_size=test_ratio, random_state=42, shuffle=True)
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_four, Y_four, test_size=test_ratio, random_state=42, shuffle=True)
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_haar, Y_haar, test_size=test_ratio, random_state=42, shuffle=True)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_can, Y_can, test_size=test_ratio, random_state=42, shuffle=True)
    
    st.write('X_train, Y_train', X_train_d.shape, y_train_d.shape)
    st.write('X_test, Y_test', X_test_d.shape, y_test_d.shape)
    
    # st.write(os.listdir('cache/dataset/train'))
    # st.write(optimizer, loss, metrics)

    
    # for percent_complete in range(100):
    #     time.sleep(0.1)
    #     my_bar.progress(percent_complete + 1)
    #     if percent_complete == 99:
    #         percent_complete = 0
    final_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model_d = final_model
    model_f = final_model
    model_h = final_model
    model_c = final_model
    my_bar = st.progress(0)
    hist_d = model_d.fit(X_train_d, y_train_d, epochs=10, validation_data=(X_test_d, y_test_d))
    my_bar.progress(25)
    hist_f = model_f.fit(X_train_f, y_train_f, epochs=10, validation_data=(X_test_f, y_test_f))
    my_bar.progress(50)
    hist_h = model_h.fit(X_train_h, y_train_h, epochs=10, validation_data=(X_test_h, y_test_h))
    my_bar.progress(75)
    hist_c = model_c.fit(X_train_c, y_train_c, epochs=10, validation_data=(X_test_c, y_test_c))
    my_bar.progress(100)
    
    st.write(hist_d.history.keys())

    save_data(hist_d, hist_f, hist_h, hist_c)
    show_plots()
    

plot_button_status = True
st.title('Image Transform Prediction Model Analysis')

files = st.file_uploader('Upload here', type=['zip'], accept_multiple_files=False)
if files is not None:
    with ZipFile(files,"r") as z:
        z.extractall('data')
    classes = os.listdir('data')
    listing = {}
    for i in classes:
        listing[i] = len(os.listdir('data/'+i))
    # listing = {'Class':list(listing.keys()),
    #            'Count':list(listing.values())}
    # count_df = pd.DataFrame(listing)
    # st.write('Classes found : ', count_df)
    col = list(st.columns(len(list(listing.keys()))))
    with st.container():
        for index, i in enumerate(col):
            with i:
                name = list(listing.keys())[index]
                count = list(listing.values())[index]
                st.metric(name, count)
model_name = st.selectbox('Select CNN Model',['VGG16', 'ResNet', 'InceptionNet', 'XceptionNet'], index = 0)
model = None
button_status = True
plot_button_status = True
if model_name and files:
    if model_name == 'VGG16':
        from keras.applications.vgg16 import VGG16
        model = VGG16(include_top=False, input_shape=(128, 128, 3), classes = len(classes))
    elif model_name == 'ResNet':
        from keras.applications.resnet import ResNet50
        model = ResNet50(include_top=False, input_shape=(128, 128, 3), classes = len(classes))
    elif model_name == 'InceptionNet':
        from keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(include_top=False, input_shape=(128, 128, 3), classes = len(classes))
    elif model_name == 'XceptionNet':
        from keras.applications.xception import Xception
        model = Xception(include_top=False, input_shape=(128, 128, 3), classes = len(classes))
    else:
        pass
    trainable = st.checkbox('Trainable')
    model.trainable = trainable
    visualkeras.layered_view(model, to_file='cache/imported_model.png', background_fill = '#0e1117')
    image = Image.open('cache/imported_model.png')
    st.image(image, caption=model_name)
    with st.expander("Expand model"):
        model.summary(print_fn=lambda x: st.text(x))
    neurons = []
    activation = []
    with st.container():
        choice , layers = st.columns(2)
        with choice:
            hidden_layers = st.number_input(label = 'Number of hidden layers', min_value=0, value=0, step=1)
            neurons = [0 for i in range(hidden_layers)]
            activation = ['relu' for i in range(hidden_layers)]
        with layers:
                for i in range(hidden_layers):
                    with st.expander("Layer : "+ str(i+1)):
                        neurons[i] = st.number_input(label = 'Number of neurons', min_value=2, value=len(classes), step=1, key = i)
                        activation[i] = st.radio('Activation Function', ['ReLU', 'sigmoid', 'tanh'], index=0, key=i+hidden_layers)

    final_model = Sequential()
    if model:
        final_model.add(model)
        
        final_model.add(Flatten())
        for i in range(hidden_layers):
            final_model.add(Dense(neurons[i], activation = activation[i]))
        final_model.add(Dense(len(classes), activation = 'softmax'))


    with st.expander("Final model"):
        final_model.summary(print_fn=lambda x: st.text(x))
    
    test_ratio = st.slider('Testing split', min_value=0, max_value=100, value=10, step=1)
    val_ratio = st.slider('Validation split', min_value=0, max_value=100-test_ratio, value=10, step=1)
    
    with st.container():
        optimizer, loss, metrics = st.columns(3)
        
        with optimizer:
            optimizer = st.radio('Optimizer', ['SGD', 'RMSprop', 'Adam', 'Adagrad'], index=2)
        
        with loss:
            loss = st.radio('Loss Function', ['CategoricalCrossentropy', 'SparseCategoricalCrossentropy'], index=0)
            
        with metrics:
            options = ['Accuracy', 'BinaryAccuracy', 'CategoricalAccuracy', 'AUC', 'Precision', 'Recall', 'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives', 'PrecisionAtRecall', 'SensitivityAtSpecificity', 'SpecificityAtSensitivity']
            if loss == 'SparseCategoricalCrossentropy':
                options.insert(3, 'SparseCategoricalAccuracy')
            metrics = st.multiselect('Metrics', options = options, default = ['Accuracy'])
            
            
    button_status = not button_status 

def save_data(hist_d, hist_f, hist_h, hist_c):
    
    plt.plot(hist_d.history['Accuracy'])
    plt.plot(hist_d.history['val_Accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cache/daubechies_acc.png')
    plt.figure().clear()
    plt.plot(hist_d.history['loss'])
    plt.plot(hist_d.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cache/daubechies_loss.png')
    plt.figure().clear()

    plt.plot(hist_f.history['Accuracy'])
    plt.plot(hist_f.history['val_Accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cache/fourier_acc.png')
    plt.figure().clear()
    plt.plot(hist_f.history['loss'])
    plt.plot(hist_f.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cache/fourier_loss.png')
    plt.figure().clear()

    plt.plot(hist_h.history['Accuracy'])
    plt.plot(hist_h.history['val_Accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cache/haar_acc.png')
    plt.figure().clear()
    plt.plot(hist_h.history['loss'])
    plt.plot(hist_h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cache/haar_loss.png')
    plt.figure().clear()

    plt.plot(hist_c.history['Accuracy'])
    plt.plot(hist_c.history['val_Accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cache/canny_acc.png')
    plt.figure().clear()
    plt.plot(hist_c.history['loss'])
    plt.plot(hist_c.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cache/canny_loss.png')
    plt.figure().clear()

if st.button('Start Training', disabled = button_status):
    start_training(classes, final_model, test_ratio=test_ratio/100, val_ratio=val_ratio/100, optimizer=optimizer, loss=loss, metrics=metrics)
    plot_button_status = False


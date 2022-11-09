import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import patoolib
from zipfile import ZipFile
from PIL import Image
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
import splitfolders
import shutil
import visualkeras
from PIL import Image

def start_training(final_model, test_ratio, val_ratio, optimizer, loss, metrics):
    try:
        shutil.rmtree('dataset')
    finally:
        os.mkdir('dataset')
    splitfolders.ratio('data', 'dataset', ratio=(1-test_ratio-val_ratio, val_ratio, test_ratio))
    st.write(os.listdir('dataset/train'))
    st.write(optimizer, loss, metrics)
    # model.compile(
    # optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # # Loss function to minimize
    # loss=keras.losses.SparseCategoricalCrossentropy(),
    # # List of metrics to monitor
    # metrics=[keras.metrics.SparseCategoricalAccuracy()],
    # )
    


st.title('Wavelet Transform Prediction Model Analysis')

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
    visualkeras.layered_view(model, to_file='imported_model.png', background_fill = '#0e1117')
    image = Image.open('imported_model.png')
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
            metrics = st.multiselect('Metrics', options = options)
            
            
    button_status = not button_status 
if st.button('Start Training', disabled = button_status):
    start_training(final_model, test_ratio=test_ratio/100, val_ratio=val_ratio/100)
# CV-Project 
## Prediction Model Analysis
This project acts as a pipeline to perform certain image transformations as a method to preprocess images, and feeding those transformed images though CNN models. Depending on the input classes, different features will be extracted, and a good predictive model can be determined for the usecase. 

Four transforms are used in this project : 
+ Fast Fourier Transform
+ Daubechies Transform
+ Haar Wavelet Transform
+ Canny Edge Detection

Four pre-trained models (with learnable and un-learnable parameters) have been implemented as well : 
+ VGG16
+ Resnet50
+ InceptionNet
+ XceptionNet

This project has the following features : 
+ The input can contain any number of classes. The models will adjust accordingly.
+ The model has been deployed on a web-based UI hosted on streamlit for ease of use.
+ We get a learning graph at the end as outputs for comparing the learning of the models form these four models.

## Input Format
The input should be a `.zip` file containing number of folders the same as the number of classes in the dataset. Each of those folders need to contain images of those classes.

![image](https://user-images.githubusercontent.com/67522615/204631834-c1ce72f0-785d-4437-89dd-28408e70768b.png)
![image](https://user-images.githubusercontent.com/67522615/204631879-eed4a164-8716-4b3c-abb0-9a5f0bb6dc77.png)
![image](https://user-images.githubusercontent.com/67522615/204631928-3c6dbcb2-08e0-4c0d-ac5a-7f95b5da6566.png)

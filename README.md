# CV-Project 
## Prediction Model Analysis
This project acts as a pipeline to perform certain image transformations as a method to preprocess images, and feeding those transformed images though CNN models. Depending on the input classes, different features will be extracted, and a good predictive model can be determined for the usecase. 

Four transforms are used in this project : 
+ [Fast Fourier Transform](https://towardsdatascience.com/fast-fourier-transform-937926e591cb)
+ [Daubechies Transform](https://medium.com/image-vision/2d-dwt-a-brief-intro-89e9ef1698e3)
+ [Haar Wavelet Transform](https://medium.com/@digitalpadm/image-compression-haar-wavelet-transform-5d7be3408aa)
+ [Canny Edge Detection](https://medium.com/simply-dev/what-is-canny-edge-detection-cfefa272a8d0)

Four pre-trained models (with learnable and un-learnable parameters) have been implemented as well : 
+ [VGG16](https://arxiv.org/abs/1409.1556)
+ [Resnet50](https://arxiv.org/abs/1512.03385)
+ [InceptionNet](https://arxiv.org/abs/1409.4842)
+ [XceptionNet](https://arxiv.org/abs/1610.02357)

This project has the following features : 
+ The input can contain any number of classes. The models will adjust accordingly.
+ The model has been deployed on a web-based UI hosted on streamlit for ease of use.
+ We get a learning graph at the end as outputs for comparing the learning of the models form these four models.

## Input Format
The input should be a `.zip` file containing number of folders the same as the number of classes in the dataset. Each of those folders need to contain images of those classes.

![image](https://user-images.githubusercontent.com/67522615/204631834-c1ce72f0-785d-4437-89dd-28408e70768b.png)
![image](https://user-images.githubusercontent.com/67522615/204631879-eed4a164-8716-4b3c-abb0-9a5f0bb6dc77.png)
![image](https://user-images.githubusercontent.com/67522615/204631928-3c6dbcb2-08e0-4c0d-ac5a-7f95b5da6566.png)

## Output
The output is in the form of learning graph showing learning curve of the transformed images as the base data.
![canny_acc](https://user-images.githubusercontent.com/67522615/204634672-8a451155-a382-4f71-94bb-111f4edc3e23.png)
![canny_loss](https://user-images.githubusercontent.com/67522615/204634682-cfe85fe6-c9fd-4824-93c9-c735ceb07784.png)

## Original setup
1. Windows 10 64-bit.
2. Python 3.10.2
3. Nvidia GTX 1050m

## Project Setup
1. Clone this repository using `git clone https://github.com/accelbia/CV-Project-Wavelet-Transform-Prediction-Model-Analysis`.
2. Navigate to the current cloned repository.
3. Install the necessary requirements using `pip install -r requirements.txt`.
4. Run the streamlit application by running `streamlit run Home.py`.

## Trial Run 

1. Upload the dataset by clicking on the 'Browse files'.
   ![image](https://user-images.githubusercontent.com/67522615/204639170-4453fa21-81c7-413c-a215-632aa2d1890c.png)

2. The classes contained in the zip files are displayed. Choose the model you want to use and add in the relevant layers and information.
   ![image](https://user-images.githubusercontent.com/67522615/204639958-51e233d2-e07d-47f7-a5b0-6fa10c946011.png)
   
   

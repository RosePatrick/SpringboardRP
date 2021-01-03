## Distracted Driver Detection

### Project Definition
The aim of this project is given the dataset of driver images, each taken in a car with a driver doing something in the car (texting, eating, talking on the phone, makeup, reaching behind, etc), predict the likelihood of what the driver is doing in each picture. The 10 classes to predict are:
* c0: safe driving
* c1: texting - right
* c2: talking on the phone - right
* c3: texting - left
* c4: talking on the phone - left
* c5: operating the radio
* c6: drinking
* c7: reaching behind
* c8: hair and makeup
* c9: talking to passenger

### Dataset
The dataset used for this problem is the State Farm Distracted Driver dataset available on Kaggle.The dataset contains 102,150 JPG images divided into training and test data. Each image is sized at 640 x 480 pixels. The total dataset file size is 4GB.
Source: https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

### Files
**Capstone_Project_Distracted_Driver_Detection Rev1.ipynb** contains the complete Keras model for this project.  

**Capstone_Project_Distracted_Driver_Detection.ipynb** is the preliminary project work to load and explore the dataset as well as some data augmentation.  

**Distracted_Driver_AutoML.ipynb** uses the same dataset which was then uploaded to Google Cloud Platform where an AutoML model was derived and tested. The results of the AutoML implementation of this dataset is illustrated here.

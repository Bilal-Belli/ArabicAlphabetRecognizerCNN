# Arabic Alphabet Recognizer Convolutional Neural Network
This repository contains a classification model made by python for Arabic Alphabet Detection.
### Dataset
- For classification dataset, I have installed a set of images from this [Dataset]().
- After that i perform data augmentation step.
### Language And Libraries
- Python - TensorFlow - Keras - OpenCV.
### Architecture
This model is built using 7 layers.
<div align="center">
   <img  width="550" src="">
</div>

The model uses the following optimization algorithm:
- Adam ('adam') for training.
### Statistiques
#### Accuracy
- The accuracy for the model with 5 classes: 62,14% .
- The accuracy is not perfect but it is good.
#### Confusion Matrix
<div align="center">
   <img  width="550" src="">
</div>

#### True Positive Rate - False Positive Rate
<div align="center">
   <img  width="550" src="">
</div>

### Model Testing
#### Input
<div align="center">
   <img width="404" alt="1" src="https://user-images.githubusercontent.com/74218805/229004154-4018b2b9-0efe-47de-b73f-b1feb3e579fa.PNG">
</div>

#### Output
<div align="center">
   <img width="302" alt="2" src="https://user-images.githubusercontent.com/74218805/229004167-1b3850db-3ec0-457d-89e6-fa6406a4b737.PNG">
</div>

### License
This repository is licensed under the MIT License.

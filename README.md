# Image classifier using the state of the art deep learning models 

In this project, we provided a complete development cycle of modern AI apps which exploits the state of the art deep learning pretrained models to solve a specific problem via  transfer learning. 

## Main problem
This application is developed to recgonize the flower type from different 102 categories by training on [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). 

## Repo structure
### 1- Development story: 
In this part, we described the efforts done upon completing the app along with all the necessary details to reproduce the results. 
The files for this part, are Image_Classifier_Project_colab.ipynb, Image_Classifier_Project_colab.html
### 2- Complete app: 
In this part, a Python app ready to be deployed for any platform which can be run via command window. 
The files for this part are in app subfolder. 
#### App options
1- No arguments: Error expected 
Example: python predict.py 
2- Image, model : Expected returning default number of classes; currently = 5
Example: python predict.py \path\for\testimage \path\for\model
3- Image, model, top_k: Expected returning optional number of classes
Example: python predict.py \path\for\testimage \path\for\model 3 
4- Image, model, labelsfile: Expected returning the labels of the classes
Example: python predict.py \path\for\testimage \path\for\model \path\for\json\file\have\mapping\for\labels

## Used Libraries 
This method is developed using the lastet available version of TensorFlow which is 2.4.0
## Results
The model is GPU-trained to acheive 70% accuracy on the test data set via google coolab. 
 

# Image classifier using the state of the art deep learning models 
A complete development cycle of modern AI apps which exploits the state of the art deep learning pretrained models to solve a specific problem via transfer learning. 

## Objective
This application is developed to recgonize the flower type from 102 different categories by training on [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). 

## Repo structure
### 1- Development story: 
In this part, we described the efforts done upon completing the app along with all the necessary details to reproduce the results. 
The files for this part, are Image_Classifier_Project_colab.ipynb, Image_Classifier_Project_colab.html
### 2- Complete app: 
In this part, a Python app ready to be deployed for any platform which can be run via command window. 
The files for this part are in `app` subfolder. 
#### App options
By executing the command:
`python predict.py -h`  
you can view all the available options. For now,   

```
usage: predict_V2.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                     image model

List of Arguments

positional arguments:
  image                 Image path
  model                 Saved model path

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         Top K most likely classes
  --category_names CATEGORY_NAMES
                        Input JSON file with category names
```
                        
## Used Libraries 
This app is developed using the lastet available version of TensorFlow which is `2.4.0` and [`MobileNet`](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4) pre-trained model from `TensorFlow Hub`
## Results
The model is GPU-trained to acheive 70% accuracy on the test data set via google coolab. 
 

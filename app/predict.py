# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:30:03 2020

@author: Omar Khater
"""
from utils import predict
import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')


image_path = 'D:/ML/courses/MachineLearningWithTF/codes/projects/p2_image_classifier/test_images/cautleya_spicata.jpg'
model_path = 'D:/ML/courses/MachineLearningWithTF/codes/projects/p2_image_classifier/BestClassifier.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
probs, classes = predict(image_path, model, 5 )
    
# if __name__ == '__main__':
#     main()
    
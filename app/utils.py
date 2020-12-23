# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:18:37 2020

@author: Omar Khater
"""
import tensorflow as tf
from PIL import Image
import numpy as np
import json

def process_image(image,image_size = 224):
    """
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    image_size : TYPE, optional
        DESCRIPTION. The default is 224.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    imten = tf.convert_to_tensor(image)
    imten = tf.image.resize(imten, (image_size,image_size))
    imten = tf.cast(imten, tf.float32)
    imten /= 255
    return imten.numpy()

def predict(image_path , model , image_size = 224, top_k = 5,
            MapLabels = 'D:/ML/courses/MachineLearningWithTF/codes/projects/p2_image_classifier/label_map.json'):
    """
    
    Predict the label of unseen image by given model. 
    Return the most top_k probable labels 
    
    Parameters
    ----------
    image_path : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    top_k : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image, image_size = image_size)
    all_probs =  model.predict(np.expand_dims(processed_test_image, axis = 0))
    all_classes = np.argsort(all_probs)
    probs = all_probs[:,all_classes[:,-top_k:]]
    classes = all_classes[:,-top_k:]
    
    with open(MapLabels, 'r') as f:
        class_names = json.load(f)
        
    mappednames = [class_names[str(i+1)] for i in classes]
    return probs[0][0], classes[0],mappednames
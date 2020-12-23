# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:18:37 2020

@author: Omar Khater
"""
import tensorflow as tf
from PIL import Image
import numpy as np

def normalize(image, label, image_size = 224):
    """
    Normalize input image to [0,1] range using given size. 
    Return the normalized image with associated label.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.
    image_size : TYPE, optional
        DESCRIPTION. The default is 224.

    Returns
    -------
    image : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.

    """
    image = tf.cast(image, tf.float32)
    image /= 255
    image = tf.image.resize_with_crop_or_pad(image, target_height = image_size, target_width = image_size)
    return image, label

def augment_data(image, label):
    """
    Perform data augmentation on given data set to enhance the model performance     

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.

    Returns
    -------
    image : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.

    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, 0.4, 1.0)
    image = tf.image.rot90(image)
      # Add more augmentation of your choice
    return image, label

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

def predict(image_path , model , image_size = 224, top_k = 5):
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
    return probs[0][0], classes[0]
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:36:49 2020

@author: Omar Khater
"""

import argparse
from utils import predict
import tensorflow as tf
import tensorflow_hub as hub

def main():
    
    try:
        
        parser = argparse.ArgumentParser(description='List of Arguments')
        parser.add_argument('image', type=str,  help='Image path')
        parser.add_argument('model', type=str, help='Saved model path')
        parser.add_argument("--top_k", default = 1, type=int, help='Top K most likely classes')
        parser.add_argument("--category_names", 
                            default ='D:/ML/courses/MachineLearningWithTF/codes/projects/p2_image_classifier/label_map.json',
                            type= str, 
                            help='Input JSON file with category names')
        
        args = parser.parse_args()
        
        model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})

        probs, classes, mappednames = predict(args.image, model, top_k = args.top_k, 
                                    MapLabels = args.category_names)
        
        print('Predicting done!')
        
    except Exception as e:
          print (e)
    
if __name__ == '__main__':
    main()
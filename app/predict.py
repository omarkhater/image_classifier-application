# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:30:03 2020

@author: Omar Khater
"""
from utils import predict
import tensorflow as tf
import tensorflow_hub as hub
import getopt, sys
import warnings
warnings.filterwarnings('ignore')

def main():
    
    try:
        _, command = getopt.getopt(sys.argv[0:], "t:")
        
        if len(command) == 3 :
            
            image_path = command[1]
            model_path = command[2]
            model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
            probs, classes, _ = predict(image_path, model)
            print('Returning default number of classes = 5')
            print('Classes = {}\nPropabilities = {} '.format(probs, classes))
            
        elif len(command) == 4:
            
            image_path = command[1]
            model_path = command[2]
            option = command[3]
            if option.isnumeric(): # if the option is --top_k
            
                model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
                probs, classes , _ = predict(image_path, model ,top_k = int(option))
                print('Returning optional number of classes = {}'.format(option))
                print('Classes = {}\nPropabilities = {} '.format(probs, classes))
                
            
            else: # if the option is --category_names 
                
                model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
                probs, classes, labels = predict(image_path, model , MapLabels = option)
                print('Returning labels = {}'.format(option))
                print('Classes = {}\nPropabilities = {}\Labels = {}'.format(probs, classes, labels))

            
        elif len(command) < 2:
            
            sys.exit('Please provide at least 2 arguments\nA path for the test image as well as the model')

    
    except Exception as e:
          print (e)
    
if __name__ == '__main__':
    main()

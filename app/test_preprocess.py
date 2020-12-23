# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 00:10:24 2020

@author: Omar Khater
"""

from PIL import Image
from utils import process_image
import numpy as np
import matplotlib.pyplot as plt

#%% 
image_path = 'D:/ML/courses/MachineLearningWithTF/codes/projects/p2_image_classifier/test_images/cautleya_spicata.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()
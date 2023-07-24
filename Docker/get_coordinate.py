# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:22:34 2023

@author: SABARI
"""

import cv2
import numpy as np

from skimage.filters import threshold_sauvola

import tensorflow as tf
from tensorflow.keras import backend as K
import json

def sauvola_thresholding(grayImage_,window_size=15):
    
    """"
    Sauvola thresholds are local thresholding techniques that are 
    useful for images where the background is not uniform, especially for text recognition
    
    grayImage--- Input image should be in 2-Dimension Gray Scale format
    window_size --- It represents the filter window size 
    
    """
    thresh_sauvolavalue = threshold_sauvola(grayImage_, window_size=window_size)

    thresholdImage_=(grayImage_>thresh_sauvolavalue)
    
    return  1- np.uint8(np.array(thresholdImage_)*1)

# Function to get coordinates of the object
def get_object_coordinates(image):
    
    # Convert the image from BGR to GRAY  color space
    grayImage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
 
    # Create a mask using the specified color range
    thresholdedImage=sauvola_thresholding(grayImage)
    
    kernel = np.ones((35, 1), np.uint8)
    dilated_image = cv2.dilate(thresholdedImage, kernel, iterations=1)
    # Find contours in the mask
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coordinate = []
    # Check if any contours were found
    if len(contours) > 0:
        for i in range(len(contours)):
    
            # Get the largest contour (assuming it's the object of interest)
#            largest_contour = max(contours, key=cv2.contourArea)
    
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contours[i])
            
            coordinate.append([x,y,x+w,y+h])
            # Calculate the center coordinates of the object
    #        center_x = x + w // 2
    #        center_y = y + h // 2
    
        return coordinate,thresholdedImage
    else:
        # Return None if no object was found
        return None,thresholdedImage

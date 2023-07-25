# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:42:56 2023

@author: SABARI
"""

import numpy as np
import pandas as pd
import cv2
import argparse

from keras.models import Model
from keras import optimizers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from custom_models import lightNetHOCR
import tensorflow as tf
import os

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Script to get input arguments for train and test data.')
    parser.add_argument('--train_images', type=str, required=True,
                        help='Path to the training images (.npz).')
    parser.add_argument('--train_labels', type=str, required=True,
                        help='Path to the training labels (.npz).')
    parser.add_argument('--test_images', type=str, required=True,
                        help='Path to the test images (.npz).')
    parser.add_argument('--test_labels', type=str, required=True,
                        help='Path to the test labels (.npz).')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16).')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs for training (default: 150).')
    args = parser.parse_args()

    # Check if paths end with ".npz"
    for path in [args.train_images, args.train_labels, args.test_images, args.test_labels]:
        if not path.lower().endswith('.npz'):
            raise ValueError("All paths should end with '.npz'")
    

    
    # initialize the variables
    
    train_images_path = args.train_images
    train_labels_path = args.train_labels
    test_images_path = args.test_images
    test_labels_path = args.test_labels
    
    
    train_images = np.load(train_images_path)['arr_0']
    
    # resize the image from 28x28 to 64x64
    train_images = np.array([cv2.resize(train_images[i],(64,64)) for i in range(len(train_images))])
    train_images = np.reshape(train_images,(len(train_images),64,64,1))
    print("train_images shape: ",train_images.shape)
    
    test_images = np.load(test_images_path)['arr_0']
    
    test_images = np.array([cv2.resize(test_images[i],(64,64)) for i in range(len(test_images))])
    
    print("test_images shape: ",test_images.shape)
    
    
    test_images = np.reshape(test_images,(len(test_images),64,64,1))
    
    train_labels = np.load(train_labels_path)['arr_0']
    test_labels = np.load(test_labels_path)['arr_0']
    
    
    
    BATCH_SIZE=args.batch_size
    EPOCHS=args.epochs
    LEARNING_RATE=0.001
    
    path="./model/"

    # check OS folder is exist or not
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created successfully.")

    #
    model = lightNetHOCR((64,64,1),learning_rate=0.00001)
    

    filepath=path+"weights-lightNetHOCR--{epoch:02d}---val_loss-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min')
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode = 'min',
                                      factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    callbacks_list = [checkpoint,reduce_lr]
    
    history=model.fit(train_images, train_labels, epochs=EPOCHS,validation_data=(test_images,test_labels),callbacks=callbacks_list,verbose=1)

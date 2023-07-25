# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:04:30 2023

@author: SABARI
"""

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import math

kaiming_normal = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

from tensorflow.keras import backend as K

def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    # Compute the channel attention

    channel = input_feature.get_shape()[-1]

    shared_layer_one = tf.keras.layers.Dense(channel//ratio,activation='relu',kernel_initializer='he_normal',
                            use_bias=True,bias_initializer='zeros')
    shared_layer_two = tf.keras.layers.Dense(channel,kernel_initializer='he_normal',use_bias=True,
                                    bias_initializer='zeros')

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = tf.keras.layers.Reshape((1,1,channel))(avg_pool)
    assert avg_pool.get_shape()[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.get_shape()[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.get_shape()[1:] == (1,1,channel)

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)
    max_pool = tf.keras.layers.Reshape((1,1,channel))(max_pool)
    assert max_pool.get_shape()[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.get_shape()[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.get_shape()[1:] == (1,1,channel)

    cbam_feature = tf.keras.layers.Add()([avg_pool,max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = tf.keras.layers.Permute((3, 1, 2))(cbam_feature)

    return tf.keras.layers.multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    # Spatial Attention implementation

    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.get_shape()[-1]
        cbam_feature = tf.keras.layers.Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.get_shape()[-1]
        cbam_feature = input_feature

    avg_pool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.get_shape()[-1] == 1
    max_pool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.get_shape()[-1] == 1
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.get_shape()[-1] == 2
    cbam_feature = tf.keras.layers.Conv2D(filters = 1,
                    kernel_size=kernel_size,strides=1,padding='same',activation='sigmoid',
                    kernel_initializer='he_normal',use_bias=False)(concat)
    assert cbam_feature.get_shape()[-1] == 1
    if K.image_data_format() == "channels_first":
        cbam_feature = tf.keras.layers.Permute((3, 1, 2))(cbam_feature)

    return tf.keras.layers.multiply([input_feature, cbam_feature])



def BatchActivate(x):
    # Apply batch normalization followed by LeakyReLU activation
    x = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

def convolution_block(x, filters, size=(3,3), strides=(1,1), padding='same', activation=True,DILATION_VALUE=1):
    # Apply Convolution operation followed by BatchActivate activation
    x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding,kernel_initializer=kaiming_normal,dilation_rate=DILATION_VALUE)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def Squeeze_excitation_layer(input_x):
    # Squeeze-Excitation layer for channel-wise feature recalibration

    ratio = 4
    out_dim =  int(np.shape(input_x)[-1])
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_x)
    excitation = tf.keras.layers.Dense(units=int(out_dim / ratio))(squeeze)
    excitation = tf.keras.layers.Activation('relu')(excitation)
    excitation = tf.keras.layers.Dense(units=out_dim)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([-1,1,out_dim])(excitation)
    scale = tf.keras.layers.multiply([input_x, excitation])

    return scale

def residual_block(blockInput, num_filters=16, batch_activate = False):
    # Residual block with Squeeze-Excitation layer

    x_side = convolution_block(blockInput, num_filters,(3,3))

    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) ,activation=True)
    x = convolution_block(x, num_filters, (3,3), activation=True)

    x = convolution_block(x, num_filters, (3,3), activation=True)

    x=Squeeze_excitation_layer(x)
    x = tf.keras.layers.Add()([x,x_side,blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


def lightNetHOCR(input_shape,learning_rate=0.001):
    # lightweight_modelimplementation

    start_neurons = 32
    
    inputs = tf.keras.layers.Input(input_shape)

    #x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)
    
    
    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation='elu', padding="same")(x)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation='elu', padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv2D(start_neurons * 3, (3, 3), activation='elu', padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 3)
    conv3 = residual_block(conv3,start_neurons * 3, True)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation='elu', padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 4)
    conv4 = residual_block(conv4,start_neurons * 4, True)
#    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    
    conv4 = tf.keras.layers.Add()([cbam_block(conv4),conv4])
    
    GA = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(conv4)
    initializer = tf.keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    
    x_b = tf.keras.layers.BatchNormalization()(GA)
    
    x_d = tf.keras.layers.Dropout(0.1)(x_b)

    output=tf.keras.layers.Dense(49,activation='softmax', kernel_initializer=initializer,name='recognition')(x_d)
    
    
    model=tf.keras.models.Model(inputs,[output])
        
    c = tf.keras.optimizers.Adam(lr =learning_rate)
      
    model.compile(optimizer=c, loss=["sparse_categorical_crossentropy"],metrics=["accuracy","mae"])
    
    
    return model



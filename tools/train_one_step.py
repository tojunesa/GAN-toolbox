import numpy as np
import PIL
import pandas as pd
import matplotlib.pyplot as plt
import re

import os

import tensorflow as tf

from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Add
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Concatenate, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow import identity

def d_training(real_img):
    #One discriminator training step with gradient penalty
    with tf.GradientTape() as tape_d:
        fake_img = net_g(real_img)
        loss_real = tf.reduce_mean(net_d(real_img))
        loss_fake = tf.reduce_mean(net_d(fake_img))
        
        #Calculate gradient penalty
        with tf.GradientTape() as tape_penalty:
            epsilon = tf.random.uniform([batch_size], 0, 1)
            epsilon = tf.reshape(epsilon, (-1,1,1,1))
            interpolated_img = epsilon*real_img + (1-epsilon)*fake_img
            tape_penalty.watch(interpolated_img)
            interpolated_out = net_d(interpolated_img)
            grad_interpolated = tape_penalty.gradient(interpolated_out, interpolated_img)
            grad_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(grad_interpolated), axis = [1,2,3]))
            grad_penalty = tf.reduce_mean(tf.math.square(grad_norm-1))
        
        loss = loss_fake - loss_real + gp_coef*grad_penalty
        grad_d = tape_d.gradient(loss, net_d.trainable_weights)
        optimizer_d.apply_gradients(zip(grad_d, net_d.trainable_weights))
    
    return (loss_real, loss_fake, loss)

def g_training(real_img):
    #One generator training step with gradient penalty
    with tf.GradientTape() as tape_g:
        gen_img = net_g(real_img)
        residual_loss = tf.reduce_mean(tf.math.abs(real_img-gen_img))
        loss = -tf.reduce_mean(net_d(gen_img))
        grad_g = tape_g.gradient(loss, net_g.trainable_weights)
        optimizer_g.apply_gradients(zip(grad_g, net_g.trainable_weights))
    return (residual_loss, loss)
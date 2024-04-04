# model builder
# import model_build

# refer to
# https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/tutorials/quickstart/advanced

import math
import numpy as np
import logging
import pandas as pd

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import SimpleRNN, Attention, AdditiveAttention, TimeDistributed, MultiHeadAttention
from tensorflow.keras import Input, Model

N_times = 14
N_feature = 14
N_outputs = 8

## **************************************
def embedding_f(inputx, layer_n=3, unit=128):
    he = tf.keras.initializers.HeNormal()
    
    x = inputx
    for i in range(layer_n):
        # x = Dense(units=unit//2**(layer_n-i-1),activation="relu", kernel_initializer=he)(x)
        x = Dense(units=unit, activation="relu", kernel_initializer=he)(x)
        x = BatchNormalization()(x)
    
    return x
## **************************************
def decoder_f(inputx, layer_n=3, unit=128):
    he = tf.keras.initializers.HeNormal()
    
    x = inputx
    for i in range(layer_n):
        # x = Dense(units=unit//2**i,activation="relu", kernel_initializer=he)(x)
        x = Dense(units=unit, activation="relu", kernel_initializer=he)(x)
        x = BatchNormalization()(x)
    
    return x

def positional_encoding(length, depth):
    depth = depth // 2
    
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    
    return tf.cast(pos_encoding, dtype=tf.float32)
##************************************************************************************************************
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

## check test_mask.py to see why this is like adding two new axises
def create_padding_mask(inputs, mask_value=0):
    seq = tf.cast(tf.math.not_equal(inputs[:, :, 0], mask_value), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# layern=1; units=64; n_times=95; n_times_out=95; n_feature=2; n_head=8; is_batch=False; drop=0; mask_value = 3.2767;  active="linear"
# inputs = X_rnn_train[:16,:,:] # no filled data
# inputs = [train_x_ran[:16,:,:],train_y_ran[:16,:].reshape(16,-1,1)] # no filled data
# https://www.tensorflow.org/text/tutorials/transformer

def get_transformer_new_att0(n_times=14, n_feature=2, n_out=9, layern=3, units=128, n_head=4, drop=0.1, is_batch=True, is_att=True, mask_value=-9999.0, active="softmax"):
    """
    using AveragePooling1D with mask
    after averagepooling layer, add attention or not"""
    inputs = Input(shape=(n_times, n_feature,))  
    ## test linear or relu      
    embedding = Dense(units, activation="relu")
    embedding_p = Dense(units, activation="relu")
    ## *******************
    # positional
    from keras import backend as K
    b = K.ones_like(inputs[:, :, :1])
    xp = np.arange(n_times)[:, np.newaxis] / n_times  # (seq, 1)
    xpp = b * xp
   
    x = inputs
   
    mask_multi = tf.cast(tf.math.not_equal(x,mask_value), tf.float32)
    x = x * mask_multi
    x = embedding(x)
    x = x + embedding_p(xpp)
    padding_mask = create_padding_mask(inputs=inputs, mask_value=mask_value)
    # encoder
    for i in range(layern):
        attn_output, attn4 = MultiHeadAttention(key_dim=units // n_head, num_heads=n_head)(query=x, value=x, key=x,
                                                                                           return_attention_scores=True,
                                                                                           attention_mask=padding_mask)
        if drop > 0:
            attn_output = Dropout(drop)(attn_output)
        
        out1 = x + attn_output
        if is_batch == True:
            out1 = LayerNormalization(epsilon=1e-6)(out1)
        
        ffn_output = point_wise_feed_forward_network(units, units * 4)(out1)
        if drop > 0:
            ffn_output = Dropout(drop)(ffn_output)
        
        out2 = out1 + ffn_output
        if is_batch == True:
            out2 = LayerNormalization(epsilon=1e-6)(out2)
        
        x = out2

    enc_output = x    
    # enc_output1 = MaxPooling1D(pool_size=n_times)(enc_output)
    # enc_output2 = AveragePooling1D(pool_size=n_times)(enc_output)
    enc_output2 = tf.math.multiply(enc_output,mask_multi[:,:,:1])
    enc_output2 =  K.sum(enc_output2, axis=1) / K.sum(mask_multi[:,:,:1], axis=1)
    enc_output2 = enc_output2[:,tf.newaxis,]
    if is_att:
        attn_output, attn4 = MultiHeadAttention(key_dim=units//n_head, num_heads=n_head)(query=enc_output2, value=x, key=x, return_attention_scores=True, attention_mask=padding_mask)
        # enc_output = enc_output1 + enc_output2
        enc_output = attn_output
        enc_output = tf.reshape(enc_output, [-1, enc_output.shape[2]])
    else:
        enc_output = enc_output2 
    output = Dense(n_out, activation=active)(enc_output)
    model = Model(inputs, output)    
    return model

def get_transformer_cls (n_times=14, n_feature=2, n_out=9, layern=3, units=128, n_head=4, drop=0.1, is_batch=True, mask_value=-9999.0, active="softmax"):
    """borrowing BERT CLS token"""
    gu = tf.keras.initializers.GlorotUniform()
    gn = tf.keras.initializers.GlorotNormal()
    he = tf.keras.initializers.HeNormal()
    inputs = Input(shape=(n_times, n_feature,))    
    embedding = Dense(units, activation="relu")
    embedding_p = Dense(units, activation="relu")
    ## *******************
    # positional
    from keras import backend as K
    b = K.ones_like(inputs[:, :, :1])
    xp = np.arange(n_times)[:, np.newaxis] / n_times  # (seq, 1)
    xpp = b * xp
    # xpp = tf.identity(inputs[:,:,:1] )
    # xpp[0,:,:] = xp
    # x = Masking(mask_value=mask_value, input_shape=(n_times, n_feature)) (inputs)
    x = inputs
    # mask_multi = tf.cast(x != mask_value, tf.float32)
    mask_multi = tf.cast(tf.math.not_equal(x,mask_value), tf.float32)
    x = x * mask_multi
    x = embedding(x)
    x = x + embedding_p(xpp)
    ## add a cls_token
    cls_token = K.zeros_like(x[:, :1, :])
    x2 = tf.concat([cls_token,x],axis=1)
    x = x2
    ## tested work not well
    # if drop>0:
    # x=Dropout(drop)(x)
    ## add token bert method
    cls_token_mask = tf.cast(tf.math.not_equal(K.zeros_like(x[:, :1, :1]), 1), tf.float32)
    cls_token_mask = cls_token_mask[:, tf.newaxis, :, :]  # (batch_size, 1, 1, seq_len)
    padding_mask = create_padding_mask(inputs=inputs, mask_value=mask_value)
    padding_mask = tf.concat([cls_token_mask,padding_mask],axis=3)
    # encoder
    for i in range(layern):
        # temp_mha = MultiHeadAttention(key_dim=units//n_head, num_heads=n_head)
        # attn_output, attn4 = temp_mha(query=x,value=x,key=x,return_attention_scores=True, attention_mask=padding_mask)
        attn_output, attn4 = MultiHeadAttention(key_dim=units // n_head, num_heads=n_head)(query=x, value=x, key=x,
                                                                                           return_attention_scores=True,
                                                                                           attention_mask=padding_mask)
        if drop > 0:
            attn_output = Dropout(drop)(attn_output)
        
        out1 = x + attn_output
        if is_batch == True:
            out1 = LayerNormalization(epsilon=1e-6)(out1)
        
        ffn_output = point_wise_feed_forward_network(units, units * 4)(out1)
        if drop > 0:
            ffn_output = Dropout(drop)(ffn_output)
        
        out2 = out1 + ffn_output
        if is_batch == True:
            out2 = LayerNormalization(epsilon=1e-6)(out2)
        
        x = out2
    
    enc_output = x[:,0,:]
    output = Dense(n_out, activation=active)(enc_output)
    model = Model(inputs, output)
    
    return model
##***************************************************BACK UP FUNCTIONS*************************************************************************************************






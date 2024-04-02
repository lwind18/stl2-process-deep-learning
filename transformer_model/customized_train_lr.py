## import customized_train
import math
import logging

import numpy as np 
import tensorflow as tf 
# import tensorflow-addons as tfa 
import tensorflow_addons as tfa
# import customized_loss

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        # return optimizer.lr
        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr
        return current_lr

from keras import backend as K
from keras.callbacks import TensorBoard

# https://github.com/tensorflow/tensorflow/issues/39782
# https://github.com/tensorflow/tensorflow/issues/39782
class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        optimizer = self.model.optimizer
        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr
        
        logs.update({'lr': K.eval(current_lr)})
        super().on_epoch_end(epoch, logs)

 
def my_train_schedule(model,train_datasetx,train_datasety,epochs=10,start_rate=0.001,loss=np.nan,per_epoch=100,split_epoch=4,option=0):
    momentum = 0.9   # good for RMSprop not good for RMSprop but OK for Adam
    decay = 0.5
    decay = 0.05
    decay = 1e-5
    print_str = "momentum="+str(momentum) + "\tlearning rate="+str(start_rate)+"  decay="+str(decay) 
    print(print_str); logging.info(print_str)
    ##*****************************************************************************************************
    ## warm up
    warm_up = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=start_rate/split_epoch/per_epoch,decay_steps=split_epoch*per_epoch,end_learning_rate=start_rate,name='Decay_linear')
    if option==0:
        print ('tfa.optimizers.RMSprop '); optimizer=tf.keras.optimizers.RMSprop(learning_rate=warm_up, rho=momentum    ); 
    elif option==1:                                                                           
        print ('tfa.optimizers.Adam ');   optimizer=tf.keras.optimizers.Adam   (learning_rate=warm_up, beta_1=momentum, beta_2=0.999, epsilon=1e-07)
    else:
        print ('tfa.optimizers.AdamW ');  optimizer=tfa.optimizers.AdamW       (weight_decay=decay,learning_rate=warm_up, beta_1=momentum)

    # metric = get_lr_metric(optimizer)
    lr_metric = get_lr_metric(optimizer)
    # model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.RootMeanSquaredError(),lr_metric])
    # model.compile(optimizer=optimizer, loss=loss)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy',lr_metric])
    # model_history = model.fit(train_dataset, validation_data=validation_dataset, epochs=split_epoch, verbose=2)
    model_history = model.fit(x=train_datasetx, y=train_datasety, validation_split=0.04, epochs=split_epoch, verbose=2)
    
    ##*****************************************************************************************************
    ## consine
    cos_dec1 = tf.keras.optimizers.schedules.CosineDecay(start_rate, decay_steps=(epochs-split_epoch)*per_epoch, alpha=0, name='Cosine_Decay_1')
    if option==0:
        print ('tfa.optimizers.RMSprop'); optimizer=tf.keras.optimizers.RMSprop(learning_rate=cos_dec1, rho=momentum    ); 
    elif option==1:                                                                           
        print ('tfa.optimizers.Adam ') ; optimizer=tf.keras.optimizers.Adam   (learning_rate=cos_dec1, beta_1=momentum, beta_2=0.999, epsilon=1e-07)
    else:
        print ('tfa.optimizers.AdamW '); optimizer=tfa.optimizers.AdamW       (weight_decay=decay,learning_rate=cos_dec1, beta_1=momentum)
    
    # lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy',lr_metric])
    model_history = model.fit(x=train_datasetx, y=train_datasety, validation_split=0.04, epochs=epochs-split_epoch, verbose=2)
    return model_history

## training with validation data using RMSprop
def my_train_RMSprop(model,train_dataset, validation_dataset, epochs=2, BATCH_SIZE=4,start_rate=0.001,loss=np.nan):
    whichMAX = 2
    each_EPOCH = 30
    EPOCHS_ALL = [50,30,20]
    # if BATCH_SIZE<20:
        # EPOCHS_ALL = [30,40,30]
    
    # EPOCHS_ALL = [20,20,20]
    
    # EPOCHS_ALL = [50,50,50]
    # whichMAX = 1
    # each_EPOCH = 50
    momentum = 0.9   # good for RMSprop not good for RMSprop but OK for Adam
    print_str = "momentum="+str(momentum)
    print(print_str)
    logging.info(print_str)
    whichi = 0
    # for traini in range(0,epochs,each_EPOCH):
    for traini in range(whichMAX+1):
        rate = start_rate*(5**(whichi*-1) )
        # epoch_n_i = min(epochs-traini,each_EPOCH)
        epoch_n_i = EPOCHS_ALL[traini]
        print_str = "learning_rate="+str(rate)
        print(print_str)
        logging.info(print_str)
        # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=rate, rho=momentum), loss=loss, metrics=['accuracy'])
        print ('tf.keras.optimizers.RMSprop')
        # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=rate, momentum=momentum), loss=loss, metrics=['accuracy'])
        model_history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epoch_n_i,verbose=2)
        whichi = whichi+1 if whichi<whichMAX else whichi

## training with validation data using Adam
def my_train_Adam(model,train_dataset, validation_dataset, epochs=2, BATCH_SIZE=4,start_rate=0.001):
# def my_train2(model,train_dataset, validation_dataset, epochs=2, BATCH_SIZE=4,start_rate=0.0001):
    whichMAX = 2
    whichMAX = 1
    each_EPOCH = 20
    momentum = 0.9   # not good for RMSprop but OK for Adam
    # momentum = 0.99  # good 
    # momentum = 0.999 # not good v5.1
    # momentum = 0.995 # had no difference with 0.99 so 0.99 is used 
    print_str = "momentum="+str(momentum)
    print(print_str)
    logging.info(print_str)
    whichi = 0
    for traini in range(0,epochs,each_EPOCH):
        rate = start_rate*10**(whichi*-1)
        print_str = "learning_rate="+str(rate)
        print(print_str)
        logging.info(print_str)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=rate, beta_1=momentum), loss=loss, metrics=['accuracy'])
        print ('tf.keras.optimizers.Adam')
        # model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=rate, rho=momentum), loss=loss, metrics=['accuracy'])
        # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=rate, momentum=momentum), loss=loss, metrics=['accuracy'])
        model_history = model.fit(train_dataset, validation_data=validation_dataset, epochs=each_EPOCH,verbose=2)
        whichi = whichi+1 if whichi<whichMAX else whichi

def my_loss(y_pred,y_true):
    one_hot_y = tf.one_hot(y_true,N_CLASS)
    y_conv = tf.nn.softmax(y_pred)   
    # loss1 = -1*tf.math.log(y_conv)*one_hot_y
    # return tf.math.reduce_mean(loss1)
    multi = tf.math.multiply(tf.math.multiply(tf.math.log(y_conv),one_hot_y),-1)
    loss1_sum = tf.math.reduce_sum (multi,axis=1)
    return tf.math.reduce_mean(loss1_sum)
    
def test_accuacy(model,input_images,y_test):
    logits = model.predict(input_images)
    classesi = np.argmax(logits,axis=1).astype(np.uint8).reshape(y_test.shape)
    accuracy = (y_test==classesi).sum()/classesi.size
    return accuracy,classesi

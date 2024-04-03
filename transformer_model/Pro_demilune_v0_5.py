"""
Created on Sat Mar 30 10:24:25 2024
@author: Dong.Luo
transformer-att: average pool + correct mask, attention, 
"""
import os
import logging
import socket
import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
import rasterio
import train_test 
import customized_train_lr
import transformer_encoder

import importlib
def get_mean_acc_f1 (rst_arr):
    """rst_arr: filtered arr"""
    rst_true = rst_arr[:,5].astype(int)
    rst_run0 = rst_arr[:,8].astype(int)
    rst_run1 = rst_arr[:,9].astype(int)
    rst_run2 = rst_arr[:,10].astype(int)
    
    oa0 = accuracy_score(rst_true, rst_run0)
    oa1 = accuracy_score(rst_true, rst_run1)
    oa2 = accuracy_score(rst_true, rst_run2)
        
    cm0_f1 = f1_score(rst_true, rst_run0, average=None)
    cm1_f1 = f1_score(rst_true, rst_run1, average=None)
    cm2_f1 = f1_score(rst_true, rst_run2, average=None)
        
    oa_all = np.stack((oa0, oa1, oa2), axis=0)
    acc_mean = np.mean(oa_all, axis=0)
        
    cm_f1_all = np.stack((cm0_f1, cm1_f1, cm2_f1), axis=0)
    cm_f1_all_mean = np.mean(cm_f1_all, axis=0)
    np.set_printoptions(suppress=True)     # showing decimal notation instead scientific notation
    return acc_mean, cm_f1_all_mean
##################################################################################################################
############################################model training part###################################################
##################################################################################################################
## load csv file
csv_dir = './demilune_niger_20to23_bd_times.csv'  # year 2020 to 2023

new_file = csv_dir.replace('bd_times', 'bd_times_train')

class_field = 'label'

if not os.path.exists(new_file):
    data_per_all = pd.read_csv(csv_dir)
    data_per_all_yr = data_per_all[data_per_all['train']!='task']
    data_per_all_yr.to_csv(new_file) 

data_per = pd.read_csv(new_file)
yclasses = data_per[class_field]

#*****************************************************************************************************************
## split training & testing data with 80% for train and 20% for test
import train_test 
importlib.reload(train_test)

orders = train_test.random_split(data_per.shape[0],split_n=10)

index_train = orders>1
index_test =  orders<=0

unique_yclass = np.unique(yclasses)
print (np.unique(yclasses[index_train]))
print (np.unique(yclasses[index_test ]))

N_CLASS = np.unique(yclasses[index_train]).size

##*****************************************************************************************************************
## construct training and testing data
## IMG_HEIGHT is time, IMG_WIDTH is featrue
IMG_HEIGHT2 = 10   ; IMG_WIDTH2 = 10; IMG_BANDS2=1

BATCH_SIZE = 16;
LEARNING_RATE = 0.001;  LAYER_N = 5; EPOCH = 5; ITERS = 1; L2 = 1e-3; METHOD=0; GPUi = 0

MODEL_DIR = "./model/"
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

import sys
if __name__ == "__main__":
    print ("sys.argv n: " + str(len(sys.argv)))
    LEARNING_RATE = float(sys.argv[1])
    EPOCH         = int(sys.argv[2] )
    METHOD        = int(sys.argv[3] )
    L2            = float(sys.argv[4])
    if len(sys.argv)>5:
        LAYER_N       = int(sys.argv[5])        
    if len(sys.argv)>6:
        GPUi       = int(sys.argv[6])
    #*****************************************************************************************************************
    ## set GPU
    if '__file__' in globals():
        base_name = os.path.basename(__file__)+socket.gethostname()
        print(os.path.basename(__file__))
    
    base_name = base_name+'.layer'+str(LAYER_N)+'.dim'+str(IMG_HEIGHT2)+'.METHOD'+str(METHOD)+'.LR'+str(LEARNING_RATE)+'.EPOCH'+str(EPOCH)+'.L2'+str(L2)
    if METHOD==0:
        logging.basicConfig(filename=base_name+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    print (GPUi)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[GPUi], 'GPU')  
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)    
    
    print(tf.config.get_visible_devices())
    logging.info (tf.config.get_visible_devices()[0])
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()] 
    #*****************************************************************************************************************
    ## get train and testing data
    importlib.reload(train_test)
    # input_images_train_norm3 is for 1d purpose, input_images_train_norm2 is for 2d purpose
    ## generate input_images_train_norm2 is for 2d purpose, then input_images_train_norm3 getted from input_images_train_norm2        
    y_train, y_test, input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out, mean_train2, std_train2 = \
        train_test.get_train_test(data_per,orders,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion = 0.8)
                                                                        
    print(f"training data (2d) shape: {input_images_train_norm2.shape}")  # (N, 6, 23, 2)
    train_n = input_images_train_norm2.shape[0]
    test_n  = input_images_test_norm2 .shape[0]
    
    # get mask and 1d data from output
    masks = input_images_train_norm2[:,:,:,1].copy()
    data1 = input_images_train_norm2[:,:,:,0].copy()
    data1[masks==0] = -9999.0
    input_images_train_norm3 = np.moveaxis(data1,1,2) 
    
    masks = input_images_test_norm2[:,:,:,1].copy()
    data1 = input_images_test_norm2[:,:,:,0].copy()
    data1[masks==0] = -9999.0
    input_images_test_norm3 = np.moveaxis(data1,1,2) 
    
    print(f"training data (1d) shape: {input_images_train_norm3.shape}")
    
    #*****************************************************************************************************************
    ## save mean and std 
    mean_name=MODEL_DIR+base_name+'.demilune_2023_mean.csv'
    mean = mean_train2.copy()
    std  = std_train2.copy()
    # !!!!!!! transpose & reshape are different 
    arr = np.concatenate((mean.reshape(1,mean.shape[0]*mean.shape[1]), std.reshape(1,mean.shape[0]*mean.shape[1]) )).transpose()
    header = 'mean,std'
    np.savetxt(mean_name, arr, fmt="%s", header=header, delimiter=",")   
    #*****************************************************************************************************************    
    # testi = 0
    #*****************************************************************************************************************
    ## model
    accuracylist2 = list()
    importlib.reload(customized_train_lr)
    per_epoch = train_n//BATCH_SIZE
    
    for i in range(ITERS):
        print_str = "\n 1: transformer model 10*10 *****************************************************************************************************************"
        print (print_str); logging.info (print_str)
        importlib.reload(transformer_encoder)
        model = transformer_encoder.get_transformer_new_att0(n_times=IMG_HEIGHT2,n_feature=IMG_WIDTH2,n_out=N_CLASS, layern=3, units=64, n_head=4, drop=0.1, is_att=True) 
        if i==0:
            print (model.summary())
        
        # model_history = customized_train.my_train_schedule(model,train_dataset,validation_dataset,epochs=180,start_rate=learning_rate,loss=loss,per_epoch=per_epoch,split_epoch=20,option=0)
        model_history = customized_train_lr.my_train_schedule(model,input_images_train_norm3,y_train,epochs=EPOCH,start_rate=LEARNING_RATE,loss=loss,per_epoch=per_epoch,split_epoch=5,option=METHOD)
        
        accuracy,classesi = customized_train_lr.test_accuacy(model,input_images_test_norm3,y_test)
        print (">>>>>>>>>>>>>>>tranfatt" + '  {:0.4f}'.format(accuracy) )
        accuracylist2.append (accuracy)
        dat_out['predicted_cnn' + str(i)] = classesi                
        model_name = MODEL_DIR+base_name+'.tranfatt.model.h5'
    model.save(model_name)
     
    # #*****************************************************************************************************************
    print (accuracylist2)
    if accuracylist2!=[]:
        print ("accuracylist2 2d mean" + '  {:4.2f}'.format(np.array(accuracylist2).mean()*100) + "\nstd" + '  {:4.2f}'.format(np.array(accuracylist2).std()*100) )
    #*****************************************************************************************************************
    ##################################################################################################################
    ############################################model validation part#################################################
    ##################################################################################################################
    rst_arr0 = pd.read_csv(file_name).to_numpy()
    acc_all, f1_all = get_mean_acc_f1(rst_arr0)
    print(f"overal accuracy: {acc_all}")
    print(f"F1-score for each class: {f1_all}")
    print(confusion_matrix(rst_arr0[:,5].astype(int), rst_arr0[:,8].astype(int)))
    print(confusion_matrix(rst_arr0[:,5].astype(int), rst_arr0[:,9].astype(int)))
    print(confusion_matrix(rst_arr0[:,5].astype(int), rst_arr0[:,10].astype(int)))
    ##################################################################################################################
    ############################################model predict part#################################################
    ##################################################################################################################
    rst_dir    = './demilune_niger_aoicellis1.tif'
    input_dir  = './demilune_niger_2023_bd_times.csv'
    bn = os.path.basename(mean_name)[2:-23]    
    x_mean = 0
    x_std = 1
    if os.path.exists(mean_name):
        dat_temp = np.loadtxt(open(mean_name, "rb"), dtype='<U30', delimiter=",", skiprows=1)
        arr = dat_temp.astype(np.float64)
        x_mean,x_std = arr[:,0],arr[:,1]
    else:
        print("Error !!!! mean file not exists " + mean_name)
    
    rst = rasterio.open(rst_dir)
    rst_prof = rst.profile
    
    inputdt = pd.read_csv(input_dir)
    input_arr = inputdt.to_numpy()    
    ###########################################################################
    # shape (N, bands, times)
    train_fields = list()
    for bandi in ('b01', 'b02', 'b03', 'b04','b05', 'b06','b07','b08','b11','b12'):
        for ni in ('00','01','02','03','04','05','06','07','08','09'):    
            train_fields.append(ni+'.'+bandi)
    input_dt = inputdt[train_fields].to_numpy()
    input_ready = (input_dt-x_mean)/x_std
    input_2d = input_ready.reshape(input_ready.shape[0], 10, 10)
    ## *********************************************************************
    ## load model
    # model = tf.keras.models.load_model(model_path, compile=False)     
    ## **********************************************************************
    logits = model.predict(input_2d)
    rst_list = np.argmax(logits,axis=1).astype(np.uint8)
    ref_dt = input_arr[:,0:4]  
    final_rst = np.append(ref_dt, np.expand_dims(rst_list, axis=1), axis=1)
    heads = ['FID', 'ld', 'gridcode', 'Name', 'result'] 
    pd.DataFrame(final_rst).to_csv('./final_result.csv',  header=heads, index=False)        
    ################################################################################
    ## save raster data
    rst_arr = np.array(rst_list)
    rst_2d = rst_arr.reshape(38, 63)
    naip_metai = rst_prof.copy()
    naip_metai['driver'] = 'GTiff'       
    naip_metai['count'] = 1
    OUT = './demilune_result2023.' + bn + '.tif'          
    with rasterio.open(OUT, 'w', **naip_metai) as dst:
        dst.write(rst_2d, 1)    

    

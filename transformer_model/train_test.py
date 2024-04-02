"""
Dong adapted from LAMAP purpose
note:
    csv band refletance already in [0,1]
    to use this function, csv column order should be 23 blue band, 23 green band, 23 red band, etc.  
"""
# import train_test 
import os 
import math 
import numpy as np  
import pandas as pd 


SPLIT_DIR = "./split/"

if not os.path.isdir(SPLIT_DIR):
    os.makedirs(SPLIT_DIR)

## *************************************************************************
## training and validation data split 
## invoked by customized_train.py
def random_split_train_validation (X_train,y_train,pecentage = 0.04):
    """
    split train into training and validating
    Used in the "customized_train.py" 
    """
    total_n = y_train.shape[0]
    sample_n = math.ceil(total_n*pecentage)
    split_n  = math.ceil(total_n/sample_n)
    file_index = SPLIT_DIR+"split.total_n"+str(total_n)+".for.validation.txt"
    
    if os.path.isfile(file_index):
        print ('file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
        
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i==0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        
        orders = orders[range(total_n)]  
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    
    validation_index = orders==0
    training_index   = orders!=0
    sum(validation_index)
    sum(training_index  )
    return X_train[training_index],y_train[training_index],X_train[validation_index],y_train[validation_index],training_index,validation_index

## *************************************************************************
## training and testing data split 
## invoked by main function 
def random_split (total_n, split_n):
    """
    creat split index by split total_n into train and test based on split_n
    test: orders==0
    train: order!=0
    """
    sample_n = math.ceil(total_n/split_n)
    file_index = SPLIT_DIR+"index.total_n"+str(total_n)+".for.random.txt"
    if os.path.isfile(file_index):
        print ('file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
        
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i==0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        
        orders = orders[range(total_n)]  
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    
    return orders
####################################################################################
##########CORE FUCNTION TO GENERATE model input data################################
####################################################################################
# # IMG_HEIGHT = COMPOSITE_N; IMG_WIDTH=6; IMG_BANDS=1
# data_all = data_per
# train_fields = train_metric
# test_field = class_field
# IMG_HEIGHT = IMG_HEIGHT2
# IMG_WIDTH = IMG_WIDTH2
# IMG_BANDS = IMG_BANDS2; is_single_norm=True;

def construct_composite_train_test(data_all,index_train,index_test,train_fields,test_field,
                                   IMG_HEIGHT,IMG_WIDTH,IMG_BANDS,mean_train=0,std_train=1, 
                                   is_train_test_com=True):
    """
    Perpare train and test from composite csv file (simple version of 'construct_metric_train_test')
    input_images_train shape: (trainx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    creat a mask_train and mask_test
    mean_train,std_train shape: (IMG_WIDTH, 1,1) (1) when processing step50 data, 0 and 1, (2) when processing 25000, it used mean and std from step50 data
    """   
    trainx2 = np.array(data_all[train_fields][index_train]).astype(np.float32)
    input_images_train = trainx2.reshape(trainx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    y_train = np.array(data_all[test_field][index_train]).astype(np.int32)
    
    testx2 = np.array(data_all[train_fields][index_test]).astype(np.float32)
    input_images_test = testx2.reshape(testx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    y_test = np.array(data_all[test_field][index_test]).astype(np.int32)
    train_n = input_images_train.shape[0]
    test_n  = input_images_test .shape[0]
    
    print(train_n)
    print(test_n )
      
    ## change nan value to 0 
    input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_BANDS)
    masks_train = np.logical_not(np.isnan(input_images_train)).astype(np.float32)
    input_images_train0 = np.concatenate((input_images_train,masks_train),axis=3)
    input_images_train0[np.isnan(input_images_train[:,:,:,0]),:] = 0
    masks_test = np.logical_not(np.isnan(input_images_test)).astype(np.float32)
    input_images_test0 = np.concatenate((input_images_test,masks_test),axis=3)
    input_images_test0[np.isnan(input_images_test[:,:,:,0]),:] = 0
    
    ## normalize 
    input_images_train_norm0 = input_images_train0.copy()
    input_images_test_norm0  = input_images_test0 .copy()
    
    ## this norm turn out to be very important 
    if is_train_test_com:
        a = np.ma.array(np.concatenate((input_images_train0[:,:,:,0],input_images_test0[:,:,:,0])), \
            mask=np.concatenate((input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0,input_images_test0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)))
    else:
        a = np.ma.array(input_images_train0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
    if IMG_BANDS==0:
        a = np.concatenate((input_images_train[:,:,:],input_images_test[:,:,:]))
        mean_train = a.mean(axis=0).reshape(a.shape[1],a.shape[2])
        std_train  = a.std (axis=0).reshape(a.shape[1],a.shape[2])
        input_images_train_norm0[:,:,:] = (input_images_train[:,:,:] - mean_train)/std_train
        input_images_test_norm0 [:,:,:] = (input_images_test [:,:,:] - mean_train)/std_train
    else:
        a = np.concatenate((input_images_train[:,:,:,0],input_images_test[:,:,:,0]))
        mean_train = a.mean(axis=0).reshape(a.shape[1],a.shape[2],1)
        std_train  = a.std (axis=0).reshape(a.shape[1],a.shape[2],1)
        input_images_train_norm0[:,:,:,:IMG_BANDS] = (input_images_train[:,:,:,:IMG_BANDS] - mean_train)/std_train
        input_images_test_norm0 [:,:,:,:IMG_BANDS] = (input_images_test [:,:,:,:IMG_BANDS] - mean_train)/std_train       
    return input_images_train_norm0,input_images_test_norm0,input_images_train0,input_images_test0,y_train,y_test,mean_train,std_train
## *************************************************************************
def get_train_test (data_per,orders,
                    IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,
        					  class_field,proportion=0.8, total_days=10, 
        					  ):
    """
    used construct_composite_train_test to get train and test data for model input
    training_location: normalize based on x and y by using mean_train_pre,std_train_pre
    """      
    if proportion==0.5:
        index_train = orders<5
        index_test  = orders>=5
    elif proportion==0.8:
        index_train = orders<8
        index_test  = orders>=8
    elif proportion==0.9:
        index_train = orders<9
        index_test  = orders>=9
    else:
        index_train = orders==0
        index_test  = orders!=0  
    train_metric = list()
    bandslist = ['b01', 'b02', 'b03', 'b04','b05', 'b06', 'b07', 'b08', 'b11', 'b12']        
    for bandi in bandslist:       
        for nii in range(total_days):
            ni = '{:02d}'.format(nii)            
            train_metric.append(ni+'.'+bandi)
        
    input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train,std_train \
        = construct_composite_train_test(data_per,index_train,index_test,train_metric,class_field,
		                                 IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,)                                    									 
    dat_out = pd.DataFrame()
    for propertyi in data_per.keys():
        if '0' not in propertyi:
            dat_out[propertyi] = (data_per[propertyi][index_test]).copy()
    
    dat_out['predicted_cnn0'] = 255
    dat_out['predicted_cnn1'] = 255
    dat_out['predicted_cnn2'] = 255
    dat_out['predicted_cnn3'] = 255
    dat_out['predicted_cnn4'] = 255

    return y_train,y_test,\
            input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train

###################################################################################################################################################
###################################################################################################################################################

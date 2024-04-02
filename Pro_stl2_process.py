"""
Created on Fri Mar 29 12:30:59 2024
@author: Dong.Luo
unzip all downloaded data and perpare data to cut to AOI
per-prepare:
    (1) download all avalible L2A products from online and unzip
    (2) years from 2016-01-01 to 2023-12-31
NOTE:
    (1) bands: b1, b2, b3, b4, b5, b6, b7, b8, b11, b12, SCL
    (2) SCL and band filled value = 0
    (4) start from baseline"_N04??" it has offset (-1000)
OUTPUT:
    (1) 10 m products folder 
    (2) shape (38, 63)
"""
import os
# import glob2
# import shutil

import numpy as np 
import importlib
import rasterio
from rasterio.enums import Resampling 
import scipy.ndimage


tile_name = '31PDR'

UTM_DIR = './' + tile_name + '/unzip/'   

r10m_DIR = './' + tile_name + '/cut_boa10m/'      # composite bands
SCL_DIR = './'  + tile_name + '/cut_scl10m/'       
# https://stackoverflow.com/questions/2212643/python-recursive-folder-read  
# ##************************************************************************************************
## create dir 
def create_dir (dir1, dir2=''):
    if not os.path.isdir(dir1):
        os.makedirs(dir1)
    
    if dir2!='' and not os.path.isdir(dir2):
        os.makedirs(dir2)         

create_dir (dir1=r10m_DIR, dir2=SCL_DIR)

# find single file using pattern parameter        
def find_one_file (dir1, pattern=''):
    find_file = ''
    for root, dirs, files in os.walk(dir1):
        for file in files:
            # print(file)
            if pattern in file:                
                find_file = os.path.join(root, file)    
    return find_file

def find_img_folder (safe_path):
        img_folder = ''   
        for root, subdir, files in os.walk(safe_path):
            for folder in subdir:
                # print(folder)
                if 'IMG_DATA' in folder:
                    img_folder = os.path.join(root, folder)           
        return img_folder
              
##************************************************************************************************
## generate file list  
l2a_list = list()
for root, dirs, files in os.walk(UTM_DIR):
   for safe in dirs:
       if '.SAFE' in safe:
           l2a_list.append(os.path.join(root, safe))          
print(len(l2a_list))
n_file = len(l2a_list)

if n_file==0:
    print ("!!!No file is found")
# # ##************************************************************************************************
# ## utm process 
for i in range(n_file):
    print(i)
    l2a_file = l2a_list[i]
    basenamei = os.path.basename(l2a_file)[0:60] 
    ##********************************************* 
    ## find img folder and QI folder                
    img_folder = find_img_folder (l2a_file)     
    ##*********************************************
    ##copy SCL layer  
    scl_file = find_one_file (img_folder, pattern='SCL_20m')
    dataset = rasterio.open(scl_file)
    ref0_profile = dataset.profile
    scl10m = dataset.read(out_shape =(dataset.count, 
                                        int(dataset.shape[0]*int(2)),
                                        int(dataset.shape[1]*int(2))),
                                          resampling=Resampling.nearest)
    scl10m1 = np.squeeze(scl10m)[9771:9809, 1580:1643]
    if np.count_nonzero(scl10m1==0) != scl10m1.shape[0]*scl10m1.shape[1]:
        scl10m_dir = SCL_DIR + basenamei + '_10m_SCL.jp2'
        meta = ref0_profile.copy()
        meta['diver'] = 'JP2OpenJPEG'
        meta['width']  = scl10m1.shape[1]   
        meta['height'] = scl10m1.shape[0]
        meta['count'] = 1
        meta['dtype'] = 'uint8'
        meta['transform'] = rasterio.transform.Affine(10,0,415760.0,0,-10,1502310.0)
        with rasterio.open(scl10m_dir, 'w', **meta) as dst:
            dst.write(scl10m1, 1)
    ##*********************************************
    ##process 20m BOA and save to the folder     
    imgb05 = find_one_file (img_folder, pattern='B05_20m')
    dataset = rasterio.open(imgb05)
    boa_b05 = dataset.read(out_shape =(dataset.count, 
                                        int(dataset.shape[0]*int(2)),
                                        int(dataset.shape[1]*int(2))),
                                          resampling=Resampling.nearest)
    
    imgb06 = find_one_file (img_folder, pattern='B06_20m')
    dataset = rasterio.open(imgb06)
    boa_b06 = dataset.read(out_shape =(dataset.count, 
                                        int(dataset.shape[0]*int(2)),
                                        int(dataset.shape[1]*int(2))),
                                          resampling=Resampling.nearest)
    
    imgb07 = find_one_file (img_folder, pattern='B07_20m')
    dataset = rasterio.open(imgb07)
    boa_b07 = dataset.read(out_shape =(dataset.count, 
                                        int(dataset.shape[0]*int(2)),
                                        int(dataset.shape[1]*int(2))),
                                          resampling=Resampling.nearest)
    
    imgb11 = find_one_file (img_folder, pattern='B11_20m')
    dataset = rasterio.open(imgb11)
    boa_b11 = dataset.read(out_shape =(dataset.count, 
                                        int(dataset.shape[0]*int(2)),
                                        int(dataset.shape[1]*int(2))),
                                          resampling=Resampling.nearest)
    
    imgb12 = find_one_file (img_folder, pattern='B12_20m')
    dataset = rasterio.open(imgb12)
    boa_b12 = dataset.read(out_shape =(dataset.count, 
                                        int(dataset.shape[0]*int(2)),
                                        int(dataset.shape[1]*int(2))),
                                          resampling=Resampling.nearest)
    ##*********************************************
    ##process 10m BOA and save to the folder
    imgb02 = find_one_file (img_folder, pattern='B02_10m')
    dataset = rasterio.open(imgb02)          
    ref_profile = dataset.profile
    boa_b02 = dataset.read()
    
    imgb03 = find_one_file (img_folder, pattern='B03_10m')
    boa_b03 = rasterio.open(imgb03).read()
    
    imgb04 = find_one_file (img_folder, pattern='B04_10m')
    boa_b04 = rasterio.open(imgb04).read()
    
    imgb08 = find_one_file (img_folder, pattern='B08_10m')
    boa_b08 = rasterio.open(imgb08).read()
    ##*********************************************
    ##process 60m BOA and save to the folder
    imgb01 = find_one_file (img_folder, pattern='B01_60m')
    dataset = rasterio.open(imgb01)          
    boa_b01 = dataset.read(out_shape =(dataset.count, 
                                        int(dataset.shape[0]*int(6)),
                                        int(dataset.shape[1]*int(6))),
                                        resampling=Resampling.nearest)      
    boa_all = np.stack([np.squeeze(boa_b01),np.squeeze(boa_b02),np.squeeze(boa_b03),np.squeeze(boa_b04),\
                        np.squeeze(boa_b05),np.squeeze(boa_b06),np.squeeze(boa_b07),np.squeeze(boa_b08),\
                        np.squeeze(boa_b11),np.squeeze(boa_b12)],axis = 0).astype(np.int16)
    if "_N030" in  basenamei:
        boa_all = boa_all           
    else:
        boa_all = boa_all -1000    
    boa_all1 = boa_all[:, 9771:9809, 1580:1643]
    if np.count_nonzero(scl10m1==0) != scl10m1.shape[0]*scl10m1.shape[1]:
        boa_10m_dir = r10m_DIR + basenamei + '_10m_BOA.jp2'
        meta = ref_profile.copy()
        meta['diver'] = 'JP2OpenJPEG'
        meta['width']  = boa_all1.shape[2]   
        meta['height'] = boa_all1.shape[1] 
        meta['count'] = 10
        meta['dtype'] = 'int16'
        meta['transform'] = rasterio.transform.Affine(10,0,415760.0,0,-10,1502310.0)
        with rasterio.open(boa_10m_dir, 'w', **meta) as dst:
            dst.write(boa_all1)
##*************************************************************************************************************       


                

        
        




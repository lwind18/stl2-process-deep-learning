"""
Created on Fri Mar 29 18:18:23 2024
@author: Dong.Luo
generate point value
shape (38, 63)
"""
import os
import pandas as pd
import numpy as np
import rasterio


def file_list (DIR ,pattern0="", pattern1 = ""):
    boa_scl_list = list()    
    for root, dirs, files in os.walk(DIR):
        for file in files:
            if pattern0 in file and pattern1 in file:
                boa_scl_list.append(os.path.join(root, file))                
    return boa_scl_list

month_dir = 'G:/GRI_interview/cut_boa10m_comp'

yr_list = ['S2_MSIL2A_2020', 'S2_MSIL2A_2021', 'S2_MSIL2A_2022', 'S2_MSIL2A_2023']
yr_str = 'S2_MSIL2A_2023'
image_list = file_list (month_dir, pattern0 = yr_str, pattern1 = 'T31PDR_comp.tif')
image_list.sort()

month_list = []
total_n = len(image_list)
for i in range(total_n):
    monthi = image_list[i]
    month_arri = rasterio.open(monthi).read()
    month_arri2d = month_arri.reshape(month_arri.shape[0], month_arri.shape[1]*month_arri.shape[2])
    ## need shape (2394, b1*10, b2*10, b10*10). 2394: left to right, top to down
    month_arri2d_need = np.moveaxis(month_arri2d, 1, 0)
    month_list.append(month_arri2d_need)

month_all = np.array(month_list)
## shape needs to be (pixel number, bands, number of image)
month_allneed0 = np.moveaxis(month_all, 0, 2)
month_all1d0 = month_allneed0.reshape(month_allneed0.shape[0], month_allneed0.shape[1]*month_allneed0.shape[2])
## generate heads
csv_heads0 = list()
for bandi in ('b01', 'b02', 'b03', 'b04','b05', 'b06','b07','b08','b11','b12'):
    for ni in ('00','01','02','03','04','05','06','07','08','09'):    
        csv_heads0.append(ni+'.'+bandi)

####################################################################
lab_dir = 'G:/GRI_interview/csvs/all_cell_train_test.csv'
dtty = pd.read_csv(lab_dir)
arrsyy = dtty.to_numpy()
print("Total pixels:" + str(arrsyy.shape[0]))  
y_heads = dtty.columns.tolist()

yr_arr = np.full([arrsyy.shape[0],1], fill_value=yr_str[-4:], dtype=int)
arr_xy0 =np.hstack((arrsyy, yr_arr, month_all1d0))
arr_xy1 =np.hstack((arrsyy, yr_arr, month_all1d1))
          
##************************************************************************************************************    
outfiles0 = 'G:/GRI_interview/csvs/' + 'demilune_niger_' + str(yr_str[-4:]) +'_bd_times.csv'     
head0 = y_heads + ['year'] + csv_heads0 
pd.DataFrame(arr_xy0).to_csv(outfiles0,  header=head0, index=False)    # header=None, 

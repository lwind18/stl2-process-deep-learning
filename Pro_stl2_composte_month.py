"""
Created on Fri Mar 29 17:40:55 2024
@author: Dong.Luo
process composite image per month
shape (38, 63)
ESA:2016:0; 2017:4, 2018:16, 2019:17, 2020:31, 2021:36, 2022:29, 2023:24     Total: 157
GEE:2016:0; 2017:0, 2018:2,  2019:32, 2020:37, 2021:40, 2022:30, 2023:23     Total: 164
"""
import os
import numpy as np
import rasterio
import gc

TOTALBANDS = 10
DIM_H = 38; DIM_W = 63

def file_list (DIR ,pattern0="", pattern1 = ""):
    boa_scl_list = list()    
    for root, dirs, files in os.walk(DIR):
        for file in files:
            if pattern0 in file and pattern1 in file:
                boa_scl_list.append(os.path.join(root, file))                
    return boa_scl_list

scl_dir = 'G:/GRI_interview/cut_scl10m'
b10_dir = 'G:/GRI_interview/cut_boa10m'

out_img = 'G:/GRI_interview/cut_boa10m_comp/'

def clear_stl2 (scl):
    nodata     = scl!=0
    cld_shadow = scl!=3
    cld_medium = scl!=8
    cld_high   = scl!=9
    cirrus     = scl!=10
    snow       = scl!=11
    all_clear = cld_shadow & cld_medium & cld_high & cirrus & snow & nodata
    return all_clear

## generate files in each designed date list
def get_composite (ddir, pattern1 = '.tif', pattern2 = '.tif'):
    """
    function to find files in the designed days
    ddir: directory
    """            
    file_path = []
    for root, dirs, files in os.walk(ddir):
        for file in files:                         
            if pattern1 in file and pattern2 in file:            
                find_file = os.path.join(root, file)
                # print(find_file)
                file_path.append(find_file)     
    return file_path

month = ['MSIL2A_202301', 'MSIL2A_202302', 'MSIL2A_202303', 'MSIL2A_202304', 'MSIL2A_202305', 'MSIL2A_202306',
         'MSIL2A_202307', 'MSIL2A_202308', 'MSIL2A_202309', 'MSIL2A_202310', 'MSIL2A_202311', 'MSIL2A_202312']

month = ['MSIL2A_202201', 'MSIL2A_202202', 'MSIL2A_202203', 'MSIL2A_202204', 'MSIL2A_202205', 'MSIL2A_202206',
          'MSIL2A_202207', 'MSIL2A_202208', 'MSIL2A_202209', 'MSIL2A_202210', 'MSIL2A_202211', 'MSIL2A_202212']

month = ['MSIL2A_202101', 'MSIL2A_202102', 'MSIL2A_202103', 'MSIL2A_202104', 'MSIL2A_202105', 'MSIL2A_202106',
          'MSIL2A_202107', 'MSIL2A_202108', 'MSIL2A_202109', 'MSIL2A_202110', 'MSIL2A_202111', 'MSIL2A_202112']

month = ['MSIL2A_202001', 'MSIL2A_202002', 'MSIL2A_202003', 'MSIL2A_202004', 'MSIL2A_202005', 'MSIL2A_202006',
          'MSIL2A_202007', 'MSIL2A_202008', 'MSIL2A_202009', 'MSIL2A_202010', 'MSIL2A_202011', 'MSIL2A_202012']

for mn in month:
    scl_list = file_list(scl_dir, pattern0 = mn, pattern1 ='_10m_SCL.jp2')
    scl_list.sort()
    b10_list = file_list(b10_dir, pattern0 = mn, pattern1 ='_10m_BOA.jp2')
    b10_list.sort()
    
    total_n = len(scl_list)
    if total_n > 0:
        basename = os.path.basename(scl_list[0])[0:16] + os.path.basename(scl_list[0])[31:43]
        rsti = rasterio.open(b10_list[0])
        rst_prof = rsti.profile
        
        Image_time_series = np.full([total_n, TOTALBANDS, DIM_H, DIM_W], fill_value=0, dtype=np.uint16)
        for i in range(total_n):   
                                    
            scli = np.squeeze(rasterio.open(scl_list[i]).read())
            scl_lri = clear_stl2(scli)
            no_filled_msk = scl_lri == 1
            b10i = rasterio.open(b10_list[i]).read() 
            b10i[:,np.logical_not(no_filled_msk)] = 0
            Image_time_series[i,:,:,:] = b10i     
           
        ## median method: chagne 0 to np.nan. then use np.nanmedian    
        FINAL_IMG = np.full([TOTALBANDS, DIM_H, DIM_W], fill_value=0, dtype=np.int16)
        if Image_time_series.shape[0] >=2:
            for bd in range(Image_time_series.shape[1]):
                time_series_bandi = Image_time_series[:,bd,:,:]
                time_series_bandii = np.where(time_series_bandi==0, np.nan, time_series_bandi)
                FINAL_IMG[bd, :, :] = np.nanmedian(time_series_bandii, axis=0)
        else:
            FINAL_IMG = np.squeeze(Image_time_series)        
        
        del Image_time_series
        gc.collect()
        
        naip_metai = rst_prof.copy()
        naip_metai['driver'] = 'GTiff'       
        naip_metai['count'] = TOTALBANDS
        naip_metai['width']  = DIM_W  
        naip_metai['height'] = DIM_H
        naip_metai['dtype'] = 'uint16' 
        OUT = out_img + basename +'_comp.tif'          
        with rasterio.open(OUT, 'w', **naip_metai) as dst:
            dst.write(FINAL_IMG)



























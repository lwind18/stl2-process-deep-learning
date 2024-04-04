"""
Created on Fri Mar 29 13:35:18 2024
@author: Dong.Luo
process NDVI and NDMI
(1) bands: b1, b2, b3, b4, b5, b6, b7, b8, b11, b12, SCL
(2) SCL and band filled value = 0
total number of image: 157
ESA:2016:0; 2017:4, 2018:16, 2019:17, 2020:31, 2021:36, 2022:29, 2023:24
shape (38, 63)
"""
import os
import numpy as np
import rasterio
import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

scl_dir = 'G:/GRI_interview/cut_scl10m'
b10_dir = 'G:/GRI_interview/cut_boa10m'
mask_dir = 'G:/GRI_interview/demilune_niger_mask.tif'

def find_files(DIR, pattern=".TIF"):
    file_list = list()
    for root, dirs, files in os.walk(DIR):
        for file in files:
            if  pattern in file:             
                file_list.append(os.path.join(root, file))               
    return file_list

def clear_stl2 (scl):
    nodata     = scl!=0
    cld_shadow = scl!=3
    water      = scl!=6
    cld_medium = scl!=8
    cld_high   = scl!=9
    cirrus     = scl!=10
    snow       = scl!=11
    all_clear = cld_shadow & cld_medium & cld_high & cirrus & snow & nodata & water
    return all_clear

mask_arr = np.squeeze(rasterio.open(mask_dir).read())
nofill_msk = mask_arr != 65535

scl_list = find_files(scl_dir, pattern ='_10m_SCL.jp2')
scl_list.sort()
b10_list = find_files(b10_dir, pattern ='_10m_BOA.jp2')
b10_list.sort()

total_n = len(scl_list)

ndvi_bands = np.full([total_n, np.count_nonzero(nofill_msk!=0)],fill_value=np.nan,dtype=np.float32)
ndmi_bands = np.full([total_n, np.count_nonzero(nofill_msk!=0)],fill_value=np.nan,dtype=np.float32)
# scl_bands  = np.full([total_n, np.count_nonzero(nofill_msk!=0)], fill_value=False ,dtype=np.int8)
doys       = np.full ([total_n],fill_value=0,dtype=object)

for i in range(total_n):
    scli = scl_list[i]
    doys[i] = os.path.basename(scli)[10:18]
    # doys[i] = datetime.datetime.strptime(datei, '%Y%m%d').timetuple().tm_yday
    b10i = b10_list[i]
    
    scl_arri = np.squeeze(rasterio.open(scli).read())
    scl_cleari = clear_stl2(scl_arri)
    scl_clrii = scl_cleari[nofill_msk]
    ## check total usable pixels
    print(f"usable total number of pixels: {np.count_nonzero(scl_clrii==1)} vs 228")
    ## band order: b1, b10, b2, b3, b4, b5, b6, b7, b8, b9
    b08_arri = np.squeeze(rasterio.open(b10i).read(8))
    b04_arri = np.squeeze(rasterio.open(b10i).read(4))
    b11_arri = np.squeeze(rasterio.open(b10i).read(9))
    ndvii = (b08_arri[nofill_msk] - b04_arri[nofill_msk])/(b08_arri[nofill_msk] + b04_arri[nofill_msk])
    ndmii = (b08_arri[nofill_msk] - b11_arri[nofill_msk])/(b08_arri[nofill_msk] + b11_arri[nofill_msk])
    if np.count_nonzero(scl_clrii==1) < 228:
        ndvi_bands[i,:] = ndvii[~scl_clrii] = np.nan
        ndmi_bands[i,:] = ndmii[~scl_clrii] = np.nan
    else:
        ndvi_bands[i,:] = ndvii[scl_clrii]
        ndmi_bands[i,:] = ndmii[scl_clrii]

xx = range(len(doys))
 
fig1, ax1 = plt.subplots(figsize=(20,6))
ax1.scatter(xx, np.nanmean(ndvi_bands, axis=1), marker='o', s=10, color ='blue', label = 'NDVI time series')
ax1.plot(xx, np.nanmean(ndvi_bands, axis=1), color = 'blue', linestyle='dashed', lw=0.3)
ax1.set_xticks([4, 20, 37, 68, 104, 133])
ax1.set_xticklabels(['2018.01', '2019.01', '2020.01', '2021.01', '2022.01', '2023.01']) 


ax1.tick_params(axis='both', which = 'major', labelsize='large')
ax1.set_xlabel("years", fontsize='large')
ax1.set_ylabel("NDVI", fontsize='large')  
ax1.legend(ncol=1, loc='upper left', fontsize='medium')  
fig1.savefig('./GRI_rst/demilune_ndvi.png', dpi=150)

fig2, ax2 = plt.subplots(figsize=(20,6))
ax2.scatter(xx, np.nanmean(ndmi_bands, axis=1), marker='o', s=10, color ='green', label = 'NDMI time series')
ax2.plot(xx, np.nanmean(ndmi_bands, axis=1), color = 'green', linestyle='dashed', lw=0.3)
ax2.set_xticks([4, 20, 37, 68, 104, 133])
ax2.set_xticklabels(['2018.01', '2019.01', '2020.01', '2021.01', '2022.01', '2023.01']) 

ax2.tick_params(axis='both', which = 'major', labelsize='large')
ax2.set_xlabel("years", fontsize='large')
ax2.set_ylabel("NDMI", fontsize='large') 
ax2.legend(ncol=1, loc='upper left', fontsize='medium')
fig2.savefig('./GRI_rst/demilune_ndmi.png', dpi=150)



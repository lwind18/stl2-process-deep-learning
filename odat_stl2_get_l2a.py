"""
Created on Fri Jul 28 08:56:26 2023
@author: Dong.Luo
new way to download sentinel data
from Copernicus Data Space Ecosystem APIs 
ODate API: https://documentation.dataspace.copernicus.eu/APIs/OData.html
note:
    Search options should always be preceded with $ and consecutive options should be separated with &.
    Consecutive filters within filter option should be separated with and or or. Not operator can also be used
    filter can use &$ for consecutive filters such as "&$orderby=ContentDate/Start desc"; "&$skip=23"; "&$count=True"; "&$top=100"
    if response.status_code == 200, the file is avariable and can be download
    seems that this method cannot check existed files in the folder and download files not in the folder because it takes token when do checking
    token issue: have reached the maximum number of sessions  >> each month 1st will refresh!
    search tiles between two dates using ContentDate (Acquistition Dates): ContentDate/Start gt 2019-02-01T00:00:00.000Z and ContentDate/Start lt 2019-02-28T00:00:00.000Z 
    date uses: ISO 8601 Datetime  
"""
import os
import sys
import numpy as np
import pandas as pd
import requests
import time

## get key token function
def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
        }
    try:
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
            )
    return r.json()["access_token"]


tiles_na = ['31PDR']         

conDate = ["ContentDate/Start gt 2016-01-01T00:00:00.000Z and ContentDate/Start lt 2016-01-31T00:00:00.000Z",
           "ContentDate/Start gt 2016-02-01T00:00:00.000Z and ContentDate/Start lt 2016-02-29T00:00:00.000Z",
           "ContentDate/Start gt 2016-03-01T00:00:00.000Z and ContentDate/Start lt 2016-03-31T00:00:00.000Z",
           "ContentDate/Start gt 2016-04-01T00:00:00.000Z and ContentDate/Start lt 2016-04-30T00:00:00.000Z",
           "ContentDate/Start gt 2016-05-01T00:00:00.000Z and ContentDate/Start lt 2016-05-31T00:00:00.000Z",
           "ContentDate/Start gt 2016-06-01T00:00:00.000Z and ContentDate/Start lt 2016-06-30T00:00:00.000Z",
           "ContentDate/Start gt 2016-07-01T00:00:00.000Z and ContentDate/Start lt 2016-07-31T00:00:00.000Z",
           "ContentDate/Start gt 2016-08-01T00:00:00.000Z and ContentDate/Start lt 2016-08-31T00:00:00.000Z",
           "ContentDate/Start gt 2016-09-01T00:00:00.000Z and ContentDate/Start lt 2016-09-30T00:00:00.000Z",
           "ContentDate/Start gt 2016-10-01T00:00:00.000Z and ContentDate/Start lt 2016-10-31T00:00:00.000Z",
           "ContentDate/Start gt 2016-11-01T00:00:00.000Z and ContentDate/Start lt 2016-11-30T00:00:00.000Z",
           "ContentDate/Start gt 2016-12-01T00:00:00.000Z and ContentDate/Start lt 2016-12-31T00:00:00.000Z"]
           
start_time = time.time()
for tileid in tiles_na:
    print("tileid is:" + tileid)
    SAVE_DIR = './T' + tileid + '/need'
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)    
    files = os.listdir(SAVE_DIR)
    keys_list = []
    for fk in range(len(files)):
        filei = files[fk]
        filei_date = filei[0:26] 
        filei_obt_id = filei[33:60]        
        tileid_keys = filei_date + '_N????_' + filei_obt_id
        keys_list.append(tileid_keys)
    tileid_keys_unique = np.unique(np.array(keys_list))
    for cd in conDate:
        ## product name, L2A, cloud cover and month       
        json = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-2' and contains(Name,'{tileid}') \
                            and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') \
                            and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt 10.00) \
                            and {cd}").json()           
        product_list = pd.DataFrame.from_dict(json['value'])
        product_arr = product_list.to_numpy()
        total_n = product_arr.shape[0]
        print(total_n)                
        for i in range(total_n):
            producti = product_arr[i,:]
            product_id = producti[1]
            product_name = producti[2]
            product_keys = product_name[0:26] + '_N????_' + product_name[33:60]
            if '_N0500_' in product_name:
                access_token = get_keycloak ("???", "???")                                
                session = requests.Session()
                session.headers.update({'Authorization': f'Bearer {access_token}'})
                url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
                response = session.get(url, allow_redirects=False)
                while response.status_code in (301, 302, 303, 307):
                    url = response.headers['Location']
                    response = session.get(url, allow_redirects=False)
                get_file_info = session.get(url, verify=False, allow_redirects=True)            
                if get_file_info.status_code == 200:
                    print(product_name)
                    out_file = SAVE_DIR + '/' + product_name + '.zip'
                    with open(out_file, 'wb') as p:
                        p.write(get_file_info.content)                 
                else:
                    print(f"{product_name} cannot be download!!!")
print("--- %s seconds ---" % (time.time() - start_time))            
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:34:07 2023

@author: oak
"""

"""
missing files:
4/18: (26 - 42)
    20220418_084000
    20220418_090000
    ...
    20220418_140000

5/13: (28, 34-45)
    20220513_092000
    
    20220513_112000
    20220513_114000
    ...
    20220513_150000
    
5/17: (47-71)
    20220517_154000
    20220517_160000
    ...
    20220517_234000
    
5/18: (00-48)
    20220518_000000
    20220518_002000
    ...
    20220518_160000
    
5/19: (24-27)
    20220519_080000
    20220519_082000
    ...
    20220519_090000

6/04: (00-71)
    20220604_000000
    20220604_002000
    ...
    20220604_234000
    
6/05: (00-71)
    20220605_000000
    20220605_002000
    ...
    20220605_234000
    
6/06: (00-33)
    20220605_000000
    20220605_002000
    ...
    20220605_110000
"""

#%%

import pandas as pd
import numpy as np

print('done with imports')

#%%


hours_arr = np.array(["00", "01", "02", "03", "04", "05", "06", 
                      "07", "08", "09", "10", "11", "12", 
                      "13", "14", "15", "16", "17", "18", 
                      "19", "20", "21", "22", "23", ])
print(len(hours_arr))
minutes_arr = np.array(["00", "20","40"])
seconds_arr = np.array(['00'])

print(str(hours_arr[4])+str(minutes_arr[0]))



full_time_list = []

for i in range(len(hours_arr)):
    for j in range(3):
        full_time_ij = str(hours_arr[i])+str(minutes_arr[j])+str(seconds_arr[0])
        full_time_list.append(full_time_ij)

print(full_time_list[2])



#%%
ports = np.array([1,2,3,4,5,6,7])
print(ports[0])
print(len(ports))

#%%
import os
import natsort

filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
filename_generalDate = []
filename_port1 = []
filename_port2 = []
filename_port3 = []
filename_port4 = []
filename_port5 = []
filename_port6 = []
filename_port7 = []
for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]
        fiilename_general_name_only = filename[12:-4]
        if filename.startswith("mNode_Port1"):
            filename_port1.append(filename_only)
            filename_generalDate.append(fiilename_general_name_only)
            if not filename.endswith("00.txt"):
                print(str(filename))
            else:
                continue
        if filename.startswith("mNode_Port2"):
            filename_port2.append(filename_only)
        if filename.startswith("mNode_Port3"):
            filename_port3.append(filename_only)
        if filename.startswith("mNode_Port4"):
            filename_port4.append(filename_only)
        if filename.startswith("mNode_Port5"):
            filename_port5.append(filename_only)
        if filename.startswith("mNode_Port6"):
            filename_port6.append(filename_only)
        if filename.startswith("mNode_Port7"):
            filename_port7.append(filename_only)
        else:
            continue
#%%       
test_var = filename_port1[2471][11:]        
#%%
for i in range(2000,2500):
    if filename_port1[i][11:] == filename_port7[i][11:]:
        continue
    else:
        print('False, index #'+str(i))

#%%
file_day_name = "_20220418_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220418/"
bad_column_values = ['xxxx']

bad_file_index_start = 26
bad_file_index_end = 42
for j in range(len(ports)):
    for i in range(bad_file_index_start, bad_file_index_end+1):
        file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
        file_df.to_csv(file_save_path + "mNode_Port" + str(ports[j])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with 4/18 missing/bad dates')    

#%%
file_day_name = "_20220513_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220513/"
bad_column_values = ['xxxx']

for j in range(len(ports)):
    file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
    file_df.to_csv(file_save_path + "mNode_Port" + str(ports[j])+ file_day_name +str(full_time_list[28])+ ".dat")

bad_file_index_start = 34
bad_file_index_end = 45
for j in range(len(ports)):
    for i in range(bad_file_index_start, bad_file_index_end+1):
        file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
        file_df.to_csv(file_save_path + "mNode_Port" + str(ports[j])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with 5/13 missing/bad dates') 


#%%
file_day_name = "_20220517_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220517/"
bad_column_values = ['xxxx']


bad_file_index_start = 47
bad_file_index_end = 71
for j in range(len(ports)):
    for i in range(bad_file_index_start, bad_file_index_end+1):
        file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
        file_df.to_csv(file_save_path + "mNode_Port" + str(ports[j])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with 5/17 missing/bad dates') 


#%%
file_day_name = "_20220518_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220518/"
bad_column_values = ['xxxx']


bad_file_index_start = 00
bad_file_index_end = 48
for j in range(len(ports)):
    for i in range(bad_file_index_start, bad_file_index_end+1):
        file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
        file_df.to_csv(file_save_path + "mNode_Port" + str(ports[j])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with 5/18 missing/bad dates')


#%%
file_day_name = "_20220519_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220519/"
bad_column_values = ['xxxx']


bad_file_index_start = 24
bad_file_index_end = 27
for j in range(len(ports)):
    for i in range(bad_file_index_start, bad_file_index_end+1):
        file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
        file_df.to_csv(file_save_path + "mNode_Port" + str(ports[j])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with 5/19 missing/bad dates')


#%%
file_day_name = "_20220604_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220604/"
bad_column_values = ['xxxx']


bad_file_index_start = 00
bad_file_index_end = 71
for j in range(len(ports)):
    for i in range(bad_file_index_start, bad_file_index_end+1):
        file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
        file_df.to_csv(file_save_path + "mNode_Port" + str(ports[j])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with 6/04 missing/bad dates')

#%%
file_day_name = "_20220605_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220605/"
bad_column_values = ['xxxx']


bad_file_index_start = 00
bad_file_index_end = 71
for j in range(len(ports)):
    for i in range(bad_file_index_start, bad_file_index_end+1):
        file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
        file_df.to_csv(file_save_path + "mNode_Port" + str(ports[j])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with 6/05 missing/bad dates')

#%%
file_day_name = "_20220606_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220606/"
bad_column_values = ['xxxx']


bad_file_index_start = 00
bad_file_index_end = 33
for j in range(len(ports)):
    for i in range(bad_file_index_start, bad_file_index_end+1):
        file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
        file_df.to_csv(file_save_path + "mNode_Port" + str(ports[j])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with 6/06 missing/bad dates')

#%%
file_day_name = "_20220415_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220415/"
bad_column_values = ['xxxx']


bad_file_index_start = 00
bad_file_index_end = 71

for i in range(bad_file_index_start, bad_file_index_end+1):
    file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
    file_df.to_csv(file_save_path + "mNode_Port" + str(ports[6])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with port 7 4/15 missing dates')

#%%
file_day_name = "_20220416_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220416/"
bad_column_values = ['xxxx']


bad_file_index_start = 00
bad_file_index_end = 71

for i in range(bad_file_index_start, bad_file_index_end+1):
    file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
    file_df.to_csv(file_save_path + "mNode_Port" + str(ports[6])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with port 7 4/16 missing dates')

#%%
file_day_name = "_20220417_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220417/"
bad_column_values = ['xxxx']


bad_file_index_start = 00
bad_file_index_end = 71

for i in range(bad_file_index_start, bad_file_index_end+1):
    file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
    file_df.to_csv(file_save_path + "mNode_Port" + str(ports[6])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with port 7 4/17 missing dates')

#%%
file_day_name = "_20220418_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220418/"
bad_column_values = ['xxxx']


bad_file_index_start = 43
bad_file_index_end = 49

for i in range(bad_file_index_start, bad_file_index_end+1):
    file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
    file_df.to_csv(file_save_path + "mNode_Port" + str(ports[6])+ file_day_name +str(full_time_list[i])+ ".dat")
print('done with port 7 4/18 missing dates')

#%%
file_day_name = "_20220519_"
file_save_path = r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode\mN220519/"
bad_column_values = ['xxxx']

file_df = pd.DataFrame({'missing or time-corrupted offset data':bad_column_values})
file_df.to_csv(file_save_path + "mNode_Port" + str(ports[6])+ file_day_name +str(full_time_list[23])+ ".dat")
print('done with port 7 5/19 missing dates')
        

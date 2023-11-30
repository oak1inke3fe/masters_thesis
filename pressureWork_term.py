# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:07:39 2023

@author: oak
"""

#%%
import numpy as np
import pandas as pd
# from pandas import rolling_median
import os
import matplotlib.pyplot as plt
import natsort
# import statistics
import time
import datetime
import math
# from scipy import interpolate
# import re
import scipy.signal as signal
# import pickle5 as pickle
# os.chdir(r'E:\mNode_test2folders\test')
print('done with imports')

#%%
### function start
#######################################################################################
# Function for interpolating the RMY sensor (freq = 32 Hz)
def interp_sonics123(df_sonics123):
    sonics123_xnew = np.arange(0, 32*60*20)   # this will be the number of points per file based
    df_align_interp= df_sonics123.reindex(sonics123_xnew).interpolate(limit_direction='both')
    return df_align_interp
#######################################################################################
### function end
# returns: df_align_interp
print('done with interp_sonics123 simple function')

### function start
#######################################################################################
# Function for interpolating the paros sensor (freq = 16 Hz)
def interp_paros(df_paros):
    paros_xnew = np.arange(0, 16*60*20)   # this will be the number of points per file based
    df_paros_interp = df_paros.reindex(paros_xnew).interpolate(limit_direction='both')
    return df_paros_interp
#######################################################################################
### function end
# returns: df_paros_interp
print('done with interp_paros function')

### function start
#######################################################################################
# Function for interpolating the sonics to the same frequency as the pressure heads (downsample to 16 Hz)
def interp_sonics2paros(df_despiked_sonics):
    sonic2paros_xnew = np.arange(0, 16*60*20)   # this will be the number of points per file based
    df_sonic2paros_interp= df_despiked_sonics.reindex(sonic2paros_xnew).interpolate(limit_direction='both')
    return df_sonic2paros_interp
#######################################################################################
### function end
# returns: df_sonic2paros_interp
print('done with interp_sonic2paros function')

#%%
# need to first interpolate to the paros frequency then

filepath_sonics = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level2_analysis\despiked_sonics"
w_prime_20minFile_port1 = pd.DataFrame()
w_prime_20minFile_port2 = pd.DataFrame()
w_prime_20minFile_port3 = pd.DataFrame()

for root, dirnames, filenames in os.walk(filepath_sonics): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        filename_only = filename[:-4]
        file = os.path.join(root, filename)
        if filename.startswith('mNode_Port1'):
            sampling_frequency_sonics = 32
            sampling_frequency_paros = 16
            s1_df = pd.read_csv(file, index_col=0, header=0)
            if len(s1_df)>= (0.9*(sampling_frequency_sonics*60*20)): #making sure there is at least 90% of a complete file before interpolating
                s1_df_interp = interp_sonics123(s1_df) # for upsampling paros to sonics
                s1_df_interp2paros = interp_sonics2paros(s1_df) # for downsampling sonics to paros
                w_prime_port1 = np.array(signal.detrend(s1_df_interp['Wr']))
                w_prime_port1_interpSonic2paros = np.array(signal.detrend(s1_df_interp2paros['Wr']))
            else:
                w_prime_port1 = np.full((sampling_frequency_sonics*60*20,1),np.nan)
                w_prime_port1_interpSonic2paros = np.full((sampling_frequency_paros*60*20,1),np.nan)
            # w_prime_20minFile_port1['Wp1_'+filename_only[14:-2]] = w_prime_port1_interpSonic2paros #for downsampling sonics to paros
            w_prime_20minFile_port1['Wp1_'+filename_only[14:-2]] = w_prime_port1 #for upsampling paros to sonics
            print(str(filename_only))
        if filename.startswith('mNode_Port2'):
            sampling_frequency_sonics = 32
            sampling_frequency_paros = 16
            s2_df = pd.read_csv(file, index_col=0, header=0)
            if len(s2_df)>= (0.9*(sampling_frequency_sonics*60*20)): #making sure there is at least 90% of a complete file before interpolating
                s2_df_interp = interp_sonics123(s2_df) # for upsampling paros to sonics
                s2_df_interp2paros = interp_sonics2paros(s2_df) # for downsampling sonics to paros
                w_prime_port2 = np.array(signal.detrend(s2_df_interp['Wr']))
                w_prime_port2_interpSonic2paros = np.array(signal.detrend(s2_df_interp2paros['Wr']))
            else:
                w_prime_port2 = np.full((sampling_frequency_sonics*60*20,1),np.nan)
                w_prime_port2_interpSonic2paros = np.full((sampling_frequency_paros*60*20,1),np.nan)
            # w_prime_20minFile_port2['Wp2_'+filename_only[14:-2]] = w_prime_port2_interpSonic2paros #for downsampling sonics to paros
            w_prime_20minFile_port2['Wp2_'+filename_only[14:-2]] = w_prime_port2 #for upsampling paros to sonics
            print(str(filename_only))
        if filename.startswith('mNode_Port3'):
            sampling_frequency_sonics = 32
            sampling_frequency_paros = 16
            s3_df = pd.read_csv(file, index_col=0, header=0)
            if len(s3_df)>= (0.9*(sampling_frequency_sonics*60*20)): #making sure there is at least 90% of a complete file before interpolating
                s3_df_interp = interp_sonics123(s3_df) # for upsampling paros to sonics
                s3_df_interp2paros = interp_sonics2paros(s3_df) # for downsampling sonics to paros
                w_prime_port3 = np.array(signal.detrend(s3_df_interp['Wr']))
                w_prime_port3_interpSonic2paros = np.array(signal.detrend(s3_df_interp2paros['Wr']))
            else:
                w_prime_port3 = np.full((sampling_frequency_sonics*60*20,1),np.nan)
                w_prime_port3_interpSonic2paros = np.full((sampling_frequency_paros*60*20,1),np.nan)
            # w_prime_20minFile_port3['Wp3_'+filename_only[14:-2]] = w_prime_port3_interpSonic2paros #for downsampling sonics to paros
            w_prime_20minFile_port3['Wp3_'+filename_only[14:-2]] = w_prime_port3 #for upsampling paros to sonics
            print(str(filename_only))
        else:
            continue
#%%
p1_prime_20minFile_port6 = pd.DataFrame()
p2_prime_20minFile_port6 = pd.DataFrame()
p3_prime_20minFile_port6 = pd.DataFrame()
filepath_paros = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
for root, dirnames, filenames in os.walk(filepath_paros): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]
        if filename.startswith('mNode_Port6'):
            sampling_frequency_paros = 16
            sampling_frequency_sonics = 32
            filename_only = filename[:-4]            
            s6_df = pd.read_csv(file,index_col=None, header = None) #read into a df
            s6_df.columns =['sensor','p'] #rename columns
            s6_df= s6_df[s6_df['sensor'] != 0] #get rid of any rows where the sensor is 0 because this is an error row
            s6_df_1 = s6_df[s6_df['sensor'] == 1] # make a df just for sensor 1
            s6_df_2 = s6_df[s6_df['sensor'] == 2] # make a df just for sensor 2
            s6_df_3 = s6_df[s6_df['sensor'] == 3] # make a df just for sensor 3
            if len(s6_df_1)>= (0.9*(sampling_frequency_paros*60*20)): #making sure there is at least 90% of a complete file before interpolating
                s6_df_1_interp = interp_paros(s6_df_1) 
                p1_prime = np.array(signal.detrend(s6_df_1_interp['p']))
                p1_extendedArr = np.repeat(p1_prime,2) #match length of sonics, for upsampling to sonics                
            else:
                p1_prime = np.full((sampling_frequency_paros*60*20,1),np.nan)
                p1_extendedArr = np.full((sampling_frequency_sonics*60*20,1),np.nan)
            # p1_prime_20minFile_port6['p1_prime_'+filename_only[14:-2]] = p1_prime #for downsampling to paros
            p1_prime_20minFile_port6['p1_prime_'+filename_only[14:-2]] = p1_extendedArr #for upsampling to sonics
            
            if len(s6_df_2)>= (0.9*(sampling_frequency_paros*60*20)): #making sure there is at least 90% of a complete file before interpolating
                s6_df_2_interp = interp_paros(s6_df_2) 
                p2_prime = np.array(signal.detrend(s6_df_2_interp['p']))
                p2_extendedArr = np.repeat(p2_prime,2) #match length of sonics, for upsampling to sonics                
            else:
                p2_prime = np.full((sampling_frequency_paros*60*20,1),np.nan)
                p2_extendedArr = np.full((sampling_frequency_sonics*60*20,1),np.nan)
            # p2_prime_20minFile_port6['p2_prime_'+filename_only[14:-2]] = p2_prime #for downsampling to paros
            p2_prime_20minFile_port6['p2_prime_'+filename_only[14:-2]] = p2_extendedArr #for upsampling to sonics
            
            if len(s6_df_3)>= (0.9*(sampling_frequency_paros*60*20)): #making sure there is at least 90% of a complete file before interpolating
                s6_df_3_interp = interp_paros(s6_df_3) 
                p3_prime = np.array(signal.detrend(s6_df_3_interp['p']))
                p3_extendedArr = np.repeat(p3_prime,2) #match length of sonics, for upsampling to sonics                
            else:
                p3_prime = np.full((sampling_frequency_paros*60*20,1),np.nan)
                p3_extendedArr = np.full((sampling_frequency_sonics*60*20,1),np.nan)
            # p3_prime_20minFile_port6['p3_prime_'+filename_only[14:-2]] = p3_prime #for downsampling to paros
            p3_prime_20minFile_port6['p3_prime_'+filename_only[14:-2]] = p3_extendedArr #for upsampling to sonics
            
        else:
            continue
        
#%%
#need to average between sonic levels
WpPp_1 = pd.DataFrame()
WpPp_2 = pd.DataFrame()
WpPp_3 = pd.DataFrame()

n = w_prime_20minFile_port1.shape[1]
for i in range(0,n):
    w1 = np.array(w_prime_20minFile_port1.iloc[:,i])
    w2 = np.array(w_prime_20minFile_port2.iloc[:,i])
    w3 = np.array(w_prime_20minFile_port3.iloc[:,i])
    p1 = np.array(p1_prime_20minFile_port6.iloc[:,i])
    p2 = np.array(p2_prime_20minFile_port6.iloc[:,i])
    p3 = np.array(p3_prime_20minFile_port6.iloc[:,i])
    colname = w_prime_20minFile_port1.columns[i]
    col_date = colname[2:]
    WpPp_1["WpPp_1_"+str(col_date)]= w1*p1
    WpPp_2["WpPp_2_"+str(col_date)]= w2*p2
    WpPp_3["WpPp_3_"+str(col_date)]= w3*p3
print('done')
#%%
WpPp_bar1 = np.array(WpPp_1.mean())
WpPp_bar2 = np.array(WpPp_2.mean())
WpPp_bar3 = np.array(WpPp_1.mean()) 
print('done')
#%%
PW_bar_term = pd.DataFrame()
PW_bar_term['WpPp_1'] = WpPp_bar1
PW_bar_term['WpPp_2'] = WpPp_bar2
PW_bar_term['WpPp_3'] = WpPp_bar3
print('done')
#%%
def despikeThis(input_df,n_std):
    n = input_df.shape[1]
    output_df = pd.DataFrame()
    for i in range(0,n):
        elements_input = input_df.iloc[:,i]
        elements = elements_input
        mean = np.mean(elements)
        sd = np.std(elements)
        extremes = np.abs(elements-mean)>(n_std*sd)
        elements[extremes]=np.NaN
        despiked = np.array(elements)
        colname = input_df.columns[i]
        output_df[str(colname)]=despiked

    return output_df
#%%
PW_despiked = despikeThis(PW_bar_term,5)
#%%
save_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
PW_despiked.to_csv(save_path+'PW_bar_terms_octoberOnly.csv')

print('done')
#%%
# adding in the d/dz
dz_LI = 1.8161 #fall SEPT 2022 deployment
dz_LII = 3.2131 #fall SEPT 2022 deployment
dUbar_dz_LI_arr = []
dUbar_dz_LII_arr = []

PWbar_LI = ((np.array(PW_despiked['WpPp_2'])-np.array(PW_despiked['WpPp_1']))/2)/dz_LI
PWbar_LII = ((np.array(PW_despiked['WpPp_3'])-np.array(PW_despiked['WpPp_2']))/2)/dz_LII
print('done with d/dz')

PW_divergence = pd.DataFrame()
PW_divergence['WpPp_bar_LI'] = PWbar_LI
PW_divergence['WpPp_bar_LII'] = PWbar_LII
        
#%%
save_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
PW_divergence.to_csv(save_path+'PW_bar_divergence_terms_octoberOnly.csv')
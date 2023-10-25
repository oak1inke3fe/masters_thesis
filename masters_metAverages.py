#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:17:58 2022

@author: OAKLIN KEEFE
"""

# This code is for getting all the variables we need for COARE that come from
# out met sensors: t_air, rh, P_air, sw_dn, lw_dn, rain rate

# we just need to average all the variables and put the averages combined into 
# another file to use for COARE

#%% IMPORTS
import os
import numpy as np
import pandas as pd
import natsort
# import pyrsktools
import matplotlib.pyplot as plt
print('done with imports')
#%%
filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-despike-interp/"
met_df_test = pd.read_csv(filepath + 'mNode_Port5_20220510_000000_1.csv')
print(met_df_test.columns)
#%%
t1_avg = []
t2_avg = []
rh1_avg = []
rh2_avg = []
p_air_avg = []
sw_dn_avg = [] #shortwave radiation (SW)
lw_dn_avg = [] #longwave radiation (IR)

for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-6]
        # path_save = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level2_analysis\port5/"
        if filename.startswith('mNode_Port5'):
        #     # Yday, Batt V, Tpan, Tair1, Tair2,  TIR, Pair, RH1, RH2, Solar, IR, IR ratio, Fix, GPS, Nsat
        #     # EX lines of data:
        #     # 106.4999,12.02,10.18,9.63,9.75,10.8,1053,75.83,75.53,323.1,-83.8,0.646,0,0,0
        #     # 106.4999,12.02,10.18,9.69,9.78,10.8,1053,75.83,75.26,323.1,-83.9,0.646,0,0,0
            # filename_only = filename[:-4]
            # path_save = r"E:\ASIT-research\BB-ASIT\Level1_align-despike-interp\port5/"
            met_df = pd.read_csv(file, index_col=None)
            # met_20MinMean = met_df.groupby(np.arange(len(met_df))//(1*60*20)).mean()
            t1 = met_df['T1'].mean()            
            t2 = met_df['T2'].mean()
            rh1 = met_df['RH1'].mean()
            rh2 = met_df['RH2'].mean()
            p_air = met_df['p_air'].mean()
            sw_dn = met_df['SW'].mean()     
            lw_dn= met_df['IRt'].mean()
            
            
            t1_avg.append(t1)
            t2_avg.append(t2)
            rh1_avg.append(rh1)
            rh2_avg.append(rh2)
            p_air_avg.append(p_air)
            sw_dn_avg.append(sw_dn)
            lw_dn_avg.append(lw_dn)
            print(filename_only)


#%%
    
t1_avg = np.array(t1_avg)
t2_avg = np.array(t2_avg)
rh1_avg = np.array(rh1_avg)
rh2_avg = np.array(rh2_avg)
p_air_avg = np.array(p_air_avg)
sw_dn_avg = np.array(sw_dn_avg)
lw_dn_avg = np.array(lw_dn_avg)
print('done with turning lists into arrays')
#%%
met_avg_df = pd.DataFrame()

met_avg_df['rh1']=rh1_avg
met_avg_df['rh2']=rh2_avg
met_avg_df['sw_dn']=sw_dn_avg
met_avg_df['lw_dn']=lw_dn_avg
met_avg_df['t1 [C]']=t1_avg
met_avg_df['t2 [C]']=t2_avg
met_avg_df['t1 [K]']=t1_avg+273.15
met_avg_df['t2 [K]']=t2_avg+273.15
met_avg_df['p_air [mb]']=p_air_avg
met_avg_df['p_air [Pa]']=p_air_avg/100


print('done with creating dataframe of all averaged met variables')
#%% save this file where the other files we are pulling from will come from
path_save = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
met_avg_df.to_csv(path_save+"metAvg_CombinedAnalysis.csv")
print('file saved')
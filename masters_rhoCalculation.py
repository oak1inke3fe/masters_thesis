#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:45:16 2023

@author: oak
"""


import numpy as np
import pandas as pd
import math
import os
import natsort
from scipy import interpolate
import matplotlib.pyplot as plt
print('done with imports')

#%%

# p from paros
# R is the ideal gas constant
R = 287.053 #[Pa K^-1 m^3 kg^-1]
# T from RHT sensors
# rho = p/(R*T)
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
Temp_df = pd.read_csv(file_path+"metAvg_CombinedAnalysis.csv")
T_1 = Temp_df['t1 [K]']
T_2 = Temp_df['t2 [K]']
p_df = pd.read_csv(file_path+"parosAvg_combinedAnalysis.csv")

rho_bar_1 = p_df['p1_avg [Pa]']/(R*T_1)
rho_bar_2 = p_df['p2_avg [Pa]']/(R*T_1)
rho_bar_3 = p_df['p3_avg [Pa]']/(R*T_2)


rho_df = pd.DataFrame()
rho_df['rho_bar_1'] = rho_bar_1
rho_df['rho_bar_2'] = rho_bar_2
rho_df['rho_bar_3'] = rho_bar_3

rho_df.to_csv(file_path + 'rhoAvg_CombinedAnalysis.csv')


#%%
# ### function start
# #######################################################################################
# # Function for interpolating the met sensor (freq = 1 Hz)
# def interp_met(df_met):    
#     met_xnew = np.arange(0, 1200)   # this will be the number of points per file based
#     s5_df_met_interp= df_met.reindex(met_xnew).interpolate(limit_direction='both')
#     return s5_df_met_interp
# #######################################################################################
# ### function end
# # returns: s5_df_met_interp
# print('done with interp_met function')

# #%%
# ### function start
# #######################################################################################
# # Function for interpolating the paros sensor (freq = 16 Hz)
# def interp_paros(df_paros):
#     paros_xnew = np.arange(0, 19200)   # this will be the number of points per file based
#     df_paros_interp = df_paros.reindex(paros_xnew).interpolate(limit_direction='both')
#     return df_paros_interp
# #######################################################################################
# ### function end
# # returns: df_paros_interp
# print('done with interp_paros function')

# #%% Test on one file
# Tv1_prime_20minFile_port5 = pd.DataFrame()
# Tv1_prime_20minFile_port5['Index'] = np.arange(0,38400)

# Tv2_prime_20minFile_port5 = pd.DataFrame()
# Tv2_prime_20minFile_port5['Index'] = np.arange(0,38400)

# test_file = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved\mNode_Port5_20221010_144000.txt"

# df_s5 = pd.read_csv(test_file)
# df_s5.columns =['full_date', 'YYYY', 'MM','DD','full_time','hh','mm','ss',
#                 'yearDay', 'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
#                 'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
#                 'fix', 'GPS', 'Nsat']

# df_s5_interp = interp_met(df_s5)
# Tv_1_C = df_s5_interp['T1']
# Tv_1 = df_s5_interp['T1']-273.15 #convert to K
# Tv_1_extendedArr = np.repeat(Tv_1,16) #match length of paros
# Tv_1_extendedArr = np.repeat(Tv_1,16) #match length of paros

# # Tv1_prime_20minFile_port5['Tv1_'+filename_only[14:-2]] = np.array(Tv1_extendedArr)
# #%%

# Tv1_20minFile_port5 = pd.DataFrame()
# Tv2_20minFile_port5 = pd.DataFrame()

# filepath_met = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
# for root, dirnames, filenames in os.walk(filepath_met): #this is for looping through files that are in a folder inside another folder
#     for filename in natsort.natsorted(filenames):
#         file = os.path.join(root, filename)
#         filename_only = filename[:-4]

#         if filename.startswith("mNode_Port5"):
#             df_s5 = pd.read_csv(file)
#             df_s5.columns =['full_date', 'YYYY', 'MM','DD','full_time','hh','mm','ss',
#                             'yearDay', 'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
#                             'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
#                             'fix', 'GPS', 'Nsat']
#             if len(df_s5)>= (0.9*(1*60*20)): #making sure there is at least 90% of a complete file before interpolating
#                 df_s5_interp = interp_met(df_s5)
#                 Tv_1 = df_s5_interp['T1']-273.15
#                 Tv_1_extendedArr = np.repeat(Tv_1,16) #converting to kelvin
#                 Tv_2 = df_s5_interp['T2']-273.15
#                 Tv_2_extendedArr = np.repeat(Tv_2,16) #converting to kelvin                              
#             else:
#                 Tv_1_extendedArr= np.full((19200,1),np.nan) #match length of paros
#                 Tv_2_extendedArr= np.full((19200,1),np.nan) #match length of paros

#             Tv1_20minFile_port5['Tv1_'+filename_only[14:-2]] = Tv_1_extendedArr
#             Tv2_20minFile_port5['Tv2_'+filename_only[14:-2]] = Tv_2_extendedArr
#             print(str(filename_only))
#         else:
#             continue
# print('done')
# #%%
# p1_20minFile_port6 = pd.DataFrame()
# p2_20minFile_port6 = pd.DataFrame()
# p3_20minFile_port6 = pd.DataFrame()


# filepath_paros = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
# for root, dirnames, filenames in os.walk(filepath_paros): #this is for looping through files that are in a folder inside another folder
#     for filename in natsort.natsorted(filenames):
#         file = os.path.join(root, filename)
#         filename_only = filename[:-4]
#         if filename.startswith('mNode_Port6'):
#             filename_only = filename[:-4]            
#             s6_df = pd.read_csv(file,index_col=None, header = None) #read into a df
#             s6_df.columns =['sensor','p'] #rename columns
#             s6_df= s6_df[s6_df['sensor'] != 0] #get rid of any rows where the sensor is 0 because this is an error row
#             s6_df_1 = s6_df[s6_df['sensor'] == 1] # make a df just for sensor 1
#             s6_df_2 = s6_df[s6_df['sensor'] == 2] # make a df just for sensor 2
#             s6_df_3 = s6_df[s6_df['sensor'] == 3] # make a df just for sensor 3
#             if len(s6_df_1)>= (0.9*(16*60*20)): #making sure there is at least 90% of a complete file before interpolating
#                 s6_df_1_interp = interp_paros(s6_df_1)
#                 p1 = s6_df_1_interp['p']*1000 #converting from given hPa to Pa
#             else:
#                 p1 = np.full((19200,1),np.nan)
#             p1_20minFile_port6['p1_'+filename_only[14:-2]] = p1
            
#             if len(s6_df_2)>= (0.9*(16*60*20)): #making sure there is at least 90% of a complete file before interpolating
#                 s6_df_2_interp = interp_paros(s6_df_2)
#                 p2 = s6_df_2_interp['p']*1000 #converting from given hPa to Pa
#             else:
#                 p2 = np.full((19200,1),np.nan)
#             p2_20minFile_port6['p2_'+filename_only[14:-2]] = p2
            
#             if len(s6_df_3)>= (0.9*(16*60*20)): #making sure there is at least 90% of a complete file before interpolating
#                 s6_df_3_interp = interp_paros(s6_df_3)
#                 p3 = s6_df_3_interp['p']*1000 #converting from given hPa to Pa
#             else:
#                 p3 = np.full((19200,1),np.nan)
#             p3_20minFile_port6['p3_'+filename_only[14:-2]] = p3
            
#         else:
#             continue
# #%%
# #Calculating rho
# # rho = p/(R*T)
# rho_1 = pd.DataFrame()
# rho_2 = pd.DataFrame()
# rho_3 = pd.DataFrame()

# n = p1_20minFile_port6.shape[1]
# for i in range(0,n):
#     T1 = np.array(Tv1_20minFile_port5.iloc[:,i])
#     T2 = np.array(Tv2_20minFile_port5.iloc[:,i])
#     p1 = np.array(p1_20minFile_port6.iloc[:,i])
#     p2 = np.array(p2_20minFile_port6.iloc[:,i])
#     p3 = np.array(p3_20minFile_port6.iloc[:,i])
#     colname1 = p1_20minFile_port6.columns[i]
#     col_date1 = colname1[2:]
#     rho_1['rho'+str(col_date1)] = p1/(R*T1)
#     rho_2['rho'+str(col_date1)] = p2/(R*T2)
#     rho_3['rho'+str(col_date1)] = p3/(R*T2)
# print('done')
# #%%

# rho_bar1 = np.array(rho_1.mean())
# rho_bar2 = np.array(rho_2.mean())
# rho_bar3 = np.array(rho_3.mean())


# #%%
# rho_bar_term = pd.DataFrame()
# rho_bar_term['rho_1'] = rho_bar1
# rho_bar_term['rho_2'] = rho_bar2
# rho_bar_term['rho_3'] = rho_bar3
# #%%
# def despikeThis(input_df,n_std):
#     n = input_df.shape[1]
#     output_df = pd.DataFrame()
#     for i in range(0,n):
#         elements_input = input_df.iloc[:,i]
#         elements = elements_input
#         mean = np.mean(elements)
#         sd = np.std(elements)
#         extremes = np.abs(elements-mean)>(n_std*sd)
#         elements[extremes]=np.NaN
#         despiked = np.array(elements)
#         colname = input_df.columns[i]
#         output_df[str(colname)]=despiked

#     return output_df
# #%%
# rho_despiked = despikeThis(rho_bar_term,5)
# #%%
# save_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
# rho_bar_term.to_csv(save_path+'rho_bar_terms_octoberOnly.csv')


# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:11:38 2023

@author: oak
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:54:35 2022

@author: oaklin keefe

This is Level 0 pipeline: taking quality controlled Level00 data, making sure there is enough
'good' data, aligning it to the wind (for applicable sensors) then despiking it (aka getting 
rid of the outliers), and finally interpolating it to the correct sensor sampling frequency. 
Edited files are saved to the Level 1 folder, in their respective "port" sub-folder as .csv 
files.                                                                                          

Input:
    .txt files per 20 min period per port from Level1_errorLinesRemoved and sub-port folder
Output:
    .csv files per 20 min period per port into LEVEL_1 folder
    wind has been aligned to mean wind direction
    files have been despiked and interpolated to all be the same size
    all units the same as the input raw units
    
    
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
from hampel import hampel
# from scipy import interpolate
# import re
# import scipy.signal as signal
# import pickle5 as pickle
# os.chdir(r'E:\mNode_test2folders\test')
print('done with imports')

#%%

### function start
#######################################################################################
# Function for aligning the U,V,W coordinaes to the mean wind direction
def alignwind(wind_df):
    # try: 
    wind_df = wind_df.replace('NAN', np.nan)
    wind_df['u'] = wind_df['u'].astype(float)
    wind_df['v'] = wind_df['v'].astype(float)
    wind_df['w'] = wind_df['w'].astype(float)
    Ub = wind_df['u'].mean()
    Vb = wind_df['v'].mean()
    Wb = wind_df['w'].mean()
    Sb = math.sqrt((Ub**2)+(Vb**2))
    beta = math.atan2(Wb,Sb)
    beta_arr = np.ones(len(wind_df))*(beta*180/math.pi)
    alpha = math.atan2(Vb,Ub)
    alpha_arr = np.ones(len(wind_df))*(alpha*180/math.pi)
    x1 = wind_df.index
    x = np.array(x1)
    Ur = wind_df['u']*math.cos(alpha)*math.cos(beta)+wind_df['v']*math.sin(alpha)*math.cos(beta)+wind_df['w']*math.sin(beta)
    Ur_arr = np.array(Ur)
    Vr = wind_df['u']*(-1)*math.sin(alpha)+wind_df['v']*math.cos(alpha)
    Vr_arr = np.array(Vr)
    Wr = wind_df['u']*(-1)*math.cos(alpha)*math.sin(beta)+wind_df['v']*(-1)*math.sin(alpha)*math.sin(beta)+wind_df['w']*math.cos(beta)     
    Wr_arr = np.array(Wr)
    T_arr = np.array(wind_df['T'])
    u_arr = np.array(wind_df['u'])
    v_arr = np.array(wind_df['v'])
    w_arr = np.array(wind_df['w'])

    df_aligned = pd.DataFrame({'base_index':x,'Ur':Ur_arr,'Vr':Vr_arr,'Wr':Wr_arr,'T':T_arr,'u':u_arr,'v':v_arr,'w':w_arr,'alpha':alpha_arr,'beta':beta_arr})

    return df_aligned
#######################################################################################
### function end
# returns: df_aligned (index, Ur, Vr, Wr, T, u, v, w, alpha, beta)
print('done with alignwind function')


#%%
### function start
#######################################################################################
# Function for interpolating the RMY sensor (freq = 32 Hz)
def interp_sonics123(df_sonics123):
    sonics123_xnew = np.arange(0, (32*60*20))   # this will be the number of points per file based
    df_align_interp= df_sonics123.reindex(sonics123_xnew).interpolate(limit_direction='both')
    return df_align_interp
#######################################################################################
### function end
# returns: df_align_interp
print('done with interp_sonics123 simple function')
#%%
### function start
#######################################################################################
# Function for interpolating the Gill sensor (freq = 20 Hz)
def interp_sonics4(df_sonics4):
    sonics4_xnew = np.arange(0, (20*60*20))   # this will be the number of points per file based
    df_align_interp_s4= df_sonics4.reindex(sonics4_xnew).interpolate(limit_direction='both')
    return df_align_interp_s4
#######################################################################################
### function end
# returns: df_align_interp_s4
print('done with interp_sonics4 function')
#%%
### function start
#######################################################################################
# Function for interpolating the paros sensor (freq = 16 Hz)
def interp_paros(df_paros):
    paros_xnew = np.arange(0, (16*60*20))   # this will be the number of points per file based
    df_paros_interp = df_paros.reindex(paros_xnew).interpolate(limit_direction='both')
    return df_paros_interp
#######################################################################################
### function end
# returns: df_paros_interp
print('done with interp_paros function')
#%%
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
### function start
#######################################################################################
# Function for interpolating the sonics to the same frequency as the pressure heads (downsample to 16 Hz)
def interp_paros2met(df_paros):
    paros2met_xnew = np.arange(0, 1*60*20)   # this will be the number of points per file based
    df_paros2met_interp= df_paros.reindex(paros2met_xnew).interpolate(limit_direction='both')
    return df_paros2met_interp
#######################################################################################
### function end
# returns: df_sonic2paros_interp
print('done with interp_paros2met function')
#%%
### function start
#######################################################################################
# Function for interpolating the met sensor (freq = 1 Hz)
def interp_met(df_met):    
    met_xnew = np.arange(0, (1*60*20))   # this will be the number of points per file based
    s5_df_met_interp= df_met.reindex(met_xnew).interpolate(limit_direction='both')
    return s5_df_met_interp
#######################################################################################
### function end
# returns: s5_df_met_interp
print('done with interp_met function')
#%%
### function start
#######################################################################################
# Function for interpolating the lidar sensor (freq = 20 Hz)
def interp_lidar(df_lidar):    
    lidar_xnew = np.arange(0, 20*60*20)   # this will be the number of points per file based
    s7_df_interp= df_lidar.reindex(lidar_xnew).interpolate(limit_direction='both')
    return s7_df_interp
#######################################################################################
### function end
# returns: s7_df_interp
print('done with interp_lidar function')

#%%
### function start
#######################################################################################
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
#######################################################################################
### function end
# returns: output_df
print('done with despike_this function')
#%%
# filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
# filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_errorLinesRemoved/"

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
print('port 1 length = '+ str(len(filename_port1)))
print('port 2 length = '+ str(len(filename_port2)))
print('port 3 length = '+ str(len(filename_port3)))
print('port 4 length = '+ str(len(filename_port4)))
print('port 5 length = '+ str(len(filename_port5)))
print('port 6 length = '+ str(len(filename_port6)))
print('port 7 length = '+ str(len(filename_port7)))
#%% THIS CODE ALIGNS, INTERPOLATES, THEN DESPIKES THE RAW DATA W/REMOVED ERR LINES
# filepath= r"E:\ASIT-research\BB-ASIT\test_Level1_errorLinesRemoved"
# filepath= r"E:\ASIT-research\BB-ASIT\Level1_errorLinesRemoved"

start=datetime.datetime.now()

Ubar_s1_arr = []
Tbar_s1_arr = []
UpWp_bar_s1_arr = []
VpWp_bar_s1_arr = []
WpTp_bar_s1_arr = []
WpEp_bar_s1_arr = []
Umedian_s1_arr = []
Tmedian_s1_arr = []
U_horiz_bar_s1_arr = []
U_streamwise_bar_s1_arr = []
TKE_bar_s1_arr = []

# Ubar_s2_arr = []
# Tbar_s2_arr = []
# UpWp_bar_s2_arr = []
# VpWp_bar_s2_arr = []
# WpTp_bar_s2_arr = []
# WpEp_bar_s2_arr = []
# Umedian_s2_arr = []
# Tmedian_s2_arr = []
# U_horiz_s2_arr = []
# U_streamwise_s2_arr = []

# Ubar_s3_arr = []
# Tbar_s3_arr = []
# UpWp_bar_s3_arr = []
# VpWp_bar_s3_arr = []
# WpTp_bar_s3_arr = []
# WpEp_bar_s3_arr = []
# Umedian_s3_arr = []
# Tmedian_s3_arr = []
# U_horiz_s3_arr = []
# U_streamwise_s3_arr = []

# Ubar_s4_arr = []
# Tbar_s4_arr = []
# UpWp_bar_s4_arr = []
# VpWp_bar_s4_arr = []
# WpTp_bar_s4_arr = []
# WpEp_bar_s4_arr = []
# Umedian_s4_arr = []
# Tmedian_s4_arr = []
# U_horiz_s4_arr = []
# U_streamwise_s4_arr = []

# len_dfOutput = []
# port7_singleFile_nans_sum = []

# filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
# filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved"
filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_errorLinesRemoved/"

# path_save = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp/"
# path_save = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp/"
path_save = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-despike-interp/"

for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]
        if filename.startswith("mNode_Port1"):
            s1_df = pd.read_csv(file)
            if len(s1_df)>= (0.75*(32*60*20)): #making sure there is at least 75% of a complete file before interpolating
                s1_df.columns =['u', 'v', 'w', 'T', 'err_code','chk_sum'] #set column names to the variable
                s1_df = s1_df[['u', 'v', 'w', 'T',]]            
                s1_df['u']=s1_df['u'].astype(float) 
                s1_df['v']=s1_df['v'].astype(float)            
                s1_df['u']=-1*s1_df['u']
                s1_df['w']=-1*s1_df['w']
                df_s1_aligned = alignwind(s1_df)
                df_s1_aligned['Ur'][np.abs(df_s1_aligned['Ur']) >=55 ] = np.nan
                df_s1_aligned['Vr'][np.abs(df_s1_aligned['Vr']) >=20 ] = np.nan
                df_s1_aligned['Wr'][np.abs(df_s1_aligned['Wr']) >=20 ] = np.nan                            
                df_s1_interp = interp_sonics123(df_s1_aligned)
                # Ur_s1 = df_s1_interp['Ur']
                # Ur_s1_outlier_in_Ts = hampel(Ur_s1, window_size=10, n=3, imputation=True) # Outlier Imputation with rolling median
                # df_s1_interp['Ur'] = Ur_s1_outlier_in_Ts
                # Vr_s1 = df_s1_interp['Vr']
                # Vr_s1_outlier_in_Ts = hampel(Vr_s1, window_size=10, n=3, imputation=True) # Outlier Imputation with rolling median
                # df_s1_interp['Vr'] = Vr_s1_outlier_in_Ts
                # Wr_s1 = df_s1_interp['Wr']
                # Wr_s1_outlier_in_Ts = hampel(Wr_s1, window_size=10, n=3, imputation=True) # Outlier Imputation with rolling median
                # df_s1_interp['Wr'] = Wr_s1_outlier_in_Ts
                U_horiz_s1 = np.sqrt((np.array(df_s1_interp['Ur'])**2)+(np.array(df_s1_interp['Vr'])**2))
                U_streamwise_s1 = np.sqrt((np.array(df_s1_interp['Ur'])**2)+(np.array(df_s1_interp['Vr'])**2)+(np.array(df_s1_interp['Wr'])**2))
                Up_s1 = df_s1_interp['Ur']-df_s1_interp['Ur'].mean()
                Vp_s1 = df_s1_interp['Vr']-df_s1_interp['Vr'].mean()
                Wp_s1 = df_s1_interp['Wr']-df_s1_interp['Wr'].mean()
                Tp_s1 = (df_s1_interp['T']+273.15)-(df_s1_interp['T']+273.15).mean()
                TKEp_s1 = 0.5*((Up_s1**2)+(Vp_s1**2)+(Wp_s1**2))                
                Tbar_s1 = (df_s1_interp['T']+273.15).mean()
                UpWp_bar_s1 = np.nanmean(Up_s1*Wp_s1)
                VpWp_bar_s1 = np.nanmean(Vp_s1*Wp_s1)
                WpTp_bar_s1 = np.nanmean(Tp_s1*Wp_s1)
                WpEp_bar_s1 = np.nanmean(TKEp_s1*Wp_s1)
                Ubar_s1 = df_s1_interp['Ur'].mean()
                Umedian_s1 = df_s1_interp['Ur'].median()
                Tmedian_s1 = (df_s1_interp['T']+273.15).median()
                TKE_bar_s1 = np.nanmean(TKEp_s1)
                U_horiz_bar_s1 = np.nanmean(U_horiz_s1)
                U_streamwise_bar_s1 = np.nanmean(U_streamwise_s1)
            else:
                df_s1_interp = pd.DataFrame(np.nan, index=[0,1], columns=['base_index','Ur','Vr','Wr','T','u','v','w','alpha','beta'])
                Tbar_s1 = np.nan
                UpWp_bar_s1 = np.nan
                VpWp_bar_s1 = np.nan
                WpTp_bar_s1 = np.nan
                WpEp_bar_s1 = np.nan
                Ubar_s1 = np.nan
                Umedian_s1 = np.nan
                Tmedian_s1 = np.nan
                TKE_bar_s1 = np.nan
                U_horiz_bar_s1 = np.nan
                U_streamwise_bar_s1 = np.nan
            Tbar_s1_arr.append(Tbar_s1)
            UpWp_bar_s1_arr.append(UpWp_bar_s1)
            VpWp_bar_s1_arr.append(VpWp_bar_s1)
            WpTp_bar_s1_arr.append(WpTp_bar_s1)
            WpEp_bar_s1_arr.append(WpEp_bar_s1)
            Ubar_s1_arr.append(Ubar_s1)
            Umedian_s1_arr.append(Umedian_s1)
            Tmedian_s1_arr.append(Tmedian_s1)
            TKE_bar_s1_arr.append(TKE_bar_s1)
            U_horiz_bar_s1_arr.append(U_horiz_bar_s1)
            U_streamwise_bar_s1_arr.append(U_streamwise_bar_s1)
            df_s1_interp.to_csv(path_save+filename_only+"_1.csv")
            print('Port 1 ran: '+filename)
            ## BELOW ARE PLOTTING LINES IF YOU WANT TO CHECK OUTPUTS AS THE CODE RUNS
            ## CAN ADD BELOW OTHER PORTS TO CHECK THEIR OUTPUTS TOO
        #     plt.plot(df_aligned['Ur'], label="Ur")
        #     plt.plot(df_aligned['Vr'], label="Vr")
        #     plt.plot(df_aligned['Wr'], label="Wr")                         
        #     # plt.ylim(-4,10)
        #     plt.legend(loc='upper left',prop={'size': 6})
        #     plt.title(str(filename))
        #     plt.draw()
        #     plt.pause(0.0001)
        #     plt.clf()
            
        # if filename.startswith("mNode_Port2"):
        # # if filename.startswith("mNode_Port2"):
        #     s2_df = pd.read_csv(file)
        #     if len(s2_df)>= (0.75*(32*60*20)):
        #         s2_df.columns =['u', 'v', 'w', 'T', 'err_code','chk_sum'] #set column names to the variable
        #         s2_df = s2_df[['u', 'v', 'w', 'T',]]            
        #         s2_df['u']=s2_df['u'].astype(float) 
        #         s2_df['v']=s2_df['v'].astype(float)            
        #         s2_df['u']=-1*s2_df['u']
        #         s2_df['w']=-1*s2_df['w']
        #         df_s2_aligned = alignwind(s2_df)
        #         df_s2_aligned['Ur'][np.abs(df_s2_aligned['Ur']) >=55 ] = np.nan
        #         df_s2_aligned['Vr'][np.abs(df_s2_aligned['Vr']) >=20 ] = np.nan
        #         df_s2_aligned['Wr'][np.abs(df_s2_aligned['Wr']) >=20 ] = np.nan
        #         df_s2_interp = interp_sonics123(df_s2_aligned)
        #         U_horiz = np.sqrt((np.array(df_s2_interp['Ur'])**2)+(np.array(df_s2_interp['Vr'])**2))
        #         U_streamwise = np.sqrt((np.array(df_s2_interp['Ur'])**2)+(np.array(df_s2_interp['Vr'])**2)+(np.array(df_s2_interp['Wr'])**2))
        #         Up_s2 = df_s2_interp['Ur']-df_s2_interp['Ur'].mean()
        #         Vp_s2 = df_s2_interp['Vr']-df_s2_interp['Vr'].mean()
        #         Wp_s2 = df_s2_interp['Wr']-df_s2_interp['Wr'].mean()
        #         Tp_s2 = (df_s2_interp['T']+273.15)-(df_s2_interp['T']+273.15).mean()
        #         TKEp_s2 = 0.5*((Up_s2**2)+(Vp_s2**2)+(Wp_s2**2))
        #         Tbar_s2 = (df_s2_interp['T']+273.15).mean()
        #         UpWp_bar_s2 = np.mean(Up_s2*Wp_s2)
        #         VpWp_bar_s2 = np.mean(Vp_s2*Wp_s2)
        #         WpTp_bar_s2 = np.mean(Tp_s2*Wp_s2)
        #         WpEp_bar_s2 = np.mean(TKEp_s2*Wp_s2)
        #         Ubar_s2 = df_s2_interp['Ur'].mean()
        #         Umedian_s2 = df_s2_interp['Ur'].median()
        #         Tmedian_s2 = (df_s2_interp['T']+273.15).median()
        #     else:
        #         df_s2_interp = pd.DataFrame(np.nan, index=[0,1], columns=['base_index','Ur','Vr','Wr','T','u','v','w','alpha','beta'])
        #         Tbar_s2 = np.nan
        #         UpWp_bar_s2 = np.nan
        #         VpWp_bar_s2 = np.nan
        #         WpTp_bar_s2 = np.nan
        #         WpEp_bar_s2 = np.nan
        #         Ubar_s2 = np.nan
        #         Umedian_s2 = np.nan
        #         Tmedian_s2 = np.nan 
        #         U_horiz = np.nan
        #         U_streamwise = np.nan
        #     Tbar_s2_arr.append(Tbar_s2)
        #     UpWp_bar_s2_arr.append(UpWp_bar_s2)
        #     VpWp_bar_s2_arr.append(VpWp_bar_s2)
        #     WpTp_bar_s2_arr.append(WpTp_bar_s2)
        #     WpEp_bar_s2_arr.append(WpEp_bar_s2)
        #     Ubar_s2_arr.append(Ubar_s2)
        #     Umedian_s2_arr.append(Umedian_s2)
        #     Tmedian_s2_arr.append(Tmedian_s2)
        #     U_horiz_s2_arr.append(U_horiz)
        #     U_streamwise_s2_arr.append(U_streamwise)
        #     df_s2_interp.to_csv(path_save+filename_only+"_1.csv")
        #     print('Port 2 ran: '+filename)
     
        # if filename.startswith("mNode_Port3"):
        #     s3_df = pd.read_csv(file)
        #     if len(s3_df)>= (0.75*(32*60*20)):
        #         s3_df.columns =['u', 'v', 'w', 'T', 'err_code','chk_sum'] #set column names to the variable
        #         s3_df = s3_df[['u', 'v', 'w', 'T',]]            
        #         s3_df['u']=s3_df['u'].astype(float) 
        #         s3_df['v']=s3_df['v'].astype(float) 
        #         df_s3_aligned = alignwind(s3_df)
        #         df_s3_aligned['Ur'][np.abs(df_s3_aligned['Ur']) >=55 ] = np.nan
        #         df_s3_aligned['Vr'][np.abs(df_s3_aligned['Vr']) >=20 ] = np.nan
        #         df_s3_aligned['Wr'][np.abs(df_s3_aligned['Wr']) >=20 ] = np.nan
        #         df_s3_interp = interp_sonics123(df_s3_aligned)
        #         U_horiz = np.sqrt((np.array(df_s3_interp['Ur'])**2)+(np.array(df_s3_interp['Vr'])**2))
        #         U_streamwise = np.sqrt((np.array(df_s3_interp['Ur'])**2)+(np.array(df_s3_interp['Vr'])**2)+(np.array(df_s3_interp['Wr'])**2))
        #         Up_s3 = df_s3_interp['Ur']-df_s3_interp['Ur'].mean()
        #         Vp_s3 = df_s3_interp['Vr']-df_s3_interp['Vr'].mean()
        #         Wp_s3 = df_s3_interp['Wr']-df_s3_interp['Wr'].mean()
        #         Tp_s3 = (df_s3_interp['T']+273.15)-(df_s3_interp['T']+273.15).mean()
        #         TKEp_s3 = 0.5*((Up_s3**2)+(Vp_s3**2)+(Wp_s3**2))
        #         Tbar_s3 = (df_s3_interp['T']+273.15).mean()
        #         UpWp_bar_s3 = np.mean(Up_s3*Wp_s3)
        #         VpWp_bar_s3 = np.mean(Vp_s3*Wp_s3)
        #         WpTp_bar_s3 = np.mean(Tp_s3*Wp_s3)
        #         WpEp_bar_s3 = np.mean(TKEp_s3*Wp_s3)
        #         Ubar_s3 = df_s3_interp['Ur'].mean()
        #         Umedian_s3 = df_s3_interp['Ur'].median()
        #         Tmedian_s3 = (df_s3_interp['T']+273.15).median()
        #     else:
        #         df_s3_interp = pd.DataFrame(np.nan, index=[0,1], columns=['base_index','Ur','Vr','Wr','T','u','v','w','alpha','beta'])
        #         Tbar_s3 = np.nan
        #         UpWp_bar_s3 = np.nan
        #         VpWp_bar_s3 = np.nan
        #         WpTp_bar_s3 = np.nan
        #         WpEp_bar_s3 = np.nan
        #         Ubar_s3 = np.nan
        #         Umedian_s3 = np.nan
        #         Tmedian_s3 = np.nan
        #         U_horiz = np.nan
        #         U_streamwise = np.nan
        #     Tbar_s3_arr.append(Tbar_s3)
        #     UpWp_bar_s3_arr.append(UpWp_bar_s3)
        #     VpWp_bar_s3_arr.append(VpWp_bar_s3)
        #     WpTp_bar_s3_arr.append(WpTp_bar_s3)
        #     WpEp_bar_s3_arr.append(WpEp_bar_s3)
        #     Ubar_s3_arr.append(Ubar_s3)
        #     Umedian_s3_arr.append(Umedian_s3)
        #     Tmedian_s3_arr.append(Tmedian_s3)
        #     U_horiz_s3_arr.append(U_horiz)
        #     U_streamwise_s3_arr.append(U_streamwise)
        #     df_s3_interp.to_csv(path_save+filename_only+"_1.csv")
        #     print('Port 3 ran: '+filename)

            
        # if filename.startswith("mNode_Port4"):
        #     s4_df = pd.read_csv(file)
        #     if len(s4_df)>= (0.75*(20*60*20)):
        #         s4_df =pd.read_csv(file, index_col=None, header = None)
        #         s4_df.columns =['chk_1','chk_2','u', 'v', 'w', 'T', 'err_code']
        #         s4_df = s4_df[['u', 'v', 'w', 'T',]]                         
        #         s4_df['u']=s4_df['u'].astype(float) 
        #         s4_df['v']=s4_df['v'].astype(float) 
        #         df_s4_aligned = alignwind(s4_df)
        #         df_s4_aligned['Ur'][np.abs(df_s4_aligned['Ur']) >=55 ] = np.nan
        #         df_s4_aligned['Vr'][np.abs(df_s4_aligned['Vr']) >=20 ] = np.nan
        #         df_s4_aligned['Wr'][np.abs(df_s4_aligned['Wr']) >=20 ] = np.nan
        #         df_s4_interp = interp_sonics4(df_s4_aligned)
        #         U_horiz = np.sqrt((np.array(df_s4_interp['Ur'])**2)+(np.array(df_s4_interp['Vr'])**2))
        #         U_streamwise = np.sqrt((np.array(df_s4_interp['Ur'])**2)+(np.array(df_s4_interp['Vr'])**2)+(np.array(df_s4_interp['Wr'])**2))
        #         Up_s4 = df_s4_interp['Ur']-df_s4_interp['Ur'].mean()
        #         Vp_s4 = df_s4_interp['Vr']-df_s4_interp['Vr'].mean()
        #         Wp_s4 = df_s4_interp['Wr']-df_s4_interp['Wr'].mean()
        #         Tp_s4 = (df_s4_interp['T']+273.15)-(df_s4_interp['T']+273.15).mean()
        #         TKEp_s4 = 0.5*((Up_s4**2)+(Vp_s4**2)+(Wp_s4**2))
        #         Tbar_s4 = (df_s4_interp['T']+273.15).mean()
        #         UpWp_bar_s4 = np.mean(Up_s4*Wp_s4)
        #         VpWp_bar_s4 = np.mean(Vp_s4*Wp_s4)
        #         WpTp_bar_s4 = np.mean(Tp_s4*Wp_s4)
        #         WpEp_bar_s4 = np.mean(TKEp_s4*Wp_s4)
        #         Ubar_s4 = df_s4_interp['Ur'].mean()
        #         Umedian_s4 = df_s4_interp['Ur'].median()
        #         Tmedian_s4 = (df_s4_interp['T']+273.15).median()
        #     else:
        #         df_s4_interp = pd.DataFrame(np.nan, index=[0,1], columns=['base_index','Ur','Vr','Wr','T','u','v','w','alpha','beta'])
        #         Tbar_s4 = np.nan
        #         UpWp_bar_s4 = np.nan
        #         VpWp_bar_s4 = np.nan
        #         WpTp_bar_s4 = np.nan
        #         WpEp_bar_s4 = np.nan
        #         Ubar_s4 = np.nan
        #         Umedian_s4 = np.nan
        #         Tmedian_s4 = np.nan
        #         U_horiz = np.nan
        #         U_streamwise = np.nan
        #     Tbar_s4_arr.append(Tbar_s4)
        #     UpWp_bar_s4_arr.append(UpWp_bar_s4)
        #     VpWp_bar_s4_arr.append(VpWp_bar_s4)
        #     WpTp_bar_s4_arr.append(WpTp_bar_s4)
        #     WpEp_bar_s4_arr.append(WpEp_bar_s4)
        #     Ubar_s4_arr.append(Ubar_s4)
        #     Umedian_s4_arr.append(Umedian_s4)
        #     Tmedian_s4_arr.append(Tmedian_s4)
        #     U_horiz_s4_arr.append(U_horiz)
        #     U_streamwise_s4_arr.append(U_streamwise)
        #     df_s4_interp.to_csv(path_save+filename_only+"_1.csv")
        #     print('Port 4 ran: '+filename)
        
        # if filename.startswith('mNode_Port5_202204'):
        #     # Yday, Batt V, Tpan, Tair1, Tair2,  TIR, Pair, RH1, RH2, Solar, IR, IR ratio, Fix, GPS, Nsat
        #     # EX lines of data:
        #     # 106.4999,12.02,10.18,9.63,9.75,10.8,1053,75.83,75.53,323.1,-83.8,0.646,0,0,0
        #     # 106.4999,12.02,10.18,9.69,9.78,10.8,1053,75.83,75.26,323.1,-83.9,0.646,0,0,0
            
        #     # path_save = r"E:\ASIT-research\BB-ASIT\Level1_align-despike-interp\port5/"
        #     s5_df = pd.read_csv(file, index_col=None, header = None)
        #     s5_df.columns =['yearDay', 
        #                     'bat_volt','pannel_T', 'T1', 'T2','TIR',
        #                     'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                     'fix', 'GPS', 'Nsat'] 
        #     if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating
        #         # df_despiked = despikeThis(s5_df,5) #despike doesn't work unless it's a columns of all number
        #         s5_df_met_interp = interp_met(s5_df)
        #     else:
        #         s5_df = pd.DataFrame(np.nan, index=[0,1], columns=['yearDay', 
        #                                                             'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
        #                                                             'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                                                             'fix', 'GPS', 'Nsat'])
        #     s5_df.to_csv(path_save+str(filename_only)+'_1.csv')
        #     print('Port 5 ran: '+filename)
        # if filename.startswith('mNode_Port5_202205'):
        #     # Yday, Batt V, Tpan, Tair1, Tair2,  TIR, Pair, RH1, RH2, Solar, IR, IR ratio, Fix, GPS, Nsat
        #     # EX lines of data:
        #     # 106.4999,12.02,10.18,9.63,9.75,10.8,1053,75.83,75.53,323.1,-83.8,0.646,0,0,0
        #     # 106.4999,12.02,10.18,9.69,9.78,10.8,1053,75.83,75.26,323.1,-83.9,0.646,0,0,0
            
        #     # path_save = r"E:\ASIT-research\BB-ASIT\Level1_align-despike-interp\port5/"
        #     s5_df = pd.read_csv(file, index_col=None, header = None)
        #     s5_df.columns =['yearDay', 
        #                     'bat_volt','pannel_T', 'T1', 'T2','TIR',
        #                     'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                     'fix', 'GPS', 'Nsat'] 
        #     if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating
        #         # df_despiked = despikeThis(s5_df,5) #despike doesn't work unless it's a columns of all number
        #         s5_df_met_interp = interp_met(s5_df)
        #     else:
        #         s5_df = pd.DataFrame(np.nan, index=[0,1], columns=['yearDay', 
        #                                                             'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
        #                                                             'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                                                             'fix', 'GPS', 'Nsat'])
        #     s5_df.to_csv(path_save+str(filename_only)+'_1.csv')
        #     print('Port 5 ran: '+filename)
        # if filename.startswith('mNode_Port5_202206'):
        #     # Yday, Batt V, Tpan, Tair1, Tair2,  TIR, Pair, RH1, RH2, Solar, IR, IR ratio, Fix, GPS, Nsat
        #     # EX lines of data:
        #     # 106.4999,12.02,10.18,9.63,9.75,10.8,1053,75.83,75.53,323.1,-83.8,0.646,0,0,0
        #     # 106.4999,12.02,10.18,9.69,9.78,10.8,1053,75.83,75.26,323.1,-83.9,0.646,0,0,0
            
        #     # path_save = r"E:\ASIT-research\BB-ASIT\Level1_align-despike-interp\port5/"
        #     s5_df = pd.read_csv(file, index_col=None, header = None)
        #     s5_df.columns =['yearDay', 
        #                     'bat_volt','pannel_T', 'T1', 'T2','TIR',
        #                     'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                     'fix', 'GPS', 'Nsat'] 
        #     if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating
        #         # df_despiked = despikeThis(s5_df,5) #despike doesn't work unless it's a columns of all number
        #         s5_df_met_interp = interp_met(s5_df)
        #     else:
        #         s5_df = pd.DataFrame(np.nan, index=[0,1], columns=['yearDay', 
        #                                                             'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
        #                                                             'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                                                             'fix', 'GPS', 'Nsat'])
        #     s5_df.to_csv(path_save+str(filename_only)+'_1.csv')
        #     print('Port 5 ran: '+filename)
            
        # if filename.startswith('mNode_Port5_202209'):
        #     # Yday, Batt V, Tpan, Tair1, Tair2,  TIR, Pair, RH1, RH2, Solar, IR, IR ratio, Fix, GPS, Nsat
        #     # EX lines of data:
        #     # 106.4999,12.02,10.18,9.63,9.75,10.8,1053,75.83,75.53,323.1,-83.8,0.646,0,0,0
        #     # 106.4999,12.02,10.18,9.69,9.78,10.8,1053,75.83,75.26,323.1,-83.9,0.646,0,0,0
            
        #     # path_save = r"E:\ASIT-research\BB-ASIT\Level1_align-despike-interp\port5/"
        #     s5_df = pd.read_csv(file, index_col=None, header = None)
        #     s5_df.columns =['date','YYYY','MM','DD','time',
        #                     'hh','mm','ss','yearDay', 
        #                     'bat_volt','pannel_T', 'T1', 'T2','TIR',
        #                     'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                     'fix', 'GPS', 'Nsat'] 
        #     if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating
        #         # df_despiked = despikeThis(s5_df,5) #despike doesn't work unless it's a columns of all number
        #         s5_df_met_interp = interp_met(s5_df)
        #     else:
        #         s5_df = pd.DataFrame(np.nan, index=[0,1], columns=['date','YYYY','MM','DD','time',
        #                                                             'hh','mm','ss','yearDay', 
        #                                                             'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
        #                                                             'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                                                             'fix', 'GPS', 'Nsat'])
        #     s5_df.to_csv(path_save+str(filename_only)+'_1.csv')
        #     print('Port 5 ran: '+filename)
        # if filename.startswith('mNode_Port5_202210'):
        #     # Yday, Batt V, Tpan, Tair1, Tair2,  TIR, Pair, RH1, RH2, Solar, IR, IR ratio, Fix, GPS, Nsat
        #     # EX lines of data:
        #     # 106.4999,12.02,10.18,9.63,9.75,10.8,1053,75.83,75.53,323.1,-83.8,0.646,0,0,0
        #     # 106.4999,12.02,10.18,9.69,9.78,10.8,1053,75.83,75.26,323.1,-83.9,0.646,0,0,0
            
        #     # path_save = r"E:\ASIT-research\BB-ASIT\Level1_align-despike-interp\port5/"
        #     s5_df = pd.read_csv(file, index_col=None, header = None)
        #     s5_df.columns =['date','YYYY','MM','DD','time',
        #                     'hh','mm','ss','yearDay', 
        #                     'bat_volt','pannel_T', 'T1', 'T2','TIR',
        #                     'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                     'fix', 'GPS', 'Nsat'] 
        #     if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating
        #         # df_despiked = despikeThis(s5_df,5) #despike doesn't work unless it's a columns of all number
        #         s5_df_met_interp = interp_met(s5_df)
        #     else:
        #         s5_df = pd.DataFrame(np.nan, index=[0,1], columns=['date','YYYY','MM','DD','time',
        #                                                             'hh','mm','ss','yearDay', 
        #                                                             'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
        #                                                             'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                                                             'fix', 'GPS', 'Nsat'])
        #     s5_df.to_csv(path_save+str(filename_only)+'_1.csv')
        #     print('Port 5 ran: '+filename)
        # if filename.startswith('mNode_Port5_202211'):
        #     # Yday, Batt V, Tpan, Tair1, Tair2,  TIR, Pair, RH1, RH2, Solar, IR, IR ratio, Fix, GPS, Nsat
        #     # EX lines of data:
        #     # 106.4999,12.02,10.18,9.63,9.75,10.8,1053,75.83,75.53,323.1,-83.8,0.646,0,0,0
        #     # 106.4999,12.02,10.18,9.69,9.78,10.8,1053,75.83,75.26,323.1,-83.9,0.646,0,0,0
            
        #     # path_save = r"E:\ASIT-research\BB-ASIT\Level1_align-despike-interp\port5/"
        #     s5_df = pd.read_csv(file, index_col=None, header = None)
        #     s5_df.columns =['date','YYYY','MM','DD','time',
        #                     'hh','mm','ss','yearDay', 
        #                     'bat_volt','pannel_T', 'T1', 'T2','TIR',
        #                     'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                     'fix', 'GPS', 'Nsat'] 
        #     if len(s5_df)>= (0.75*(1*60*20)): #making sure there is at least 75% of a complete file before interpolating
        #         # df_despiked = despikeThis(s5_df,5) #despike doesn't work unless it's a columns of all number
        #         s5_df_met_interp = interp_met(s5_df)
        #     else:
        #         s5_df = pd.DataFrame(np.nan, index=[0,1], columns=['date','YYYY','MM','DD','time',
        #                                                             'hh','mm','ss','yearDay', 
        #                                                             'bat_volt', 'pannel_T', 'T1', 'T2','TIR',
        #                                                             'p_air', 'RH1', 'RH2', 'SW', 'IR', 'IR_ratio', 
        #                                                             'fix', 'GPS', 'Nsat'])
        #     s5_df.to_csv(path_save+str(filename_only)+'_1.csv')
        #     print('Port 5 ran: '+filename)
                
        # if filename.startswith('mNode_Port6'):

        #     # path_save = r"E:\ASIT-research\BB-ASIT\Level1_align-despike-interp\port6/"
        #     s6_df = pd.read_csv(file,index_col=None, header = None) #read into a df
        #     s6_df.columns =['sensor','p'] #rename columns            
        #     s6_df= s6_df[s6_df['sensor'] != 0] #get rid of any rows where the sensor is 0 because this is an error row
        #     s6_df_1 = s6_df[s6_df['sensor'] == 1] # make a df just for sensor 1
        #     s6_df_2 = s6_df[s6_df['sensor'] == 2] # make a df just for sensor 2
        #     s6_df_3 = s6_df[s6_df['sensor'] == 3] # make a df just for sensor 3
        #     df_despiked_1 = pd.DataFrame()
        #     df_despiked_2 = pd.DataFrame()
        #     df_despiked_3 = pd.DataFrame()
        #     if len(s6_df_1)>= (0.75*(16*60*20)): #making sure there is at least 75% of a complete file before interpolating
        #         s6_df_1.loc[((s6_df_1['p']>=2000)|(s6_df_1['p']<=100)) , 'p'] = np.nan #PUT RESTRICTION ON REASONABLE PRESSURE OBSERVATIONS (hPa)
        #         # don't worry about the "warning" it gives when doing this ^
        #         df_despiked_1 = despikeThis(s6_df_1,5)
        #         df_paros_interp = interp_paros(df_despiked_1) #interpolate to proper frequency
        #         s6_df_1_interp = df_paros_interp #rename                
        #     else: #if not enough points, make a df of NaNs that is the size of a properly interpolated df
        #         s6_df_1_interp = pd.DataFrame(np.nan, index=[0,1], columns=['sensor','p']) 
        #     s6_df_1_interp.to_csv(path_save+"L1_"+str(filename_only)+'_1.csv') #save as csv
        #     print('done with paros 1 '+filename)
        #     if len(s6_df_2)>= (0.75*(16*60*20)): #making sure there is at least 75% of a complete file before interpolating
        #         s6_df_2.loc[((s6_df_2['p']>=2000)|(s6_df_2['p']<=100)) , 'p'] = np.nan    #PUT RESTRICTION ON REASONABLE PRESSURE OBSERVATIONS (hPa) 
        #         # don't worry about the "warning" it gives when doing this ^
        #         df_despiked_2 = despikeThis(s6_df_2,5)
        #         df_paros_interp = interp_paros(df_despiked_2) #interpolate to proper frequency
        #         s6_df_2_interp = df_paros_interp #rename                
        #     else: #if not enough points, make a df of NaNs that is the size of a properly interpolated df
        #         s6_df_2_interp = pd.DataFrame(np.nan, index=[0,1], columns=['sensor','p'])
        #     s6_df_2_interp.to_csv(path_save+"L2_"+str(filename_only)+'_1.csv') #save as csv
        #     print('done with paros 2 '+filename)
        #     if len(s6_df_3)>= (0.75*(16*60*20)): #making sure there is at least 75% of a complete file before interpolating
        #         s6_df_3.loc[((s6_df_3['p']>=2000)|(s6_df_3['p']<=100)) , 'p'] = np.nan  #PUT RESTRICTION ON REASONABLE PRESSURE OBSERVATIONS (hPa)
        #         # don't worry about the "warning" it gives when doing this ^
        #         df_despiked_3 = despikeThis(s6_df_3,5)
        #         df_paros_interp = interp_paros(df_despiked_3) #interpolate to proper frequency
        #         s6_df_3_interp = df_paros_interp #rename                
        #     else: #if not enough points, make a df of NaNs that is the size of a properly interpolated df
        #         s6_df_3_interp = pd.DataFrame(np.nan, index=[0,1], columns=['sensor','p'])
        #     s6_df_3_interp.to_csv(path_save+"L3_"+str(filename_only)+'_1.csv') #save as csv
        #     print('done with paros '+filename)

        # port7_singleFile_nans = []
        # if filename.startswith('mNode_Port7'):
        #     filename_only = filename[:-4]
        #     # path_save = r"E:\ASIT-research\BB-ASIT\test_Level1_align-despike-interp/"
        #     s7_df = pd.read_csv(file,index_col=None, header = None)
        #     s7_df.columns =['range','amplitude','quality']
        #     # df_despiked = pd.DataFrame()
        #     if s7_df['range'].isna().sum()<(0.50*(1*60*20)): #make sure at least 50% of 20 minutes (at 1Hz frequency because of wave dropout) is recorded
        #         # s7_df = s7_df['all'].str.split(';', expand=True) #now separate into different columns
        #         # s7_df.columns =['range','amplitude','quality'] # name the columns
        #         # s7_df['range'] = s7_df['range'].str.lstrip('r') #get rid of leading 'r' in range column
        #         # s7_df['amplitude'] = s7_df['amplitude'].str.lstrip('a') #get rid of leading 'a' in amplitude column
        #         # s7_df['quality'] = s7_df['quality'].str.lstrip('q') #get rid of leading 'q' in quality column
        #         # df_despiked = despikeThis(s7_df,5)
        #         s7_df_interp = interp_lidar(s7_df) #interpolate to Lidar's sampling frequency
        #         port7_singleFile_nans_i=0
        #         port7_singleFile_nans.append(port7_singleFile_nans_i)
                
        #     else:
        #         s7_df_interp = pd.DataFrame(np.nan, index=[0,1], columns=['range','amplitude','quality'])
        #         port7_singleFile_nans_i = 1
        #         port7_singleFile_nans.append(port7_singleFile_nans_i)
        #     port7_singleFile_nans_sum_i = sum(port7_singleFile_nans)
        #     port7_singleFile_nans_sum.append(port7_singleFile_nans_i)
        #     # s7_df_interp.to_csv(path_save+str(filename_only)+'_1.csv') #save as csv
        #     print('Port 7 ran for file: '+ filename)
        #     len_file = len(s7_df_interp)
        #     len_dfOutput.append(len_file)
        
        else:
            # print("file doesn't start with mNode_Port 1-7")
            continue

end = datetime.datetime.now()
print('done')
print(start)
print(end)

# import winsound
# duration = 3000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration)




#%%
# path_save_L4 = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level4/"
path_save_L4 = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/"

Ubar_s1_arr = np.array(Ubar_s1_arr)
U_horiz_s1_arr = np.array(U_horiz_bar_s1_arr)
U_streamwise_s1_arr = np.array(U_streamwise_bar_s1_arr)
Umedian_s1_arr = np.array(Umedian_s1_arr)
Tbar_s1_arr = np.array(Tbar_s1_arr)
Tmedian_s1_arr = np.array(Tmedian_s1_arr)
UpWp_bar_s1_arr = np.array(UpWp_bar_s1_arr)
VpWp_bar_s1_arr = np.array(VpWp_bar_s1_arr)
WpTp_bar_s1_arr = np.array(WpTp_bar_s1_arr)
WpEp_bar_s1_arr = np.array(WpEp_bar_s1_arr)
TKE_bar_s1_arr = np.array(TKE_bar_s1_arr)


combined_s1_df = pd.DataFrame()
combined_s1_df['Ubar_s1'] = Ubar_s1_arr
combined_s1_df['U_horiz_s1'] = U_horiz_s1_arr
combined_s1_df['U_streamwise_s1'] = U_streamwise_s1_arr
combined_s1_df['Umedian_s1'] = Umedian_s1_arr
combined_s1_df['Tbar_s1'] = Tbar_s1_arr
combined_s1_df['Tmedian_s1'] = Tmedian_s1_arr
combined_s1_df['UpWp_bar_s1'] = UpWp_bar_s1_arr
combined_s1_df['VpWp_bar_s1'] = VpWp_bar_s1_arr
combined_s1_df['WpTp_bar_s1'] = WpTp_bar_s1_arr
combined_s1_df['WpEp_bar_s1'] = WpEp_bar_s1_arr
combined_s1_df['TKE_bar_s2'] = TKE_bar_s1_arr

combined_s1_df.to_csv(path_save_L4 + "s1_turbulenceTerms_andMore_combined.csv")


print('done')

# # path_save_L4 = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
# path_save_L4 = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level4/"


# VpWp_bar_df = pd.DataFrame()
# VpWp_bar_df["date"] = np.array(filename_generalDate)
# VpWp_bar_df["VpWp_bar_s1"] = np.array(VpWp_bar_s1_arr)
# VpWp_bar_df["VpWp_bar_s2"] = np.array(VpWp_bar_s2_arr)
# VpWp_bar_df["VpWp_bar_s3"] = np.array(VpWp_bar_s3_arr)
# VpWp_bar_df["VpWp_bar_s4"] = np.array(VpWp_bar_s4_arr)
# # VpWp_bar_df.to_csv(path_save_L4+"VpWp_bar_allSonics_allFall.csv")
# VpWp_bar_df.to_csv(path_save_L4+"VpWp_bar_allSonics_CombinedAnalysis.csv")


# Tbar_df = pd.DataFrame()
# Tbar_df["date"] = np.array(filename_generalDate)
# Tbar_df["Tbar_s1"] = np.array(Tbar_s1_arr)
# Tbar_df["Tbar_s2"] = np.array(Tbar_s2_arr)
# Tbar_df["Tbar_s3"] = np.array(Tbar_s3_arr)
# Tbar_df["Tbar_s4"] = np.array(Tbar_s4_arr)
# # Tbar_df.to_csv(path_save_L4+"Tbar_allSonics_allFall.csv")
# Tbar_df.to_csv(path_save_L4+"Tbar_allSonics_CombinedAnalysis.csv")

# Ubar_df = pd.DataFrame()
# Ubar_df["date"] = np.array(filename_generalDate)
# Ubar_df["Ubar_s1"] = np.array(Ubar_s1_arr)
# Ubar_df["Ubar_s2"] = np.array(Ubar_s2_arr)
# Ubar_df["Ubar_s3"] = np.array(Ubar_s3_arr)
# Ubar_df["Ubar_s4"] = np.array(Ubar_s4_arr)
# # Ubar_df.to_csv(path_save_L4+"Ubar_allSonics_allFall.csv")
# Ubar_df.to_csv(path_save_L4+"Ubar_allSonics_CombinedAnalysis.csv")

# UpWp_bar_df = pd.DataFrame()
# UpWp_bar_df["date"] = np.array(filename_generalDate)
# UpWp_bar_df["UpWp_bar_s1"] = np.array(UpWp_bar_s1_arr)
# UpWp_bar_df["UpWp_bar_s2"] = np.array(UpWp_bar_s2_arr)
# UpWp_bar_df["UpWp_bar_s3"] = np.array(UpWp_bar_s3_arr)
# UpWp_bar_df["UpWp_bar_s4"] = np.array(UpWp_bar_s4_arr)
# # UpWp_bar_df.to_csv(path_save_L4+"UpWp_bar_allSonics_allFall_hampel.csv")
# UpWp_bar_df.to_csv(path_save_L4+"UpWp_bar_allSonics_CombinedAnalysis_hampel.csv")

# WpTp_bar_df = pd.DataFrame()
# WpTp_bar_df["date"] = np.array(filename_generalDate)
# WpTp_bar_df["WpTp_bar_s1"] = np.array(WpTp_bar_s1_arr)
# WpTp_bar_df["WpTp_bar_s2"] = np.array(WpTp_bar_s2_arr)
# WpTp_bar_df["WpTp_bar_s3"] = np.array(WpTp_bar_s3_arr)
# WpTp_bar_df["WpTp_bar_s4"] = np.array(WpTp_bar_s4_arr)
# # WpTp_bar_df.to_csv(path_save_L4+"WpTp_bar_allSonics_allFall.csv")
# WpTp_bar_df.to_csv(path_save_L4+"WpTp_bar_allSonics_CombinedAnalysis.csv")

# WpEp_bar_df = pd.DataFrame()
# WpEp_bar_df["date"] = np.array(filename_generalDate)
# WpEp_bar_df["WpEp_bar_s1"] = np.array(WpEp_bar_s1_arr)
# WpEp_bar_df["WpEp_bar_s2"] = np.array(WpEp_bar_s2_arr)
# WpEp_bar_df["WpEp_bar_s3"] = np.array(WpEp_bar_s3_arr)
# WpEp_bar_df["WpEp_bar_s4"] = np.array(WpEp_bar_s4_arr)
# # WpEp_bar_df.to_csv(path_save_L4+"WpEp_bar_allSonics_allFall.csv")
# WpEp_bar_df.to_csv(path_save_L4+"WpEp_bar_allSonics_CombinedAnalysis.csv")

# Tmedian_df = pd.DataFrame()
# Tmedian_df["date"] = np.array(filename_generalDate)
# Tmedian_df["Tmedian_s1"] = np.array(Tmedian_s1_arr)
# Tmedian_df["Tmedian_s2"] = np.array(Tmedian_s2_arr)
# Tmedian_df["Tmedian_s3"] = np.array(Tmedian_s3_arr)
# Tmedian_df["Tmedian_s4"] = np.array(Tmedian_s4_arr)
# # Tmedian_df.to_csv(path_save_L4+"Tmedian_allSonics_allFall.csv")
# Tmedian_df.to_csv(path_save_L4+"Tmedian_allSonics_CombinedAnalysis.csv")

# Umedian_df = pd.DataFrame()
# Umedian_df["date"] = np.array(filename_generalDate)
# Umedian_df["Umedian_s1"] = np.array(Umedian_s1_arr)
# Umedian_df["Umedian_s2"] = np.array(Umedian_s2_arr)
# Umedian_df["Umedian_s3"] = np.array(Umedian_s3_arr)
# Umedian_df["Umedian_s4"] = np.array(Umedian_s4_arr)
# # Umedian_df.to_csv(path_save_L4+"Umedian_allSonics_allFall.csv")
# Umedian_df.to_csv(path_save_L4+"Umedian_allSonics_CombinedAnalysis.csv")

# U_horiz_df = pd.DataFrame()
# U_horiz_df["date"] = np.array(filename_generalDate)
# U_horiz_df["U_horiz_s1"] = np.array(U_horiz_s1_arr)
# U_horiz_df["U_horiz_s2"] = np.array(U_horiz_s2_arr)
# U_horiz_df["U_horiz_s3"] = np.array(U_horiz_s3_arr)
# U_horiz_df["U_horiz_s4"] = np.array(U_horiz_s4_arr)
# # U_horiz_df.to_csv(path_save_L4+"U_horiz_allSonics_allFall.csv")
# U_horiz_df.to_csv(path_save_L4+"U_horiz_allSonics_CombinedAnalysis.csv")

# U_streamwise_df = pd.DataFrame()
# U_streamwise_df["date"] = np.array(filename_generalDate)
# U_streamwise_df["U_streamwise_s1"] = np.array(U_streamwise_s1_arr)
# U_streamwise_df["U_streamwise_s2"] = np.array(U_streamwise_s2_arr)
# U_streamwise_df["U_streamwise_s3"] = np.array(U_streamwise_s3_arr)
# U_streamwise_df["U_streamwise_s4"] = np.array(U_streamwise_s4_arr)
# # U_streamwise_df.to_csv(path_save_L4+"U_streamwise_allSonics_allFall.csv")
# U_streamwise_df.to_csv(path_save_L4+"U_streamwise_allSonics_CombinedAnalysis.csv")

# print('done')

# import winsound
# duration = 3000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration)

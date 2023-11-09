#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:47:07 2023

@author: oaklinkeefe
"""

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import binsreg
import seaborn as sns
print('done with imports')

g = -9.81
kappa = 0.40 # von Karman constant used by Edson (1998) Similarity Theory paper
break_index = 3959 #index is 3959, full length is 3960
print('done with setting gravity (g = -9.81) and von-karman (kappa = 4), break index between spring and fall deployment = 3959')

#%%
# file_path = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/"
file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
prod_df = pd.read_csv(file_path+"prodTerm_combinedAnalysis.csv")
prod_df = prod_df.drop(['Unnamed: 0'], axis=1)

date_df = pd.read_csv(file_path + "date_combinedAnalysis.csv")
print(date_df.columns)
print(date_df['datetime'][10])

windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_wCameraFlags_combinedAnalysis.csv")
# windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_120to196.csv")
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)

sonic1_df = pd.read_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_combined.csv')
sonic2_df = pd.read_csv(file_path + 'despiked_s2_turbulenceTerms_andMore_combined.csv')
sonic3_df = pd.read_csv(file_path + 'despiked_s3_turbulenceTerms_andMore_combined.csv')
sonic4_df = pd.read_csv(file_path + 'despiked_s4_turbulenceTerms_andMore_combined.csv')


plt.figure()
plt.plot(np.array(sonic3_df['Ubar'][:break_index+1]), label = 's3')
plt.plot(np.array(sonic4_df['Ubar'][:break_index+1]), label = 's4')
plt.legend()
plt.title('spring timeseries Ubar')
#%%
plt.figure()
plt.plot(np.array(sonic3_df['Ubar'])[0:500], label = 's3')
plt.plot(np.array(sonic4_df['Ubar'])[0:500], label = 's4')
plt.legend()
#wind dir around 235 (sw flow) at index 100
start = 0
stop = 500
fig, ax1 = plt.subplots(figsize=(8, 2))
ax2 = ax1.twinx()
ax1.plot(np.arange(len(sonic4_df[start:stop])), sonic4_df['Ubar'][start:stop], color = 'red', label = 'S')
ax2.plot(np.arange(len(sonic4_df[start:stop])), windDir_df['alpha_s4'][start:stop],color = 'gray', label = 'D')


plt.figure()
plt.plot(np.array(sonic3_df['Ubar'])[1500:2000], label = 's3')
plt.plot(np.array(sonic4_df['Ubar'])[1500:2000], label = 's4')
plt.legend()
#wind dir around 45 (NE-ENE flow) at index 1750
start = 1500
stop = 2000
fig, ax1 = plt.subplots(figsize=(8, 2))
ax2 = ax1.twinx()
ax1.plot(np.arange(len(sonic4_df[start:stop])), sonic4_df['Ubar'][start:stop], color = 'red', label = 'S')
ax2.plot(np.arange(len(sonic4_df[start:stop])), windDir_df['alpha_s4'][start:stop],color = 'gray', label = 'D')


plt.figure()
plt.plot(np.array(sonic3_df['Ubar'])[3000:3500], label = 's3')
plt.plot(np.array(sonic4_df['Ubar'])[3000:3500], label = 's4')
plt.legend()
#wind dir 200 (SSW flow) at index 3300
start = 3000
stop = 3500
fig, ax1 = plt.subplots(figsize=(8, 2))
ax2 = ax1.twinx()
ax1.plot(np.arange(len(sonic4_df[start:stop])), sonic4_df['Ubar'][start:stop], color = 'red', label = 'S')
ax2.plot(np.arange(len(sonic4_df[start:stop])), windDir_df['alpha_s4'][start:stop],color = 'gray', label = 'D')


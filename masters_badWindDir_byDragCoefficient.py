#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:06:13 2023

@author: oaklinkeefe
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import binsreg
import seaborn as sns
print('done with imports')
#%%
# Cd = (u_star**2)/(ubar**2)  equation for drag coeeficient fron Stull (1998) pg. 262

file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
break_index = 3959

date_df = pd.read_csv(file_path + "date_combinedAnalysis.csv")
print(date_df.columns)
print(date_df['datetime'][10])

windDir_df = pd.read_csv(file_path + "windDir_IncludingBad_combinedAnalysis.csv")
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)

sonic1_df = pd.read_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_combined.csv')
sonic2_df = pd.read_csv(file_path + 'despiked_s2_turbulenceTerms_andMore_combined.csv')
sonic3_df = pd.read_csv(file_path + 'despiked_s3_turbulenceTerms_andMore_combined.csv')
sonic4_df = pd.read_csv(file_path + 'despiked_s4_turbulenceTerms_andMore_combined.csv')
#%%
Ubar_s1 =[]
Ubar_s2 =[]
Ubar_s3 =[]
Ubar_s4 =[]

for i in range(len(sonic1_df)):
    if sonic1_df['Ubar'][i] < 3:
        Ubar_s1_i = np.nan
    else:
        Ubar_s1_i = sonic1_df['Ubar'][i]
    Ubar_s1.append(Ubar_s1_i)

for i in range(len(sonic2_df)):
    if sonic2_df['Ubar'][i] < 3:
        Ubar_s2_i = np.nan
    else:
        Ubar_s2_i = sonic2_df['Ubar'][i]
    Ubar_s2.append(Ubar_s2_i)

for i in range(len(sonic3_df)):
    if sonic3_df['Ubar'][i] < 3:
        Ubar_s3_i = np.nan
    else:
        Ubar_s3_i = sonic3_df['Ubar'][i]
    Ubar_s3.append(Ubar_s3_i)

for i in range(len(sonic4_df)):
    if sonic4_df['Ubar'][i] < 3:
        Ubar_s4_i = np.nan
    else:
        Ubar_s4_i = sonic4_df['Ubar'][i]
    Ubar_s4.append(Ubar_s4_i)
#%%
usr_df = pd.read_csv(file_path + "usr_combinedAnalysis.csv")
usr_s1 = usr_df['usr_s1']
usr_s2 = usr_df['usr_s2']
usr_s3 = usr_df['usr_s3']
usr_s4 = usr_df['usr_s4']

#%%
cd_s1 = (usr_s1**2)/(np.array(Ubar_s1)**2)
cd_s2 = (usr_s2**2)/(np.array(Ubar_s2)**2)
cd_s3 = (usr_s3**2)/(np.array(Ubar_s3)**2)
cd_s4 = (usr_s4**2)/(np.array(Ubar_s4)**2)

#%%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
fig.suptitle('Determining bad wind directions from drag coefficients \n SPRING ONLY')
ax1.scatter(windDir_df['alpha_s1'][:break_index+1],cd_s1[:break_index+1]*1000, s = 1, color = 'darkgreen')
ax1.set_ylim(0,5)
ax1.set_ylabel('$C_{d1}x10^3$')
ax2.scatter(windDir_df['alpha_s2'][:break_index+1],cd_s2[:break_index+1]*1000, s = 1, color = 'darkgreen')
ax2.set_ylim(0,5)
ax2.set_ylabel('$C_{d2}x10^3$')
ax3.scatter(windDir_df['alpha_s3'][:break_index+1],cd_s3[:break_index+1]*1000, s = 1, color = 'darkgreen')
ax3.set_ylim(0,5)
ax3.set_ylabel('$C_{d3}x10^3$')
ax4.scatter(windDir_df['alpha_s4'][:break_index+1],cd_s4[:break_index+1]*1000, s = 1, color = 'darkgreen')
ax4.set_ylim(0,5)
ax4.set_ylabel('$C_{d4}x10^3$')
ax4.set_xlabel('Wind Direction')

#%%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
fig.suptitle('Determining bad wind directions from drag coefficients \n FALL ONLY')
ax1.scatter(windDir_df['alpha_s1'][break_index+1:],cd_s1[break_index+1:]*1000, s = 1, color = 'darkorange')
ax1.set_ylim(0,5)
ax1.set_ylabel('$C_{d1}x10^3$')
ax2.scatter(windDir_df['alpha_s2'][break_index+1:],cd_s2[break_index+1:]*1000, s = 1, color = 'darkorange')
ax2.set_ylim(0,5)
ax2.set_ylabel('$C_{d2}x10^3$')
ax3.scatter(windDir_df['alpha_s3'][break_index+1:],cd_s3[break_index+1:]*1000, s = 1, color = 'darkorange')
ax3.set_ylim(0,5)
ax3.set_ylabel('$C_{d3}x10^3$')
ax4.scatter(windDir_df['alpha_s4'][break_index+1:],cd_s4[break_index+1:]*1000, s = 1, color = 'darkorange')
ax4.set_ylim(0,5)
ax4.set_ylabel('$C_{d4}x100$')
ax4.set_xlabel('Wind Direction')

#%%


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:04:45 2023

@author: oaklinkeefe



This file is used to determine "bad wind directions" by examining times when the vertical profile deviates noticeably from the log-profile
We restrict the inputs to neutral stability and wind directions unobstructed by the tower

INPUT files:
    sonics files from Level1_align-despike-interp files (for first time running)
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    date_combinedAnalysis.csv
    windDir_IncludingBad_wS4rotation_combinedAnalysis.csv
    
    
We also set:
    base_index= 3959 as the last point in the spring deployment to separate spring and fall datasets so the hampel filter is not 
    corrupted by data that is not in consecutive time sequence.

    
OUTPUT files:

    
    
"""
#%%

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from windrose import WindroseAxes

print('done with imports')

#%%

file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
plot_save_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'

sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)

sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)

sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)

sonic_file4 = "despiked_s4_turbulenceTerms_andMore_combined.csv"
sonic4_df = pd.read_csv(file_path+sonic_file4)

windDir_includingBadDirs_file = "windDir_IncludingBad_wS4rotation_combinedAnalysis.csv"
windDir_includingBad_df = pd.read_csv(file_path + windDir_includingBadDirs_file)

windDir_file = "windDir_withBadFlags_110to155_within15degRequirement_combinedAnalysis.csv"
windDir_df = pd.read_csv(file_path + windDir_file)

date_file = "date_combinedAnalysis.csv"
date_df = pd.read_csv(file_path + date_file)
print(date_df.columns)

zL_file = "ZoverL_combinedAnalysis.csv"
zL_df = pd.read_csv(file_path + zL_file)
zL_df.head(15)

z_file = "z_air_side_combinedAnalysis.csv"
z_df = pd.read_csv(file_path + z_file)
z_df.columns

plt.figure()
plt.plot(z_df['z_sonic4'])
plt.title('Z sonic 4; combined analysis')

usr_file = "usr_combinedAnalysis.csv"
usr_df = pd.read_csv(file_path + usr_file)


break_index = 3959

#%% Mask the DFs to only keep the good wind directions
windDir_index_array = np.arange(len(windDir_df))
windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, windDir_index_array)
mask_goodWindDir = np.isin(windDir_df['new_index_arr'],windDir_index_array)

windDir_df[mask_goodWindDir] = np.nan

sonic1_df[mask_goodWindDir] = np.nan
sonic2_df[mask_goodWindDir] = np.nan
sonic3_df[mask_goodWindDir] = np.nan
sonic4_df[mask_goodWindDir] = np.nan

zL_df[mask_goodWindDir] = np.nan

z_df[mask_goodWindDir] = np.nan

usr_df[mask_goodWindDir] = np.nan

print('done with setting up good wind direction only dataframes')

#%% Mask the DFs to only keep the near neutral stabilities

zL_index_array = np.arange(len(zL_df))
zL_df['new_index_arr'] = np.where((np.abs(zL_df['zL_I_dc'])<=0.5)&(np.abs(zL_df['zL_II_dc'])<=0.5), np.nan, zL_index_array)
mask_neutral_zL = np.isin(zL_df['new_index_arr'],zL_index_array)

zL_df[mask_neutral_zL] = np.nan

windDir_df[mask_neutral_zL] = np.nan

sonic1_df[mask_neutral_zL] = np.nan
sonic2_df[mask_neutral_zL] = np.nan
sonic3_df[mask_neutral_zL] = np.nan
sonic4_df[mask_neutral_zL] = np.nan

z_df[mask_neutral_zL] = np.nan

usr_df[mask_neutral_zL] = np.nan

print('done with setting up near-neautral stability with good wind directions dataframes')

#%% Copy the original (neutral and good wind direction applied) dataframes so we can do separate masks per deployment
zL_df_spring_copy1 = zL_df.copy()
zL_df_spring_copy2 = zL_df.copy()
zL_df_fall_copy1 = zL_df.copy()
zL_df_fall_copy2 = zL_df.copy()

z_df_spring_copy1 = z_df.copy()
z_df_spring_copy2 = z_df.copy()
z_df_fall_copy1 = z_df.copy()
z_df_fall_copy2 = z_df.copy()

windDir_df_spring_copy1 = windDir_df.copy()
windDir_df_spring_copy2 = windDir_df.copy()
windDir_df_fall_copy1 = windDir_df.copy()
windDir_df_fall_copy2 = windDir_df.copy()

sonic1_df_spring_copy1 = sonic1_df.copy()
sonic1_df_spring_copy2 = sonic1_df.copy()
sonic1_df_fall_copy1 = sonic1_df.copy()
sonic1_df_fall_copy2 = sonic1_df.copy()

sonic2_df_spring_copy1 = sonic2_df.copy()
sonic2_df_spring_copy2 = sonic2_df.copy()
sonic2_df_fall_copy1 = sonic2_df.copy()
sonic2_df_fall_copy2 = sonic2_df.copy()

sonic3_df_spring_copy1 = sonic3_df.copy()
sonic3_df_spring_copy2 = sonic3_df.copy()
sonic3_df_fall_copy1 = sonic3_df.copy()
sonic3_df_fall_copy2 = sonic3_df.copy()

sonic4_df_spring_copy1 = sonic4_df.copy()
sonic4_df_spring_copy2 = sonic4_df.copy()
sonic4_df_fall_copy1 = sonic4_df.copy()
sonic4_df_fall_copy2 = sonic4_df.copy()

usr_df_spring_copy1 = usr_df.copy()
usr_df_spring_copy2 = usr_df.copy()
usr_df_fall_copy1 = usr_df.copy()
usr_df_fall_copy2 = usr_df.copy()



#%% Spring Copy 1

sonic_df_arr = [sonic1_df_spring_copy1[:break_index+1], 
                sonic2_df_spring_copy1[:break_index+1], 
                sonic3_df_spring_copy1[:break_index+1], 
                sonic4_df_spring_copy1[:break_index+1],]
z_df_arr = [z_df_spring_copy1[:break_index+1],]
usr_df_arr = [usr_df_spring_copy1[:break_index+1]]
windDir_df_arr = [windDir_df_spring_copy1[:break_index+1]]
lower = 200
upper = 250
print(lower, upper)

restricted_arr = np.where((lower <= windDir_df_spring_copy1['alpha_s4'][:break_index+1]) & (windDir_df_spring_copy1['alpha_s4'][:break_index+1] < upper), windDir_df_spring_copy1['alpha_s4'][:break_index+1],np.nan)
print('restricted_arr made')

mask_restricted_arr = np.isin(restricted_arr, np.array(windDir_df_spring_copy1['alpha_s4'][:break_index+1]))
print('mask_restricted_arr made')

restricted_wd_df_sonic1 = sonic_df_arr[0][mask_restricted_arr]
# restricted_wd_df_sonic1['new_index'] = np.arange(len(restricted_wd_df_sonic1))
restricted_wd_df_sonic1 = restricted_wd_df_sonic1.reset_index(drop=True)
restricted_wd_df_sonic2 = sonic_df_arr[1][mask_restricted_arr]
restricted_wd_df_sonic2 = restricted_wd_df_sonic2.reset_index(drop=True)
restricted_wd_df_sonic3 = sonic_df_arr[2][mask_restricted_arr]
restricted_wd_df_sonic3 = restricted_wd_df_sonic3.reset_index(drop=True)
restricted_wd_df_sonic4 = sonic_df_arr[3][mask_restricted_arr]
restricted_wd_df_sonic4 = restricted_wd_df_sonic4.reset_index(drop=True)
restricted_wd_df_z = z_df_arr[0][mask_restricted_arr]
restricted_wd_df_z = restricted_wd_df_z.reset_index(drop=True)
restricted_wd_df_usr = usr_df_arr[0][mask_restricted_arr]
restricted_wd_df_usr = restricted_wd_df_usr.reset_index(drop=True)
restricted_wd_df_windDir = windDir_df_arr[0][mask_restricted_arr]
restricted_wd_df_windDir = restricted_wd_df_windDir.reset_index(drop=True)
print('restricted dfs made')


#%%
for i in range(0,1):
    # ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic1['Ubar'][i]/usr_df_fall_copy1['usr_s1'][i]), restricted_wd_df_z['z_sonic1'][i], color = 'red', label = 's1')
    # ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic2['Ubar'][i]/usr_df_fall_copy1['usr_s2'][i]), restricted_wd_df_z['z_sonic2'][i], color = 'darkorange', label = 's2')
    # ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic3['Ubar'][i]/usr_df_fall_copy1['usr_s3'][i]), restricted_wd_df_z['z_sonic3'][i], color = 'green', label = 's3')
    # ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic4['Ubar'][i]/usr_df_fall_copy1['usr_s4'][i]), restricted_wd_df_z['z_sonic4'][i], color = 'blue', label = 's4')
    print('start')
    # uBar_arr_i = [0.0001*np.exp(0.4*restricted_wd_df_sonic1['Ubar'][i]/usr_df_fall_copy1['usr_s1'][i]), 0.0001*np.exp(0.4*restricted_wd_df_sonic2['Ubar'][i]/usr_df_fall_copy1['usr_s2'][i]), 0.0001*np.exp(0.4*restricted_wd_df_sonic3['Ubar'][i]/usr_df_fall_copy1['usr_s3'][i]), 0.0001*np.exp(0.4*restricted_wd_df_sonic4['Ubar'][i]/usr_df_fall_copy1['usr_s4'][i])]
    uBar_arr_i = [restricted_wd_df_sonic1['Ubar'][i], restricted_wd_df_sonic2['Ubar'][i], restricted_wd_df_sonic3['Ubar'][i], restricted_wd_df_sonic4['Ubar'][i]]
    print(uBar_arr_i)
    zBar_arr_i = [restricted_wd_df_z['z_sonic1'][i], restricted_wd_df_z['z_sonic2'][i], restricted_wd_df_z['z_sonic3'][i], restricted_wd_df_z['z_sonic4'][i]]
    print(zBar_arr_i)
print('done')
#%%
fig1 = plt.figure() #figure object
ax1 = fig1.gca() #axis object
ax1.set_xlabel('$\overline{u} [m/s]$')
ax1.set_ylabel('$z [m]$')
# ax1.plot(np.log(np.arange(0,50)), np.arange(0,50), color = 'k')
ax1.set_ylim(-0.5,11)
ax1.set_xlim(-5,20)
i = 50
# ax1.scatter(restricted_wd_df_sonic1['Ubar'][i], 0.0001*np.exp(0.4*restricted_wd_df_sonic1['Ubar'][i]/usr_df_fall_copy1['usr_s1'][i]), color = 'red', label = 's1')
# ax1.scatter(restricted_wd_df_sonic2['Ubar'][i], 0.0001*np.exp(0.4*restricted_wd_df_sonic2['Ubar'][i]/usr_df_fall_copy1['usr_s2'][i]), color = 'darkorange', label = 's2')
# ax1.scatter(restricted_wd_df_sonic3['Ubar'][i], 0.0001*np.exp(0.4*restricted_wd_df_sonic3['Ubar'][i]/usr_df_fall_copy1['usr_s3'][i]), color = 'green', label = 's3')
# ax1.scatter(restricted_wd_df_sonic4['Ubar'][i], 0.0001*np.exp(0.4*restricted_wd_df_sonic4['Ubar'][i]/usr_df_fall_copy1['usr_s4'][i]), color = 'blue', label = 's4')

# ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic1['Ubar'][i]/usr_df_fall_copy1['usr_s1'][i]), restricted_wd_df_z['z_sonic1'][i], color = 'red', label = 's1')
# ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic2['Ubar'][i]/usr_df_fall_copy1['usr_s2'][i]), restricted_wd_df_z['z_sonic2'][i], color = 'darkorange', label = 's2')
# ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic3['Ubar'][i]/usr_df_fall_copy1['usr_s3'][i]), restricted_wd_df_z['z_sonic3'][i], color = 'green', label = 's3')
# ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic4['Ubar'][i]/usr_df_fall_copy1['usr_s4'][i]), restricted_wd_df_z['z_sonic4'][i], color = 'blue', label = 's4')
input_arr = np.arange(0,50,0.02)

for i in range(20,21):
    # ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic1['Ubar'][i]/usr_df_fall_copy1['usr_s1'][i]), restricted_wd_df_z['z_sonic1'][i], color = 'red', label = 's1')
    # ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic2['Ubar'][i]/usr_df_fall_copy1['usr_s2'][i]), restricted_wd_df_z['z_sonic2'][i], color = 'darkorange', label = 's2')
    # ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic3['Ubar'][i]/usr_df_fall_copy1['usr_s3'][i]), restricted_wd_df_z['z_sonic3'][i], color = 'green', label = 's3')
    # ax1.scatter(0.0001*np.exp(0.4*restricted_wd_df_sonic4['Ubar'][i]/usr_df_fall_copy1['usr_s4'][i]), restricted_wd_df_z['z_sonic4'][i], color = 'blue', label = 's4')
    # uBar_arr_i = [0.0001*np.exp(0.4*restricted_wd_df_sonic1['Ubar'][i]/usr_df_fall_copy1['usr_s1'][i]), 0.0001*np.exp(0.4*restricted_wd_df_sonic2['Ubar'][i]/usr_df_fall_copy1['usr_s2'][i]), 0.0001*np.exp(0.4*restricted_wd_df_sonic3['Ubar'][i]/usr_df_fall_copy1['usr_s3'][i]), 0.0001*np.exp(0.4*restricted_wd_df_sonic4['Ubar'][i]/usr_df_fall_copy1['usr_s4'][i])]
    ax1.plot(np.log(np.arange(0,50,0.02))+restricted_wd_df_sonic4['Ubar'][i]-np.log(restricted_wd_df_z['z_sonic4'][i]), np.arange(0,50,0.02), color = 'k', label = 'log curve')
    
    uBar_arr_i = [restricted_wd_df_sonic1['Ubar'][i], restricted_wd_df_sonic2['Ubar'][i], restricted_wd_df_sonic3['Ubar'][i], restricted_wd_df_sonic4['Ubar'][i]]
    # print(uBar_arr_i)
    zBar_arr_i = [restricted_wd_df_z['z_sonic1'][i], restricted_wd_df_z['z_sonic2'][i], restricted_wd_df_z['z_sonic3'][i], restricted_wd_df_z['z_sonic4'][i]]
    # print(zBar_arr_i)
    ax1.scatter(uBar_arr_i, zBar_arr_i,color = 'cyan',edgecolor = 'navy', s=50, label = 'observations')
    ax1.plot(uBar_arr_i, zBar_arr_i,color = 'navy')
    # ax1.plot(input_arr, np.exp(input_arr))
ax1.legend()
    # ax1.plot(np.log(np.arange(0,10)), np.arange(0,10), color = k)
#%%
# ax1 = WindroseAxes.from_ax()
# ax1.bar(restricted_alpha_df['alpha_s4'][:break_index+1], restricted_wd_df_sonic4['Ubar'][:break_index+1], bins=np.arange(3, 18, 3), normed = True)
# ax1.set_title('Wind Rose Soinc 4: wind range ('+str(lower)+", "+str(upper)+")")
# ax1.set_legend(bbox_to_anchor=(0.9, -0.1))

vert_array_mean = [np.nanmean(restricted_wd_df_sonic1['Ubar']), 
                   np.nanmean(restricted_wd_df_sonic2['Ubar']), 
                   np.nanmean(restricted_wd_df_sonic3['Ubar']), 
                   np.nanmean(restricted_wd_df_sonic4['Ubar'])]
# vert_mean_df['wd_'+str(lower)+"_"+str(upper-1)] = np.array(vert_array_mean)
vert_array_median = [np.nanmedian(restricted_wd_df_sonic1['Ubar']), 
                     np.nanmedian(restricted_wd_df_sonic2['Ubar']), 
                     np.nanmedian(restricted_wd_df_sonic3['Ubar']), 
                     np.nanmedian(restricted_wd_df_sonic4['Ubar'])]
# vert_median_df['wd_'+str(lower)+"_"+str(upper-1)] = np.array(vert_array_median)

z_avg_arr_spring = [1.842, 4.537, 7.332, 9.747]
# z_avg_arr_fall = [2.288, 4.116, 7.332, 9.800]
#%%
input_ref = np.arange(10)
log_input_ref= np.log(input_ref)
plt.figure()
plt.plot(vert_array_mean, z_avg_arr_spring)
plt.scatter(vert_array_mean, z_avg_arr_spring, label = 'mean')
plt.plot(vert_array_median, z_avg_arr_spring)
plt.scatter(vert_array_median, z_avg_arr_spring, label = 'median')
plt.plot(np.arange(10), 0.001*np.e*(np.arange(10)*0.4/(0.01)), color = 'k')
plt.xlabel('mean wind speed')
plt.ylabel('height')
plt.title('meanU: wind range ('+str(lower)+", "+str(upper)+")")
plt.legend()

slope_43 = (np.array(vert_array_mean)[3] - np.array(vert_array_mean)[2])/(np.array(z_avg_arr_spring)[3]-np.array(z_avg_arr_spring)[2])
slope_32 = (np.array(vert_array_mean)[2] - np.array(vert_array_mean)[1])/(np.array(z_avg_arr_spring)[2]-np.array(z_avg_arr_spring)[1])
slope_21 = (np.array(vert_array_mean)[1] - np.array(vert_array_mean)[0])/(np.array(z_avg_arr_spring)[1]-np.array(z_avg_arr_spring)[0])
print("slope sonic 4-3 = " +str(slope_43))
print("slope sonic 3-1 = " +str(slope_32))
print("slope sonic 2-1 = " +str(slope_21))


#%%
sonic_df_arr = [sonic1_df_spring_copy2[:break_index+1], 
                sonic2_df_spring_copy2[:break_index+1], 
                sonic3_df_spring_copy2[:break_index+1], 
                sonic4_df_spring_copy2[:break_index+1],]

lower = 30
upper = 90
print(lower, upper)

restricted_arr = np.where((lower <= windDir_df_spring_copy2['alpha_s4'][:break_index+1]) & (windDir_df_spring_copy2['alpha_s4'][:break_index+1] < upper), windDir_df_spring_copy2['alpha_s4'][:break_index+1],np.nan)
print('restricted_arr made')

mask_restricted_arr = np.isin(restricted_arr, np.array(windDir_df_spring_copy2['alpha_s4'][:break_index+1]))
print('mask_restricted_arr made')

restricted_wd_df_sonic1 = sonic_df_arr[0][mask_restricted_arr]
restricted_wd_df_sonic2 = sonic_df_arr[1][mask_restricted_arr]
restricted_wd_df_sonic3 = sonic_df_arr[2][mask_restricted_arr]
restricted_wd_df_sonic4 = sonic_df_arr[3][mask_restricted_arr]
restricted_alpha_df = windDir_df_spring_copy2[:break_index+1][mask_restricted_arr]

print('restricted dfs made')


# ax1 = WindroseAxes.from_ax()
# ax1.bar(restricted_alpha_df['alpha_s4'][:break_index+1], restricted_wd_df_sonic4['Ubar'][:break_index+1], bins=np.arange(3, 18, 3), normed = True)
# ax1.set_title('Wind Rose Soinc 4: wind range ('+str(lower)+", "+str(upper)+")")
# ax1.set_legend(bbox_to_anchor=(0.9, -0.1))

vert_array_mean = [np.nanmean(restricted_wd_df_sonic1['Ubar']), 
                   np.nanmean(restricted_wd_df_sonic2['Ubar']), 
                   np.nanmean(restricted_wd_df_sonic3['Ubar']), 
                   np.nanmean(restricted_wd_df_sonic4['Ubar'])]
# vert_mean_df['wd_'+str(lower)+"_"+str(upper-1)] = np.array(vert_array_mean)
vert_array_median = [np.nanmedian(restricted_wd_df_sonic1['Ubar']), 
                     np.nanmedian(restricted_wd_df_sonic2['Ubar']), 
                     np.nanmedian(restricted_wd_df_sonic3['Ubar']), 
                     np.nanmedian(restricted_wd_df_sonic4['Ubar'])]
# vert_median_df['wd_'+str(lower)+"_"+str(upper-1)] = np.array(vert_array_median)

z_avg_arr_spring = [1.842, 4.537, 7.332, 9.747]
# z_avg_arr_fall = [2.288, 4.116, 7.332, 9.800]

plt.figure()
plt.plot(vert_array_mean, z_avg_arr_spring)
plt.scatter(vert_array_mean, z_avg_arr_spring, label = 'mean')
plt.plot(vert_array_median, z_avg_arr_spring)
plt.scatter(vert_array_median, z_avg_arr_spring, label = 'median')
plt.xlabel('mean wind speed')
plt.ylabel('height')
plt.title('meanU: wind range ('+str(lower)+", "+str(upper)+")")
plt.legend()

slope_43 = (np.array(vert_array_mean)[3] - np.array(vert_array_mean)[2])/(np.array(z_avg_arr_spring)[3]-np.array(z_avg_arr_spring)[2])
slope_32 = (np.array(vert_array_mean)[2] - np.array(vert_array_mean)[1])/(np.array(z_avg_arr_spring)[2]-np.array(z_avg_arr_spring)[1])
slope_21 = (np.array(vert_array_mean)[1] - np.array(vert_array_mean)[0])/(np.array(z_avg_arr_spring)[1]-np.array(z_avg_arr_spring)[0])
print("slope sonic 4-3 = " +str(slope_43))
print("slope sonic 3-1 = " +str(slope_32))
print("slope sonic 2-1 = " +str(slope_21))






#%%
sonic_df_arr = [sonic1_df, sonic2_df, sonic3_df, sonic4_df,]


range_var = 25

vert_mean_df = pd.DataFrame()
vert_median_df = pd.DataFrame()

for i in range(1,int((375/range_var)+1)):
    lower = (i*range_var)-range_var
    upper = (i*range_var)
    print(lower, upper)
    
    restricted_arr = np.where((lower <= adjusted_alpha_df['alpha_s4']) & (adjusted_alpha_df['alpha_s4'] < upper),adjusted_alpha_df['alpha_s4'],np.nan)
    print('restricted_arr made')
    
    mask_restricted_arr = np.isin(restricted_arr, np.array(adjusted_alpha_df['alpha_s4']))
    print('mask_restricted_arr made')
    
    restricted_wd_df_sonic1 = sonic_df_arr[0][mask_restricted_arr]
    restricted_wd_df_sonic2 = sonic_df_arr[1][mask_restricted_arr]
    restricted_wd_df_sonic3 = sonic_df_arr[2][mask_restricted_arr]
    restricted_wd_df_sonic4 = sonic_df_arr[3][mask_restricted_arr]
    restricted_alpha_df = adjusted_alpha_df[mask_restricted_arr]
    
    print('restricted dfs made')
    
    
    ax1 = WindroseAxes.from_ax()
    ax1.bar(restricted_alpha_df['alpha_s4'], restricted_wd_df_sonic4['Ubar'], bins=np.arange(3, 18, 3), normed = True)
    ax1.set_title('Wind Rose Soinc 4: wind range ('+str(lower)+", "+str(upper)+")")
    ax1.set_legend(bbox_to_anchor=(0.9, -0.1))
    
    vert_array_mean = [np.nanmean(restricted_wd_df_sonic1['Ubar']), 
                       np.nanmean(restricted_wd_df_sonic2['Ubar']), 
                       np.nanmean(restricted_wd_df_sonic3['Ubar']), 
                       np.nanmean(restricted_wd_df_sonic4['Ubar'])]
    vert_mean_df['wd_'+str(lower)+"_"+str(upper-1)] = np.array(vert_array_mean)
    vert_array_median = [np.nanmedian(restricted_wd_df_sonic1['Ubar']), 
                         np.nanmedian(restricted_wd_df_sonic2['Ubar']), 
                         np.nanmedian(restricted_wd_df_sonic3['Ubar']), 
                         np.nanmedian(restricted_wd_df_sonic4['Ubar'])]
    vert_median_df['wd_'+str(lower)+"_"+str(upper-1)] = np.array(vert_array_median)
    
    z_avg_arr_fall = [2.288, 4.117, 7.333, 9.801]
    
    plt.figure()
    plt.plot(vert_array_mean, z_avg_arr_fall)
    plt.scatter(vert_array_mean, z_avg_arr_fall, label = 'mean')
    plt.plot(vert_array_median, z_avg_arr_fall)
    plt.scatter(vert_array_median, z_avg_arr_fall, label = 'median')
    plt.xlabel('mean wind speed')
    plt.ylabel('height')
    plt.title('meanU: wind range ('+str(lower)+", "+str(upper)+")")
    plt.legend()
    
    
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle( 'wind direction range ('+str(lower)+", "+str(upper)+") [deg]")
    # ax1 = WindroseAxes.from_ax()
    # ax1.bar(restricted_alpha_df['alpha_s4'], restricted_wd_df_sonic4['Ubar'], bins=np.arange(3, 18, 3), normed = True)
    # ax1.set_title('Wind Rose Sonic 4')
    # ax1.set_legend(bbox_to_anchor=(0.9, -0.1))
    
    # ax2.plot(vert_array_mean, z_avg_arr)
    # ax2.scatter(vert_array_mean, z_avg_arr, label = 'mean')
    # ax2.plot(vert_array_median, z_avg_arr)
    # ax2.scatter(vert_array_median, z_avg_arr, label = 'median')
    # ax2.set_xlabel('mean wind speed')
    # ax2.set_ylabel('height')
    # ax2.set_title('<u>')
    # ax2.legend()
#%%
plt.figure()
plt.plot(vert_mean_df['wd_0_24'], z_avg_arr_fall, label = '0-25', color = 'black')
plt.plot(vert_mean_df['wd_25_49'], z_avg_arr_fall, label = '25-50', color = 'black')
plt.plot(vert_mean_df['wd_50_74'], z_avg_arr_fall, label = '50-75', color = 'black')
plt.plot(vert_mean_df['wd_75_99'], z_avg_arr_fall, label = '75-100', color = 'black')
plt.plot(vert_mean_df['wd_100_124'], z_avg_arr_fall, label = '100-125', color = 'black')
plt.plot(vert_mean_df['wd_125_149'], z_avg_arr_fall, label = '125-150', color = 'black')
plt.plot(vert_mean_df['wd_150_174'], z_avg_arr_fall, label = '150-175', color = 'black')
plt.plot(vert_mean_df['wd_175_199'], z_avg_arr_fall, label = '175-200', color = 'blue')
plt.plot(vert_mean_df['wd_200_224'], z_avg_arr_fall, label = '200-225', color = 'red')
plt.plot(vert_mean_df['wd_225_249'], z_avg_arr_fall, label = '225-250', color = 'orange')
plt.plot(vert_mean_df['wd_250_274'], z_avg_arr_fall, label = '250-275', color = 'yellow')
plt.plot(vert_mean_df['wd_275_299'], z_avg_arr_fall, label = '275-300', color = 'green')
plt.plot(vert_mean_df['wd_300_324'], z_avg_arr_fall, label = '300-325', color = 'pink')
plt.plot(vert_mean_df['wd_325_349'], z_avg_arr_fall, label = '325-350', color = 'purple')
plt.plot(vert_mean_df['wd_350_374'], z_avg_arr_fall, label = '350-365', color = 'brown')
plt.legend(bbox_to_anchor=(1.25, 1.2))
plt.xlabel('mean wind speed')
plt.ylabel('height')
plt.title('FALL: vertical wind profiles separated by wind direction')
#%%
plt.figure()
plt.plot(vert_median_df['wd_0_24'], z_avg_arr_fall, label = '0-25', color = 'black')
plt.plot(vert_median_df['wd_25_49'], z_avg_arr_fall, label = '25-50', color = 'black')
plt.plot(vert_median_df['wd_50_74'], z_avg_arr_fall, label = '50-75', color = 'black')
plt.plot(vert_median_df['wd_75_99'], z_avg_arr_fall, label = '75-100', color = 'green')
plt.plot(vert_median_df['wd_100_124'], z_avg_arr_fall, label = '100-125', color = 'blue')
plt.plot(vert_median_df['wd_125_149'], z_avg_arr_fall, label = '125-150', color = 'red')
plt.plot(vert_median_df['wd_150_174'], z_avg_arr_fall, label = '150-175', color = 'black')
plt.plot(vert_median_df['wd_175_199'], z_avg_arr_fall, label = '175-200', color = 'black')
plt.plot(vert_median_df['wd_200_224'], z_avg_arr_fall, label = '200-225', color = 'black')
plt.plot(vert_median_df['wd_225_249'], z_avg_arr_fall, label = '225-250', color = 'black')
plt.plot(vert_median_df['wd_250_274'], z_avg_arr_fall, label = '250-275', color = 'black')
plt.plot(vert_median_df['wd_275_299'], z_avg_arr_fall, label = '275-300', color = 'black')
plt.plot(vert_median_df['wd_300_324'], z_avg_arr_fall, label = '300-325', color = 'black')
plt.plot(vert_median_df['wd_325_349'], z_avg_arr_fall, label = '325-350', color = 'black')
plt.plot(vert_median_df['wd_350_374'], z_avg_arr_fall, label = '350-365', color = 'black')
plt.legend(bbox_to_anchor=(1.25, 1.2))
plt.xlabel('MEDIAN wind speed')
plt.ylabel('height')
plt.title('FALL: MEDIAN vertical wind profiles separated by wind direction')


#%%
#calculate slope

slope1 = (vert_mean_df['wd_0_24'].loc[3]-vert_mean_df['wd_0_24'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope2 = (vert_mean_df['wd_25_49'].loc[3]-vert_mean_df['wd_25_49'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope3 = (vert_mean_df['wd_50_74'].loc[3]-vert_mean_df['wd_50_74'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope4 = (vert_mean_df['wd_75_99'].loc[3]-vert_mean_df['wd_75_99'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope5 = (vert_mean_df['wd_100_124'].loc[3]-vert_mean_df['wd_100_124'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope6 = (vert_mean_df['wd_125_149'].loc[3]-vert_mean_df['wd_125_149'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope7 = (vert_mean_df['wd_150_174'].loc[3]-vert_mean_df['wd_150_174'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope8 = (vert_mean_df['wd_175_199'].loc[3]-vert_mean_df['wd_175_199'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope9 = (vert_mean_df['wd_200_224'].loc[3]-vert_mean_df['wd_200_224'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope10 = (vert_mean_df['wd_225_249'].loc[3]-vert_mean_df['wd_225_249'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope11 = (vert_mean_df['wd_250_274'].loc[3]-vert_mean_df['wd_250_274'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope12 = (vert_mean_df['wd_275_299'].loc[3]-vert_mean_df['wd_275_299'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope13 = (vert_mean_df['wd_300_324'].loc[3]-vert_mean_df['wd_300_324'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope14 = (vert_mean_df['wd_325_349'].loc[3]-vert_mean_df['wd_325_349'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])
slope15 = (vert_mean_df['wd_350_374'].loc[3]-vert_mean_df['wd_350_374'].loc[2])/(np.array(z_avg_arr_fall)[3]-np.array(z_avg_arr_fall)[2])

mean_slopes_arr = [slope1, 
                   slope2, 
                   slope3, 
                   slope4, #problem 75-99
                   slope5, #problem 100-124
                   slope6, #problem 125-149
                   slope7, 
                   slope8, 
                   slope9,
                   slope10,
                   slope11,
                   slope12,
                   slope13,
                   slope14,
                   slope15,]
x_arr = np.arange(1,len(mean_slopes_arr)+1)
plt.figure()
plt.scatter(x_arr, mean_slopes_arr)
plt.grid()

#%%
fig, ax1 = plt.subplots()

color = 'skyblue'
ax1.set_ylabel('wind direction [deg]', color=color)  # we already handled the x-label with ax1
ax1.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s4'], s=10, color=color)
ax1.axhline(y=180, color='skyblue', linestyle='--')
ax1.axhline(y=270, color='skyblue', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('Wind Speed and Direction')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'navy'
ax2.set_xlabel('time')
ax2.set_ylabel('wind speed [m/s]', color=color)
ax2.plot(adjusted_alpha_df['time'], sonic4_df['Ubar_s4'], color=color)
ax2.axhline(y=10, color='navy', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
"""

#%%
#for october 2-4 wind event
"""
mask_Ubar = (Ubar_df['time'] >= 731) & (Ubar_df['time'] <= 913)
Ubar_df = Ubar_df.loc[mask_Ubar]

mask_alpha = (adjusted_alpha_df['time'] >= 731) & (adjusted_alpha_df['time'] <= 913)
adjusted_alpha_df = adjusted_alpha_df.loc[mask_alpha]


#%%
fig = plt.figure()
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s1'], color = 'r', label = 's1')
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s2'], color = 'orange', label = 's2')
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s3'], color = 'g', label = 's3')
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s4'], color = 'b', label = 's4')
plt.legend()
plt.title('Wind Direction')
plt.ylabel('wind direction (deg)')
#%%
fig = plt.figure()

plt.plot(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s4'], color = 'skyblue', label = 's4')
plt.plot(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s3'], color = 'silver', label = 's3')
plt.plot(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s2'], color = 'darkorange', label = 's2')
plt.plot(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s1'], color = 'darkgreen', label = 's1')

plt.legend()
plt.ylim(0,360)
plt.title('Wind Direction')
plt.ylabel('wind direction (deg)')

#%%

ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df['alpha_s1'], Ubar_df['Ubar_s1'], bins=np.arange(9, 18, 1), normed = True)
ax.set_title('Sonic 1 Wind Rose')
ax.set_legend(bbox_to_anchor=(0.9, -0.1))
#%%
ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df['alpha_s2'], Ubar_df['Ubar_s2'], bins=np.arange(9, 18, 1), normed = True)
ax.set_title('Sonic 2 Wind Rose')
ax.set_legend(bbox_to_anchor=(0.9, -0.1))

#%%
ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df['alpha_s3'], Ubar_df['Ubar_s3'], bins=np.arange(9, 18, 1), normed = True)
ax.set_title('Sonic 3 Wind Rose')
ax.set_legend(bbox_to_anchor=(0.9, -0.1))

#%%
ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df['alpha_s4'], Ubar_df['Ubar_s4'], bins=np.arange(9, 18, 1), normed = True)
ax.set_title('Sonic 4 Wind Rose')
ax.set_legend(bbox_to_anchor=(0.9, -0.1))
# ax.legend(bbox_to_anchor=(1.2 , -0.1)



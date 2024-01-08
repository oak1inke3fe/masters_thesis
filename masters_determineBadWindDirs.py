#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:05:49 2023

@author: oaklin keefe

NOTE: step one of this file needs to be run on the remote desktop (the step where it says "for first time running").

This file is used to determine "bad wind directions" where the wind direcion may cause turbulence to be formed from interaction with the tower or
other tower components and that is unrelated to the flow if the tower was not present.

INPUT files:
    sonics files from Level1_align-despike-interp files (for first time running)
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    date_combinedAnalysis.csv
    
    
We also set:
    base_index= 3959 as the last point in the spring deployment to separate spring and fall datasets so the hampel filter is not 
    corrupted by data that is not in consecutive time sequence.

    
OUTPUT files:
    alpha_combinedAnalysis.csv (this is a file with all the wind directions between -180,+180 for the full spring/fall deployment. 0 degrees is
                                coming from the E, +/-180 is coming from the W, +90 is coming from the N, -90 is coming from the S)
    windDir_withBadFlags_combinedAnalysis.csv (this has teh adjusted alpha such that 0 degrees is N, 90 E, 180 S, 270 W; it also has binary flags
                                               for when a wind direction is blowing through or near the tower)
    WindRose_spring.png
    WindRose_fall.png
    WindRose_combinedAnalysis.png

    
    
"""



#%%

import numpy as np
import pandas as pd
import os
import natsort
import matplotlib.pyplot as plt
from windrose import WindroseAxes

print('done with imports')


#%% For first time running

# alpha_s1 = []
# alpha_s2 = []
# alpha_s3 = []
# alpha_s4 = []

# beta_s1 = []
# beta_s2 = []
# beta_s3 = []
# beta_s4 = []

# # filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp/"
# filepath = r"/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-despike-interp/"
# for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
#     for filename in natsort.natsorted(filenames):
#         file = os.path.join(root, filename)
#         filename_only = filename[:-6]
        
#         if filename.startswith("mNode_Port1"):
#             df = pd.read_csv(file)
#             # #alpha = np.nanmean(df['alpha']) #we don't need to do this... see next line
#             alpha = df['alpha'][1] #first rotation 
#             alpha_s1.append(alpha)
#             beta = df['beta'][1] #second rotation
#             beta_s1.append(beta)
#             print(filename_only)
#         if filename.startswith("mNode_Port2"):
#             df = pd.read_csv(file)
#             alpha = df['alpha'][1] 
#             alpha_s2.append(alpha)
#             beta = df['beta'][1] 
#             beta_s2.append(beta)
#             print(filename_only)
#         if filename.startswith("mNode_Port3"):
#             df = pd.read_csv(file)
#             alpha = df['alpha'][1] 
#             alpha_s3.append(alpha)
#             beta = df['beta'][1] 
#             beta_s3.append(beta)
#             print(filename_only)
#         if filename.startswith("mNode_Port4"):
#             df = pd.read_csv(file)
#             alpha = df['alpha'][1] 
#             alpha_s4.append(alpha)
#             beta = df['beta'][1] 
#             beta_s4.append(beta)
#             print(filename_only)
#         else:
#             continue


# alpha_df = pd.DataFrame()
# alpha_df['alpha_s1'] = alpha_s1
# alpha_df['alpha_s2'] = alpha_s2
# alpha_df['alpha_s3'] = alpha_s3
# alpha_df['alpha_s4'] = alpha_s4

# beta_df = pd.DataFrame()
# beta_df['beta_s1'] = beta_s1
# beta_df['beta_s2'] = beta_s2
# beta_df['beta_s3'] = beta_s3
# beta_df['beta_s4'] = beta_s4

# # file_save_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
# file_save_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
# alpha_df.to_csv(file_save_path + "alpha_combinedAnalysis.csv")
# beta_df.to_csv(file_save_path + "beta_combinedAnalysis.csv")

# print('done')


#%%
# file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
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


alpha_df = pd.read_csv(file_path+"alpha_combinedAnalysis.csv")
beta_df = pd.read_csv(file_path+"beta_combinedAnalysis.csv")

time_arr = np.arange(0,len(alpha_df))

date_df = pd.read_csv(file_path + "date_combinedAnalysis.csv")
print(date_df.columns)
print(date_df['datetime'][10])


alpha_df['time'] = time_arr
alpha_df['datetime'] = date_df['datetime']
alpha_df['MM'] = date_df['MM']
alpha_df['hh'] = date_df['hh']

beta_df['time'] = time_arr
beta_df['datetime'] = date_df['datetime']
beta_df['MM'] = date_df['MM']
beta_df['hh'] = date_df['hh']

print(alpha_df.columns)
#%%
#This step to see if the sonics were facing the same way

fig = plt.figure()
plt.scatter(alpha_df['time'], alpha_df['alpha_s1'], color = 'r', label = 's1')
plt.scatter(alpha_df['time'], alpha_df['alpha_s2'], color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], alpha_df['alpha_s3'], color = 'g', label = 's3')
plt.scatter(alpha_df['time'], alpha_df['alpha_s4'], color = 'b', label = 's4')
plt.title('Combined Analysis (ALPHA)')
plt.legend()

fig = plt.figure()
plt.scatter(beta_df['time'], beta_df['beta_s1'], color = 'r', label = 's1')
plt.scatter(beta_df['time'], beta_df['beta_s2'], color = 'orange', label = 's2')
plt.scatter(beta_df['time'], beta_df['beta_s3'], color = 'g', label = 's3')
plt.scatter(beta_df['time'], beta_df['beta_s4'], color = 'b', label = 's4')
plt.title('Combined Analysis (BETA)')
plt.legend()


fig = plt.figure()
plt.scatter(alpha_df['time'], alpha_df['alpha_s1'], color = 'r', label = 's1')
plt.scatter(alpha_df['time'], alpha_df['alpha_s2'], color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], alpha_df['alpha_s3'], color = 'g', label = 's3')
plt.scatter(alpha_df['time'], alpha_df['alpha_s4'], color = 'b', label = 's4')
plt.xlim(0,3959)
# plt.xlim(2140,2150)
plt.title('Spring Deployment')
plt.legend()
#here we see they are aligned relatively aligned... off by about 5-10 degrees to eachother... I fix this in the next step

fig = plt.figure()
plt.scatter(alpha_df['time'], alpha_df['alpha_s1'], color = 'r', label = 's1')
plt.scatter(alpha_df['time'], alpha_df['alpha_s2'], color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], alpha_df['alpha_s3'], color = 'g', label = 's3')
plt.scatter(alpha_df['time'], alpha_df['alpha_s4'], color = 'b', label = 's4')
plt.xlim(3960,8279)
plt.title('Fall Deployment')
plt.legend()
#here we see sonic 4 is aligned differently
#%%
#here we adjust to a 0-360 degree circle, with 0 degrees as true north
# adjusted_arr = [adjusted_a_s1, adjusted_a_s2, adjusted_a_s3, adjusted_a_s4]
alpha_s1 = np.array(alpha_df['alpha_s1'])
alpha_s2 = np.array(alpha_df['alpha_s2'])
alpha_s3 = np.array(alpha_df['alpha_s3'])
alpha_s4 = np.array(alpha_df['alpha_s4'])
adjusted_arr = [alpha_s1, alpha_s2, alpha_s3, alpha_s4]

for arr in adjusted_arr:
    for i in range(0,len(arr)):
        if arr[i] > 360:
            arr[i] = arr[i]-360
        elif arr[i] < 0:
            arr[i] = 360 + arr[i]
        else:
            arr[i] = arr[i]
print('done')

fig = plt.figure()
plt.scatter(alpha_df['time'], alpha_s1, color = 'r', label = 's1')
plt.scatter(alpha_df['time'], alpha_s2, color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], alpha_s3, color = 'g', label = 's3')
plt.scatter(alpha_df['time'], alpha_s4, color = 'b', label = 's4')
plt.title('Combined Analysis on 360º direction')
plt.legend()
#%%
#now we need to adjust them to true north

break_index = 3959 #(length of spring is 3960, index is 3959)

#making slight adjustments here to make them uniform to sonic 4 after rotating to true north
# adjusted_a_s1_spring = np.array(alpha_df['alpha_s1'][:break_index+1])+260-180
# adjusted_a_s2_spring = np.array(alpha_df['alpha_s2'][:break_index+1])+264-180
# adjusted_a_s3_spring = np.array(alpha_df['alpha_s3'][:break_index+1])+262-180
# adjusted_a_s4_spring = np.array(alpha_df['alpha_s4'][:break_index+1])+270-180


# 30 - alpha to match high frequency waves
adjusted_a_s1_spring = 30- np.array(alpha_df['alpha_s1'][:break_index+1])
adjusted_a_s2_spring = 30- np.array(alpha_df['alpha_s2'][:break_index+1])
adjusted_a_s3_spring = 30- np.array(alpha_df['alpha_s3'][:break_index+1])
adjusted_a_s4_spring = 30- np.array(alpha_df['alpha_s4'][:break_index+1])

adjusted_a_s1_fall = 30- np.array(alpha_df['alpha_s1'][break_index+1:])
adjusted_a_s2_fall = 30- np.array(alpha_df['alpha_s2'][break_index+1:])
adjusted_a_s3_fall = 30- np.array(alpha_df['alpha_s3'][break_index+1:])
adjusted_a_s4_fall = 30- np.array(alpha_df['alpha_s4'][break_index+1:])


adjusted_a_s1 = np.concatenate([adjusted_a_s1_spring, adjusted_a_s1_fall], axis = 0)
adjusted_a_s2 = np.concatenate([adjusted_a_s2_spring, adjusted_a_s2_fall], axis = 0)
adjusted_a_s3 = np.concatenate([adjusted_a_s3_spring, adjusted_a_s3_fall], axis = 0)
adjusted_a_s4 = np.concatenate([adjusted_a_s4_spring, adjusted_a_s4_fall], axis = 0)


fig = plt.figure()
plt.scatter(alpha_df['time'], adjusted_a_s1, color = 'r', label = 's1')
plt.scatter(alpha_df['time'], adjusted_a_s2, color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], adjusted_a_s3, color = 'g', label = 's3')
plt.scatter(alpha_df['time'], adjusted_a_s4, color = 'b', label = 's4')
plt.hlines(y=360, xmin=0, xmax=len(alpha_df), color = 'k')
plt.hlines(y=0, xmin=0, xmax=len(alpha_df), color = 'k')
plt.legend()
plt.xlim(0, 3959)
plt.title('Spring Deployment')


fig = plt.figure()
plt.scatter(alpha_df['time'], adjusted_a_s1, color = 'r', label = 's1')
plt.scatter(alpha_df['time'], adjusted_a_s2, color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], adjusted_a_s3, color = 'g', label = 's3')
plt.scatter(alpha_df['time'], adjusted_a_s4, color = 'b', label = 's4')
plt.hlines(y=360, xmin=0, xmax=len(alpha_df), color = 'k')
plt.hlines(y=0, xmin=0, xmax=len(alpha_df), color = 'k')
plt.legend()
plt.xlim(3959, 8279)
plt.title('Fall Deployment')

# adjusted_a_s1 = np.array(adjusted_a_s1)-55
# adjusted_a_s2 = np.array(adjusted_a_s2)-55
# adjusted_a_s3 = np.array(adjusted_a_s3)-55
# adjusted_a_s4 = np.array(adjusted_a_s4)-55

# fig = plt.figure()
# plt.scatter(alpha_df['time'], adjusted_a_s1, color = 'r', label = 's1')
# plt.scatter(alpha_df['time'], adjusted_a_s2, color = 'orange', label = 's2')
# plt.scatter(alpha_df['time'], adjusted_a_s3, color = 'g', label = 's3')
# plt.scatter(alpha_df['time'], adjusted_a_s4, color = 'b', label = 's4')
# plt.hlines(y=360, xmin=0, xmax=4422, color = 'k')
# plt.hlines(y=0, xmin=0, xmax=4422, color = 'k')
# plt.legend()

#%%
#here we adjust to a 0-360 degree circle, with 0 degrees as true north
adjusted_arr = [adjusted_a_s1, adjusted_a_s2, adjusted_a_s3, adjusted_a_s4]

for arr in adjusted_arr:
    for i in range(0,len(arr)):
        if arr[i] > 360:
            arr[i] = arr[i]-360
        elif arr[i] < 0:
            arr[i] = 360 + arr[i]
        else:
            arr[i] = arr[i]
print('done')

fig = plt.figure()
plt.scatter(alpha_df['time'], adjusted_a_s1, color = 'r', label = 's1')
plt.scatter(alpha_df['time'], adjusted_a_s2, color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], adjusted_a_s3, color = 'g', label = 's3')
plt.scatter(alpha_df['time'], adjusted_a_s4, color = 'b', label = 's4')
plt.hlines(y=360, xmin=0, xmax=len(alpha_df), color = 'k')
plt.hlines(y=0, xmin=0, xmax=len(alpha_df), color = 'k')
plt.legend()
plt.xlim(0, 3959)
plt.title('Spring Deployment')


fig = plt.figure()
plt.scatter(alpha_df['time'], adjusted_a_s1, color = 'r', label = 's1')
plt.scatter(alpha_df['time'], adjusted_a_s2, color = 'orange', label = 's2')
plt.scatter(alpha_df['time'], adjusted_a_s3, color = 'g', label = 's3')
plt.scatter(alpha_df['time'], adjusted_a_s4, color = 'b', label = 's4')
plt.hlines(y=360, xmin=0, xmax=len(alpha_df), color = 'k')
plt.hlines(y=0, xmin=0, xmax=len(alpha_df), color = 'k')
plt.legend()
plt.xlim(3959, 8279)
plt.title('Fall Deployment')

adjusted_alpha_df = pd.DataFrame()
adjusted_alpha_df['alpha_s1'] = adjusted_a_s1
adjusted_alpha_df['alpha_s2'] = adjusted_a_s2
adjusted_alpha_df['alpha_s3'] = adjusted_a_s3
adjusted_alpha_df['alpha_s4'] = adjusted_a_s4
adjusted_alpha_df['time'] = alpha_df['time']
adjusted_alpha_df['date'] = alpha_df['datetime']
adjusted_alpha_df['MM'] = alpha_df['MM']
adjusted_alpha_df['hh'] = alpha_df['hh']
# adjusted_alpha_df.to_csv(file_path+"windDir_IncludingBad_combinedAnalysis.csv")
#%%
#compare adjustment to period when we know the wind was consistently from the SW and
#check the output is from the SW
#here we will pick May 14 1800 [index 2142] where the wind was from the SSW-SW
plt.figure()
plt.plot(adjusted_a_s1, label ='s1', color = 'r')
plt.plot(adjusted_a_s2, label ='s2', color = 'orange')
plt.plot(adjusted_a_s3, label ='s3', color = 'g')
plt.plot(adjusted_a_s4, label ='s4', color = 'b')
plt.legend()
# plt.xlim(2140,2150)
# plt.title ('checking flow is from the SSW-SW')
plt.xlim(925,975)
plt.ylim(300,360)
plt.title ('checking difference between s4 and s1 directions \n at different points in time')


# plt.figure()
# plt.plot(alpha_df['alpha_s1'][:break_index+1], label ='s1', color = 'r')
# plt.plot(alpha_df['alpha_s2'][:break_index+1], label ='s2', color = 'orange')
# plt.plot(alpha_df['alpha_s3'][:break_index+1], label ='s3', color = 'g')
# plt.plot(alpha_df['alpha_s4'][:break_index+1], label ='s4', color = 'b')
# plt.legend()
# plt.xlim(2140,2150)
# plt.title ('checking flow is from the SSW-SW')

# fig = plt.figure()
# plt.scatter(alpha_df['time'], adjusted_alpha_df['alpha_s1'], color = 'r', label = 's1')
# plt.scatter(alpha_df['time'], adjusted_alpha_df['alpha_s2'], color = 'orange', label = 's2')
# plt.scatter(alpha_df['time'], adjusted_alpha_df['alpha_s3'], color = 'g', label = 's3')
# plt.scatter(alpha_df['time'], adjusted_alpha_df['alpha_s4'], color = 'b', label = 's4')
# plt.legend()
# plt.xlim(0,3959)
# # plt.ylim(300,350)
# plt.title('Spring Deployment Adjusted to 360')
#%%
#compare adjustment to period when we know the wind was consistently from the NE and
#check the output is from the NE
#here we will pick Oct 02 1400 [index 4650] where the wind was from the NE
fig = plt.figure()
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s1'], color = 'r', label = 's1')
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s2'], color = 'orange', label = 's2')
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s3'], color = 'g', label = 's3')
plt.scatter(adjusted_alpha_df['time'], adjusted_alpha_df['alpha_s4'], color = 'b', label = 's4')
plt.legend()
plt.xlim(4650,4800)
# plt.ylim(300,350)
plt.title('Fall Deployment Adjusted to 360')
#%%
#s4 is still off so we will rotate it to match:
adjusted_a_s1_fall_s4rotation = adjusted_a_s1_fall
adjusted_a_s2_fall_s4rotation = adjusted_a_s2_fall
adjusted_a_s3_fall_s4rotation = adjusted_a_s3_fall
adjusted_a_s4_fall_s4rotation = adjusted_a_s4_fall + 90

adjusted_arr = [adjusted_a_s1_fall_s4rotation, 
                adjusted_a_s2_fall_s4rotation, 
                adjusted_a_s3_fall_s4rotation, 
                adjusted_a_s4_fall_s4rotation]

for arr in adjusted_arr:
    for i in range(0,len(arr)):
        if arr[i] > 360:
            arr[i] = arr[i]-360
        elif arr[i] < 0:
            arr[i] = 360 + arr[i]
        else:
            arr[i] = arr[i]
print('done')

adjusted_a_s1_wS4rotation = np.concatenate([np.array(adjusted_a_s1[:break_index+1]), adjusted_a_s1_fall_s4rotation], axis = 0)
adjusted_a_s2_wS4rotation = np.concatenate([np.array(adjusted_a_s2[:break_index+1]), adjusted_a_s2_fall_s4rotation], axis = 0)
adjusted_a_s3_wS4rotation = np.concatenate([np.array(adjusted_a_s3[:break_index+1]), adjusted_a_s3_fall_s4rotation], axis = 0)
adjusted_a_s4_wS4rotation = np.concatenate([np.array(adjusted_a_s4[:break_index+1]), adjusted_a_s4_fall_s4rotation], axis = 0)

adjusted_alpha_df_final = pd.DataFrame()
adjusted_alpha_df_final['alpha_s1'] = adjusted_a_s1_wS4rotation
adjusted_alpha_df_final['alpha_s2'] = adjusted_a_s2_wS4rotation
adjusted_alpha_df_final['alpha_s3'] = adjusted_a_s3_wS4rotation
adjusted_alpha_df_final['alpha_s4'] = adjusted_a_s4_wS4rotation
adjusted_alpha_df_final['time'] = alpha_df['time']
adjusted_alpha_df_final['date'] = alpha_df['datetime']
adjusted_alpha_df_final['MM'] = alpha_df['MM']
adjusted_alpha_df_final['hh'] = alpha_df['hh']
adjusted_alpha_df_final.to_csv(file_path+"windDir_IncludingBad_wS4rotation_combinedAnalysis.csv")


fig = plt.figure()
plt.plot(adjusted_alpha_df_final['time'], adjusted_alpha_df_final['alpha_s1'], color = 'r', label = 's1')
plt.plot(adjusted_alpha_df_final['time'], adjusted_alpha_df_final['alpha_s2'], color = 'orange', label = 's2')
plt.plot(adjusted_alpha_df_final['time'], adjusted_alpha_df_final['alpha_s3'], color = 'g', label = 's3')
plt.plot(adjusted_alpha_df_final['time'], adjusted_alpha_df_final['alpha_s4'], color = 'b', label = 's4')
plt.legend()
plt.xlim(4650,4680)
plt.ylim(40,60)
plt.title('Fall Deployment Adjusted to 360')
#%%
fig, ax = plt.subplots(1,1)
ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df_final['alpha_s4'], sonic4_df['Ubar'], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Wind Velocity [$ms^{-1}$]', fontsize=20)
ax.set_legend(bbox_to_anchor=(0.9, -0.1),fontsize=20)

plt.savefig(plot_save_path+ "windRoseAllWindDirections_combinedAnalysis.png", dpi=300)
plt.savefig(plot_save_path+ "windRoseAllWindDirections_combinedAnalysis.pdf")

#%% make windroses of daylight conditions 
df_daylight = pd.DataFrame()

plt.figure()
plt.scatter(adjusted_alpha_df_final['alpha_s4'], sonic4_df['Ubar'], s=1)

index_array = np.arange(len(adjusted_alpha_df))
adjusted_alpha_df_final['new_index_arr'] = np.where((adjusted_alpha_df_final['hh']>=7)&(adjusted_alpha_df_final['hh']<=17), np.nan, index_array)
mask_Daylight = np.isin(adjusted_alpha_df_final['new_index_arr'],index_array)
adjusted_alpha_df_final[mask_Daylight] = np.nan
sonic4_df[mask_Daylight] = np.nan

plt.figure()
plt.scatter(adjusted_alpha_df_final['alpha_s4'], sonic4_df['Ubar'],s=1)

ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df_final['alpha_s4'], sonic4_df['Ubar'], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('DAYLIGHT Wind Velocity [$ms^{-1}$] \n Combined Deployments', fontsize=20)
ax.set_legend(bbox_to_anchor=(0.9, -0.1),fontsize=20)

plt.savefig(plot_save_path+ "windRose_DAYLIGHTcombinedAnalysis.png", dpi=300)
plt.show()
plt.savefig(plot_save_path+ "windRose_DAYLIGHTcombinedAnalysis.pdf")

#%% daylight by month (just May and Oct.) to compare to Sailflow

may_daylight_df = pd.DataFrame()
may_daylight_df['index'] = np.arange(len(adjusted_alpha_df_final))
may_daylight_df['alpha_s4'] = np.array(adjusted_alpha_df_final['alpha_s4'])
may_daylight_df['Ubar_s4'] = np.array(sonic4_df['Ubar'])
plt.figure()
plt.scatter(may_daylight_df['alpha_s4'], may_daylight_df['Ubar_s4'],s=1)
plt.title('May daylight Before')

oct_daylight_df = pd.DataFrame()
oct_daylight_df['index'] = np.arange(len(adjusted_alpha_df_final))
oct_daylight_df['alpha_s4'] = np.array(adjusted_alpha_df_final['alpha_s4'])
oct_daylight_df['Ubar_s4'] = np.array(sonic4_df['Ubar'])
plt.figure()
plt.scatter(oct_daylight_df['alpha_s4'], oct_daylight_df['Ubar_s4'],s=1)
plt.title('Oct. daylight Before')
#%%
may_daylight_df['may_daylight']= np.where(adjusted_alpha_df_final['MM']==5, np.nan, index_array)
oct_daylight_df['oct_daylight']= np.where(adjusted_alpha_df_final['MM']==10, np.nan, index_array)
mask_mayDaylight = np.isin(may_daylight_df['may_daylight'],index_array)
mask_octDaylight = np.isin(oct_daylight_df['oct_daylight'],index_array)


may_daylight_df[mask_mayDaylight] = np.nan
plt.figure()
plt.scatter(may_daylight_df['alpha_s4'], may_daylight_df['Ubar_s4'],s=1)
plt.title('May daylight AFTER')

oct_daylight_df[mask_octDaylight] = np.nan
plt.figure()
plt.scatter(oct_daylight_df['alpha_s4'], oct_daylight_df['Ubar_s4'],s=1)
plt.title('Oct. daylight After')

#%%
ax = WindroseAxes.from_ax()
ax.bar(may_daylight_df['alpha_s4'], may_daylight_df['Ubar_s4'], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Daylight Wind Velocity [$ms^{-1}$] \n MAY', fontsize=20)
ax.set_legend(bbox_to_anchor=(0.9, -0.1),fontsize=20)

plt.savefig(plot_save_path+ "windRose_DaylightMay.png", dpi=300)
plt.show()
plt.savefig(plot_save_path+ "windRose_DaylightMay.pdf")

ax = WindroseAxes.from_ax()
ax.bar(oct_daylight_df['alpha_s4'], oct_daylight_df['Ubar_s4'], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Daylight Wind Velocity [$ms^{-1}$] \n OCTOBER', fontsize=20)
ax.set_legend(bbox_to_anchor=(0.9, -0.1),fontsize=20)

plt.savefig(plot_save_path+ "windRose_DaylightOct.png", dpi=300)
plt.show()
plt.savefig(plot_save_path+ "windRose_DaylightOct.pdf")
#%%



#%% this is original method of flagging bad wind directions by setting a pre-set range of bad wind directions
sonic_head_arr = [1.27, 1.285, 1.23]
sonic_min = np.min(sonic_head_arr)
b = 0.641
c = 0.641+sonic_min
A = 60
import math
a_len = math.sqrt(b**2+c**2-(2*b*c*math.cos(A*math.pi/180)))
angle_offset_rad = math.asin(b*math.sin(A*math.pi/180)/a_len)
angle_offset = angle_offset_rad*180/math.pi
print("angle offset = " + str(angle_offset))
angle_start= 125+angle_offset
angle_end = 125

print(angle_start)
print(angle_end)
#%%


angle_start_spring= 110
angle_end_spring = 160 #bad angle range to get rid of (110-149)

angle_start_fall = 110
angle_end_fall = 160 #bad angle range to get rid of (125-150)

print("SPRING: good angle start = " + str(angle_start_spring))
print("SPRING good angle end = " + str(angle_end_spring))
print("FALL: good angle start = " + str(angle_start_fall))
print("FALL good angle end = " + str(angle_end_fall))

adjusted_a_s1_copy = np.copy(adjusted_a_s1_wS4rotation)
adjusted_a_s2_copy = np.copy(adjusted_a_s2_wS4rotation)
adjusted_a_s3_copy = np.copy(adjusted_a_s3_wS4rotation)
adjusted_a_s4_copy = np.copy(adjusted_a_s4_wS4rotation)
#%%
plt.figure()
plt.plot(adjusted_a_s1_copy)
plt.plot(adjusted_a_s2_copy)
plt.plot(adjusted_a_s3_copy)
plt.plot(adjusted_a_s4_copy)
#%%
adjusted_a_s1_copy_spring = np.array(adjusted_a_s1_wS4rotation[: break_index+1])
adjusted_a_s2_copy_spring = np.array(adjusted_a_s2_wS4rotation[: break_index+1])
adjusted_a_s3_copy_spring = np.array(adjusted_a_s3_wS4rotation[: break_index+1])
adjusted_a_s4_copy_spring = np.array(adjusted_a_s4_wS4rotation[: break_index+1])

adjusted_a_s1_copy_fall = np.array(adjusted_a_s1_wS4rotation[break_index+1:])
adjusted_a_s2_copy_fall = np.array(adjusted_a_s2_wS4rotation[break_index+1:])
adjusted_a_s3_copy_fall = np.array(adjusted_a_s3_wS4rotation[break_index+1:])
adjusted_a_s4_copy_fall = np.array(adjusted_a_s4_wS4rotation[break_index+1:])

good_flag_spring = np.ones(len(adjusted_a_s1_copy_spring), dtype = bool)
good_flag_fall = np.ones(len(adjusted_a_s1_copy_fall), dtype = bool)

good_flag_1_spring = np.where((adjusted_a_s1_copy_spring >= angle_start_spring)&(adjusted_a_s1_copy_spring <= angle_end_spring), 'False', good_flag_spring)   #this is if we are using bad anlge range
good_flag_2_spring = np.where((adjusted_a_s2_copy_spring >= angle_start_spring)&(adjusted_a_s2_copy_spring <= angle_end_spring), 'False', good_flag_spring)
good_flag_3_spring = np.where((adjusted_a_s3_copy_spring >= angle_start_spring)&(adjusted_a_s3_copy_spring <= angle_end_spring), 'False', good_flag_spring)
good_flag_4_spring = np.where((adjusted_a_s4_copy_spring >= angle_start_spring)&(adjusted_a_s4_copy_spring <= angle_end_spring), 'False', good_flag_spring)

good_flag_1_fall = np.where((adjusted_a_s1_copy_fall >= angle_start_fall)&(adjusted_a_s1_copy_fall <= angle_end_fall), 'False', good_flag_fall) 
good_flag_2_fall = np.where((adjusted_a_s2_copy_fall >= angle_start_fall)&(adjusted_a_s2_copy_fall <= angle_end_fall), 'False', good_flag_fall)
good_flag_3_fall = np.where((adjusted_a_s3_copy_fall >= angle_start_fall)&(adjusted_a_s3_copy_fall <= angle_end_fall), 'False', good_flag_fall)
good_flag_4_fall = np.where((adjusted_a_s4_copy_fall >= angle_start_fall)&(adjusted_a_s4_copy_fall <= angle_end_fall), 'False', good_flag_fall)

good_flag_1 = np.concatenate([good_flag_1_spring, good_flag_1_fall], axis=0)
good_flag_2 = np.concatenate([good_flag_2_spring, good_flag_2_fall], axis=0)
good_flag_3 = np.concatenate([good_flag_3_spring, good_flag_3_fall], axis=0)
good_flag_4 = np.concatenate([good_flag_4_spring, good_flag_4_fall], axis=0)

print('done')

adjusted_alpha_df_final['good_wind_dir'] = np.array(good_flag_4)
print('done adding good flag array to df')

#%% This is for flagging the bad wind directions by calling a direction "bad" if it deviates from sonic4 by more than 15 degrees
test_badWindDir_s1 = []
test_badWindDir_s2 = []
test_badWindDir_s3 = []
test_badWindDir_s4 = []
original_badWindDir_s4 = adjusted_alpha_df_final['alpha_s4']

import math
for i in range(len(adjusted_alpha_df_final)):
    if math.isnan(adjusted_alpha_df_final['alpha_s4'][i]) == False:
        if np.abs(adjusted_alpha_df_final['alpha_s4'][i] - adjusted_alpha_df_final['alpha_s3'][i]) > 15:
            test_badWindDir_s4_i = 'False'
        else:
            test_badWindDir_s4_i = 'True'
    else:
        if np.abs(adjusted_alpha_df_final['alpha_s2'][i] - adjusted_alpha_df_final['alpha_s3'][i]) > 15:
            test_badWindDir_s4_i = 'False'
        else:
            test_badWindDir_s4_i = 'True'
    test_badWindDir_s4.append(test_badWindDir_s4_i)

plt.figure()
plt.scatter(np.arange(len(test_badWindDir_s4)), test_badWindDir_s4)
plt.title('when s3 is within 15 deg of s4 or s2 if s4 is nan; combined analysis')
#%%
adjusted_alpha_df_final['windDir_within_15_deg'] = np.array(test_badWindDir_s4)

final_good_windDir = []
for i in range(len(adjusted_alpha_df_final)):
    if adjusted_alpha_df_final['windDir_within_15_deg'][i] == 'True':
        if adjusted_alpha_df_final['good_wind_dir'][i] == 'True':
            final_good_windDir_i = 'True'
        else:
            final_good_windDir_i = 'False'
    else:
        final_good_windDir_i = 'False'
    final_good_windDir.append(final_good_windDir_i)

adjusted_alpha_df_final['windDir_final'] = np.array(final_good_windDir)       



adjusted_alpha_df_final['time'] = alpha_df['time']
adjusted_alpha_df_final['datetime'] = alpha_df['datetime']       
# adjusted_alpha_df.to_csv(file_path+"windDir_withBadFlags_combinedAnalysis.csv")
adjusted_alpha_df_final.to_csv(file_path+"windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv")


#%%

# adjusted_alpha_df['offshore_NtoW'] = np.array(offshore_NtoW_flag_4)
# adjusted_alpha_df['onshore_WtoS'] = np.array(onshore_WtoS_flag_4)


#%%
groups_good = adjusted_alpha_df_final.groupby('good_wind_dir')
# groups_good = adjusted_alpha_df.groupby('good_wind_dir_springCameras')
# groups_meh = adjusted_alpha_df.groupby('potential_good_wind_dir')
plt.figure()
for name, group in groups_good:
    plt.plot(group['time'], group['alpha_s4'], marker='o', linestyle='', markersize=2, label=name)
# for name, group in groups_meh:
    # plt.plot(group['time'], group['alpha_s4'], marker='o', linestyle='', markersize=2, label=name)
plt.xlabel('time')
plt.ylabel('wind direction (deg)')
plt.vlines(x=3959, ymin = 0, ymax=360, color = 'k', label = 'break')
# plt.xlim(2000,2200)
plt.legend(loc = 'lower left')
plt.title("Wind Dir. (Good wind directions = False)")
#%%
ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df_final['alpha_s4'][:break_index+1], sonic4_df['Ubar'][:break_index+1], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Wind Rose Spring Dataset')
ax.set_legend(bbox_to_anchor=(0.9, -0.1))
fig.savefig(plot_save_path + 'WindRose_spring.png', dpi = 300)
fig.savefig(plot_save_path + 'WindRose_spring.pdf')

ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df_final['alpha_s4'][break_index+1:], sonic4_df['Ubar'][break_index+1:], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Wind Rose Fall Dataset')
ax.set_legend(bbox_to_anchor=(0.9, -0.1))
fig.savefig(plot_save_path + 'WindRose_fall.png', dpi = 300)
fig.savefig(plot_save_path + 'WindRose_fall.pdf')

ax = WindroseAxes.from_ax()
ax.bar(adjusted_alpha_df_final['alpha_s4'], sonic4_df['Ubar'], bins=np.arange(3, 18, 3), normed = True)
ax.set_title('Wind Rose Combined Datasets')
ax.set_legend(bbox_to_anchor=(0.9, -0.1))
fig.savefig(plot_save_path + 'WindRose_combinedAnalysis.png', dpi = 300)
fig.savefig(plot_save_path + 'WindRose_combinedAnalysis.pdf')

print('figures saved')


#%% make plts based on wind directions 
# test on singular range first in one deployment

sonic_df_arr = [sonic1_df[:break_index+1], sonic2_df[:break_index+1], sonic3_df[:break_index+1], sonic4_df[:break_index+1],]
lower = 110
upper = 155
print(lower, upper)

restricted_arr = np.where((lower <= adjusted_alpha_df_final['alpha_s4'][:break_index+1]) & (adjusted_alpha_df_final['alpha_s4'][:break_index+1] < upper),adjusted_alpha_df_final['alpha_s4'][:break_index+1],np.nan)
print('restricted_arr made')

mask_restricted_arr = np.isin(restricted_arr, np.array(adjusted_alpha_df_final['alpha_s4'][:break_index+1]))
print('mask_restricted_arr made')

restricted_wd_df_sonic1 = sonic_df_arr[0][mask_restricted_arr]
restricted_wd_df_sonic2 = sonic_df_arr[1][mask_restricted_arr]
restricted_wd_df_sonic3 = sonic_df_arr[2][mask_restricted_arr]
restricted_wd_df_sonic4 = sonic_df_arr[3][mask_restricted_arr]
restricted_alpha_df = adjusted_alpha_df_final[:break_index+1][mask_restricted_arr]

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

"""

#RESUME HERE

"""
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


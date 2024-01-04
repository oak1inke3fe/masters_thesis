# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:56:44 2023

@author: oak
"""

#%% IMPORTS
# import os
# import pyrsktools
import numpy as np 
# import regex as re
import pandas as pd
from mat4py import loadmat
# import pyrsktools # Import the library
print('done with imports')

import matplotlib.pyplot as plt

#%%
# file_path = r"Z:\Fall_Deployment\OaklinCopy_waveData/"
file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'

#%%
# data = pyrsktools.open(file_path + '075965_20221214_1600.rsk') # Load up an rk file
file_spring = "BBASIT_Spring_waves.mat"
data_spring = loadmat(file_path+file_spring)
print(data_spring.keys())

file_fall = "BBASIT_Fall_waves.mat"
data_fall = loadmat(file_path+file_fall)
print(data_fall.keys())

#%%
sigH_spring = np.array(data_spring['Hsig'])
T_period_spring = np.array(data_spring['T'])
Cp_spring = np.array(data_spring['c'])
dn_arr_spring = np.array(data_spring['mday'])
print('done spring')

sigH_fall = np.array(data_fall['Hsig'])
T_period_fall = np.array(data_fall['T'])
Cp_fall = np.array(data_fall['c'])
dn_arr_fall = np.array(data_fall['mday'])
print('done fall')


#%%
wave_df_spring = pd.DataFrame()
wave_df_spring['index_arr'] = np.arange(len(dn_arr_spring))
wave_df_spring['date_time'] = dn_arr_spring
wave_df_spring['sigH'] = sigH_spring
wave_df_spring['T_period'] = T_period_spring
wave_df_spring['Cp'] = Cp_spring
print('done spring wave df')

wave_df_fall = pd.DataFrame()
wave_df_fall['index_arr'] = np.arange(len(dn_arr_fall))
wave_df_fall['date_time'] = dn_arr_fall
wave_df_fall['sigH'] = sigH_fall
wave_df_fall['T_period'] = T_period_fall
wave_df_fall['Cp'] = Cp_fall
print('done fall wave df')


#%%
import datetime as dt
python_datetime_spring = []
for i in range(len(dn_arr_spring)):
    python_datetime_spring_i = dt.datetime.fromordinal(int(np.array(wave_df_spring['date_time'][i]))) + dt.timedelta(days=np.array(wave_df_spring['date_time'][i])%1) - dt.timedelta(days = 366)
    python_datetime_spring.append(python_datetime_spring_i)
wave_df_spring['date_time'] = python_datetime_spring
print('done spring datetime')


python_datetime_fall = []
for i in range(len(dn_arr_fall)):
    python_datetime_fall_i = dt.datetime.fromordinal(int(np.array(wave_df_fall['date_time'][i]))) + dt.timedelta(days=np.array(wave_df_fall['date_time'][i])%1) - dt.timedelta(days = 366)
    python_datetime_fall.append(python_datetime_fall_i)
wave_df_fall['date_time'] = python_datetime_fall
print('done fall datetime')

#%%
print(wave_df_spring.head(5))
print('above, spring head 5')

print(wave_df_spring.tail(5))
print('above, spring tail 5')



wave_df_spring.index = wave_df_spring['date_time']
del wave_df_spring['date_time']
# Resample to 20-minute data starting on the hour
wave_df_spring_resampled = wave_df_spring.resample('20T').mean()

# Display the original and resampled data
print("Original SPRING Data:")
print(wave_df_spring.head(9))

print("\nResampled SPRING Data:")
print(wave_df_spring_resampled.head(9))

#%%
print(wave_df_fall.head(5))
print('above, fall head 5')

print(wave_df_fall.tail(5))
print('above, fall tail 5')



wave_df_fall.index = wave_df_fall['date_time']
del wave_df_fall['date_time']
# Resample to 20-minute data starting on the hour
wave_df_fall_resampled = wave_df_fall.resample('20T').mean()

# Display the original and resampled data
print("Original FALL Data:")
print(wave_df_fall.head(9))

print("\nResampled FALL Data:")
print(wave_df_fall_resampled.head(9))
#%% interpolate to fill the NaNs

wave_df_spring_interpolated = wave_df_spring_resampled.interpolate(method='linear')
print('done spring interpolation')

print("Resampled SPRING Data:")
print(wave_df_spring_resampled.head(9))

print("\nInterpolated SPRING Data:")
print(wave_df_spring_interpolated.head(9))

#%%
wave_df_fall_interpolated = wave_df_fall_resampled.interpolate(method='linear')
print('done fall interpolation')

print("Resampled FALL Data:")
print(wave_df_fall_resampled.head(9))

print("\nInterpolated FALL Data:")
print(wave_df_fall_interpolated.head(9))
#%%
print(wave_df_spring_interpolated.head(5))
print('above, spring head 5')

'''
- we see that it starts at 4-15-2022 08:00:00, and goes in 20 min intervals
- we need to start it at 4-15-2022 00:00:00, so we will extend the start by 8*3 NaN entries (3, 20-min entries/hour)
'''


print(wave_df_spring_interpolated.tail(5))
print('above, spring tail 5')

'''
- we see that it ends at 5-31-2022 18:20:00, 
- we need to end it at 6-08-2022 23:40:00, so we will extend it by 8*24*3+17  NaN entries (plus 17 to get to june 1 then plus 8*24*3 (8 days w/24 hours and 3 20-min entries/hour) to get to end date)
'''
#%%
a_start = np.empty(8*3,)
a_end = np.empty(8*3*24+17-1,)
a_start[:] = np.nan
a_end[:] = np.nan
df_extension_start = pd.DataFrame({'index_arr': a_start,
                             'sigH': a_start,
                             'T_period': a_start,
                             'Cp': a_start,})
df_extension_end = pd.DataFrame({'index_arr': a_end,
                             'sigH': a_end,
                             'T_period': a_end,
                             'Cp': a_end,})
wave_df_spring_fullStart = pd.concat([df_extension_start, wave_df_spring_interpolated],axis=0)
wave_df_spring_full = pd.concat([wave_df_spring_fullStart, df_extension_end], axis=0)

print("Un-extended SPRING Data:")
print(wave_df_spring_interpolated.head(9))

print("\nExtended SPRING Data:")
print(wave_df_spring_full.head(9))
#add new row to end of DataFrame
# wave_df_interpolated = wave_df_interpolated.append(df_extension, ignore_index = True)
# wave_df_interpolated = wave_df_interpolated.append(df_extension, ignore_index = False)
#%%
print(wave_df_fall_interpolated.head(5))
print('above, fall head 5')

'''
- we see that it starts at 9-22-2022 00:10:00, and goes in 20 min intervals

'''


print(wave_df_fall_interpolated.tail(5))
print('above, fall tail 5')

'''
- we see that it ends at 11-19-2022 00:00:00, 
- we need to end it at 11-21-2022 23:40:00, so we will extend it by 2*24*3  NaN entries (2*24*3 (2 days w/24 hours and 3 20-min entries/hour) to get to end date)
'''
#%%
# a_start = np.empty(8*3,)
a_end = np.empty(2*24*3-1,)
# a_start[:] = np.nan
a_end[:] = np.nan
# df_extension_start = pd.DataFrame({'index_arr': a_start,
#                              'sigH': a_start,
#                              'T_period': a_start,
#                              'Cp': a_start,})
df_extension_end = pd.DataFrame({'index_arr': a_end,
                             'sigH': a_end,
                             'T_period': a_end,
                             'Cp': a_end,})
# wave_df_fall_fullStart = pd.concat([df_extension_start, wave_df_fall_interpolated],axis=0)
wave_df_fall_full = pd.concat([wave_df_fall_interpolated, df_extension_end], axis=0)

print("Un-extended FALL Data:")
print(wave_df_fall_interpolated.tail(9))

print("\nExtended FALL Data:")
print(wave_df_fall_full.tail(9))
#%%
wave_df_combined = pd.concat([wave_df_spring_full, wave_df_fall_full], axis=0)
wave_df_combined['new_index_arr'] = np.arange(len(wave_df_combined))
wave_df_combined.set_index('new_index_arr', inplace=True)


dates_df = pd.read_csv(file_path + 'date_combinedAnalysis.csv')

wave_df_combined['new_datetime'] = dates_df['datetime']
print(wave_df_combined.head(9))

# wave_df_combined = wave_df_combined.drop('datetime', axis=1)
# wave_df_combined = wave_df_combined.drop('index_arr', axis=1)
# print(wave_df_combined.head(9))

#%%
wave_df_combined.to_csv(file_path +'waveData_combinedAnalysis.csv')
print('done. Saved to .csv')

#%%
#some test plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, axs = plt.subplots(3, sharex = True, sharey=False)
fig.suptitle('Wave Data Time Series Fall')
axs[0].plot(wave_df_combined['sigH'], label = 'sigH')
plt.ylabel('$H_{sig}$ [m]')
axs[1].plot(wave_df_combined['T_period'], label = 'T')
plt.ylabel('T [s]')
axs[2].plot(wave_df_combined['Cp'], label = '$c_p$')
plt.ylabel('$c_p$ [m/s]')
axs[2].tick_params(axis = 'x', rotation=90)

#%%
# trying seaborn to see if it's "prettier"
# import seaborn as sns
# fig, axes = plt.subplots(3, sharex=True)
# #create lineplot in each subplot
# sns.lineplot(data=wave_df_interpolated, x='index_arr', y='sigH', ax=axes[0])
# # sns.lineplot(data=wave_df_interpolated, x='index_arr', y='sigH')
# sns.lineplot(data=wave_df_interpolated, x='index_arr', y='T_period', ax=axes[1])
# sns.lineplot(data=wave_df_interpolated, x='index_arr', y='Cp', ax=axes[2])
# axes[2].set_xticklabels(rotation=45)
# # 
#%%
break_index = 3959
date_df = pd.read_csv(file_path+'date_combinedAnalysis.csv')
dates_arr = np.array(pd.to_datetime(date_df['datetime']))
# SPRING
s=5
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, figsize=(4,4))
fig.suptitle('Wave Data: Spring Deployment Period', fontsize=8)
fig.tight_layout()
fig.subplots_adjust(top=0.90)
ax1.scatter(dates_arr[:break_index+1], wave_df_combined['sigH'][:break_index+1], s=s, color = 'navy', label = '$H_{sig}$')

ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=5)
# ax1.legend(fontsize=7, loc='lower left')
ax1.set_title('$H_{sig}$ [m]', fontsize=6)
ax1.tick_params(axis='y', labelsize=6)
ax1.set_ylim([0,2])

ax2.scatter(dates_arr[:break_index+1], wave_df_combined['Cp'][:break_index+1], s=s, color = 'dimgray', label = '$c_{p}$')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=5)
# ax2.legend(fontsize=7)
ax2.set_title('$c_{p}$ [m/s]', fontsize=6)
ax2.tick_params(axis='y', labelsize=6)
ax2.set_ylim([0,12])

ax3.scatter(dates_arr[:break_index+1], wave_df_combined['T_period'][:break_index+1], s=s, color = 'darkslategray', label = '$T$')
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax3.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=5)
# ax3.legend(fontsize=7)
ax3.set_title('$T$ [s]', fontsize=6)
ax3.tick_params(axis='y', labelsize=6)
ax3.set_ylim([0,10])

ax1.set_ylabel('$H_{sig}$ [m]', fontsize=6)
ax2.set_ylabel('$c_{p}$ [m/s]', fontsize=6)
ax3.set_ylabel('$T$ [s]', fontsize=6)


plt.show()


plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/plots/'
fig.savefig(plot_savePath + "timeseries_WaveData_Spring.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_WaveData_Spring.pdf")
#%%


# FALL
s=5
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, figsize=(4,4))
fig.suptitle('Wave Data: Fall Deployment Period', fontsize=8)
fig.tight_layout()
fig.subplots_adjust(top=0.90)
ax1.scatter(dates_arr[break_index+1:], wave_df_combined['sigH'][break_index+1:], s=s, color = 'navy', label = '$H_{sig}$')

ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=5)
# ax1.legend(fontsize=7, loc='lower left')
ax1.set_title('$H_{sig}$ [m]', fontsize=6)
ax1.tick_params(axis='y', labelsize=6)
ax1.set_ylim([0,2])

ax2.scatter(dates_arr[break_index+1:], wave_df_combined['Cp'][break_index+1:], s=s, color = 'dimgray', label = '$c_{p}$')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=5)
# ax2.legend(fontsize=7)
ax2.set_title('$c_{p}$ [m/s]', fontsize=6)
ax2.tick_params(axis='y', labelsize=6)
ax2.set_ylim([0,12])

ax3.scatter(dates_arr[break_index+1:], wave_df_combined['T_period'][break_index+1:], s=s, color = 'darkslategray', label = '$T$')
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax3.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center', fontsize=5)
# ax3.legend(fontsize=7)
ax3.set_title('$T$ [s]', fontsize=6)
ax3.tick_params(axis='y', labelsize=6)
ax3.set_ylim([0,10])

ax1.set_ylabel('$H_{sig}$ [m]', fontsize=6)
ax2.set_ylabel('$c_{p}$ [m/s]', fontsize=6)
ax3.set_ylabel('$T$ [s]', fontsize=6)


plt.show()


plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/plots/'
fig.savefig(plot_savePath + "timeseries_WaveData_Fall.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_WaveData_Fall.pdf")








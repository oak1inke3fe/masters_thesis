#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:32:10 2023

@author: oaklin keefe


This file is used to make timeseries plots of wind direction and wind speed for both deployments. The resulting
plots will go in the Results & Analysis section of the thesis.
NOTE: only sonic 3 is used for simplicity, and as a proxy for the heighest sonic since sonic 4 speed is often different

INPUT files:
    despiked_s3_turbulenceTerms_andMore_combined.csv
    windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv
    date_combinedAnalysis.csv
    
OUTPUT files:
    Only figures:
        

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


print('done with imports')
#%%

file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'

sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)

windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv")
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)

break_index = 3959

print('done')


#%% First, get rid of bad wind directions first

plt.figure()
plt.scatter(windDir_df.index, windDir_df['alpha_s4'], color='gray',label='before')
plt.title('Wind direction BEFORE mask')
plt.xlabel('index')
plt.ylabel('Wind direction [deg]')
plt.legend(loc='lower right')
plt.show()

windDir_index_array = np.arange(len(windDir_df))
windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, windDir_index_array)
mask_goodWindDir = np.isin(windDir_df['new_index_arr'],windDir_index_array)

windDir_df[mask_goodWindDir] = np.nan

sonic3_df[mask_goodWindDir] = np.nan


print('done with setting up  good wind direction only dataframes')

# plt.figure()
plt.scatter(windDir_df.index, windDir_df['alpha_s4'], color = 'green',label = 'after')
plt.title('Wind direction AFTER mask')
plt.xlabel('index')
plt.ylabel('Wind direction [deg]')
plt.legend(loc='lower right')
# plt.xlim(4000,5000)

#%% Then, pLot the timeseries

date_df = pd.read_csv(file_path+'date_combinedAnalysis.csv')
dates_arr = np.array(pd.to_datetime(date_df['datetime']))

# SPRING
fig, ax1 = plt.subplots(figsize=(8,3))
fig.suptitle('Wind Speed and Direction: Spring')
ax2 = ax1.twinx()
ax1.plot(dates_arr[:break_index+1], sonic3_df['Ubar'][:break_index+1], color = 'black', label = 'Wind Speed $\overline{u}$ [m/s]')
ax2.scatter(dates_arr[:break_index+1], windDir_df['alpha_s3'][:break_index+1], color = 'forestgreen', edgecolor = 'darkgreen', s=10, label = 'Wind Direction [º]')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax1.set_ylabel('Speed ($\overline{u}$) [m/s]', color='black')
ax2.set_ylabel('Direction [º]', color='forestgreen')
ax2.tick_params(colors='forestgreen', which='both')

plt.show()

fig.savefig(plot_savePath + "timeseries_UbarWindDir_Spring.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_UbarWindDir_Spring.pdf")



# FALL
fig, ax1 = plt.subplots(figsize=(8,3))
fig.suptitle('Wind Speed and Direction: Fall')
ax2 = ax1.twinx()
ax1.plot(dates_arr[break_index+1:], sonic3_df['Ubar'][break_index+1:], color = 'black', label = 'Wind Speed $\overline{u}$ [m/s]')
ax2.scatter(dates_arr[break_index+1:], windDir_df['alpha_s3'][break_index+1:], color = 'darkorange', edgecolor = 'darkorange', s=10, label = 'Wind Direction [º]')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=0, horizontalalignment='center')
ax1.set_ylabel('Speed ($\overline{u}$) [m/s]', color='black')
ax2.set_ylabel('Direction [º]', color='darkorange')
ax2.tick_params(colors='darkorange', which='both')

plt.show()

fig.savefig(plot_savePath + "timeseries_UbarWindDir_Fall.png",dpi=300)
fig.savefig(plot_savePath + "timeseries_UbarWindDir_Fall.pdf")
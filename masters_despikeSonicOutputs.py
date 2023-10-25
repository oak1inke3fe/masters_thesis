#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:02:34 2023

@author: oak
"""
#%%

import numpy as np
import pandas as pd
from hampel import hampel
import matplotlib.pyplot as plt

print('done with imports')

#%%
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'

sonic_file1 = "s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)
sonic1_df = sonic1_df.drop(['Unnamed: 0'], axis=1)
# print(sonic1_df.columns)
sonic1_df = sonic1_df.rename(columns={'Ubar_s1': 'Ubar', 
                                      'U_horiz_s1': 'U_horiz',
                                      'U_streamwise_s1':'U_streamwise',
                                      'Umedian_s1':'Umedian',
                                      'Tbar_s1':'Tbar',
                                      'Tmedian_s1':'Tmedian',
                                      'UpWp_bar_s1':'UpWp_bar',
                                      'VpWp_bar_s1':'VpWp_bar',
                                      'WpTp_bar_s1':'WpTp_bar',
                                      'WpEp_bar_s1':'WpEp_bar',
                                      'TKE_bar_s1':'TKE_bar'})

sonic_file2 = "s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)
sonic2_df = sonic2_df.drop(['Unnamed: 0'], axis=1)
sonic2_df = sonic2_df.rename(columns={'Ubar_s2': 'Ubar', 
                                      'U_horiz_s2': 'U_horiz',
                                      'U_streamwise_s2':'U_streamwise',
                                      'Umedian_s2':'Umedian',
                                      'Tbar_s2':'Tbar',
                                      'Tmedian_s2':'Tmedian',
                                      'UpWp_bar_s2':'UpWp_bar',
                                      'VpWp_bar_s2':'VpWp_bar',
                                      'WpTp_bar_s2':'WpTp_bar',
                                      'WpEp_bar_s2':'WpEp_bar',
                                      'TKE_bar_s2':'TKE_bar'})

sonic_file3 = "s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)
sonic3_df = sonic3_df.drop(['Unnamed: 0'], axis=1)
sonic3_df = sonic3_df.rename(columns={'Ubar_s3': 'Ubar', 
                                      'U_horiz_s3': 'U_horiz',
                                      'U_streamwise_s3':'U_streamwise',
                                      'Umedian_s3':'Umedian',
                                      'Tbar_s3':'Tbar',
                                      'Tmedian_s3':'Tmedian',
                                      'UpWp_bar_s3':'UpWp_bar',
                                      'VpWp_bar_s3':'VpWp_bar',
                                      'WpTp_bar_s3':'WpTp_bar',
                                      'WpEp_bar_s3':'WpEp_bar',
                                      'TKE_bar_s3':'TKE_bar'})

sonic_file4 = "s4_turbulenceTerms_andMore_combined.csv"
sonic4_df = pd.read_csv(file_path+sonic_file4)
sonic4_df = sonic4_df.drop(['Unnamed: 0'], axis=1)
sonic4_df = sonic4_df.rename(columns={'Ubar_s4': 'Ubar', 
                                      'U_horiz_s4': 'U_horiz',
                                      'U_streamwise_s4':'U_streamwise',
                                      'Umedian_s4':'Umedian',
                                      'Tbar_s4':'Tbar',
                                      'Tmedian_s4':'Tmedian',
                                      'UpWp_bar_s4':'UpWp_bar',
                                      'VpWp_bar_s4':'VpWp_bar',
                                      'WpTp_bar_s4':'WpTp_bar',
                                      'WpEp_bar_s4':'WpEp_bar',
                                      'TKE_bar_s4':'TKE_bar'})
print('done reading files and renaming columns')
#%%
break_index = 3959

sonic1_df_spring = sonic1_df[:break_index+1]
sonic1_df_spring = sonic1_df_spring.reset_index(drop = True)
sonic2_df_spring = sonic2_df[:break_index+1]
sonic2_df_spring = sonic2_df_spring.reset_index(drop = True)
sonic3_df_spring = sonic3_df[:break_index+1]
sonic3_df_spring = sonic3_df_spring.reset_index(drop = True)
sonic4_df_spring = sonic4_df[:break_index+1]
sonic4_df_spring = sonic4_df_spring.reset_index(drop = True)


sonics_df_arr_spring = [sonic1_df_spring,
                        sonic2_df_spring,
                        sonic3_df_spring,
                        sonic4_df_spring]

print('done with creating fall dataframes')

sonic1_df_despiked_spring = pd.DataFrame()
sonic2_df_despiked_spring = pd.DataFrame()
sonic3_df_despiked_spring = pd.DataFrame()
sonic4_df_despiked_spring = pd.DataFrame()
sonics_despike_arr_spring = [sonic1_df_despiked_spring, 
                             sonic2_df_despiked_spring, 
                             sonic3_df_despiked_spring, 
                             sonic4_df_despiked_spring]

column_arr = ['Ubar', 'U_horiz', 'U_streamwise', 'Umedian', 
              'Tbar', 'Tmedian', 
              'UpWp_bar', 'VpWp_bar', 'WpTp_bar', 'WpEp_bar',
              'TKE_bar']

for i in range(len(sonics_df_arr_spring)):
# for sonic in sonics_df_arr:
    for column_name in column_arr:
    
        my_array = sonics_df_arr_spring[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 10
        n = 3
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        sonics_despike_arr_spring[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('sonic '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2
    # L_coare_despike = L_despiked_2times
print('done SPRING')



#%%
sonic1_df_fall = sonic1_df[break_index+1:]
sonic1_df_fall = sonic1_df_fall.reset_index(drop = True)
sonic2_df_fall = sonic2_df[break_index+1:]
sonic2_df_fall = sonic2_df_fall.reset_index(drop = True)
sonic3_df_fall = sonic3_df[break_index+1:]
sonic3_df_fall = sonic3_df_fall.reset_index(drop = True)
sonic4_df_fall = sonic4_df[break_index+1:]
sonic4_df_fall = sonic4_df_fall.reset_index(drop = True)


sonics_df_arr_fall = [sonic1_df_fall, 
                      sonic2_df_fall,
                      sonic3_df_fall,
                      sonic4_df_fall]
print('done with creating fall dataframes')

sonic1_df_despiked_fall = pd.DataFrame()
sonic2_df_despiked_fall = pd.DataFrame()
sonic3_df_despiked_fall = pd.DataFrame()
sonic4_df_despiked_fall = pd.DataFrame()
sonics_despike_arr_fall = [sonic1_df_despiked_fall, 
                           sonic2_df_despiked_fall, 
                           sonic3_df_despiked_fall, 
                           sonic4_df_despiked_fall]
for i in range(len(sonics_df_arr_fall)):
# for sonic in sonics_df_arr:
    for column_name in column_arr:
    
        my_array = sonics_df_arr_fall[i][column_name]
        
        # Just outlier detection
        input_array = my_array
        window_size = 10
        n = 3
        
        my_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
        my_despiked_1times = my_outlier_in_Ts
        
        # plt.figure()
        # plt.plot(L_despiked_once)
    
        input_array2 = my_despiked_1times
        my_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
        # Outlier Imputation with rolling median
        my_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)
        sonics_despike_arr_fall[i][column_name] = my_outlier_in_Ts2
        print(column_name)
        print('sonic '+str(i+1))
        # L_despiked_2times = L_outlier_in_Ts2
    # L_coare_despike = L_despiked_2times
print('done FALL')

#%%
sonic1_df_despiked_spring.to_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_spring.csv')
sonic2_df_despiked_spring.to_csv(file_path + 'despiked_s2_turbulenceTerms_andMore_spring.csv')
sonic3_df_despiked_spring.to_csv(file_path + 'despiked_s3_turbulenceTerms_andMore_spring.csv')
sonic4_df_despiked_spring.to_csv(file_path + 'despiked_s4_turbulenceTerms_andMore_spring.csv')

sonic1_df_despiked_fall.to_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_fall.csv')
sonic2_df_despiked_fall.to_csv(file_path + 'despiked_s2_turbulenceTerms_andMore_fall.csv')
sonic3_df_despiked_fall.to_csv(file_path + 'despiked_s3_turbulenceTerms_andMore_fall.csv')
sonic4_df_despiked_fall.to_csv(file_path + 'despiked_s4_turbulenceTerms_andMore_fall.csv')

sonic1_df_despiked_combined = pd.concat([sonic1_df_despiked_spring,sonic1_df_despiked_fall], axis = 0)
sonic1_df_despiked_combined['new_index'] = np.arange(0, len(sonic1_df_despiked_combined))
sonic1_df_despiked_combined = sonic1_df_despiked_combined.set_index('new_index')
sonic2_df_despiked_combined = pd.concat([sonic2_df_despiked_spring,sonic2_df_despiked_fall], axis = 0)
sonic2_df_despiked_combined['new_index'] = np.arange(0, len(sonic2_df_despiked_combined))
sonic2_df_despiked_combined = sonic2_df_despiked_combined.set_index('new_index')
sonic3_df_despiked_combined = pd.concat([sonic3_df_despiked_spring,sonic3_df_despiked_fall], axis = 0)
sonic3_df_despiked_combined['new_index'] = np.arange(0, len(sonic3_df_despiked_combined))
sonic3_df_despiked_combined = sonic3_df_despiked_combined.set_index('new_index')
sonic4_df_despiked_combined = pd.concat([sonic4_df_despiked_spring,sonic4_df_despiked_fall], axis = 0)
sonic4_df_despiked_combined['new_index'] = np.arange(0, len(sonic4_df_despiked_combined))
sonic4_df_despiked_combined = sonic4_df_despiked_combined.set_index('new_index')

sonic1_df_despiked_combined.to_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_combined.csv')
sonic2_df_despiked_combined.to_csv(file_path + 'despiked_s2_turbulenceTerms_andMore_combined.csv')
sonic3_df_despiked_combined.to_csv(file_path + 'despiked_s3_turbulenceTerms_andMore_combined.csv')
sonic4_df_despiked_combined.to_csv(file_path + 'despiked_s4_turbulenceTerms_andMore_combined.csv')

print('done with saving despiked files')

#%%
plt.figure()
plt.plot(sonic1_df['UpWp_bar'], label = 'orig.', color = 'blue')
plt.plot(sonic1_df_despiked_combined['UpWp_bar'], label = 'despiked', color = 'navy')
plt.legend()
plt.title("sonic 1: Comparing original and Despiked <u'w'>")

plt.figure()
plt.plot(sonic2_df['UpWp_bar'], label = 'orig.', color = 'limegreen')
plt.plot(sonic2_df_despiked_combined['UpWp_bar'], label = 'despiked', color = 'darkgreen')
plt.legend()
plt.title("sonic 2: Comparing original and Despiked <u'w'>")

plt.figure()
plt.plot(sonic3_df['UpWp_bar'], label = 'orig.', color = 'orange')
plt.plot(sonic3_df_despiked_combined['UpWp_bar'], label = 'despiked', color = 'red')
plt.legend()
plt.title("sonic 3: Comparing original and Despiked <u'w'>")

plt.figure()
plt.plot(sonic4_df['UpWp_bar'], label = 'orig.', color = 'gray')
plt.plot(sonic4_df_despiked_combined['UpWp_bar'], label = 'despiked', color = 'black')
plt.legend()
plt.title("sonic 4: Comparing original and Despiked <u'w'>")

#%%
plt.figure()
plt.plot(sonic1_df['Ubar'], label = 'orig.', color = 'blue')
plt.plot(sonic1_df_despiked_combined['Ubar'], label = 'despiked', color = 'navy')
plt.legend()
plt.title("sonic 1: Comparing original and Despiked <u>")

plt.figure()
plt.plot(sonic2_df['Ubar'], label = 'orig.', color = 'limegreen')
plt.plot(sonic2_df_despiked_combined['Ubar'], label = 'despiked', color = 'darkgreen')
plt.legend()
plt.title("sonic 2: Comparing original and Despiked <u>")

plt.figure()
plt.plot(sonic3_df['Ubar'], label = 'orig.', color = 'orange')
plt.plot(sonic3_df_despiked_combined['Ubar'], label = 'despiked', color = 'red')
plt.legend()
plt.title("sonic 3: Comparing original and Despiked <u>")

plt.figure()
plt.plot(sonic4_df['Ubar'], label = 'orig.', color = 'gray')
plt.plot(sonic4_df_despiked_combined['Ubar'], label = 'despiked', color = 'black')
plt.legend()
plt.title("sonic 4: Comparing original and Despiked <u>")






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:37:44 2023

@author: oak
"""


#%%
import numpy as np
import pandas as pd
# from pandas import rolling_median
import os
import matplotlib.pyplot as plt
import natsort
import time
import datetime
import math
from numpy import trapz
from hampel import hampel

print('done with imports')
#%%
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
sonic1_df = pd.read_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_combined.csv')
sonic2_df = pd.read_csv(file_path + 'despiked_s2_turbulenceTerms_andMore_combined.csv')
sonic3_df = pd.read_csv(file_path + 'despiked_s3_turbulenceTerms_andMore_combined.csv')
sonic4_df = pd.read_csv(file_path + 'despiked_s4_turbulenceTerms_andMore_combined.csv')

z_df = pd.read_csv(file_path + 'z_air_side_combinedAnalysis.csv')

#%%
# z1 = np.ones(len(sonic1_df))*z_df['z_s1_avg'][1]
# z2 = np.ones(len(Tbar_df))*z_df['z_s2_avg'][1]
# z3 = np.ones(len(Tbar_df))*z_df['z_s3_avg'][1]
# z4 = np.ones(len(Tbar_df))*z_df['z_s4_avg'][1]

#make sure T is in Kelvin
Tbar_s1 = sonic1_df['Tbar']
Tbar_s2 = sonic2_df['Tbar']
Tbar_s3 = sonic3_df['Tbar']
Tbar_s4 = sonic4_df['Tbar']


# Tbar_s1 = Tbar_df['Tbar_s1']
# Tbar_s2 = Tbar_df['Tbar_s2']
# Tbar_s3 = Tbar_df['Tbar_s3']
# Tbar_s4 = Tbar_df['Tbar_s4']

WpTp_bar_s1 = sonic1_df['WpTp_bar']
WpTp_bar_s2 = sonic2_df['WpTp_bar']
WpTp_bar_s3 = sonic3_df['WpTp_bar']
WpTp_bar_s4 = sonic4_df['WpTp_bar']



# date_arr = np.array(Tbar_df['date'])

# WpTp_bar_Tbar = pd.DataFrame()
# WpTp_bar_Tbar['date'] = date_arr
# WpTp_bar_Tbar['WpTp_bar_Tbar_sonic1'] = WpTp_bar_s1/Tbar_s1
# WpTp_bar_Tbar['WpTp_bar_Tbar_sonic2'] = WpTp_bar_s2/Tbar_s2
# WpTp_bar_Tbar['WpTp_bar_Tbar_sonic3'] = WpTp_bar_s3/Tbar_s3
# WpTp_bar_Tbar['WpTp_bar_Tbar_sonic4'] = WpTp_bar_s4/Tbar_s4
# WpTp_bar_Tbar['z_s1'] = z1
# WpTp_bar_Tbar['z_s2'] = z2
# WpTp_bar_Tbar['z_s3'] = z3
# WpTp_bar_Tbar['z_s4'] = z4

# WpTp_bar_Tbar.to_csv(file_path+"WpTp_bar_Tbar_allFall.csv")

#%%
rho_df = pd.read_csv(file_path+"rhoAvg_CombinedAnalysis.csv")
rho_1 = rho_df['rho_bar_1']
rho_2 = rho_df['rho_bar_2']
rho_3 = rho_df['rho_bar_3']

g = 9.81
cp = 1004.67
# buoy_1 = g/(Tbar_s1)*WpTp_bar_df['WpTp_bar_s1'] / (rho_1*cp)
# buoy_2 = g/(Tbar_s2)*WpTp_bar_df['WpTp_bar_s2'] / (rho_2*cp)
# buoy_3 = g/(Tbar_s3)*WpTp_bar_df['WpTp_bar_s3'] / (rho_3*cp)
# buoy_4 = g/(Tbar_s4)*WpTp_bar_df['WpTp_bar_s4'] / (rho_3*cp)
buoy_1 = g/(Tbar_s1)*WpTp_bar_s1
buoy_2 = g/(Tbar_s2)*WpTp_bar_s2
buoy_3 = g/(Tbar_s3)*WpTp_bar_s3
buoy_4 = g/(Tbar_s4)*WpTp_bar_s4
buoyFlux_1 = WpTp_bar_s1*rho_1*cp
buoyFlux_2 = WpTp_bar_s2*rho_2*cp
buoyFlux_3 = WpTp_bar_s3*rho_3*cp
buoyFlux_4 = WpTp_bar_s4*rho_3*cp
 

#%%
buoy_df = pd.DataFrame()
# buoy_df['date'] = date_arr
buoy_df['buoy_1'] = buoy_1
buoy_df['buoy_2'] = buoy_2
buoy_df['buoy_3'] = buoy_3
buoy_df['buoy_4'] = buoy_4
buoy_df['buoy_I'] = (buoy_1+buoy_2)/2
buoy_df['buoy_II'] = (buoy_2+buoy_3)/2
buoy_df['buoy_III'] = (buoy_3+buoy_4)/2
buoy_df['buoyFlux_1'] =  buoyFlux_1
buoy_df['buoyFlux_2'] =  buoyFlux_2
buoy_df['buoyFlux_3'] =  buoyFlux_3
buoy_df['buoyFlux_4'] =  buoyFlux_4
buoy_df['buoyFlux_I'] = (buoyFlux_1+buoyFlux_2)/2
buoy_df['buoyFlux_II'] = (buoyFlux_2+buoyFlux_3)/2
buoy_df['buoyFlux_III'] = (buoyFlux_3+buoyFlux_4)/2

print('done with buoyancy')


buoy_df.to_csv(file_path+"buoy_terms_combinedAnalysis.csv")

print('done saving to .csv')
#%%
'''
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
prod_to_despike = pd.DataFrame() #creating a new df since the despike doesn't work on the dates column
prod_to_despike['prod_I'] = prod_I
prod_to_despike['prod_II'] = prod_II
prod_to_despike['prod_III'] = prod_III
prod_to_despike = despikeThis(prod_to_despike,3)

#%%
prod_to_despike['date'] = date_arr #put back in the date array
filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
prod_to_despike.to_csv(filepath+"New_prod_terms_DESPIKED_allFall.csv")
#%%
fig1 = plt.figure()
plt.plot(prod_to_despike['prod_III'],color = 'r', label = 'upper levels')
plt.plot(prod_to_despike['prod_II'],color = 'g', label = 'middle levels')
plt.plot(prod_to_despike['prod_I'], color = 'b', label='lower levels')
plt.legend()
plt.title('production')
plt.xlabel('time')
plt.ylabel('production m^2/s^3')
# plt.ylim(-5,0)
'''

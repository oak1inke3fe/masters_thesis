#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:45:16 2023

@author: oak


This file is used to calculate the 20min file averages of rho (density) based on 20min file averages of temperature observations from the met/port5 sensor,
and 20min file averages of pressure observations from the high-res paros p sensor.

INPUT files:
    metAvg_CombinedAnalysis.csv
    parosAvg_combinedAnalysis.csv
    R = 287.053 #[Pa K^-1 m^3 kg^-1] # R is the ideal gas constant
    
OUTPUT files:
    rhoAvg_CombinedAnalysis.csv
    
"""
#%% IMPORTS
import pandas as pd

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


print('done saving rho avg. file')

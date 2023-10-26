#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:07:04 2023

@author: oaklin keefe
This file is used to calculate ustar, (u*) also known as friction velocity.

INPUT files:
    despiked_s1_turbulenceTerms_andMore_combined.csv
    despiked_s2_turbulenceTerms_andMore_combined.csv
    despiked_s3_turbulenceTerms_andMore_combined.csv
    despiked_s4_turbulenceTerms_andMore_combined.csv
    
OUTPUT files:
    usr_combinedAnalysis.csv
        
"""
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('done with imports')

#%%
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'

sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)
sonic1_df = sonic1_df.drop(['new_index'], axis=1)
# print(sonic1_df.columns)


sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)
sonic2_df = sonic2_df.drop(['new_index'], axis=1)


sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)
sonic3_df = sonic3_df.drop(['new_index'], axis=1)


sonic_file4 = "despiked_s4_turbulenceTerms_andMore_combined.csv"
sonic4_df = pd.read_csv(file_path+sonic_file4)
sonic4_df = sonic4_df.drop(['new_index'], axis=1)

print('done reading in sonics')
#%%

# usr_s1_withRho = (1/rho_df['rho_bar_1'])*((sonic1_df_despiked['UpWp_bar'])**2+(sonic1_df_despiked['VpWp_bar'])**2)**(1/4)
usr_s1 = ((sonic1_df['UpWp_bar'])**2+(sonic1_df['VpWp_bar'])**2)**(1/4)

usr_s2 = ((sonic2_df['UpWp_bar'])**2+(sonic2_df['VpWp_bar'])**2)**(1/4)

usr_s3 = ((sonic3_df['UpWp_bar'])**2+(sonic3_df['VpWp_bar'])**2)**(1/4)

usr_s4 = ((sonic4_df['UpWp_bar'])**2+(sonic4_df['VpWp_bar'])**2)**(1/4)

USTAR_df = pd.DataFrame()
USTAR_df['usr_s1'] = np.array(usr_s1)
USTAR_df['usr_s2'] = np.array(usr_s2)
USTAR_df['usr_s3'] = np.array(usr_s3)
USTAR_df['usr_s4'] = np.array(usr_s4)
USTAR_df.to_csv(file_path + 'usr_combinedAnalysis.csv')

plt.figure()
plt.plot(usr_s1, label = "u*_{s1} = $(<u'w'>^{2} + <v'w'>^{2})^{1/4}$")
plt.legend()
plt.title('U*')


print('done with calculaitng ustar and plotting it.')


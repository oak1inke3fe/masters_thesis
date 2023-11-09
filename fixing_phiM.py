#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:38:58 2023

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


sonic1_df = pd.read_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_combined.csv')
sonic2_df = pd.read_csv(file_path + 'despiked_s2_turbulenceTerms_andMore_combined.csv')
sonic3_df = pd.read_csv(file_path + 'despiked_s3_turbulenceTerms_andMore_combined.csv')
sonic4_df = pd.read_csv(file_path + 'despiked_s4_turbulenceTerms_andMore_combined.csv')

plt.figure()
plt.plot(np.array(sonic3_df['Ubar'])[1500:3000], label = 's3')
plt.plot(np.array(sonic4_df['Ubar'])[1500:3000], label = 's4')
plt.plot(np.array(sonic4_df['Ubar'])[1500:3000]-np.array(sonic3_df['Ubar'])[1500:3000], label = 's4-s3')
plt.legend()

plt.figure()
plt.plot(np.array(sonic3_df['Ubar'][:break_index+1])-np.array(sonic2_df['Ubar'][:break_index+1]), label = 's3-s2', color = 'darkorange')
# plt.plot(np.array(sonic4_df['Ubar'][:break_index+1])-np.array(sonic3_df['Ubar'][:break_index+1]), label = 's4-s3', color = 'blue')
# plt.plot(np.array(sonic2_df['Ubar'][:break_index+1])-np.array(sonic1_df['Ubar'][:break_index+1]), label = 's2-s1', color = 'green')
plt.hlines(y=0,xmin=0,xmax=break_index+1, color = 'k')
plt.ylim(-2,5)
# plt.xlim(2000,2200)
plt.legend()



z_df_spring = pd.read_csv(file_path + "z_airSide_allSpring.csv")
z_df_spring = z_df_spring.drop(['Unnamed: 0'], axis=1)

z_df_fall = pd.read_csv(file_path + "z_airSide_allFall.csv")
z_df_fall = z_df_fall.drop(['Unnamed: 0'], axis=1)

z_df = pd.concat([z_df_spring, z_df_fall], axis = 0)
print('done with z concat')


Eps_df = pd.read_csv(file_path+"epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv")
Eps_df = Eps_df.drop(['Unnamed: 0'], axis=1)


# tke_transport_df = pd.read_csv(file_path + "tke_transport_allFall.csv")
# tke_transport_df = tke_transport_df.drop(['Unnamed: 0'], axis=1)

# windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_combinedAnalysis.csv")
windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_100to130_wCameraFlags_combinedAnalysis.csv")
# windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_120to196.csv")
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)


zL_df = pd.read_csv(file_path+'ZoverL_combinedAnalysis.csv')
zL_df = zL_df.drop(['Unnamed: 0'], axis=1)

rho_df = pd.read_csv(file_path + 'rhoAvg_combinedAnalysis.csv' )
rho_df = rho_df.drop(['Unnamed: 0'], axis=1)

# usr_coare_df = pd.read_csv(file_path+"usr_coare_allFall.csv")
# usr_coare_df = usr_coare_df.drop(['Unnamed: 0'], axis=1)

# wave_df = pd.read_csv(file_path + "waveData_allFall.csv")
# wave_df = wave_df.drop(['Unnamed: 0'], axis=1)

buoy_df = pd.read_csv(file_path+'buoy_terms_combinedAnalysis.csv')
buoy_df = buoy_df.drop(['Unnamed: 0'], axis=1)

oct_storm = False
#%%
# #for october 2-4 wind event
# oct_storm = True
# windDir_df['octStorm_index_arr'] = np.arange(len(windDir_df))
# windDir_df['octStorm_new_index_arr'] = np.where((windDir_df['octStorm_index_arr']>=704)&(windDir_df['octStorm_index_arr']<=886), np.nan, windDir_df['octStorm_index_arr'])

# mask_Oct_Storm = np.isin(windDir_df['octStorm_new_index_arr'],windDir_df['octStorm_index_arr'])

# windDir_df[mask_Oct_Storm] = np.nan

# prod_df[mask_Oct_Storm] = np.nan
# Eps_df[mask_Oct_Storm] = np.nan
# sonic1_df[mask_Oct_Storm] = np.nan
# sonic2_df[mask_Oct_Storm] = np.nan
# sonic3_df[mask_Oct_Storm] = np.nan
# sonic4_df[mask_Oct_Storm] = np.nan
# tke_transport_df[mask_Oct_Storm] = np.nan
# z_df[mask_Oct_Storm] = np.nan
# zL_df[mask_Oct_Storm] = np.nan
# rho_df[mask_Oct_Storm] = np.nan
# wave_df[mask_Oct_Storm] = np.nan

# print('done with setting up  october 2-4 wind event dataframes')
#%% Get rid of bad wind directions first
all_windDirs = True
onshore = False
offshore = False

plt.figure()
plt.scatter(np.arange(len(windDir_df)),windDir_df['alpha_s4'])
plt.title('wind dir (s4) before mask')

index_array = np.arange(len(windDir_df))
# windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, index_array)
# windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir_springCameras'])==True, np.nan, index_array)
windDir_df['new_index_arr'] = np.where((windDir_df['windDir_final'])==True, np.nan, index_array)
mask_goodWindDir = np.isin(windDir_df['new_index_arr'],index_array)

windDir_df[mask_goodWindDir] = np.nan
plt.figure()
plt.scatter(np.arange(len(windDir_df[:break_index+1])),windDir_df['alpha_s1'][:break_index+1])
plt.scatter(np.arange(len(windDir_df[:break_index+1])),windDir_df['alpha_s4'][:break_index+1], color = 'g')
# plt.xlim(1340,1400)
# plt.vlines(x=1230,ymin=0,ymax=360)
# plt.vlines(x=1260,ymin=0,ymax=360)
# plt.vlines(x=1278,ymin=0,ymax=360)
plt.title('wind dir (s4) after mask')

prod_df[mask_goodWindDir] = np.nan
Eps_df[mask_goodWindDir] = np.nan
sonic1_df[mask_goodWindDir] = np.nan
sonic2_df[mask_goodWindDir] = np.nan
sonic3_df[mask_goodWindDir] = np.nan
sonic4_df[mask_goodWindDir] = np.nan
# tke_transport_df[mask_goodWindDir] = np.nan
z_df[mask_goodWindDir] = np.nan
zL_df[mask_goodWindDir] = np.nan
rho_df[mask_goodWindDir] = np.nan
# wave_df[mask_goodWindDir] = np.nan

print('done with setting up  good wind direction only dataframes')

plt.figure()
plt.plot(np.array(sonic3_df['Ubar'][:break_index+1])-np.array(sonic2_df['Ubar'][:break_index+1]), label = 's3-s2', color = 'darkorange')
plt.plot(np.array(sonic4_df['Ubar'][:break_index+1])-np.array(sonic3_df['Ubar'][:break_index+1]), label = 's4-s3', color = 'blue')
# plt.plot(np.array(sonic2_df['Ubar'][:break_index+1])-np.array(sonic1_df['Ubar'][:break_index+1]), label = 's2-s1', color = 'green')
plt.scatter(np.arange(break_index+1), windDir_df['alpha_s1'][:break_index+1], color = 'gray')
plt.hlines(y=0,xmin=0,xmax=break_index+1, color = 'k')
plt.ylim(-2,5)
plt.xlim(1340,1400)
# plt.vlines(x=1230,ymin=-2,ymax=5)
# plt.vlines(x=1260,ymin=-2,ymax=5)
# plt.vlines(x=1278,ymin=-2,ymax=5)
plt.legend()




#%%
# If we just want to examine the high wind event from Oct2-4, 2022, use the following mask:
    
# blank_index = np.arange(0,4395)

# prod_df['index_num'] = blank_index
# Eps_df['index_num'] = blank_index
# sonic1_df['index_num'] = blank_index
# sonic2_df['index_num'] = blank_index
# sonic3_df['index_num'] = blank_index
# sonic4_df['index_num'] = blank_index
# tke_transport_df['index_num'] = blank_index
# zL_df['index_num'] = blank_index
# z_df['index_num'] = blank_index
# rho_df['index_num'] = blank_index
# wave_df['index_num'] = blank_index
# windDir_df['index_num'] = blank_index

# oct_start = 731+27
# oct_end = 913+27

# mask_prod = (prod_df['index_num'] >= oct_start) & (prod_df['index_num'] <= oct_end)
# prod_df = prod_df.loc[mask_prod]
# Eps_df = Eps_df.loc[mask_prod]
# buoy_df = buoy_df.loc[mask_prod]
# sonic1_df = sonic1_df.loc[mask_prod]
# sonic2_df = sonic2_df.loc[mask_prod]
# sonic3_df = sonic3_df.loc[mask_prod]
# sonic4_df = sonic4_df.loc[mask_prod]
# tke_transport_df = tke_transport_df.loc[mask_prod]
# zL_df = zL_df.loc[mask_prod]
# z_df = z_df.loc[mask_prod]
# rho_df = rho_df.loc[mask_prod]
# wave_df = wave_df.loc[mask_prod]
# windDir_df = windDir_df.loc[mask_prod]

# # mask_buoy = (buoy_df['index_num'] >= oct_start) & (buoy_df['index_num'] <= oct_end)
# # buoy_df = buoy_df.loc[mask_buoy]



#%% Offshore setting
# all_windDirs = False
# onshore = False
# offshore = True

# windDir_df['offshore_index_arr'] = np.arange(len(windDir_df))
# windDir_df['new_offshore_index_arr'] = np.where((windDir_df['alpha_s4'] >= 270)&(windDir_df['alpha_s4'] <= 359), windDir_df['offshore_index_arr'], np.nan)

# mask_offshoreWindDir = np.isin(windDir_df['new_offshore_index_arr'],windDir_df['offshore_index_arr'])
# windDir_df = windDir_df[mask_offshoreWindDir]

# prod_df[mask_offshoreWindDir] = np.nan
# Eps_df[mask_offshoreWindDir] = np.nan
# sonic1_df[mask_offshoreWindDir] = np.nan
# sonic2_df[mask_offshoreWindDir] = np.nan
# sonic3_df[mask_offshoreWindDir] = np.nan
# sonic4_df[mask_offshoreWindDir] = np.nan
# tke_transport_df[mask_offshoreWindDir] = np.nan
# z_df[mask_offshoreWindDir] = np.nan
# zL_df[mask_offshoreWindDir] = np.nan
# wave_df[mask_offshoreWindDir] = np.nan

#%% On-shore setting
# all_windDirs = False
# onshore = True
# offshore = False

# windDir_df['onshore_index_arr'] = np.arange(len(windDir_df))
# windDir_df['new_onshore_index_arr'] = np.where((windDir_df['alpha_s4'] >= 197)&(windDir_df['alpha_s4'] <= 269), windDir_df['onshore_index_arr'], np.nan)

# mask_onshoreWindDir = np.isin(windDir_df['new_onshore_index_arr'],windDir_df['onshore_index_arr'])
# windDir_df = windDir_df[mask_onshoreWindDir]

# prod_df[mask_onshoreWindDir] = np.nan
# Eps_df[mask_onshoreWindDir] = np.nan
# sonic1_df[mask_onshoreWindDir] = np.nan
# sonic2_df[mask_onshoreWindDir] = np.nan
# sonic3_df[mask_onshoreWindDir] = np.nan
# sonic4_df[mask_onshoreWindDir] = np.nan
# tke_transport_df[mask_onshoreWindDir] = np.nan
# z_df[mask_onshoreWindDir] = np.nan
# zL_df[mask_onshoreWindDir] = np.nan
# wave_df[mask_onshoreWindDir] = np.nan



#%%
# figure titles based on wind directions
if all_windDirs == True:
    title_windDir = 'All wind directions; '
    wave_df_save_name = '_allGoodWindDirs_'
elif onshore == True:
    title_windDir = 'Onshore winds; '
    wave_df_save_name = '_onshoreWindDirs_'
elif offshore == True:
    title_windDir = 'Offshore winds; '
    wave_df_save_name = '_offshoreWindDirs_'
if oct_storm == True:
    oct_addition = "Oct 2-4: "
else:
    oct_addition = ''

#%%
usr_df = pd.read_csv(file_path + "usr_combinedAnalysis.csv")
usr_s1 = usr_df['usr_s1']
usr_s2 = usr_df['usr_s2']
usr_s3 = usr_df['usr_s3']
usr_s4 = usr_df['usr_s4']

usr_LI = np.array(usr_s1+usr_s2)/2
Tbar_LI = np.array(sonic1_df['Tbar']+sonic2_df['Tbar'])/2
WpTp_bar_LI = -1*(np.array(sonic1_df['WpTp_bar']+sonic2_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

usr_LII = np.array(usr_s2+usr_s3)/2
Tbar_LII = np.array(sonic2_df['Tbar']+sonic3_df['Tbar'])/2
WpTp_bar_LII = -1*(np.array(sonic2_df['WpTp_bar']+sonic3_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''

usr_LIII = np.array(usr_s3+usr_s4)/2
Tbar_LIII = np.array(sonic3_df['Tbar']+sonic4_df['Tbar'])/2
WpTp_bar_LIII = -1*(np.array(sonic3_df['WpTp_bar']+sonic4_df['WpTp_bar'])/2)
'''
NOTE: because coare defines their positive fluxes opposite of us, we'll multiply -1*<w'T'> so that the L's match up
'''
usr_dc_df = pd.DataFrame()
usr_dc_df['usr_s1'] = usr_s1
usr_dc_df['usr_s2'] = usr_s2
usr_dc_df['usr_s3'] = usr_s3
usr_dc_df['usr_s4'] = usr_s4
usr_dc_df['usr_LI'] = usr_LI
usr_dc_df['usr_LII'] = usr_LII
usr_dc_df['usr_LIII'] = usr_LIII

plt.figure()
plt.plot(usr_dc_df['usr_LIII'][break_index+1:], color = 'orange', label = 'fall')
plt.plot(usr_dc_df['usr_LIII'][:break_index+1], color = 'green', label = 'spring')

# usr_dc_df.to_csv(file_path+"usr_dc_allFall.csv")
dz_LI_spring = 2.695  #sonic 2- sonic 1: spring APRIL 2022 deployment
dz_LII_spring = 2.795 #sonic 3- sonic 2: spring APRIL 2022 deployment
dz_LIII_spring = 2.415 #sonic 4- sonic 3: spring APRIL 2022 deployment
dz_LI_fall = 1.8161  #sonic 2- sonic 1: FALL SEPT 2022 deployment
dz_LII_fall = 3.2131 #sonic 3- sonic 2: FALL SEPT 2022 deployment
dz_LIII_fall = 2.468 #sonic 4- sonic 3: FALL SEPT 2022 deployment


z_LI_spring = z_df_spring['z_sonic1']+(0.5*dz_LI_spring)
z_LII_spring  = z_df_spring['z_sonic2']+(0.5*dz_LII_spring)
z_LIII_spring  = z_df_spring['z_sonic3']+(0.5*dz_LIII_spring)

z_LI_fall = z_df_fall['z_sonic1']+(0.5*dz_LI_fall)
z_LII_fall = z_df_fall['z_sonic2']+(0.5*dz_LII_fall)
z_LIII_fall = z_df_fall['z_sonic3']+(0.5*dz_LIII_fall)

z_LI = np.concatenate([z_LI_spring, z_LI_fall], axis = 0)
z_LII = np.concatenate([z_LII_spring, z_LII_fall], axis = 0)
z_LIII = np.concatenate([z_LIII_spring, z_LIII_fall], axis = 0)


#%%
dUbardz_LI_spring = np.array(sonic2_df['Ubar'][:break_index+1]-sonic1_df['Ubar'][:break_index+1])/dz_LI_spring 
dUbardz_LII_spring  = np.array(sonic3_df['Ubar'][:break_index+1]-sonic2_df['Ubar'][:break_index+1])/dz_LII_spring 

dUbardz_LIII_spring  = np.array(((sonic4_df['Ubar'][:break_index+1])*1.0)-sonic3_df['Ubar'][:break_index+1])/dz_LIII_spring 
plt.figure()
plt.plot(dUbardz_LIII_spring, color = 'green')
plt.ylim(-0.25, 0.25)
plt.hlines(y=0, xmin=0, xmax = 3959, color = 'k')
plt.title('dUbardz_LIII_spring: s4*1')

dUbardz_LIII_spring  = np.array(((sonic4_df['Ubar'][:break_index+1])*0.985)-sonic3_df['Ubar'][:break_index+1])/dz_LIII_spring 
plt.figure()
plt.plot(dUbardz_LIII_spring, color = 'green')
plt.ylim(-0.25, 0.25)
plt.hlines(y=0, xmin=0, xmax = 3959, color = 'k')
plt.title('dUbardz_LIII_spring: s4*0.985')

# dUbardz_LIII_spring  = np.array(((sonic4_df['Ubar'][:break_index+1])*1.010)-sonic3_df['Ubar'][:break_index+1])/dz_LIII_spring 
# plt.figure()
# plt.plot(dUbardz_LIII_spring, color = 'green')
# plt.ylim(-0.25, 0.25)
# plt.hlines(y=0, xmin=0, xmax = 3959, color = 'k')
# plt.title('dUbardz_LIII_spring: s4* 1.01')
#%%

dUbardz_LI_fall = np.array(sonic2_df['Ubar'][break_index+1:]-sonic1_df['Ubar'][break_index+1:])/dz_LI_fall
dUbardz_LII_fall = np.array(sonic3_df['Ubar'][break_index+1:]-sonic2_df['Ubar'][break_index+1:])/dz_LII_fall

dUbardz_LIII_fall = np.array(((sonic4_df['Ubar'][break_index+1:])*1.0)-sonic3_df['Ubar'][break_index+1:])/dz_LIII_fall
plt.figure()
plt.plot(dUbardz_LIII_fall, color = 'orange')
plt.ylim(-0.25, 0.25)
plt.hlines(y=0, xmin=0, xmax = 4320, color = 'k')
plt.title('dUbardz_LIII_fall: s4*1')

dUbardz_LIII_fall = np.array(((sonic4_df['Ubar'][break_index+1:])*0.9850)-sonic3_df['Ubar'][break_index+1:])/dz_LIII_fall
plt.figure()
plt.plot(dUbardz_LIII_fall, color = 'orange')
plt.ylim(-0.25, 0.25)
plt.hlines(y=0, xmin=0, xmax = 4320, color = 'k')
plt.title('dUbardz_LIII_fall: s4*0.985')

# dUbardz_LIII_fall = np.array(((sonic4_df['Ubar'][break_index+1:])*1.01)-sonic3_df['Ubar'][break_index+1:])/dz_LIII_fall
# plt.figure()
# plt.plot(dUbardz_LIII_fall, color = 'orange')
# plt.ylim(-0.25, 0.25)
# plt.hlines(y=0, xmin=0, xmax = 4320, color = 'k')
# plt.title('dUbardz_LIII_fall: s4* 1.01')

#%%
dUbardz_LI = np.concatenate([dUbardz_LI_spring, dUbardz_LI_fall], axis = 0)
dUbardz_LII = np.concatenate([dUbardz_LII_spring, dUbardz_LII_fall], axis = 0)
dUbardz_LIII = np.concatenate([dUbardz_LIII_spring, dUbardz_LIII_fall], axis = 0)

print('done with ustar, z/L, and dUbardz')
######################################################################
######################################################################
"""
NOW WE ARE MOVING ON TO PHI_M CALCULATIONS
"""
#%%

phi_m_I_dc_spring = kappa*np.array(z_LI_spring)/(np.array(usr_LI[:break_index+1]))*(np.array(dUbardz_LI_spring))
phi_m_II_dc_spring = kappa*np.array(z_LII_spring)/(np.array(usr_LII[:break_index+1]))*(np.array(dUbardz_LII_spring))
phi_m_III_dc_spring = kappa*np.array(z_LIII_spring)/(np.array(usr_LIII[:break_index+1]))*(np.array(dUbardz_LIII_spring))
plt.figure()
# plt.plot(z_LIII_spring, label = 'z_L')
plt.plot(dUbardz_LIII_spring, label = 'dUbardz_LIII_spring')
plt.legend()
#%%

phi_m_dc_df_spring = pd.DataFrame()
phi_m_dc_df_spring['z/L I'] = zL_df['zL_I_dc'][:break_index+1]
phi_m_dc_df_spring['z/L II'] = zL_df['zL_II_dc'][:break_index+1]
phi_m_dc_df_spring['z/L III'] = zL_df['zL_III_dc'][:break_index+1]
# phi_m_dc_df['z/L I'] = zL_df['z/L I coare']
# phi_m_dc_df['z/L II'] = zL_df['z/L II coare']
# phi_m_dc_df['z/L III'] = zL_df['z/L III coare']
phi_m_dc_df_spring['phi_m I'] = phi_m_I_dc_spring
phi_m_dc_df_spring['phi_m II'] = phi_m_II_dc_spring
phi_m_dc_df_spring['phi_m III'] = phi_m_III_dc_spring

print('done with writing phi_m SPRING via D.C. method')
print('done at line 362')

phi_m_I_dc_df_spring = pd.DataFrame()
phi_m_I_dc_df_spring['z/L'] = zL_df['zL_I_dc'][:break_index+1]
# phi_m_I_dc_df['z/L'] = zL_df['z/L I coare']
phi_m_I_dc_df_spring['phi_m'] = phi_m_I_dc_spring
#get rid of negative shear values
phi_m_I_dc_df_spring['phi_m_pos'] = np.where(phi_m_I_dc_df_spring['phi_m']>=0,phi_m_I_dc_df_spring['phi_m'],np.nan)
mask_phi_m_I_dc_df_spring = np.isin(phi_m_I_dc_df_spring['phi_m_pos'], phi_m_I_dc_df_spring['phi_m'])
phi_m_I_dc_df_final_spring = phi_m_I_dc_df_spring[mask_phi_m_I_dc_df_spring]

phi_m_I_dc_df_final_spring = phi_m_I_dc_df_final_spring.sort_values(by='z/L')
phi_m_I_dc_neg_spring = phi_m_I_dc_df_final_spring.loc[phi_m_I_dc_df_final_spring['z/L']<=0]
phi_m_I_dc_pos_spring = phi_m_I_dc_df_final_spring.loc[phi_m_I_dc_df_final_spring['z/L']>=0]

phi_m_II_dc_df_spring = pd.DataFrame()
phi_m_II_dc_df_spring['z/L'] = zL_df['zL_II_dc'][:break_index+1]
# phi_m_II_dc_df['z/L'] = zL_df['z/L II coare']
phi_m_II_dc_df_spring['phi_m'] = phi_m_II_dc_spring
#get rid of negative shear values
phi_m_II_dc_df_spring['phi_m_pos'] = np.where(phi_m_II_dc_df_spring['phi_m']>=0,phi_m_II_dc_df_spring['phi_m'],np.nan)
mask_phi_m_II_dc_df_spring = np.isin(phi_m_II_dc_df_spring['phi_m_pos'], phi_m_II_dc_df_spring['phi_m'])
phi_m_II_dc_df_final_spring = phi_m_II_dc_df_spring[mask_phi_m_II_dc_df_spring]

phi_m_II_dc_df_final_spring = phi_m_II_dc_df_final_spring.sort_values(by='z/L')
phi_m_II_dc_neg_spring = phi_m_II_dc_df_final_spring.loc[phi_m_II_dc_df_spring['z/L']<=0]
phi_m_II_dc_pos_spring = phi_m_II_dc_df_final_spring.loc[phi_m_II_dc_df_spring['z/L']>=0]

phi_m_III_dc_df_spring = pd.DataFrame()
phi_m_III_dc_df_spring['z/L'] = zL_df['zL_III_dc'][:break_index+1]
# phi_m_III_dc_df['z/L'] = zL_df['z/L III coare']
phi_m_III_dc_df_spring['phi_m'] = phi_m_III_dc_spring
#get rid of negative shear values
phi_m_III_dc_df_spring['phi_m_pos'] = np.where(phi_m_III_dc_df_spring['phi_m']>=0,phi_m_III_dc_df_spring['phi_m'],np.nan)
mask_phi_m_III_dc_df_spring = np.isin(phi_m_III_dc_df_spring['phi_m_pos'], phi_m_III_dc_df_spring['phi_m'])
phi_m_III_dc_df_final_spring = phi_m_III_dc_df_spring[mask_phi_m_III_dc_df_spring]

phi_m_III_dc_df_final_spring = phi_m_III_dc_df_final_spring.sort_values(by='z/L')
phi_m_III_dc_neg_spring = phi_m_III_dc_df_final_spring.loc[phi_m_III_dc_df_spring['z/L']<=0]
phi_m_III_dc_pos_spring = phi_m_III_dc_df_final_spring.loc[phi_m_III_dc_df_spring['z/L']>=0]
#%%

phi_m_I_dc_fall = kappa*np.array(z_LI_fall)/(np.array(usr_LI[break_index+1:]))*(np.array(dUbardz_LI_fall))
phi_m_II_dc_fall = kappa*np.array(z_LII_fall)/(np.array(usr_LII[break_index+1:]))*(np.array(dUbardz_LII_fall))
phi_m_III_dc_fall = kappa*np.array(z_LIII_fall)/(np.array(usr_LIII[break_index+1:]))*(np.array(dUbardz_LIII_fall))

phi_m_dc_df_fall = pd.DataFrame()
phi_m_dc_df_fall['z/L I'] = zL_df['zL_I_dc'][break_index+1:]
phi_m_dc_df_fall['z/L II'] = zL_df['zL_II_dc'][break_index+1:]
phi_m_dc_df_fall['z/L III'] = zL_df['zL_III_dc'][break_index+1:]
# # phi_m_dc_df['z/L I'] = zL_df['z/L I coare']
# # phi_m_dc_df['z/L II'] = zL_df['z/L II coare']
# # phi_m_dc_df['z/L III'] = zL_df['z/L III coare']
phi_m_dc_df_fall['phi_m I'] = phi_m_I_dc_fall
phi_m_dc_df_fall['phi_m II'] = phi_m_II_dc_fall
phi_m_dc_df_fall['phi_m III'] = phi_m_III_dc_fall

print('done with writing phi_m FALL via D.C. method')
print('done at line 420')

phi_m_I_dc_df_fall = pd.DataFrame()
phi_m_I_dc_df_fall['z/L'] = zL_df['zL_I_dc'][break_index+1:]
# phi_m_I_dc_df['z/L'] = zL_df['z/L I coare']
phi_m_I_dc_df_fall['phi_m'] = phi_m_I_dc_fall
#get rid of negative shear values
phi_m_I_dc_df_fall['phi_m_pos'] = np.where(phi_m_I_dc_df_fall['phi_m']>=0,phi_m_I_dc_df_fall['phi_m'],np.nan)
mask_phi_m_I_dc_df_fall = np.isin(phi_m_I_dc_df_fall['phi_m_pos'], phi_m_I_dc_df_fall['phi_m'])
phi_m_I_dc_df_final_fall = phi_m_I_dc_df_fall[mask_phi_m_I_dc_df_fall]

phi_m_I_dc_df_final_fall = phi_m_I_dc_df_final_fall.sort_values(by='z/L')
phi_m_I_dc_neg_fall = phi_m_I_dc_df_final_fall.loc[phi_m_I_dc_df_final_fall['z/L']<=0]
phi_m_I_dc_pos_fall = phi_m_I_dc_df_final_fall.loc[phi_m_I_dc_df_final_fall['z/L']>=0]

phi_m_II_dc_df_fall = pd.DataFrame()
phi_m_II_dc_df_fall['z/L'] = zL_df['zL_II_dc'][break_index+1:]
# phi_m_II_dc_df['z/L'] = zL_df['z/L II coare']
phi_m_II_dc_df_fall['phi_m'] = phi_m_II_dc_fall
#get rid of negative shear values
phi_m_II_dc_df_fall['phi_m_pos'] = np.where(phi_m_II_dc_df_fall['phi_m']>=0,phi_m_II_dc_df_fall['phi_m'],np.nan)
mask_phi_m_II_dc_df_fall = np.isin(phi_m_II_dc_df_fall['phi_m_pos'], phi_m_II_dc_df_fall['phi_m'])
phi_m_II_dc_df_final_fall = phi_m_II_dc_df_fall[mask_phi_m_II_dc_df_fall]

phi_m_II_dc_df_final_fall = phi_m_II_dc_df_final_fall.sort_values(by='z/L')
phi_m_II_dc_neg_fall = phi_m_II_dc_df_final_fall.loc[phi_m_II_dc_df_fall['z/L']<=0]
phi_m_II_dc_pos_fall = phi_m_II_dc_df_final_fall.loc[phi_m_II_dc_df_fall['z/L']>=0]

phi_m_III_dc_df_fall = pd.DataFrame()
phi_m_III_dc_df_fall['z/L'] = zL_df['zL_III_dc'][break_index+1:]
# phi_m_III_dc_df['z/L'] = zL_df['z/L III coare']
phi_m_III_dc_df_fall['phi_m'] = phi_m_III_dc_fall
plt.figure()
plt.plot(phi_m_III_dc_fall, color = 'orange', label = 'fall')
plt.plot(phi_m_III_dc_spring, color = 'green', label = 'spring')
#get rid of negative shear values
phi_m_III_dc_df_fall['phi_m_pos'] = np.where(phi_m_III_dc_df_fall['phi_m']>=0,phi_m_III_dc_df_fall['phi_m'],np.nan)
mask_phi_m_III_dc_df_fall = np.isin(phi_m_III_dc_df_fall['phi_m_pos'], phi_m_III_dc_df_fall['phi_m'])
phi_m_III_dc_df_final_fall = phi_m_III_dc_df_fall[mask_phi_m_III_dc_df_fall]

phi_m_III_dc_df_final_fall = phi_m_III_dc_df_final_fall.sort_values(by='z/L')
phi_m_III_dc_neg_fall = phi_m_III_dc_df_final_fall.loc[phi_m_III_dc_df_fall['z/L']<=0]
phi_m_III_dc_pos_fall = phi_m_III_dc_df_final_fall.loc[phi_m_III_dc_df_fall['z/L']>=0]
#%%%

phi_m_I_dc = kappa*np.array(z_LI)/(np.array(usr_LI))*(np.array(dUbardz_LI))
phi_m_II_dc = kappa*np.array(z_LII)/(np.array(usr_LII))*(np.array(dUbardz_LII))
phi_m_III_dc = kappa*np.array(z_LIII)/(np.array(usr_LIII))*(np.array(dUbardz_LIII))

phi_m_dc_df = pd.DataFrame()
phi_m_dc_df['z/L I'] = zL_df['zL_I_dc']
phi_m_dc_df['z/L II'] = zL_df['zL_II_dc']
phi_m_dc_df['z/L III'] = zL_df['zL_III_dc']
# phi_m_dc_df['z/L I'] = zL_df['z/L I coare']
# phi_m_dc_df['z/L II'] = zL_df['z/L II coare']
# phi_m_dc_df['z/L III'] = zL_df['z/L III coare']
phi_m_dc_df['phi_m I'] = phi_m_I_dc
phi_m_dc_df['phi_m II'] = phi_m_II_dc
phi_m_dc_df['phi_m III'] = phi_m_III_dc

print('done with writing phi_m via D.C. method')
print('done at line 478')

phi_m_I_dc_df = pd.DataFrame()
phi_m_I_dc_df['z/L'] = zL_df['zL_I_dc']
# phi_m_I_dc_df['z/L'] = zL_df['z/L I coare']
phi_m_I_dc_df['phi_m'] = phi_m_I_dc
#get rid of negative shear values
phi_m_I_dc_df['phi_m_pos'] = np.where(phi_m_I_dc_df['phi_m']>=0,phi_m_I_dc_df['phi_m'],np.nan)
mask_phi_m_I_dc_df = np.isin(phi_m_I_dc_df['phi_m_pos'], phi_m_I_dc_df['phi_m'])
phi_m_I_dc_df_final = phi_m_I_dc_df[mask_phi_m_I_dc_df]

phi_m_I_dc_df_final = phi_m_I_dc_df_final.sort_values(by='z/L')
phi_m_I_dc_neg = phi_m_I_dc_df_final.loc[phi_m_I_dc_df_final['z/L']<=0]
phi_m_I_dc_pos = phi_m_I_dc_df_final.loc[phi_m_I_dc_df_final['z/L']>=0]

phi_m_II_dc_df = pd.DataFrame()
phi_m_II_dc_df['z/L'] = zL_df['zL_II_dc']
# phi_m_II_dc_df['z/L'] = zL_df['z/L II coare']
phi_m_II_dc_df['phi_m'] = phi_m_II_dc
#get rid of negative shear values
phi_m_II_dc_df['phi_m_pos'] = np.where(phi_m_II_dc_df['phi_m']>=0,phi_m_II_dc_df['phi_m'],np.nan)
mask_phi_m_II_dc_df = np.isin(phi_m_II_dc_df['phi_m_pos'], phi_m_II_dc_df['phi_m'])
phi_m_II_dc_df_final = phi_m_II_dc_df[mask_phi_m_II_dc_df]

phi_m_II_dc_df_final = phi_m_II_dc_df_final.sort_values(by='z/L')
phi_m_II_dc_neg = phi_m_II_dc_df_final.loc[phi_m_II_dc_df['z/L']<=0]
phi_m_II_dc_pos = phi_m_II_dc_df_final.loc[phi_m_II_dc_df['z/L']>=0]

phi_m_III_dc_df = pd.DataFrame()
phi_m_III_dc_df['z/L'] = zL_df['zL_III_dc']
# phi_m_III_dc_df['z/L'] = zL_df['z/L III coare']
phi_m_III_dc_df['phi_m'] = phi_m_III_dc
#get rid of negative shear values
phi_m_III_dc_df['phi_m_pos'] = np.where(phi_m_III_dc_df['phi_m']>=0,phi_m_III_dc_df['phi_m'],np.nan)
mask_phi_m_III_dc_df = np.isin(phi_m_III_dc_df['phi_m_pos'], phi_m_III_dc_df['phi_m'])
phi_m_III_dc_df_final = phi_m_III_dc_df[mask_phi_m_III_dc_df]

phi_m_III_dc_df_final = phi_m_III_dc_df_final.sort_values(by='z/L')
phi_m_III_dc_neg = phi_m_III_dc_df_final.loc[phi_m_III_dc_df['z/L']<=0]
phi_m_III_dc_pos = phi_m_III_dc_df_final.loc[phi_m_III_dc_df['z/L']>=0]

#%% COARE functional form for Phi_m

coare_zL_neg = np.arange(-4,0,0.005)
coare_zL_pos = np.arange(0,2,0.005)

eq34 = (1-15*coare_zL_neg)**(-1/3) #eq 34 for negative z/L
e = 6
eq41 = 1+e*coare_zL_pos #eq 41 for posative z/L


plt.figure()
plt.plot(coare_zL_neg,eq34, color = 'g', label = 'eq34')
plt.plot(coare_zL_pos,eq41, color = 'b', label = 'eq41; e=6')

plt.xlim(-4,2)
plt.ylim(-4,4)
plt.grid()
plt.legend()
plt.title('$\phi_m$ functional form')


#%% Phi_m plots SPRING ONLY
# plt.figure()
# plt.scatter(phi_m_I_dc_df_final_spring['z/L'],phi_m_I_dc_df_final_spring['phi_m'], color = 'dodgerblue', edgecolor = 'navy', label ='$\phi_m(z/L)$ d.c. L I')
# # plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.xlabel("$z/L$")
# plt.ylabel('$\phi_m(z/L)$')
# plt.title('$\phi_m(z/L)$ Level I, SPRING ONLY')
# # plt.title(oct_addition+ title_windDir + "Level I: $\phi_m(z/L)$")
# plt.xlim(-4.5,2.5)
# # plt.ylim(-6,6)
# plt.ylim(-0.5,3)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlim(-4,2)
# plt.ylim(-4,4)
# plt.grid()
# plt.legend()

# plt.figure()
# plt.scatter(phi_m_II_dc_df_final_spring['z/L'],phi_m_II_dc_df_final_spring['phi_m'], color = 'orange', edgecolor = 'red', label ='$\phi_m(z/L)$ d.c. L II')
# # plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.xlabel("$z/L$")
# plt.ylabel('$\phi_m(z/L)$')
# plt.title('$\phi_m(z/L)$ Level II, SPRING ONLY')
# # plt.title(oct_addition+ title_windDir + "Level II: $\phi_m(z/L)$")
# plt.xlim(-4.5,2.5)
# # plt.ylim(-6,6)
# plt.ylim(-0.5,3)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlim(-4,2)
# plt.ylim(-4,4)
# plt.grid()
# plt.legend()

# plt.figure()
# plt.scatter(phi_m_III_dc_df_final_spring['z/L'],phi_m_III_dc_df_final_spring['phi_m'], color = 'seagreen', edgecolor = 'darkgreen', label ='$\phi_m(z/L)$ d.c. L III')
# # plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.xlabel("$z/L$")
# plt.ylabel('$\phi_m(z/L)$')
# plt.title('$\phi_m(z/L)$ Level III, SPRING ONLY')
# # plt.title(title_windDir + "Level III: $\phi_m(z/L)$")
# plt.xlim(-4.5,2.5)
# # plt.ylim(-6,6)
# plt.ylim(-0.5,3)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlim(-4,2)
# plt.ylim(-4,4)
# plt.grid()
# plt.legend()

#%% Phi_m plots FALL ONLY
# plt.figure()
# plt.scatter(phi_m_I_dc_df_final_fall['z/L'],phi_m_I_dc_df_final_fall['phi_m'], color = 'dodgerblue', edgecolor = 'navy', label ='$\phi_m(z/L)$ d.c. L I')
# # plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.xlabel("$z/L$")
# plt.ylabel('$\phi_m(z/L)$')
# plt.title('$\phi_m(z/L)$ Level I, FALL ONLY')
# # plt.title(oct_addition+ title_windDir + "Level I: $\phi_m(z/L)$")
# plt.xlim(-4.5,2.5)
# # plt.ylim(-6,6)
# plt.ylim(-0.5,3)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlim(-4,2)
# plt.ylim(-4,4)
# plt.grid()
# plt.legend()

# plt.figure()
# plt.scatter(phi_m_II_dc_df_final_fall['z/L'],phi_m_II_dc_df_final_fall['phi_m'], color = 'orange', edgecolor = 'red', label ='$\phi_m(z/L)$ d.c. L II')
# # plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.xlabel("$z/L$")
# plt.ylabel('$\phi_m(z/L)$')
# plt.title('$\phi_m(z/L)$ Level II, FALL ONLY')
# # plt.title(oct_addition+ title_windDir + "Level II: $\phi_m(z/L)$")
# plt.xlim(-4.5,2.5)
# # plt.ylim(-6,6)
# plt.ylim(-0.5,3)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlim(-4,2)
# plt.ylim(-4,4)
# plt.grid()
# plt.legend()

# plt.figure()
# plt.scatter(phi_m_III_dc_df_final_fall['z/L'],phi_m_III_dc_df_final_fall['phi_m'], color = 'seagreen', edgecolor = 'darkgreen', label ='$\phi_m(z/L)$ d.c. L III')
# # plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.xlabel("$z/L$")
# plt.ylabel('$\phi_m(z/L)$')
# plt.title('$\phi_m(z/L)$ Level III, FALL ONLY')
# # plt.title(title_windDir + "Level III: $\phi_m(z/L)$")
# plt.xlim(-4.5,2.5)
# # plt.ylim(-6,6)
# plt.ylim(-0.5,3)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlim(-4,2)
# plt.ylim(-4,4)
# plt.grid()
# plt.legend()

#%% Phi_m plots COMBINED DATA
# plt.figure()
# plt.scatter(phi_m_I_dc_df_final['z/L'],phi_m_I_dc_df_final['phi_m'], color = 'dodgerblue', edgecolor = 'navy', label ='$\phi_m(z/L)$ d.c. L I')
# # plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.xlabel("$z/L$")
# plt.ylabel('$\phi_m(z/L)$')
# plt.title('$\phi_m(z/L)$ Level I, Combined Analysis')
# # plt.title(oct_addition+ title_windDir + "Level I: $\phi_m(z/L)$")
# plt.xlim(-4.5,2.5)
# # plt.ylim(-6,6)
# plt.ylim(-0.5,3)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlim(-4,2)
# plt.ylim(-4,4)
# plt.grid()
# plt.legend()

# plt.figure()
# plt.scatter(phi_m_II_dc_df_final['z/L'],phi_m_II_dc_df_final['phi_m'], color = 'orange', edgecolor = 'red', label ='$\phi_m(z/L)$ d.c. L II')
# # plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.xlabel("$z/L$")
# plt.ylabel('$\phi_m(z/L)$')
# plt.title('$\phi_m(z/L)$ Level II, Combined Analysis')
# # plt.title(oct_addition+ title_windDir + "Level II: $\phi_m(z/L)$")
# plt.xlim(-4.5,2.5)
# # plt.ylim(-6,6)
# plt.ylim(-0.5,3)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlim(-4,2)
# plt.ylim(-4,4)
# plt.grid()
# plt.legend()

# plt.figure()
# plt.scatter(phi_m_III_dc_df_final['z/L'],phi_m_III_dc_df_final['phi_m'], color = 'seagreen', edgecolor = 'darkgreen', label ='$\phi_m(z/L)$ d.c. L III')
# # plt.scatter(phi_m_I_coare_df['z/L'],phi_m_I_coare_df['phi_m'], color = 'gray', edgecolor = 'k', label ='$\phi_m(z/L)$ COARE LI')
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.xlabel("$z/L$")
# plt.ylabel('$\phi_m(z/L)$')
# plt.title('$\phi_m(z/L)$ Level III, Combined Analysis')
# # plt.title(title_windDir + "Level III: $\phi_m(z/L)$")
# plt.xlim(-4.5,2.5)
# # plt.ylim(-6,6)
# plt.ylim(-0.5,3)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlim(-4,2)
# plt.ylim(-4,4)
# plt.grid()
# plt.legend()



#%%
# import binsreg

def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df_est[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est

# Estimate binsreg SPRING
df_binEstimate_phi_m_I_dc_spring = binscatter(x='z/L', y='phi_m', data=phi_m_I_dc_df_final_spring, ci=(3,3), randcut=1)
df_binEstimate_phi_m_II_dc_spring = binscatter(x='z/L', y='phi_m', data=phi_m_II_dc_df_final_spring, ci=(3,3), randcut=1)
df_binEstimate_phi_m_III_dc_spring = binscatter(x='z/L', y='phi_m', data=phi_m_III_dc_df_final_spring, ci=(3,3), randcut=1)

print('done with binning data SPRING analysis')


# Estimate binsreg FALL
df_binEstimate_phi_m_I_dc_fall = binscatter(x='z/L', y='phi_m', data=phi_m_I_dc_df_final_fall, ci=(3,3), randcut=1)
df_binEstimate_phi_m_II_dc_fall = binscatter(x='z/L', y='phi_m', data=phi_m_II_dc_df_final_fall, ci=(3,3), randcut=1)
df_binEstimate_phi_m_III_dc_fall = binscatter(x='z/L', y='phi_m', data=phi_m_III_dc_df_final_fall, ci=(3,3), randcut=1)

print('done with binning data FALL analysis')


# Estimate binsreg combined
df_binEstimate_phi_m_I_dc = binscatter(x='z/L', y='phi_m', data=phi_m_I_dc_df_final, ci=(3,3), randcut=1)
df_binEstimate_phi_m_II_dc = binscatter(x='z/L', y='phi_m', data=phi_m_II_dc_df_final, ci=(3,3), randcut=1)
df_binEstimate_phi_m_III_dc = binscatter(x='z/L', y='phi_m', data=phi_m_III_dc_df_final, ci=(3,3), randcut=1)

print('done with binning data Combined analysis')

#%% Phi_m binned scatterplot SPRING ONLY
plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'

# plt.figure()
# sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_I_dc_spring, color = 'dodgerblue', label = "binned $\phi_{M}(z/L)$: L I")
# plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_I_dc_spring, color = 'k', ls='', lw=2, alpha=0.2)
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.title(title_windDir + "Level I: $phi_{M}(z/L) (DC)$ SPRING")
# # plt.xlim(-1.2,0.8)
# plt.xlim(-2,1)
# plt.ylim(-0.5,3)
# plt.legend()

# plt.figure()
# sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_II_dc_spring, color = 'darkorange', label = "binned $\phi_{M}(z/L)$: L II")
# plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_II_dc_spring, color = 'k', ls='', lw=2, alpha=0.2)
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.title(title_windDir + "Level II: $phi_{M}(z/L) (DC)$ SPRING")
# # plt.xlim(-1.2,0.8)
# plt.xlim(-2,1)
# plt.ylim(-0.5,3)
# plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_III_dc_spring, color = 'seagreen', label = "binned $\phi_{M}(z/L)$: L III")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_III_dc_spring, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(title_windDir + "Level III: $phi_{M}(z/L) (DC)$ SPRING")
# plt.xlim(-1.2,0.8)
plt.xlim(-2,1)
plt.ylim(-0.5,3)
plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_I_dc_spring, color = 'dodgerblue', label = "L I")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_I_dc_spring, color = 'navy', ls='', lw=2, alpha=0.2, label = 'L I error')
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_II_dc_spring, color = 'orange', label = "L II")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_II_dc_spring, color = 'red', ls='', lw=2, alpha=0.2, label = 'L II error')
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_III_dc_spring, color = 'seagreen', label = "L III")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_III_dc_spring, color = 'k', ls='', lw=2, alpha=0.2, label = 'L III error')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE eq. 34, 41')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(oct_addition+ title_windDir + "$\phi_{M}(z/L) (DC)$ SPRING")
# plt.xlim(-1.2,0.8)
plt.xlim(-2,1)
plt.ylim(-0.5,3)
plt.legend()
# plt.savefig(plot_savePath + "binnedScatterplot_phiM_spring_Puu.png",dpi=300)
# plt.savefig(plot_savePath + "binnedScatterplot_phiM_spring_Puu.pdf")

#%% Phi_m binned scatterplot FALL
plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'

# plt.figure()
# sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_I_dc_fall, color = 'dodgerblue', label = "binned $\phi_{M}(z/L)$: L I")
# plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_I_dc_fall, color = 'k', ls='', lw=2, alpha=0.2)
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.title(title_windDir + "Level I: $phi_{M}(z/L) (DC)$ FALL")
# # plt.xlim(-1.2,0.8)
# plt.xlim(-2,1)
# plt.ylim(-0.5,3)
# plt.legend()

# plt.figure()
# sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_II_dc_fall, color = 'darkorange', label = "binned $\phi_{M}(z/L)$: L II")
# plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_II_dc_fall, color = 'k', ls='', lw=2, alpha=0.2)
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.title(title_windDir + "Level II: $phi_{M}(z/L) (DC)$ FALL")
# # plt.xlim(-1.2,0.8)
# plt.xlim(-2,1)
# plt.ylim(-0.5,3)
# plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_III_dc_fall, color = 'seagreen', label = "binned $\phi_{M}(z/L)$: L III")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_III_dc_fall, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(title_windDir + "Level III: $phi_{M}(z/L) (DC)$ FALL")
# plt.xlim(-1.2,0.8)
plt.xlim(-2,1)
plt.ylim(-0.5,3)
plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_I_dc_fall, color = 'dodgerblue', label = "L I")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_I_dc_fall, color = 'navy', ls='', lw=2, alpha=0.2, label = 'L I error')
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_II_dc_fall, color = 'orange', label = "L II")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_II_dc_fall, color = 'red', ls='', lw=2, alpha=0.2, label = 'L II error')
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_III_dc_fall, color = 'seagreen', label = "L III")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_III_dc_fall, color = 'k', ls='', lw=2, alpha=0.2, label = 'L III error')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE eq. 34, 41')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(oct_addition+ title_windDir + "$\phi_{M}(z/L) (DC)$ FALL")
# plt.xlim(-1.2,0.8)
plt.xlim(-2,1)
plt.ylim(-0.5,3)
plt.legend()
# plt.savefig(plot_savePath + "binnedScatterplot_phiM_fall_Puu.png",dpi=300)
# plt.savefig(plot_savePath + "binnedScatterplot_phiM_fall_Puu.pdf")

#%% Phi_m binned scatterplot combined analysis
# plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'

# plt.figure()
# sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_I_dc, color = 'dodgerblue', label = "binned $\phi_{M}(z/L)$: L I")
# plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_I_dc, color = 'k', ls='', lw=2, alpha=0.2)
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.title(title_windDir + "Level I: $phi_{M}(z/L) (DC)$")
# # plt.xlim(-1.2,0.8)
# plt.xlim(-2,1)
# plt.ylim(-0.5,3)
# plt.legend()

# plt.figure()
# sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_II_dc, color = 'darkorange', label = "binned $\phi_{M}(z/L)$: L II")
# plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_II_dc, color = 'k', ls='', lw=2, alpha=0.2)
# plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
# plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
# plt.title(title_windDir + "Level II: $phi_{M}(z/L) (DC)$")
# # plt.xlim(-1.2,0.8)
# plt.xlim(-2,1)
# plt.ylim(-0.5,3)
# plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_III_dc, color = 'seagreen', label = "binned $\phi_{M}(z/L)$: L III")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_III_dc, color = 'k', ls='', lw=2, alpha=0.2)
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE functional form')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(title_windDir + "Level III: $phi_{M}(z/L) (DC)$")
# plt.xlim(-1.2,0.8)
plt.xlim(-2,1)
plt.ylim(-0.5,3)
plt.legend()

plt.figure()
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_I_dc, color = 'dodgerblue', label = "L I")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_I_dc, color = 'navy', ls='', lw=2, alpha=0.2, label = 'L I error')
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_II_dc, color = 'orange', label = "L II")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_II_dc, color = 'red', ls='', lw=2, alpha=0.2, label = 'L II error')
sns.scatterplot(x='z/L', y='phi_m', data=df_binEstimate_phi_m_III_dc, color = 'seagreen', label = "L III")
plt.errorbar('z/L', 'phi_m', yerr='ci', data=df_binEstimate_phi_m_III_dc, color = 'k', ls='', lw=2, alpha=0.2, label = 'L III error')
plt.plot(coare_zL_neg, eq34, color = 'k',linewidth=3, label = 'COARE eq. 34, 41')
plt.plot(coare_zL_pos, eq41, color = 'k',linewidth=3)
plt.title(oct_addition+ title_windDir + "$\phi_{M}(z/L) (DC)$")
# plt.xlim(-1.2,0.8)
plt.xlim(-2,1)
plt.ylim(-0.5,3)
plt.legend()
# # plt.savefig(plot_savePath + "binnedScatterplot_phiM_Puu.png",dpi=300)
# # plt.savefig(plot_savePath + "binnedScatterplot_phiM_Puu.pdf")
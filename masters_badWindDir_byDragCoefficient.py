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
plot_save_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'
break_index = 3959

date_df = pd.read_csv(file_path + "date_combinedAnalysis.csv")
print(date_df.columns)
print(date_df['datetime'][10])

windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv")
# windDir_df = pd.read_csv(file_path + "windDir_keep090250s_075260f_combinedAnalysis.csv")
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)

sonic1_df = pd.read_csv(file_path + 'despiked_s1_turbulenceTerms_andMore_combined.csv')
sonic2_df = pd.read_csv(file_path + 'despiked_s2_turbulenceTerms_andMore_combined.csv')
sonic3_df = pd.read_csv(file_path + 'despiked_s3_turbulenceTerms_andMore_combined.csv')
sonic4_df = pd.read_csv(file_path + 'despiked_s4_turbulenceTerms_andMore_combined.csv')

Ubar_s1 = np.array(sonic1_df['Ubar'])
Ubar_s2 = np.array(sonic2_df['Ubar'])
Ubar_s3 = np.array(sonic3_df['Ubar'])
Ubar_s4 = np.array(sonic4_df['Ubar'])

#this is making suve Ubar <2m/s have been excluded
plt.figure()
plt.plot(Ubar_s1, label = 's1')
plt.plot(Ubar_s2, label = 's2')
plt.plot(Ubar_s3, label = 's3')
plt.plot(Ubar_s4, label = 's4')
plt.ylim(0,5)
plt.legend()
plt.title('Testing Ubar is excluding variable wind speeds')
plt.xlabel('index')
plt.ylabel('Ubar [m/s]')


#%% this is incase we want to exclude more small wind speeds
# Ubar_s1 =[]
# Ubar_s2 =[]
# Ubar_s3 =[]
# Ubar_s4 =[]

# for i in range(len(sonic1_df)):
#     if sonic1_df['Ubar'][i] < 3:
#         Ubar_s1_i = np.nan
#     else:
#         Ubar_s1_i = sonic1_df['Ubar'][i]
#     Ubar_s1.append(Ubar_s1_i)

# for i in range(len(sonic2_df)):
#     if sonic2_df['Ubar'][i] < 3:
#         Ubar_s2_i = np.nan
#     else:
#         Ubar_s2_i = sonic2_df['Ubar'][i]
#     Ubar_s2.append(Ubar_s2_i)

# for i in range(len(sonic3_df)):
#     if sonic3_df['Ubar'][i] < 3:
#         Ubar_s3_i = np.nan
#     else:
#         Ubar_s3_i = sonic3_df['Ubar'][i]
#     Ubar_s3.append(Ubar_s3_i)

# for i in range(len(sonic4_df)):
#     if sonic4_df['Ubar'][i] < 3:
#         Ubar_s4_i = np.nan
#     else:
#         Ubar_s4_i = sonic4_df['Ubar'][i]
#     Ubar_s4.append(Ubar_s4_i)
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
cd_df = pd.DataFrame()
cd_df['cd_s1'] = np.array(cd_s1)
cd_df['cd_s2'] = np.array(cd_s2)
cd_df['cd_s3'] = np.array(cd_s3)
cd_df['cd_s4'] = np.array(cd_s4)
cd_df.to_csv(file_path + 'dragCoefficient_combinedAnalysis.csv')

#%%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(8, 10))
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
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True,figsize=(8, 10))
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

fig,((ax4, ax8), (ax3, ax7), (ax2, ax6), (ax1, ax5)) = plt.subplots(4,2, figsize = (16,10))
fig.suptitle('Determining bad wind directions from drag coefficients \n SPRING and FALL')
ax1.scatter(windDir_df['alpha_s1'][:break_index+1],cd_s1[:break_index+1]*1000, s = 1, color = 'darkgreen')
ax1.set_ylim(0,5)
ax1.set_ylabel('$C_{D}x10^3$ \n s1')
ax2.scatter(windDir_df['alpha_s2'][:break_index+1],cd_s2[:break_index+1]*1000, s = 1, color = 'darkgreen')
ax2.set_ylim(0,5)
ax2.set_ylabel('$C_{D}x10^3$ \n s2')
ax3.scatter(windDir_df['alpha_s3'][:break_index+1],cd_s3[:break_index+1]*1000, s = 1, color = 'darkgreen')
ax3.set_ylim(0,5)
ax3.set_ylabel('$C_{D}x10^3$ \n s3')
ax4.scatter(windDir_df['alpha_s4'][:break_index+1],cd_s4[:break_index+1]*1000, s = 1, color = 'darkgreen')
ax4.set_ylim(0,5)
ax4.set_ylabel('$C_{D}x10^3$ \n s4')
ax4.set_xlabel('Wind Direction')
ax5.scatter(windDir_df['alpha_s1'][break_index+1:],cd_s1[break_index+1:]*1000, s = 1, color = 'darkorange')
ax5.set_ylim(0,5)
ax5.set_ylabel('$C_{D}x10^3$ \n s1')
ax6.scatter(windDir_df['alpha_s2'][break_index+1:],cd_s2[break_index+1:]*1000, s = 1, color = 'darkorange')
ax6.set_ylim(0,5)
ax6.set_ylabel('$C_{D}x10^3$ \n s2')
ax7.scatter(windDir_df['alpha_s3'][break_index+1:],cd_s3[break_index+1:]*1000, s = 1, color = 'darkorange')
ax7.set_ylim(0,5)
ax7.set_ylabel('$C_{D}x10^3$ \n s3')
ax8.scatter(windDir_df['alpha_s4'][break_index+1:],cd_s4[break_index+1:]*1000, s = 1, color = 'darkorange')
ax8.set_ylim(0,5)
ax8.set_ylabel('$C_{D}x10^3$ \n s4')
ax8.set_xlabel('Wind Direction')
print('done plotting')

fig.savefig(plot_save_path+'scatter_dragCoefficient_v_windDir.pdf')
# fig.savefig(plot_save_path+'scatter_dragCoefficient_v_windDir_keep090250s_075260f.pdf')
print('done saving plot')


#%% wind direction versus non-dimensional shear
kappa = 0.40

usr_LI = (usr_s1+usr_s2)/2
usr_LII = (usr_s2+usr_s3)/2
usr_LIII = (usr_s3+usr_s4)/2

dz_LI_spring = 2.695  #sonic 2- sonic 1: spring APRIL 2022 deployment
dz_LII_spring = 2.795 #sonic 3- sonic 2: spring APRIL 2022 deployment
dz_LIII_spring = 2.415 #sonic 4- sonic 3: spring APRIL 2022 deployment
dz_LI_fall = 1.8161  #sonic 2- sonic 1: FALL SEPT 2022 deployment
dz_LII_fall = 3.2131 #sonic 3- sonic 2: FALL SEPT 2022 deployment
dz_LIII_fall = 2.468 #sonic 4- sonic 3: FALL SEPT 2022 deployment

z_df_spring = pd.read_csv(file_path + "z_airSide_allSpring.csv")
z_df_spring = z_df_spring.drop(['Unnamed: 0'], axis=1)

z_df_fall = pd.read_csv(file_path + "z_airSide_allFall.csv")
z_df_fall = z_df_fall.drop(['Unnamed: 0'], axis=1)

z_df = pd.concat([z_df_spring, z_df_fall], axis = 0)
print('done with z concat')

z_LI_spring = z_df_spring['z_sonic1']+(0.5*dz_LI_spring)
z_LII_spring  = z_df_spring['z_sonic2']+(0.5*dz_LII_spring)
z_LIII_spring  = z_df_spring['z_sonic3']+(0.5*dz_LIII_spring)

z_LI_fall = z_df_fall['z_sonic1']+(0.5*dz_LI_fall)
z_LII_fall = z_df_fall['z_sonic2']+(0.5*dz_LII_fall)
z_LIII_fall = z_df_fall['z_sonic3']+(0.5*dz_LIII_fall)

z_LI = np.concatenate([z_LI_spring, z_LI_fall], axis = 0)
z_LII = np.concatenate([z_LII_spring, z_LII_fall], axis = 0)
z_LIII = np.concatenate([z_LIII_spring, z_LIII_fall], axis = 0)

dUbar_LI_spring = np.array(sonic2_df['Ubar'][:break_index+1]-sonic1_df['Ubar'][:break_index+1])
dUbar_LII_spring = np.array(sonic3_df['Ubar'][:break_index+1]-sonic2_df['Ubar'][:break_index+1])
dUbar_LIII_spring = np.array(sonic4_df['Ubar'][:break_index+1]-sonic3_df['Ubar'][:break_index+1])

dUbar_LI_fall = np.array(sonic2_df['Ubar'][break_index+1:]-sonic1_df['Ubar'][break_index+1:])
dUbar_LII_fall = np.array(sonic3_df['Ubar'][break_index+1:]-sonic2_df['Ubar'][break_index+1:])
dUbar_LIII_fall = np.array(sonic4_df['Ubar'][break_index+1:]-sonic3_df['Ubar'][break_index+1:])

dUbardz_LI_spring = np.array(sonic2_df['Ubar'][:break_index+1]-sonic1_df['Ubar'][:break_index+1])/dz_LI_spring 
dUbardz_LII_spring  = np.array(sonic3_df['Ubar'][:break_index+1]-sonic2_df['Ubar'][:break_index+1])/dz_LII_spring 
dUbardz_LIII_spring  = np.array(((sonic4_df['Ubar'][:break_index+1])*1.0)-sonic3_df['Ubar'][:break_index+1])/dz_LIII_spring 
dUbardz_LI_fall = np.array(sonic2_df['Ubar'][break_index+1:]-sonic1_df['Ubar'][break_index+1:])/dz_LI_fall
dUbardz_LII_fall = np.array(sonic3_df['Ubar'][break_index+1:]-sonic2_df['Ubar'][break_index+1:])/dz_LII_fall
dUbardz_LIII_fall = np.array(((sonic4_df['Ubar'][break_index+1:])*1.0)-sonic3_df['Ubar'][break_index+1:])/dz_LIII_fall
dUbardz_LI = np.concatenate([dUbardz_LI_spring, dUbardz_LI_fall], axis = 0)
dUbardz_LII = np.concatenate([dUbardz_LII_spring, dUbardz_LII_fall], axis = 0)
dUbardz_LIII = np.concatenate([dUbardz_LIII_spring, dUbardz_LIII_fall], axis = 0)

# phi_m_I_dc_spring = kappa*np.array(z_LI_spring)/(np.array(usr_LI[:break_index+1]))*(np.array(dUbardz_LI_spring))
# phi_m_II_dc_spring = kappa*np.array(z_LII_spring)/(np.array(usr_LII[:break_index+1]))*(np.array(dUbardz_LII_spring))
# phi_m_III_dc_spring = kappa*np.array(z_LIII_spring)/(np.array(usr_LIII[:break_index+1]))*(np.array(dUbardz_LIII_spring))

# phi_m_I_dc_fall = kappa*np.array(z_LI_fall)/(np.array(usr_LI[break_index+1:]))*(np.array(dUbardz_LI_fall))
# phi_m_II_dc_fall = kappa*np.array(z_LII_fall)/(np.array(usr_LII[break_index+1:]))*(np.array(dUbardz_LII_fall))
# phi_m_III_dc_fall = kappa*np.array(z_LIII_fall)/(np.array(usr_LIII[break_index+1:]))*(np.array(dUbardz_LIII_fall))

# phi_m_I_dc = kappa*np.array(z_LI)/(np.array(usr_LI))*(np.array(dUbardz_LI))
# phi_m_II_dc = kappa*np.array(z_LII)/(np.array(usr_LII))*(np.array(dUbardz_LII))
# phi_m_III_dc = kappa*np.array(z_LIII)/(np.array(usr_LIII))*(np.array(dUbardz_LIII))

phi_M_I = pd.read_csv(file_path + 'phiM_I_dc.csv')
phi_M_II = pd.read_csv(file_path + 'phiM_II_dc.csv')
phi_M_III = pd.read_csv(file_path + 'phiM_III_dc.csv')

#%% plotting wind direction versus kappa*uStar/z
# ND_shear_s1_spring = kappa*np.array(usr_s1[:break_index+1])/np.array(z_df_spring['z_sonic1'])
# ND_shear_s2_spring = kappa*np.array(usr_s2[:break_index+1])/np.array(z_df_spring['z_sonic2'])
# ND_shear_s3_spring = kappa*np.array(usr_s3[:break_index+1])/np.array(z_df_spring['z_sonic3'])
# ND_shear_s4_spring = kappa*np.array(usr_s4[:break_index+1])/np.array(z_df_spring['z_sonic4'])

# ND_shear_s1_fall = kappa*np.array(usr_s1[break_index+1:])/np.array(z_df_fall['z_sonic1'])
# ND_shear_s2_fall = kappa*np.array(usr_s2[break_index+1:])/np.array(z_df_fall['z_sonic2'])
# ND_shear_s3_fall = kappa*np.array(usr_s3[break_index+1:])/np.array(z_df_fall['z_sonic3'])
# ND_shear_s4_fall = kappa*np.array(usr_s4[break_index+1:])/np.array(z_df_fall['z_sonic4'])

# ND_shear_s1 = kappa*np.array(usr_s1)/np.array(z_df['z_sonic1'])
# ND_shear_s2 = kappa*np.array(usr_s2)/np.array(z_df['z_sonic2'])
# ND_shear_s3 = kappa*np.array(usr_s3)/np.array(z_df['z_sonic3'])
# ND_shear_s4 = kappa*np.array(usr_s4)/np.array(z_df['z_sonic4'])

# ymax1 = 3
# ymax = 1
# fig,((ax4, ax8), (ax3, ax7), (ax2, ax6), (ax1, ax5) ) = plt.subplots(4,2, figsize = (16,8))
# fig.suptitle('Determining bad wind directions from non-dimensional shear \n SPRING and FALL')
# ax1.scatter(windDir_df['alpha_s1'][:break_index+1],ND_shear_s1_spring*10, s = 1, color = 'darkgreen')
# ax1.set_ylim(0,ymax1)
# # ax1.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
# ax1.set_ylabel('ND shear x 10 \n s1')
# ax2.scatter(windDir_df['alpha_s2'][:break_index+1],ND_shear_s2_spring*10, s = 1, color = 'darkgreen')
# ax2.set_ylim(0,ymax)
# # ax2.set_ylabel('$$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$$')
# ax2.set_ylabel('ND shear x 10 \n s2')
# ax3.scatter(windDir_df['alpha_s3'][:break_index+1],ND_shear_s3_spring*10, s = 1, color = 'darkgreen')
# ax3.set_ylim(0,ymax)
# # ax3.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
# ax3.set_ylabel('ND shear x 10 \n s3')
# ax4.scatter(windDir_df['alpha_s4'][:break_index+1],ND_shear_s4_spring*10, s = 1, color = 'darkgreen')
# ax4.set_ylim(0,ymax)
# # ax4.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
# ax4.set_ylabel('ND shear x 10 \n s4')
# ax4.set_xlabel('Wind Direction')
# ax5.scatter(windDir_df['alpha_s1'][break_index+1:],ND_shear_s1_fall*10, s = 1, color = 'darkorange')
# ax5.set_ylim(0,ymax1)
# # ax5.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
# ax5.set_ylabel('ND shear x 10 \n s1')
# ax6.scatter(windDir_df['alpha_s2'][break_index+1:],ND_shear_s2_fall*10, s = 1, color = 'darkorange')
# ax6.set_ylim(0,ymax)
# # ax6.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
# ax6.set_ylabel('ND shear x 10 \n s2')
# ax7.scatter(windDir_df['alpha_s3'][break_index+1:],ND_shear_s3_fall*10, s = 1, color = 'darkorange')
# ax7.set_ylim(0,ymax)
# # ax7.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
# ax7.set_ylabel('ND shear x 10 \n s3')
# ax8.scatter(windDir_df['alpha_s4'][break_index+1:],ND_shear_s4_fall*10, s = 1, color = 'darkorange')
# ax8.set_ylim(0,ymax)
# # ax8.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
# ax8.set_ylabel('ND shear x 10 \n s4')
# ax8.set_xlabel('Wind Direction')
# print('done plotting')

# fig.savefig(plot_save_path+'scatter_nonDimensionalShear_v_windDir.pdf')
# print('done saving plot')

#%% plotting wind direction versus phi_m (non-dimesnional shear)

ymin = 0
ymax = 10
xmin = 0
xmax = 360

fig,((ax3, ax6), (ax2, ax5),(ax1, ax4)) = plt.subplots(3,2, figsize = (16,8))
fig.suptitle('Determining bad wind directions from $\phi_M = (\kappa u_*/z) \; (dU/dz)$ \n SPRING and FALL')
ax1.scatter(windDir_df['alpha_s1'][:break_index+1],phi_M_I['phi_m'][:break_index+1], s = 1, color = 'darkgreen')
# ax1.set_ylim(ymin,ymax)
ax1.set_xlim(xmin, xmax)
ax1.set_ylabel('$\phi_M$ LI')
ax1.set_xlabel('Wind Direction of lowest sonic in level')

ax2.scatter(windDir_df['alpha_s2'][:break_index+1],phi_M_II['phi_m'][:break_index+1], s = 1, color = 'darkgreen')
# ax2.set_ylim(ymin,ymax)
ax2.set_xlim(xmin, xmax)
ax2.set_ylabel('$\phi_M$ LII')

ax3.scatter(windDir_df['alpha_s3'][:break_index+1],phi_M_III['phi_m'][:break_index+1], s = 1, color = 'darkgreen')
# ax3.set_ylim(ymin,ymax)
ax3.set_xlim(xmin, xmax)
ax3.set_ylabel('$\phi_M$ LIII')

ax4.scatter(windDir_df['alpha_s1'][break_index+1:],phi_M_I['phi_m'][break_index+1:], s = 1, color = 'darkorange')
# ax4.set_ylim(ymin,ymax)
ax4.set_xlim(xmin, xmax)
ax4.set_ylabel('$\phi_M$ LI')
ax4.set_xlabel('Wind Direction of lowest sonic in level')

ax5.scatter(windDir_df['alpha_s2'][break_index+1:],phi_M_II['phi_m'][break_index+1:], s = 1, color = 'darkorange')
# ax5.set_ylim(ymin,ymax)
ax5.set_xlim(xmin, xmax)
ax5.set_ylabel('$\phi_M$ LII')

ax6.scatter(windDir_df['alpha_s3'][break_index+1:],phi_M_III['phi_m'][break_index+1:], s = 1, color = 'darkorange')
# ax6.set_ylim(ymin,ymax)
ax6.set_xlim(xmin, xmax)
ax6.set_ylabel('$\phi_M$ LIII')

print('done plotting')

fig.savefig(plot_save_path+'scatter_phiM_v_windDir.pdf')
fig.savefig(plot_save_path+'scatter_phiM_v_windDir.png', dpi = 300)
# fig.savefig(plot_save_path+'scatter_phiM_v_windDir_keep090250s_075260f.pdf')
# fig.savefig(plot_save_path+'scatter_phiM_v_windDir_keep090250s_075260f.png', dpi=300)
print('done saving plot')


#%% plotting wind direction versus phi_eps (non-dimesnional dissipation rate)
phi_eps_I = pd.read_csv(file_path + 'phiEps_I_dc.csv')
phi_eps_II = pd.read_csv(file_path + 'phiEps_II_dc.csv')
phi_eps_III = pd.read_csv(file_path + 'phiEps_III_dc.csv')

ymin = 0
ymax = 10
xmin = 0
xmax = 360
fig,((ax3, ax6), (ax2, ax5),(ax1, ax4)) = plt.subplots(3,2, figsize = (16,8))
fig.suptitle('Determining bad wind directions from $\phi_{\epsilon} = \epsilon (\kappa z/u_*^3) \;$ \n SPRING and FALL')
ax1.scatter(windDir_df['alpha_s1'][:break_index+1],phi_eps_I['phi_eps'][:break_index+1], s = 1, color = 'darkgreen')
ax1.set_ylim(ymin,ymax)
ax1.set_xlim(xmin, xmax)
ax1.set_ylabel('$\phi_{\epsilon}$ LI')
ax1.set_xlabel('Wind Direction of lowest sonic in level')

ax2.scatter(windDir_df['alpha_s2'][:break_index+1],phi_eps_II['phi_eps'][:break_index+1], s = 1, color = 'darkgreen')
ax2.set_ylim(ymin,ymax)
ax2.set_xlim(xmin, xmax)
ax2.set_ylabel('$\phi_{\epsilon}$ LII')

ax3.scatter(windDir_df['alpha_s3'][:break_index+1],phi_eps_III['phi_eps'][:break_index+1], s = 1, color = 'darkgreen')
ax3.set_ylim(ymin,ymax)
ax3.set_xlim(xmin, xmax)
ax3.set_ylabel('$\phi_{\epsilon}$ LIII')

ax4.scatter(windDir_df['alpha_s1'][break_index+1:],phi_eps_I['phi_eps'][break_index+1:], s = 1, color = 'darkorange')
ax4.set_ylim(ymin,ymax)
ax4.set_xlim(xmin, xmax)
ax4.set_ylabel('$\phi_{\epsilon}$ LI')
ax4.set_xlabel('Wind Direction of lowest sonic in level')

ax5.scatter(windDir_df['alpha_s2'][break_index+1:],phi_eps_II['phi_eps'][break_index+1:], s = 1, color = 'darkorange')
ax5.set_ylim(ymin,ymax)
ax5.set_xlim(xmin, xmax)
ax5.set_ylabel('$\phi_{\epsilon}$ LII')

ax6.scatter(windDir_df['alpha_s3'][break_index+1:],phi_eps_III['phi_eps'][break_index+1:], s = 1, color = 'darkorange')
ax6.set_ylim(ymin,ymax)
ax6.set_xlim(xmin, xmax)
ax6.set_ylabel('$\phi_{\epsilon}$ LIII')

print('done plotting')

fig.savefig(plot_save_path+'scatter_phiEps_v_windDir.pdf')
fig.savefig(plot_save_path+'scatter_phiEps_v_windDir.png', dpi = 300)
# fig.savefig(plot_save_path+'scatter_phiEps_v_windDir_keep090250s_075260f.pdf')
# fig.savefig(plot_save_path+'scatter_phiEps_v_windDir_keep090250s_075260f.png', dpi=300)
print('done saving plot')
#%% plotting wind direction versus dU/dz
ymaxI = 4
yminI = -5
 
ymaxII = 4
yminII = -5

ymaxIII = 4
yminIII = -5

fig,((ax3, ax6), (ax2, ax5),(ax1, ax4)) = plt.subplots(3,2, figsize = (16,8))
fig.suptitle('Determining bad wind directions from $dU/dz$ \n SPRING and FALL')
ax1.scatter(windDir_df['alpha_s1'][:break_index+1],dUbardz_LI_spring, s = 1, color = 'darkgreen')
ax1.set_ylim(yminI,ymaxI)
# ax1.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax1.set_ylabel('$dU/dz$ LI')
ax2.scatter(windDir_df['alpha_s2'][:break_index+1],dUbardz_LII_spring, s = 1, color = 'darkgreen')
ax2.set_ylim(yminII,ymaxII)
# ax2.set_ylabel('$$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$$')
ax2.set_ylabel('$dU/dz$ LII')
ax3.scatter(windDir_df['alpha_s3'][:break_index+1],dUbardz_LIII_spring, s = 1, color = 'darkgreen')
ax3.set_ylim(yminIII,ymaxIII)
# ax3.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax3.set_ylabel('$dU/dz$ LIII')
ax3.set_xlabel('Wind Direction of lowest sonic in level')
ax4.scatter(windDir_df['alpha_s1'][break_index+1:],dUbardz_LI_fall, s = 1, color = 'darkorange')
ax4.set_ylim(yminI,ymaxI)
# ax4.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax4.set_ylabel('$dU/dz$ LI')
ax5.scatter(windDir_df['alpha_s2'][break_index+1:],dUbardz_LII_fall, s = 1, color = 'darkorange')
ax5.set_ylim(yminII,ymaxII)
# ax5.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax5.set_ylabel('$dU/dz$ LII')
ax6.scatter(windDir_df['alpha_s3'][break_index+1:],dUbardz_LIII_fall, s = 1, color = 'darkorange')
ax6.set_ylim(yminIII,ymaxIII)
# ax6.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax6.set_ylabel('$dU/dz$ LIII')
ax6.set_xlabel('Wind Direction of lowest sonic in level')
print('done plotting')

fig.savefig(plot_save_path+'scatter_dUdz_v_windDir.pdf')
fig.savefig(plot_save_path+'scatter_dUdz_v_windDir.png', dpi = 300)
# fig.savefig(plot_save_path+'scatter_dUdz_v_windDir_keep090250s_075260f.pdf')
# fig.savefig(plot_save_path+'scatter_dUdz_v_windDir_keep090250s_075260f.png', dpi = 300)
print('done saving plot')


#%% plotting wind direction versus dU only -- wind dir of LOWEST SONIC
ymaxI = 4
yminI = -5
 
ymaxII = 4
yminII = -5

ymaxIII = 4
yminIII = -5

vline = np.arange(-4,4)
x_lowerLim = np.ones(8)*110
x_upperLim = np.ones(8)*155

fig,((ax3, ax6), (ax2, ax5),(ax1, ax4)) = plt.subplots(3,2, figsize = (16,8))
fig.suptitle('Determining bad wind directions from $dU$ only \n SPRING and FALL')
ax1.scatter(windDir_df['alpha_s1'][:break_index+1],dUbar_LI_spring, s = 1, color = 'darkgreen')
ax1.set_ylim(yminI,ymaxI)
ax1.plot(x_lowerLim,vline,color='k')
ax1.plot(x_upperLim,vline,color='k')
# ax1.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax1.set_ylabel('$dU$ LI')
ax1.set_xlabel('Wind Direction of lowest sonic in level')

ax2.scatter(windDir_df['alpha_s2'][:break_index+1],dUbar_LII_spring, s = 1, color = 'darkgreen')
ax2.set_ylim(yminII,ymaxII)
ax2.plot(x_lowerLim,vline,color='k')
ax2.plot(x_upperLim,vline,color='k')
# ax2.set_ylabel('$$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$$')
ax2.set_ylabel('$dU$ LII')

ax3.scatter(windDir_df['alpha_s3'][:break_index+1],dUbar_LIII_spring, s = 1, color = 'darkgreen')
ax3.set_ylim(yminIII,ymaxIII)
ax3.plot(x_lowerLim,vline,color='k')
ax3.plot(x_upperLim,vline,color='k')
# ax3.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax3.set_ylabel('$dU$ LIII')
# ax3.set_xlabel('Wind Direction of lowest sonic in level')

ax4.scatter(windDir_df['alpha_s1'][break_index+1:],dUbar_LI_fall, s = 1, color = 'darkorange')
ax4.set_ylim(yminI,ymaxI)
ax4.plot(x_lowerLim,vline,color='k')
ax4.plot(x_upperLim,vline,color='k')
# ax4.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax4.set_ylabel('$dU$ LI')
ax4.set_xlabel('Wind Direction of lowest sonic in level')

ax5.scatter(windDir_df['alpha_s2'][break_index+1:],dUbar_LII_fall, s = 1, color = 'darkorange')
ax5.set_ylim(yminII,ymaxII)
ax5.plot(x_lowerLim,vline,color='k')
ax5.plot(x_upperLim,vline,color='k')
# ax5.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax5.set_ylabel('$dU$ LII')

ax6.scatter(windDir_df['alpha_s3'][break_index+1:],dUbar_LIII_fall, s = 1, color = 'darkorange')
ax6.set_ylim(yminIII,ymaxIII)
ax6.plot(x_lowerLim,vline,color='k')
ax6.plot(x_upperLim,vline,color='k')
# ax6.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax6.set_ylabel('$dU$ LIII')
# ax6.set_xlabel('Wind Direction of lowest sonic in level')
print('done plotting')

fig.savefig(plot_save_path+'scatter_dUonly_lowSonic_v_windDir.pdf')
fig.savefig(plot_save_path+'scatter_dUonly_lowSonic_v_windDir.png', dpi = 300)
print('done saving plot')

# plotting wind direction versus dU only -- wind dir of HIGHEST SONIC
ymaxI = 4
yminI = -5
 
ymaxII = 4
yminII = -5

ymaxIII = 4
yminIII = -5

fig,((ax3, ax6), (ax2, ax5),(ax1, ax4)) = plt.subplots(3,2, figsize = (16,8))
fig.suptitle('Determining bad wind directions from $dU$ only \n SPRING and FALL')
ax1.scatter(windDir_df['alpha_s2'][:break_index+1],dUbar_LI_spring, s = 1, color = 'darkgreen')
ax1.set_ylim(yminI,ymaxI)
ax1.plot(x_lowerLim,vline,color='k')
ax1.plot(x_upperLim,vline,color='k')
# ax1.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax1.set_ylabel('$dU$ LI')
ax1.set_xlabel('Wind Direction of highest sonic in level')

ax2.scatter(windDir_df['alpha_s3'][:break_index+1],dUbar_LII_spring, s = 1, color = 'darkgreen')
ax2.set_ylim(yminII,ymaxII)
ax2.plot(x_lowerLim,vline,color='k')
ax2.plot(x_upperLim,vline,color='k')
# ax2.set_ylabel('$$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$$')
ax2.set_ylabel('$dU$ LII')

ax3.scatter(windDir_df['alpha_s4'][:break_index+1],dUbar_LIII_spring, s = 1, color = 'darkgreen')
ax3.set_ylim(yminIII,ymaxIII)
ax3.plot(x_lowerLim,vline,color='k')
ax3.plot(x_upperLim,vline,color='k')
# ax3.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax3.set_ylabel('$dU$ LIII')
# ax3.set_xlabel('Wind Direction of lowest sonic in level')

ax4.scatter(windDir_df['alpha_s2'][break_index+1:],dUbar_LI_fall, s = 1, color = 'darkorange')
ax4.set_ylim(yminI,ymaxI)
ax4.plot(x_lowerLim,vline,color='k')
ax4.plot(x_upperLim,vline,color='k')
# ax4.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax4.set_ylabel('$dU$ LI')
ax4.set_xlabel('Wind Direction of highest sonic in level')

ax5.scatter(windDir_df['alpha_s3'][break_index+1:],dUbar_LII_fall, s = 1, color = 'darkorange')
ax5.set_ylim(yminII,ymaxII)
ax5.plot(x_lowerLim,vline,color='k')
ax5.plot(x_upperLim,vline,color='k')
# ax5.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax5.set_ylabel('$dU$ LII')

ax6.scatter(windDir_df['alpha_s4'][break_index+1:],dUbar_LIII_fall, s = 1, color = 'darkorange')
ax6.set_ylim(yminIII,ymaxIII)
ax6.plot(x_lowerLim,vline,color='k')
ax6.plot(x_upperLim,vline,color='k')
# ax6.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax6.set_ylabel('$dU$ LIII')
# ax6.set_xlabel('Wind Direction of lowest sonic in level')
print('done plotting')

fig.savefig(plot_save_path+'scatter_dUonly_highSonic_v_windDir.pdf')
fig.savefig(plot_save_path+'scatter_dUonly_highSonic_v_windDir.png', dpi=300)
print('done saving plot')

#%% plotting wind direction versus Ubar

ymin = 0
ymax = 20

fig,((ax4, ax8), (ax3, ax7), (ax2, ax6), (ax1, ax5) ) = plt.subplots(4,2, figsize = (16,8))
fig.suptitle('Determining bad wind directions from wind Speed \n SPRING and FALL')
ax1.scatter(windDir_df['alpha_s1'][:break_index+1],sonic1_df['Ubar'][:break_index+1], s = 1, color = 'darkgreen')
ax1.set_ylim(ymin,ymax)
# ax1.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax1.set_ylabel('Ubar \n s1')
ax2.scatter(windDir_df['alpha_s2'][:break_index+1],sonic2_df['Ubar'][:break_index+1], s = 1, color = 'darkgreen')
ax2.set_ylim(ymin,ymax)
# ax2.set_ylabel('$$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$$')
ax2.set_ylabel('Ubar \n s2')
ax3.scatter(windDir_df['alpha_s3'][:break_index+1],sonic3_df['Ubar'][:break_index+1], s = 1, color = 'darkgreen')
ax3.set_ylim(ymin,ymax)
# ax3.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax3.set_ylabel('Ubar \n s3')
ax4.scatter(windDir_df['alpha_s4'][:break_index+1],sonic4_df['Ubar'][:break_index+1], s = 1, color = 'darkgreen')
ax4.set_ylim(ymin,ymax)
# ax4.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax4.set_ylabel('Ubar \n s4')
ax4.set_xlabel('Wind Direction')
ax5.scatter(windDir_df['alpha_s1'][break_index+1:],sonic1_df['Ubar'][break_index+1:], s = 1, color = 'darkorange')
ax5.set_ylim(ymin,ymax)
# ax5.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax5.set_ylabel('Ubar \n s1')
ax6.scatter(windDir_df['alpha_s2'][break_index+1:],sonic2_df['Ubar'][break_index+1:], s = 1, color = 'darkorange')
ax6.set_ylim(ymin,ymax)
# ax6.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax6.set_ylabel('Ubar \n s2')
ax7.scatter(windDir_df['alpha_s3'][break_index+1:],sonic3_df['Ubar'][break_index+1:], s = 1, color = 'darkorange')
ax7.set_ylim(ymin,ymax)
# ax7.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax7.set_ylabel('Ubar \n s3')
ax8.scatter(windDir_df['alpha_s4'][break_index+1:],sonic4_df['Ubar'][break_index+1:], s = 1, color = 'darkorange')
ax8.set_ylim(ymin,ymax)
# ax8.set_ylabel('$\frac{\kappa u_*}{z}\frac{d\overline{u}}{dz}$')
ax8.set_ylabel('Ubar \n s4')
ax8.set_xlabel('Wind Direction')
print('done plotting')

fig.savefig(plot_save_path+'scatter_Ubar_v_windDir.pdf')
fig.savefig(plot_save_path+'scatter_Ubar_v_windDir.png', dpi = 300)
print('done saving plot')

#%%
#lets isolate the date of 320-360º so we can look at the weather conditions


windS1_320_320_df = pd.DataFrame()
windS1_320_320_df['index'] = np.arange(len(windDir_df))
windS1_320_320_df['datetime'] = np.array(date_df['datetime'])
windS1_320_320_df['alpha'] = np.array(windDir_df['alpha_s1'])
windS1_320_320_df['Ubar'] = np.array(sonic1_df['Ubar'])

index_array = np.arange(len(windDir_df))
windS1_320_320_df['new_index_arr'] = np.where(windS1_320_320_df['alpha']>=320, np.nan, index_array)
maskS1_320_360 = np.isin(windS1_320_320_df['new_index_arr'],index_array)
windS1_320_320_df[maskS1_320_360] = np.nan

print('done with 320-360 mask Sonic 1')



windS2_320_320_df = pd.DataFrame()
windS2_320_320_df['index'] = np.arange(len(windDir_df))
windS2_320_320_df['datetime'] = np.array(date_df['datetime'])
windS2_320_320_df['alpha'] = np.array(windDir_df['alpha_s2'])
windS2_320_320_df['Ubar'] = np.array(sonic2_df['Ubar'])

index_array = np.arange(len(windDir_df))
windS2_320_320_df['new_index_arr'] = np.where(windS2_320_320_df['alpha']>=320, np.nan, index_array)
maskS2_320_360 = np.isin(windS2_320_320_df['new_index_arr'],index_array)
windS2_320_320_df[maskS2_320_360] = np.nan

print('done with 320-360 mask Sonic 2')





windS3_320_320_df = pd.DataFrame()
windS3_320_320_df['index'] = np.arange(len(windDir_df))
windS3_320_320_df['datetime'] = np.array(date_df['datetime'])
windS3_320_320_df['alpha'] = np.array(windDir_df['alpha_s3'])
windS3_320_320_df['Ubar'] = np.array(sonic3_df['Ubar'])

index_array = np.arange(len(windDir_df))
windS3_320_320_df['new_index_arr'] = np.where(windS3_320_320_df['alpha']>=320, np.nan, index_array)
maskS3_320_360 = np.isin(windS3_320_320_df['new_index_arr'],index_array)
windS3_320_320_df[maskS3_320_360] = np.nan

print('done with 320-360 mask Sonic 3')





windS4_320_320_df = pd.DataFrame()
windS4_320_320_df['index'] = np.arange(len(windDir_df))
windS4_320_320_df['datetime'] = np.array(date_df['datetime'])
windS4_320_320_df['alpha'] = np.array(windDir_df['alpha_s4'])
windS4_320_320_df['Ubar'] = np.array(sonic4_df['Ubar'])

index_array = np.arange(len(windDir_df))
windS4_320_320_df['new_index_arr'] = np.where(windS4_320_320_df['alpha']>=320, np.nan, index_array)
maskS4_320_360 = np.isin(windS4_320_320_df['new_index_arr'],index_array)
windS4_320_320_df[maskS4_320_360] = np.nan

print('done with 320-360 mask Sonic 4')

#%%
# keep only the lines that have data and exclude the NaNs
print('before excluding NaNs:')
print(len(windS1_320_320_df))
print(len(windS2_320_320_df))
print(len(windS3_320_320_df))
print(len(windS4_320_320_df))

windS1_320_320_df = windS1_320_320_df[windS1_320_320_df['alpha'].notna()]
windS2_320_320_df = windS2_320_320_df[windS2_320_320_df['alpha'].notna()]
windS3_320_320_df = windS3_320_320_df[windS3_320_320_df['alpha'].notna()]
windS4_320_320_df = windS4_320_320_df[windS4_320_320_df['alpha'].notna()]

print('after excluding NaNs:')
print(len(windS1_320_320_df))
print(len(windS2_320_320_df))
print(len(windS3_320_320_df))
print(len(windS4_320_320_df))
#%%
import matplotlib.dates as mdates

fig, ax = plt.subplots(1, 1, figsize=(10, 7), layout='constrained')
fig.suptitle('$\overline{u}$ Sonic 1 versus Date when windDir 320-360')
ax.plot('datetime', 'Ubar', data=windS1_320_320_df)
# Major ticks every half year, minor ticks every month,
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
# ax.xaxis.set_minor_locator(mdates.MonthLocator())

# Major ticks every month, minor ticks every day,
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.grid(True)
ax.set_ylabel('$\overline{u}$ [m/s]')
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
    
fig, ax = plt.subplots(1, 1, figsize=(10, 7), layout='constrained')
fig.suptitle('$\overline{u}$ Sonic 2 versus Date when windDir 320-360')
ax.plot('datetime', 'Ubar', data=windS2_320_320_df)
# Major ticks every half year, minor ticks every month,
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
# ax.xaxis.set_minor_locator(mdates.MonthLocator())

# Major ticks every month, minor ticks every day,
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.grid(True)
ax.set_ylabel('$\overline{u}$ [m/s]')
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')

fig, ax = plt.subplots(1, 1, figsize=(10, 7), layout='constrained')
fig.suptitle('$\overline{u}$ Sonic 3 versus Date when windDir 320-360')
ax.plot('datetime', 'Ubar', data=windS3_320_320_df)
# Major ticks every half year, minor ticks every month,
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
# ax.xaxis.set_minor_locator(mdates.MonthLocator())

# Major ticks every month, minor ticks every day,
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.grid(True)
ax.set_ylabel('$\overline{u}$ [m/s]')
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
    
fig, ax = plt.subplots(1, 1, figsize=(10, 7), layout='constrained')
fig.suptitle('$\overline{u}$ Sonic 2 versus Date when windDir 320-360')
ax.plot('datetime', 'Ubar', data=windS4_320_320_df)
# Major ticks every half year, minor ticks every month,
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
# ax.xaxis.set_minor_locator(mdates.MonthLocator())

# Major ticks every month, minor ticks every day,
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.grid(True)
ax.set_ylabel('$\overline{u}$ [m/s]')
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
    
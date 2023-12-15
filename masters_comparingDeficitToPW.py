#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:47:11 2023

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
# file_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
# file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
# plot_savePath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4\plots/"
# plot_savePath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/plots/'
plot_savePath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'

sonic_file1 = "despiked_s1_turbulenceTerms_andMore_combined.csv"
sonic1_df = pd.read_csv(file_path+sonic_file1)
sonic_file2 = "despiked_s2_turbulenceTerms_andMore_combined.csv"
sonic2_df = pd.read_csv(file_path+sonic_file2)
sonic_file3 = "despiked_s3_turbulenceTerms_andMore_combined.csv"
sonic3_df = pd.read_csv(file_path+sonic_file3)

windSpeed_df = pd.DataFrame()
windSpeed_df['Ubar_LI'] = (sonic1_df['Ubar']+sonic2_df['Ubar'])/2
windSpeed_df['Ubar_LII'] = (sonic2_df['Ubar']+sonic3_df['Ubar'])/2


windDir_file = "windDir_withBadFlags_110to160_within15degRequirement_combinedAnalysis.csv"
windDir_df = pd.read_csv(file_path + windDir_file)
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)


pw_df = pd.read_csv(file_path + 'pw_combinedAnalysis.csv')
pw_df = pw_df.drop(['Unnamed: 0'], axis=1)


prod_df = pd.read_csv(file_path+'prodTerm_combinedAnalysis.csv')
prod_df = prod_df.drop(['Unnamed: 0'], axis=1)


eps_df = pd.read_csv(file_path+"epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv")
eps_df = eps_df.drop(['Unnamed: 0'], axis=1)
# eps_df[eps_df['eps_sonic1'] > 1] = np.nan

buoy_df = pd.read_csv(file_path+'buoy_terms_combinedAnalysis.csv')
buoy_df = buoy_df.drop(['Unnamed: 0'], axis=1)

rho_df = pd.read_csv(file_path+'rhoAvg_combinedAnalysis.csv')
rho_df = rho_df.drop(['Unnamed: 0'], axis=1)

z_df_spring = pd.read_csv(file_path+'z_airSide_allSpring.csv')
z_df_spring = z_df_spring.drop(['Unnamed: 0'], axis=1)

z_df_fall = pd.read_csv(file_path+'z_airSide_allFall.csv')
z_df_fall = z_df_fall.drop(['Unnamed: 0'], axis=1)

z_df = pd.concat([z_df_spring, z_df_fall], axis=0)

zL_df = pd.read_csv(file_path + 'ZoverL_combinedAnalysis.csv')
zL_df = zL_df.drop(['Unnamed: 0'], axis=1)

usr_df = pd.read_csv(file_path + "usr_combinedAnalysis.csv")
usr_df = usr_df.drop(['Unnamed: 0'], axis=1)

break_index = 3959

print('done with setting up dataframes')
#%%
plt.figure()
plt.plot(-1*sonic1_df['UpWp_bar'], label = 's1')
# plt.plot(-1*sonic2_df['UpWp_bar'], label = 's2')
plt.plot(-1*sonic3_df['UpWp_bar'], label = 's3')
# plt.plot(-1*sonic2_df['UpWp_bar']-(-1*sonic1_df['UpWp_bar']), label = 'difference')
plt.plot(-1*sonic3_df['UpWp_bar']-(-1*sonic1_df['UpWp_bar']), label = 'difference')
plt.hlines(y=0,xmin=0,xmax=break_index,color = 'k')
plt.xlim(1500, 2000)
plt.ylim(-0.2,0.7)
plt.legend()
plt.title("$-\overline{u'w'}$")
#%%
plt.figure()
plt.plot(sonic1_df['Ubar'], label = 's1')
# plt.plot(sonic2_df['Ubar'], label = 's2')
plt.plot(sonic3_df['Ubar'], label = 's3')
# plt.plot(sonic2_df['Ubar']-sonic1_df['Ubar'], label = 'difference')
plt.plot(sonic3_df['Ubar']-sonic1_df['Ubar'], label = 'difference')
plt.hlines(y=0,xmin=0,xmax=break_index,color = 'k')
plt.xlim(1500, 2000)
plt.legend()
plt.title("$\overline{u}$")

#%%
plt.figure()
# plt.plot((-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = 's1')
# plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']), label = 's2')
plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = '2-1 difference')
plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = '3-1 difference')
plt.title("$-\overline{u'w'} (\overline{u})$")
plt.hlines(y=0,xmin=0,xmax=break_index,color = 'k')
plt.xlim(1500,2000)
plt.ylim(-2,10)
plt.ylim(-2,2)
plt.legend()

#%%
plt.figure()
plt.plot(eps_df['epsU_sonic1_MAD'])
plt.title('Dissipation ($\epsilon$)')
plt.xlim(0,break_index)
plt.ylim(0,0.5)
#%%
plt.figure()
# plt.plot((-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = 's1')
# plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']), label = 's2')
plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = '2-1 difference')
plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = '3-1 difference')
plt.title("$-\overline{u'w'} (\overline{u})$")
plt.hlines(y=0,xmin=0,xmax=break_index,color = 'k')
plt.xlim(1700,1800)
plt.ylim(-2,10)
plt.ylim(-2,2)
plt.legend()

plt.figure()
plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = "2-1 Production as flux")
plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']), label = "3-1 Production as flux")
plt.plot(eps_df['epsU_sonic1_MAD']*z_df['z_sonic1'], label = 'Dissipation as flux')
plt.title("Dissipation Flux ($\epsilon_1*z_1$) and Momentum Flux \n $(-\overline{u'w'}_2 (\overline{u}_2)+\overline{u'w'}_1 (\overline{u}_1)$)")
plt.xlim(0,break_index)
plt.ylim(-2,2)
plt.xlim(1500,2000)
plt.ylabel("$[m^3/s^3]$")
plt.xlabel('May Storm Time Index')
plt.legend()
# plt.savefig(plot_savePath + "timeseries_MAYstorm_PandEpsFlux.png", dpi = 300)
# plt.savefig(plot_savePath + "timeseries_MAYstorm_PandEpsFlux.pdf")

#%%
plt.figure()
plt.plot(np.arange(len(z_df)),z_df['z_sonic1'])
plt.title('height of sonic 1 ($z$)')
plt.xlim(0,break_index)
plt.xlim(1500,2000)
#%%
plt.figure()
plt.plot(rho_df['rho_bar_1_dry'], label = 'dry')
plt.plot(rho_df['rho_bar_1_moist'], label = 'moist')
plt.legend()
plt.title('air density')
# plt.xlim(1500,2000)
#%%
plt.figure()
plt.plot(pw_df['PW boom-1 [m^3/s^3]'], color = 'r', label ='$T_{\widetilde{pw}}$')
plt.legend()
plt.title('$T_{\widetilde{pw}}$ flux')
plt.xlim(1500,2000)
#%%
rho_P = rho_df['rho_bar_1_dry']*((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar'])) 
rho_eps = rho_df['rho_bar_1_dry']*(np.array(eps_df['epsU_sonic1_MAD'])*np.array(z_df['z_sonic1']))
deficit = (np.array(rho_P))-np.array(rho_eps)
# deficit_minus_pw = deficit 
plt.figure()
plt.plot(rho_P, color = 'b', label = 'P')
plt.plot(rho_eps, color = 'orange', label = '$\epsilon$')
# plt.plot(deficit, color = 'k', label = 'deficit')
# plt.plot(pw_df['PW boom-1 [m^3/s^3]'],color='red', label = 'PW')
plt.legend()
plt.ylim(-2,3)
plt.xlim(0,break_index)
plt.xlim(1500,2000)
plt.title('sonic 2-1')

plt.figure()
# plt.scatter(np.arange(len(deficit_moist)),(deficit_moist-np.array(pw_df['PW boom-1 [m^3/s^3]']))/1, color = 'blue', label = 'moist')
# plt.scatter(np.arange(len(deficit_dry)),(deficit_dry-np.array(pw_df['PW boom-1 [m^3/s^3]']))/1, color = 'darkorange', label = 'dry')
# plt.plot(deficit_moist-np.array(pw_df['PW boom-1 [m^3/s^3]']))
plt.plot(np.arange(len(deficit)), deficit, color = 'darkorange')
plt.scatter(np.arange(len(deficit)), deficit, color = 'darkorange', s=5, label = '$P-\epsilon$')

plt.plot(np.arange(len(deficit)), np.array(deficit)+np.array(pw_df['PW boom-1 [m^3/s^3]']), color = 'b')
plt.scatter(np.arange(len(deficit)), np.array(deficit)+np.array(pw_df['PW boom-1 [m^3/s^3]']), color='b', s=5, label = '$P-\epsilon+T_{\widetilde{pw}}$')

plt.plot(pw_df['PW boom-1 [m^3/s^3]'],color='red')
plt.scatter(np.arange(len(pw_df)), pw_df['PW boom-1 [m^3/s^3]'],color='red', s=5, label = '$T_{\widetilde{pw}}$')
plt.title('May Storm Dissipation Deficit Comparison')
plt.ylabel('$[m^3/s^3]$')
plt.xlabel('May Storm Index')
# plt.xlim(0,break_index)
plt.legend(loc='lower right')
plt.xlim(1550,2050)
# plt.xlim(1700,1800)
plt.ylim(-3,1.2)
plt.savefig(plot_savePath+'mayStorm_21dissipationDeficit_comparisonWithPW.pdf')

#%%
# plt.figure()
# plt.plot((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar']), label = r"s3 $(-\overline{u'w'})(\overline{u})$")
# plt.plot((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']), label = r"s2 $(-\overline{u'w'})(\overline{u})$")
# plt.title()

#average this for levels I and II, then do trapZ for sonics 1-3
rho_P_I = rho_df['rho_bar_1_dry']*((-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar'])-(-1*sonic1_df['UpWp_bar']*sonic1_df['Ubar']))
rho_P_II = rho_df['rho_bar_2_dry']*((-1*sonic3_df['UpWp_bar']*sonic3_df['Ubar'])-(-1*sonic2_df['UpWp_bar']*sonic2_df['Ubar']))
rho_P_avg = (rho_P_I + rho_P_II) /2
# rho_eps = rho_df['rho_bar_2_dry']*((np.array(eps_df['epsU_sonic2_MAD'])*np.array(z_df['z_sonic2'])) 

plt.figure()
plt.plot(rho_P_I, label='rho_P_I')
plt.plot(rho_P_II, label='rho_P_II')
# plt.plot(rho_P_avg, label='rho_P_avg')
plt.legend()
plt.ylim(-5,5)
plt.xlim(1500,2000)
plt.title(r"$\rho \cdot (-\overline{u'w'})(\overline{u}) \; [Wm^{-2}]$")
#%%
y_spring = np.vstack((eps_df['epsU_sonic1_MAD'][:break_index+1], eps_df['epsU_sonic3_MAD'][:break_index+1])).T
y_fall = np.vstack((eps_df['epsU_sonic1_MAD'][break_index+1:], eps_df['epsU_sonic3_MAD'][break_index+1:])).T
rho_eps_spring = np.array(rho_df['rho_bar_1_dry'][:break_index+1])*np.trapz(y=y_spring, x=None, dx=5.49)#do trapz for between sonics 1-3
rho_eps_fall = np.array(rho_df['rho_bar_1_dry'][break_index+1:])*np.trapz(y=y_fall, x=None, dx=5.0292)#do trapz for between sonics 1-3

rho_eps = np.concatenate((rho_eps_spring, rho_eps_fall), axis=0)
#%%
deficit = (np.array(rho_P_II))-np.array(rho_eps)
# deficit_minus_pw = deficit 
plt.figure(figsize=(8,3))
plt.scatter(np.arange(len(deficit)), rho_P_II, s=10, color = 'b', label = r'$\rho \cdot P$')
plt.plot(np.arange(len(deficit)), rho_P_II, color = 'b',)
plt.scatter(np.arange(len(deficit)), rho_eps, s=10, color = 'navy', label = r'$\rho \cdot \epsilon$')
plt.plot(np.arange(len(deficit)), rho_eps,  color = 'navy', )
plt.scatter(np.arange(len(deficit)), deficit, s=10, color = 'gray', label = r'$(\rho \cdot P)- (\rho \cdot \epsilon)$')
plt.plot(np.arange(len(deficit)), deficit,  color = 'gray',)
# plt.plot(-1*deficit/10, color = 'k', label = '-1*deficit/10')
plt.scatter(np.arange(len(deficit)), pw_df['PW boom-1 [m^3/s^3]'], s=5, color='darkorange', label = '-PW')
plt.plot(np.arange(len(deficit)), pw_df['PW boom-1 [m^3/s^3]'],color='darkorange',)
plt.scatter(np.arange(len(deficit)), pw_df['PW boom-1 [m^3/s^3]']*-10, s=10, color='red', label = '-PW*10')
plt.plot(np.arange(len(deficit)), pw_df['PW boom-1 [m^3/s^3]']*-10,color='red', )
# plt.scatter(np.arange(len(deficit)),(-1*deficit/10)-pw_df['PW boom-1 [m^3/s^3]'],color='green', label = 'deficit+PW')
plt.legend()
plt.ylim(-1,3.5)
plt.xlim(0,break_index)
plt.xlim(1500,2000)
plt.ylabel('$[m^3/s^3]$')
plt.xlabel('May Storm Index')
plt.title('Dissipation Flux Deficit versus Wave Coherent PW')
plt.savefig(plot_savePath+'mayStorm_32dissipationDeficit_comparisonWithPW.png', dpi = 300)
plt.savefig(plot_savePath+'mayStorm_32dissipationDeficit_comparisonWithPW.pdf')

# deficit_dry = np.array(rho_df['rho_bar_1_dry'])*((-1*np.array(sonic2_df['UpWp_bar'])*np.array(sonic2_df['Ubar']))-(-1*np.array(sonic1_df['UpWp_bar'])*np.array(sonic1_df['Ubar']))) - (np.array(rho_df['rho_bar_1_dry'])*np.array(eps_df['epsU_sonic1_MAD'])*np.array(z_df['z_sonic1']))
# deficit_moist = np.array(rho_df['rho_bar_1_moist'])*((-1*np.array(sonic2_df['UpWp_bar'])*np.array(sonic2_df['Ubar']))-(-1*np.array(sonic1_df['UpWp_bar'])*np.array(sonic1_df['Ubar']))) - (np.array(rho_df['rho_bar_1_moist'])*np.array(eps_df['epsU_sonic1_MAD'])*np.array(z_df['z_sonic1']))









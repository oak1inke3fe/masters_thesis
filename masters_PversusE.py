# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:36:06 2023

@author: oaklin keefe


This file is used to analyze the production, dissipation, buoyancy, and wave-coherent pw terms of the TKE budget equation.

INPUT files:
    prodTerm_combinedAnalysis.csv
    epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv
    buoy_terms_combinedAnalysis.csv
    z_airSide_allSpring.csv
    z_airSide_allFall.csv
    ZoverL_combinedAnalysis.csv
    usr_combinedAnalysis.csv
    
    
    
OUTPUT files:
    Only figures:
        

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import scipy as sp
# import seaborn as sns

print('done with imports')
#%%
# file_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
file_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
# plot_savePath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4\plots/"
plot_savePath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/plots/'



# windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_120to196.csv")
# windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)


# pw_df = pd.read_csv(file_path + 'pw_waveCoherent_allFall.csv')
# pw_df = pw_df.drop(['Unnamed: 0'], axis=1)

prod_df = pd.read_csv(file_path+'prodTerm_combinedAnalysis.csv')
prod_df = prod_df.drop(['Unnamed: 0'], axis=1)


eps_df = pd.read_csv(file_path+"epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv")
eps_df = eps_df.drop(['Unnamed: 0'], axis=1)
# eps_df[eps_df['eps_sonic1'] > 1] = np.nan

buoy_df = pd.read_csv(file_path+'buoy_terms_combinedAnalysis.csv')
buoy_df = buoy_df.drop(['Unnamed: 0'], axis=1)

z_df_spring = pd.read_csv(file_path+'z_airSide_allSpring.csv')
z_df_spring = z_df_spring.drop(['Unnamed: 0'], axis=1)

z_df_fall = pd.read_csv(file_path+'z_airSide_allFall.csv')
z_df_fall = z_df_fall.drop(['Unnamed: 0'], axis=1)

z_df = pd.concat([z_df_spring, z_df_fall], axis=0)

zL_df = pd.read_csv(file_path + 'ZoverL_combinedAnalysis.csv')
zL_df = zL_df.drop(['Unnamed: 0'], axis=1)

usr_df = pd.read_csv(file_path + "usr_combinedAnalysis.csv")
usr_df = usr_df.drop(['Unnamed: 0'], axis=1)


print('done with setting up dataframes')
#%% Get rid of bad wind directions first
all_windDirs = True
onshore = False
offshore = False

# # plt.figure()
# # plt.scatter(windDir_df.index, windDir_df['alpha_s1'])
# # plt.scatter(zL_df.index, zL_df['z/L I dc'])
# # plt.yscale('log')


# index_array = np.arange(len(windDir_df))
# windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, index_array)
# mask_goodWindDir = np.isin(windDir_df['new_index_arr'],index_array)

# windDir_df[mask_goodWindDir] = np.nan

# prod_df[mask_goodWindDir] = np.nan
# eps_df[mask_goodWindDir] = np.nan
# eps_MAD_df[mask_goodWindDir] = np.nan
# eps_newMAD_df[mask_goodWindDir] = np.nan
# buoy_df[mask_goodWindDir] = np.nan
# z_df[mask_goodWindDir] = np.nan
# zL_df[mask_goodWindDir] = np.nan
# windDir_df[mask_goodWindDir] = np.nan
# pw_df[mask_goodWindDir] = np.nan

# print('done with setting up  good wind direction only dataframes')

# # plt.figure()
# # plt.scatter(windDir_df.index, windDir_df['alpha_s1'])
# # plt.scatter(zL_df.index, zL_df['z/L I dc'])

#%% set to near-neutral stability conditions

plt.figure()
# plt.scatter(zL_df.index, zL_df['zL_II_dc'], label = 'z/L level II')
plt.scatter(zL_df.index, zL_df['zL_I_dc'], label = 'z/L level I', color = 'gray')
plt.plot(prod_df['prod_I']*10, label = 'prod LI*10', color = 'black')
plt.legend()
plt.title('z/L and Prod BEFORE neutral mask')

#%%
index_array = np.arange(len(zL_df))
zL_df['new_index_arr'] = np.where((np.abs(zL_df['zL_I_dc'])<=0.5)&(np.abs(zL_df['zL_II_dc'])<=0.5), np.nan, index_array)
# zL_df['new_index_arr'] = np.where((np.abs(zL_df['z/L II dc'])<=0.5), np.nan, index_array)

mask_neutral_zL = np.isin(zL_df['new_index_arr'],index_array)

zL_df[mask_neutral_zL] = np.nan

eps_df[mask_neutral_zL] = np.nan
prod_df[mask_neutral_zL] = np.nan

buoy_df[mask_neutral_zL] = np.nan
z_df[mask_neutral_zL] = np.nan

# windDir_df[mask_neutral_zL] = np.nan
# pw_df[mask_neutral_zL] = np.nan



plt.figure()
# plt.scatter(zL_df.index, zL_df['zL_II_dc'], label = 'z/L level II')
plt.scatter(zL_df.index, zL_df['zL_I_dc'], label = 'z/L level I', color = 'gray')
plt.plot(prod_df['prod_I']*10, label = 'prod LI*10', color = 'black')
plt.legend()
plt.title('z/L and Prod AFTER neutral mask')

# plt.figure()
# plt.scatter(zL_df.index, zL_df['z/L II dc'], label = 'level II')
# plt.scatter(zL_df.index, zL_df['z/L I dc'], label = 'level I')
# plt.legend()



print('done with setting up near-neutral dataframes')

#%% Offshore setting
# all_windDirs = False
# onshore = False
# offshore = True

# windDir_df['offshore_index_arr'] = np.arange(len(windDir_df))
# windDir_df['new_offshore_index_arr'] = np.where((windDir_df['alpha_s4'] >= 270)&(windDir_df['alpha_s4'] <= 359), windDir_df['offshore_index_arr'], np.nan)

# mask_offshoreWindDir = np.isin(windDir_df['new_offshore_index_arr'],windDir_df['offshore_index_arr'])
# windDir_df = windDir_df[mask_offshoreWindDir]

# zL_df = zL_df[mask_offshoreWindDir]
# eps_df = eps_df[mask_offshoreWindDir]
# eps_MAD_df = eps_MAD_df[mask_offshoreWindDir]
# eps_newMAD_df = eps_newMAD_df[mask_offshoreWindDir]
# prod_df = prod_df[mask_offshoreWindDir]
# buoy_df = buoy_df[mask_offshoreWindDir] 
# z_df = z_df[mask_offshoreWindDir]
# pw_df = pw_df[mask_offshoreWindDir]

#%% On-shore setting
# all_windDirs = False
# onshore = True
# offshore = False

# windDir_df['onshore_index_arr'] = np.arange(len(windDir_df))
# windDir_df['new_onshore_index_arr'] = np.where((windDir_df['alpha_s4'] >= 197)&(windDir_df['alpha_s4'] <= 269), windDir_df['onshore_index_arr'], np.nan)

# mask_onshoreWindDir = np.isin(windDir_df['new_onshore_index_arr'],windDir_df['onshore_index_arr'])
# windDir_df = windDir_df[mask_onshoreWindDir]

# zL_df = zL_df[mask_onshoreWindDir]
# eps_df = eps_df[mask_onshoreWindDir]
# eps_MAD_df = eps_MAD_df[mask_onshoreWindDir]
# eps_newMAD_df = eps_newMAD_df[mask_onshoreWindDir]
# prod_df = prod_df[mask_onshoreWindDir]
# buoy_df = buoy_df[mask_onshoreWindDir] 
# z_df = z_df[mask_onshoreWindDir]
# pw_df = pw_df[mask_onshoreWindDir]

#%%

eps_LIII = (eps_df['eps_sonic3']+eps_df['eps_sonic4'])/2
eps_LII = (eps_df['eps_sonic2']+eps_df['eps_sonic3'])/2
eps_LI = (eps_df['eps_sonic1']+eps_df['eps_sonic2'])/2
eps_df['eps_LIII'] = eps_LIII
eps_df['eps_LII'] = eps_LII
eps_df['eps_LI'] = eps_LI



buoy_LIII = buoy_df['buoy_III']
buoy_LII = buoy_df['buoy_II']
buoy_LI = buoy_df['buoy_I']



print('done with getting epsilon with LI,II,II and dfs and buoyancy dfs')

#%% Production minus dissipation (dissipation deficit if result is +)
p_minus_diss_I =  np.array(np.array(prod_df['prod_I'])-np.array(eps_df['eps_LI']))
p_minus_diss_II =  np.array(np.array(prod_df['prod_II'])-np.array(eps_df['eps_LII']))
p_minus_diss_III =  np.array(np.array(prod_df['prod_III'])-np.array(eps_df['eps_LIII']))
print('done doing production minus dissipation')

#%%
# figure titles based on wind directions
if all_windDirs == True:
    pw_timeseries_title = 'Neutral Conditions: P-$T_{\widetilde{pw}}$ vs. $T_{\widetilde{pw}}$ '
    pw_title = 'Neutral Conditions: P-$\epsilon$ vs. $T_{\widetilde{pw}}$ '
    prod_eps_title = 'Neutral Conditions: P vs. $\epsilon$ '
    prodBuoy_eps_title = 'Neutral P+B vs. $\epsilon$ '
elif onshore == True:
    pw_timeseries_title = 'Onshore Neutral Conditions: P-$T_{\widetilde{pw}}$ vs. $T_{\widetilde{pw}}$ '
    pw_title = 'Onshore Winds, Neutral Conditions: P-$\epsilon$ vs. $T_{\widetilde{pw}}$ '
    prod_eps_title = 'Onshore Winds, Neutral Conditions: P vs. $\epsilon$ '
    prodBuoy_eps_title = 'Onshore Winds, Neutral P+B vs. $\epsilon$ '
elif offshore == True:
    pw_timeseries_title = 'Offshore Neutral Conditions: P-$T_{\widetilde{pw}}$ vs. $T_{\widetilde{pw}}$ '
    pw_title = 'Offshore Winds, Neutral Conditions: P-$\epsilon$ vs. $T_{\widetilde{pw}}$ '
    prod_eps_title = 'Offshore Winds, Neutral Conditions: P vs. $\epsilon$ '
    prodBuoy_eps_title = 'Offshore Winds, Neutral P+B vs. $\epsilon$ '

#%%
#make some plots!

fig = plt.figure()
plt.plot(p_minus_diss_I, label = 'dissipation deficit')
# plt.plot(np.array(pw_df['d_dz_pw_theory_I']), label = '$T_{\widetilde{pw}}$')
plt.legend()
plt.ylim(-0.1,0.1)
# plt.title(pw_timeseries_title + "Level I")
plt.title(prod_eps_title + "Level I")

fig = plt.figure()
plt.plot(p_minus_diss_II, label = 'dissipation deficit')
# plt.plot(np.array(pw_df['d_dz_pw_theory_II']), label = '$T_{\widetilde{pw}}$')
plt.legend()
plt.ylim(-0.1,0.1)
# plt.title(pw_timeseries_title + "Level II")
plt.title(prod_eps_title + "Level II")

fig = plt.figure(figsize=(15,8))
plt.plot(p_minus_diss_II, color = 'darkorange', label = 'L II')
plt.plot(p_minus_diss_I, color = 'dodgerblue', label = 'L I')
plt.legend()
# plt.hlines(y=0, xmin=0,xmax=4395,color='k')
# plt.xlim(700,860)
plt.ylim(-0.2,0.2)
plt.xlabel('time')
plt.ylabel('$P-\epsilon$ [m^2/s^3]')
plt.title('Neutral Conditions: $P-\epsilon$ Combined Analysis')
plt.savefig(plot_savePath + "timeseries_neutralPvEps.png", dpi = 300)

#%%
plt.figure()
plt.scatter(zL_df['zL_II_dc'], p_minus_diss_II, color = 'orange', edgecolor = 'red', label = 'L II')
plt.scatter(zL_df['zL_I_dc'], p_minus_diss_I, color = 'dodgerblue', edgecolor = 'navy', label = 'L I')
plt.legend()
plt.xlim(-1,1)
plt.yscale('log')
plt.title("$P - \epsilon$ (Dissipation Deficit)")
plt.xlabel('z/L')
plt.ylabel('$P-\epsilon$ [m^2/s^3]')

#%%
plt.figure()
plt.hist(p_minus_diss_I,40,color = 'dodgerblue', edgecolor = 'white', label = 'L I; bins=40')
plt.hist(p_minus_diss_II,37,color = 'darkorange', edgecolor = 'white', label = 'L II, bins = 37')
plt.xlabel('$P-\epsilon$')
plt.ylabel('occurance')
plt.title('Histogram of $P-\epsilon$')
# plt.xlim(-0.01,0.05)
# plt.ylim(0,50)
plt.vlines(x=0, ymin=0, ymax=400, color = 'k')
plt.legend()
#%%
fig = plt.figure(figsize=(10,10))
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.plot([10**-6,1], [10**-7,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
plt.plot([10**-6,0.1], [10**-5,1],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
plt.scatter(p_minus_diss_I, np.array(pw_df['d_dz_pw_theory_I']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
# plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P - \epsilon$ [$m^2s^{-3}$]')
plt.ylabel('$T_{\widetilde{pw}}$ [$m^2s^{-3}$]')
plt.title(pw_title)
plt.legend()
plt.axis('square')


print('done with plotting dissipation deficit versus wave-coherent pw')
#%%
fig = plt.figure(figsize=(10,10))
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.plot([10**-6,1], [10**-7,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
plt.plot([10**-6,0.1], [10**-5,1],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
plt.scatter(p_minus_diss_I, np.array(pw_df['d_dz_pw_theory_I']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P - \epsilon$ [$m^2s^{-3}$]')
plt.ylabel('$T_{\widetilde{pw}}$ [$m^2s^{-3}$]')
plt.title(pw_title)
plt.legend()
plt.axis('square')
print('done plotting P vs. Eps simple diss (new)')


print('done with plotting dissipation deficit versus wave-coherent pw')
#%%
fig = plt.figure(figsize=(5,5))
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.plot([10**-6,1], [10**-7,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
plt.plot([10**-6,0.1], [10**-5,1],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
plt.scatter(p_minus_diss_I, np.array(pw_df['d_dz_pw_theory_I']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
# plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')

plt.xscale('log')
plt.yscale('log')
plt.ylabel('$P - \epsilon$ [$m^2s^{-3}$]')
plt.xlabel('$T_{\widetilde{pw}}$ [$m^2s^{-3}$]')
plt.title('Neutral Conditions: $(P-\epsilon)$ vs. $T_{\widetilde{pw}}$')
plt.legend()
# plt.axis('square')
# plt.xlim(-0.0001,0.01)
# plt.ylim(-0.01,0.01)
print('done plotting P vs. Eps simple diss (new)')


print('done with plotting dissipation deficit versus wave-coherent pw')
#%%
fig = plt.figure(figsize=(5,5))
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.plot([10**-6,1], [10**-7,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
plt.plot([10**-6,0.1], [10**-5,1],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
plt.scatter(p_minus_diss_I, np.array(pw_df['d_dz_pw_theory_I']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', )

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P - \epsilon$ [$m^2s^{-3}$]')
plt.ylabel('$T_{\widetilde{pw}}$ [$m^2s^{-3}$]')
plt.title(pw_title)
plt.legend()
plt.tight_layout()
# plt.savefig(plot_savePath + "WaveCoherent_scatterplot.png",dpi=300)
# plt.axis('square')
# plt.xlim(-0.0001,0.01)
# plt.ylim(-0.01,0.01)
print('done plotting P vs. Eps simple diss (new)')

print('done with plotting dissipation deficit versus wave-coherent pw')

#%%
fig = plt.figure(figsize=(5,5))
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.plot([10**-6,1], [10**-7,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
plt.plot([10**-6,0.1], [10**-5,1],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', label = 'level II')
plt.scatter(p_minus_diss_I, np.array(pw_df['d_dz_pw_theory_I']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
plt.scatter(p_minus_diss_II, np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red',)
plt.hlines(y=0,xmin=-10,xmax=10, color = 'gray')
plt.vlines(x=0,ymin=-10,ymax=10, color = 'gray')
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('$P - \epsilon$ [$m^2s^{-3}$]')
plt.ylabel('$T_{\widetilde{pw}}$ [$m^2s^{-3}$]')
plt.title(pw_title)
plt.legend()
# plt.axis('square')
plt.xlim(-0.0001,0.01)
plt.ylim(-0.01,0.01)
print('done plotting P vs. Eps simple diss (new)')


print('done with plotting dissipation deficit versus wave-coherent pw')
#%%
# prodPLUSbuoy= pd.DataFrame()
# prodPLUSbuoy['P+B LI'] = np.array(prod_df['prod_I'])+np.array(buoy_LI)
# prodPLUSbuoy['P+B LII'] = np.array(prod_df['prod_II'])+np.array(buoy_LII)
# prodPLUSbuoy['P+B LIII'] = np.array(prod_df['prod_III'])+np.array(buoy_LIII)
# prodPLUSbuoy['date'] = np.array(prod_df['date'])
prodPLUSbuoy= pd.DataFrame()
prodPLUSbuoy['P+B LI'] = np.array(prod_df['prod_I'])+np.array(buoy_LI)
prodPLUSbuoy['P+B LII'] = np.array(prod_df['prod_II'])+np.array(buoy_LII)
prodPLUSbuoy['P+B LIII'] = np.array(prod_df['prod_III'])+np.array(buoy_LIII)

PplusB_minus_eps_df = pd.DataFrame()
PplusB_minus_eps_df['LI'] = np.array(prodPLUSbuoy['P+B LI'])-np.array(eps_df['eps_LI'])
PplusB_minus_eps_df['LII'] = np.array(prodPLUSbuoy['P+B LII'])-np.array(eps_df['eps_LII'])

# prodPLUSbuoy_neutral= pd.DataFrame()
# prodPLUSbuoy_neutral['P+B LI'] = np.array(new_prod_df_neutral['prod_I'])+np.array(buoy_LI_neutral)
# prodPLUSbuoy_neutral['P+B LII'] = np.array(new_prod_df_neutral['prod_II'])+np.array(buoy_LII_neutral)
# prodPLUSbuoy_neutral['P+B LIII'] = np.array(new_prod_df_neutral['prod_III'])+np.array(buoy_LIII_neutral)


print('done with combining production and buoyancy')
#%%
fig = plt.figure(figsize=(5,5))
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.plot([10**-6,1], [10**-7,0.1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
plt.plot([10**-6,0.1], [10**-5,1],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(PplusB_minus_eps_df['LII'], np.array(pw_df['d_dz_pw_theory_II']), color = 'darkorange', edgecolor = 'red', label = 'level II')
plt.scatter(PplusB_minus_eps_df['LI'], np.array(pw_df['d_dz_pw_theory_I']), color = 'dodgerblue', edgecolor = 'navy',label = 'level I')
# plt.scatter(PplusB_minus_eps_df['LII'], np.array(pw_df['d_dz_pw_theory_II']), color = 'orange', edgecolor = 'red', )

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P +B - \epsilon$ [$m^2s^{-3}$]')
plt.ylabel('$T_{\widetilde{pw}}$ [$m^2s^{-3}$]')
plt.title('Neutral Conditions: $P+B-\epsilon$ vs. $T_{\widetilde{pw}}$')
plt.legend()
plt.tight_layout()
# plt.savefig(plot_savePath + "WaveCoherent_vs_dissDeficit_scatterplot.png",dpi=300)
# plt.axis('square')
# plt.xlim(-0.0001,0.01)
# plt.ylim(-0.01,0.01)
print('done plotting P vs. Eps simple diss (new)')

print('done with plotting dissipation deficit versus wave-coherent pw')
#%%
# If we just want to examine the high wind event from Oct2-4, 2022, use the following mask:
    
blank_index = np.arange(0,4395)
prod_df['index_num'] = blank_index
eps_df['index_num'] = blank_index
buoy_df['index_num'] = blank_index
z_df['index_num'] = blank_index
prodPLUSbuoy['index_num'] = blank_index

oct_start = 731+27
oct_end = 913+27

mask_prod = (prod_df['index_num'] >= oct_start) & (prod_df['index_num'] <= oct_end)
prod_df = prod_df.loc[mask_prod]

mask_buoy = (buoy_df['index_num'] >= oct_start) & (buoy_df['index_num'] <= oct_end)
buoy_df = buoy_df.loc[mask_buoy]

mask_prodPLUSbuoy = (prodPLUSbuoy['index_num'] >= oct_start) & (prodPLUSbuoy['index_num'] <= oct_end)
prodPLUSbuoy = prodPLUSbuoy.loc[mask_prodPLUSbuoy]

mask_eps = (eps_df['index_num'] >= oct_start) & (eps_df['index_num'] <= oct_end)
eps_df = eps_df.loc[mask_eps]


mask_z = (z_df['index_num'] >= oct_start) & (z_df['index_num'] <= oct_end)
z_df = z_df.loc[mask_z]

#%%
fig = plt.figure()
plt.plot(buoy_df['buoy_4'], label = 's4')
plt.plot(buoy_df['buoy_3'], label = 's3')
plt.plot(buoy_df['buoy_2'], label = 's2')
plt.plot(buoy_df['buoy_1'], label = 's1')
plt.legend()

print('done with plotting buoyancy')

#%% Correlation coefficients with Fixed ISR
PvEps_all_df = pd.DataFrame()
PvEps_all_df['P_I'] = np.array(prod_df['prod_I'])
PvEps_all_df['Eps_I'] = np.array(eps_df['eps_LI'])
PvEps_all_df['P_II'] = np.array(prod_df['prod_II'])
PvEps_all_df['Eps_II'] = np.array(eps_df['eps_LII'])
PvEps_all_df['P_III'] = np.array(prod_df['prod_III'])
PvEps_all_df['Eps_III'] = np.array(eps_df['eps_LIII'])

PvEps_I_df = pd.DataFrame()
PvEps_I_df['P_I'] = np.array(prod_df['prod_I'])
PvEps_I_df['Eps_I'] = np.array(eps_df['eps_LI'])

PvEps_II_df = pd.DataFrame()
PvEps_II_df['P_II'] = np.array(prod_df['prod_II'])
PvEps_II_df['Eps_II'] = np.array(eps_df['eps_LII'])

PvEps_III_df = pd.DataFrame()
PvEps_III_df['P_III'] = np.array(prod_df['prod_III'])
PvEps_III_df['Eps_III'] = np.array(eps_df['eps_LIII'])

break_index = 3959
PvEps_Spring_df = pd.DataFrame()
PvEps_Spring_df['P_I'] = np.array(prod_df['prod_I'][:break_index+1])
PvEps_Spring_df['Eps_I'] = np.array(eps_df['eps_LI'][:break_index+1])
PvEps_Spring_df['P_II'] = np.array(prod_df['prod_II'][:break_index+1])
PvEps_Spring_df['Eps_II'] = np.array(eps_df['eps_LII'][:break_index+1])
PvEps_Spring_df['P_III'] = np.array(prod_df['prod_III'][:break_index+1])
PvEps_Spring_df['Eps_III'] = np.array(eps_df['eps_LIII'][:break_index+1])
r_spring = PvEps_Spring_df.corr()
r_spring_I = r_spring['P_I'][1]
r_spring_I = round(r_spring_I, 3)
r_spring_II = r_spring['P_II'][3]
r_spring_II = round(r_spring_II, 3)
r_spring_III = r_spring['P_III'][5]
r_spring_III = round(r_spring_III, 3)

PvEps_Fall_df = pd.DataFrame()
PvEps_Fall_df['P_I'] = np.array(prod_df['prod_I'][break_index+1:])
PvEps_Fall_df['Eps_I'] = np.array(eps_df['eps_LI'][break_index+1:])
PvEps_Fall_df['P_II'] = np.array(prod_df['prod_II'][break_index+1:])
PvEps_Fall_df['Eps_II'] = np.array(eps_df['eps_LII'][break_index+1:])
PvEps_Fall_df['P_III'] = np.array(prod_df['prod_III'][break_index+1:])
PvEps_Fall_df['Eps_III'] = np.array(eps_df['eps_LIII'][break_index+1:])
r_fall = PvEps_Fall_df.corr()
r_fall_I = r_fall['P_I'][1]
r_fall_I = round(r_fall_I, 3)
r_fall_II = r_fall['P_II'][3]
r_fall_II = round(r_fall_II, 3)
r_fall_III = r_fall['P_III'][5]
r_fall_III = round(r_fall_III, 3)


r_I = PvEps_I_df.corr()
print(r_I)
r_I_str = r_I['P_I'][1]
r_I_str = round(r_I_str, 3)
print(r_I_str)

r_all = PvEps_all_df.corr()
print(r_all)
r_I_str = r_all['P_I'][1]
r_I_str = round(r_I_str, 3)
r_II_str = r_all['P_II'][3]
r_II_str = round(r_II_str, 3)
r_III_str = r_all['P_III'][5]
r_III_str = round(r_III_str, 3)
print(r_I_str)
print(r_II_str)
print(r_III_str)

#with buoyancy
PBvEps_all_df = pd.DataFrame()
PBvEps_all_df['PB_I'] = np.array(prodPLUSbuoy['P+B LI'])
PBvEps_all_df['Eps_I'] = np.array(eps_df['eps_LI'])
PBvEps_all_df['PB_II'] = np.array(prodPLUSbuoy['P+B LII'])
PBvEps_all_df['Eps_II'] = np.array(eps_df['eps_LII'])
PBvEps_all_df['PB_III'] = np.array(prodPLUSbuoy['P+B LIII'])
PBvEps_all_df['Eps_III'] = np.array(eps_df['eps_LIII'])

r_PB_all = PBvEps_all_df.corr()
print(r_PB_all)
r_I_PB_str = r_PB_all['PB_I'][1]
r_I_PB_str = round(r_I_PB_str, 3)
r_II_PB_str = r_PB_all['PB_II'][3]
r_II_PB_str = round(r_II_PB_str, 3)
r_III_PB_str = r_PB_all['PB_III'][5]
r_III_PB_str = round(r_III_PB_str, 3)
print(r_I_PB_str)
print(r_II_PB_str)
print(r_III_PB_str)


#%% Correlation coefficients of MAD vs Fixed ISR of eps 
Eps_comparison_df = pd.DataFrame()
Eps_comparison_df['LI_MAD'] = np.array(eps_MAD_df['eps_LI'])
Eps_comparison_df['LI_fixedISR'] = np.array(eps_df['eps_LI'])
Eps_comparison_df['LII_MAD'] = np.array(eps_MAD_df['eps_LII'])
Eps_comparison_df['LII_fixedISR'] = np.array(eps_df['eps_LII'])
Eps_comparison_df['LIII_MAD'] = np.array(eps_MAD_df['eps_LIII'])
Eps_comparison_df['LIII_fixedISR'] = np.array(eps_df['eps_LIII'])

r_epsComparison = Eps_comparison_df.corr()
print(r_epsComparison)
r_I_COMP_str = r_epsComparison['LI_MAD'][1]
r_I_COMP_str = round(r_I_COMP_str, 3)
r_II_COMP_str = r_epsComparison['LII_MAD'][3]
r_II_COMP_str = round(r_II_COMP_str, 3)
r_III_COMP_str = r_epsComparison['LIII_MAD'][5]
r_III_COMP_str = round(r_III_COMP_str, 3)
print(r_I_COMP_str)
print(r_II_COMP_str)
print(r_III_COMP_str)
#%%
plt.figure()
plt.scatter(eps_df.index, eps_df.eps_LI)
plt.xlim(2350,2400)

#%% scatterplot of Eps_MAD vs. Eps Fixed frequency

# fig = plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prod_df['prod_III'], eps_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# plt.scatter(eps_MAD_df['eps_LII'], eps_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(eps_MAD_df['eps_LI'], eps_df['eps_LI'], color = 'darkgreen', edgecolor = 'olive', label = 'level I')

# # plt.scatter(np.abs(prod_df['prod_III']), np.abs(eps_df['eps_LIII']), color = 'skyblue', edgecolor = 'navy', label = 'level III')
# # plt.scatter(np.abs(prod_df['prod_II']), np.abs(eps_df['eps_LII']), color = 'darkorange', edgecolor = 'red',label = 'level II')
# # plt.scatter(np.abs(prod_df['prod_I']), np.abs(eps_df['eps_LI']), color = 'darkgreen', edgecolor = 'olive', label = 'level I')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Eps (<u>/<z>) ')
# plt.ylabel('Eps (Fixed ISR) ')
# plt.title('Neutral Conditions comparison of Epsilon Measurements')
# plt.legend(loc = 'lower right')

# # rI, pI = sp.stats.pearsonr(x=prod_df['prod_I'], y=eps_df['eps_LI'])
# ax = plt.gca() 
# plt.text(.05, .9, "Pearson's r L II ={:.3f}".format(r_III_str), transform=ax.transAxes)
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_COMP_str), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_COMP_str), transform=ax.transAxes)


# print('done plotting P vs. Eps simple diss (new)')

#%% NEUTRAL CONDITIONS
# scatterplot of Production versus Dissipation SPRING
fig = plt.figure(figsize=(6,6))
# fig = plt.figure()
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(prod_df['prod_III'][:break_index+1], eps_df['eps_LIII'][:break_index+1], color = 'seagreen', edgecolor = 'darkgreen', label = 'level III')
plt.scatter(prod_df['prod_II'][:break_index+1], eps_df['eps_LII'][:break_index+1], color = 'darkorange', edgecolor = 'red',label = 'level II')
plt.scatter(prod_df['prod_I'][:break_index+1], eps_df['eps_LI'][:break_index+1], color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
# plt.xlim(10**-3,1)
# plt.ylim(10**-2,1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P$ [$m^2s^{-3}$]')
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
plt.title("SPRING: Neutral Conditions: $P$ vs. $\epsilon$")
plt.legend(loc = 'lower right')
ax = plt.gca() 
plt.text(.05, .9, "Pearson's r L III ={:.3f}".format(r_spring_III), transform=ax.transAxes)
plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_spring_II), transform=ax.transAxes)
plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_spring_I), transform=ax.transAxes)
plt.axis('equal')
plt.tight_layout()
plt.savefig(plot_savePath + "scatterplot_PvEps_spring.png",dpi=300)
print('done plotting P vs. Eps SPRING')



# scatterplot of Production versus Dissipation FALL
fig = plt.figure(figsize=(6,6))
# fig = plt.figure()
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
plt.scatter(prod_df['prod_III'][break_index+1:], eps_df['eps_LIII'][break_index+1:], color = 'seagreen', edgecolor = 'darkgreen', label = 'level III')
plt.scatter(prod_df['prod_II'][break_index+1:], eps_df['eps_LII'][break_index+1:], color = 'darkorange', edgecolor = 'red',label = 'level II')
plt.scatter(prod_df['prod_I'][break_index+1:], eps_df['eps_LI'][break_index+1:], color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
# plt.xlim(10**-3,1)
# plt.ylim(10**-2,1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$P$ [$m^2s^{-3}$]')
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
plt.title("FALL: Neutral Conditions: $P$ vs. $\epsilon$")
plt.legend(loc = 'lower right')
ax = plt.gca() 
plt.text(.05, .9, "Pearson's r L III ={:.3f}".format(r_fall_III), transform=ax.transAxes)
plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_fall_II), transform=ax.transAxes)
plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_fall_I), transform=ax.transAxes)
plt.axis('equal')
plt.tight_layout()
plt.savefig(plot_savePath + "scatterplot_PvEps_fall.png",dpi=300)
print('done plotting P vs. Eps FALL')
#%%
# Timeseries
import matplotlib.dates as mdates

date_df = pd.read_csv(file_path+'date_allFall.csv')
dates_arr = np.array(pd.to_datetime(date_df['datetime']))


fig, ax1 = plt.subplots(figsize=(15,4))
ax1.plot(dates_arr, p_minus_diss_I, color = 'dodgerblue', label = 'L I')
ax1.plot(dates_arr, p_minus_diss_II, color = 'darkorange', label = 'L II')
ax1.legend()
# ax1.set_title('Level I')
ax1.set_ylabel(r'$P-\epsilon$ [$m^{2}s^{-3}$]')
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax1.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
fig.suptitle(r'$P-\epsilon$', fontsize=16)

#%%

fig = plt.figure(figsize = (4,4))
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1",linewidth=3) #scale 1-to-1 line
plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10',linewidth=3) #scale line by power of 10
plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--',linewidth=3) #scale line by power of 10
# plt.scatter(prod_df['prod_III'], eps_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
plt.scatter(p_minus_diss_I, p_minus_diss_II, color = 'cyan', edgecolor = 'navy')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Level I $P-\epsilon$ [$m^2s^{-3}$]')
plt.ylabel('Level II $P-\epsilon$ [$m^2s^{-3}$]')
plt.title(r'$P-\epsilon$')
plt.legend(loc = 'upper left')
ax = plt.gca() 
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_str), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_str), transform=ax.transAxes)
plt.axis('square')

print('done plotting P vs. Eps simple diss (new)')
#%% NEUTRAL CONDITIONS
# scatterplot of Production AND Buoyancy versus fixed ISR Dissipation
fig = plt.figure(figsize = (6,6))
# fig= plt.figure()
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prodPLUSbuoy['P+B LIII'], eps_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# plt.scatter(prodPLUSbuoy['P+B LII'], eps_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
plt.scatter(prodPLUSbuoy['P+B LI'], eps_df['eps_LI'], color = 'dodgerblue', edgecolor = 'navy', label = 'level I')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('$P\; +\; B$ [$m^2s^{-3}$]')
plt.ylabel('$\epsilon$ [$m^2s^{-3}$]')
plt.title('P+B vs. $\epsilon$')
ax = plt.gca() 
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_PB_str), transform=ax.transAxes)
plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_PB_str), transform=ax.transAxes)
plt.axis('equal')
plt.tight_layout()
plt.savefig(plot_savePath + "oct2thur4_PplusBvEps_scatterplot.png",dpi=300)



#%%
# sort by z/L
zL_I_df = pd.DataFrame()
zL_I_df['zL_I'] = zL_df['z/L I dc']
zL_I_df['prod_I'] = prod_df['prod_I']
zL_I_df['pb_I'] = prodPLUSbuoy['P+B LI']
zL_I_df['buoy_I'] = buoy_df['buoy_I']
zL_I_df['eps_I'] = eps_df['eps_LI']
zL_I_df.set_index('zL_I')
zL_I_df.sort_index(ascending=True)

zL_II_df = pd.DataFrame()
zL_II_df['zL_II'] = zL_df['z/L II dc']
zL_II_df['prod_II'] = prod_df['prod_II']
zL_II_df['pb_II'] = prodPLUSbuoy['P+B LII']
zL_I_df['buoy_II'] = buoy_df['buoy_II']
zL_II_df['eps_II'] = eps_df['eps_LII']
zL_II_df.set_index('zL_II')
zL_II_df.sort_index(ascending=True)

#%%
plt.figure()
plt.scatter(zL_I_df['zL_I'], zL_I_df['eps_I'], color = 'darkorange', edgecolor = 'red', s=25, label = '$\epsilon$')
plt.scatter(zL_I_df['zL_I'], zL_I_df['prod_I'], color ='lightskyblue', edgecolor = 'navy', s = 15, label = 'P')
plt.scatter(zL_I_df['zL_I'], zL_I_df['buoy_I'], color ='green', edgecolor = 'darkgreen', s = 15, label = 'B')
# plt.hlines(y=0,xmin=-7,xmax=1,color='k')
plt.legend(loc='upper left')
# plt.xlim(-7,0)
# plt.ylim(-0.02,0.02)

# plt.xlim(-1,1)

# plt.yscale('log')
# plt.xscale('log')
plt.title('Dissipation Rate and Production: Level I')
plt.xlabel('Stability (z/L)')
plt.ylabel('$\epsilon$, P, P+B [$m^2s^{-3}$]')

#%% NEUTRAL CONDITIONS MAD P v Eps
# # scatterplot of Production versus MAD Dissipation
# fig = plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prod_df['prod_III'], eps_newMAD_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# plt.scatter(prod_df['prod_II'], eps_newMAD_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(prod_df['prod_I'], eps_newMAD_df['eps_LI'], color = 'darkgreen', edgecolor = 'olive', label = 'level I')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Production [m^2/s^3]')
# plt.ylabel('Dissipation Rate [m^2/s^3]')
# plt.title(prod_eps_title )
# plt.legend(loc = 'lower right')

# # rI, pI = sp.stats.pearsonr(x=prod_df['prod_I'], y=eps_df['eps_LI'])
# ax = plt.gca() 
# # plt.text(.05, .9, "Pearson's r L II ={:.3f}".format(r_III_str), transform=ax.transAxes)
# plt.text(.05, .85, "Pearson's r L II ={:.3f}".format(r_II_MAD_str), transform=ax.transAxes)
# plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_I_MAD_str), transform=ax.transAxes)
# print('done plotting P vs. Eps simple diss (new)')

#%% NEUTRAL CONDITIONS MAD P+B v Eps
# # scatterplot of Production AND Buoyancy versus MAD Dissipation
# fig = plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prodPLUSbuoy['P+B LIII'], eps_newMAD_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# plt.scatter(prodPLUSbuoy['P+B LII'], eps_newMAD_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(prodPLUSbuoy['P+B LI'], eps_newMAD_df['eps_LI'], color = 'darkgreen', edgecolor = 'olive', label = 'level I')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('P+B [m^2/s^3]')
# plt.ylabel('Dissipation Rate [m^2/s^3]')
# plt.title(prodBuoy_eps_title )

#%%
# # scatterplot of Production versus MAD Dissipation
# fig = plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(new_prod_df['prod_III'], eps_newMAD_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# plt.scatter(new_prod_df['prod_II'], eps_newMAD_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(new_prod_df['prod_I'], eps_newMAD_df['eps_LI'], color = 'darkgreen', edgecolor = 'olive', label = 'level I')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Production [m^2/s^3]')
# plt.ylabel('Dissipation Rate [m^2/s^3]')
# plt.title('P vs. $\epsilon$ (MAD)')
# plt.legend()

# print('done plotting P vs. Eps_mad (new)')
# #%%
# # scatterplot of Production versus Dissipation
# fig = plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(new_prod_df['prod_III'], eps_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# plt.scatter(new_prod_df['prod_II'], eps_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(new_prod_df['prod_I'], eps_df['eps_LI'], color = 'darkgreen', edgecolor = 'olive', label = 'level I')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Production [m^2/s^3]')
# plt.ylabel('Dissipation Rate [m^2/s^3]')
# plt.title('P vs. $\epsilon$ (fixed ISR)')
# plt.legend()
# print('done plotting P vs. Eps simple diss (new)')
# #%%
# # scatterplot of Production versus MAD Dissipation
# fig = plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10

# # plt.scatter(prod_df['prod_III'], eps_newMAD_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# # plt.scatter(new_prod_df['prod_III'], eps_newMAD_df['eps_LIII'], color = 'white', edgecolor = 'black', label = 'new level III')
# # plt.scatter(prod_df['prod_II'], eps_newMAD_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
# # plt.scatter(new_prod_df['prod_II'], eps_newMAD_df['eps_LII'], color = 'white', edgecolor = 'black', label = 'new level II')
# plt.scatter(new_prod_df['prod_I'], eps_MAD_df['eps_LI'][27:], color = 'darkgreen', edgecolor = 'olive', label = 'level I')
# plt.scatter(new_prod_df['prod_I'], eps_newMAD_df['eps_LI'], color = 'white', edgecolor = 'black', label = 'new level I')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Production [m^2/s^3]')
# plt.ylabel('Dissipation Rate [m^2/s^3]')
# plt.title('P vs. $\epsilon$')
# plt.legend()
# #%%
# # scatterplot of Production versus MAD Dissipation
# fig = plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10

# # plt.scatter(prod_df['prod_III'], eps_newMAD_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# plt.scatter(new_prod_df['prod_III'], eps_newMAD_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'new level III')
# # plt.scatter(prod_df['prod_II'], eps_newMAD_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(new_prod_df['prod_II'], eps_newMAD_df['eps_LII'], color = 'darkorange', edgecolor = 'red', label = 'new level II')
# # plt.scatter(prod_df['prod_I'], eps_newMAD_df['eps_LI'], color = 'darkgreen', edgecolor = 'olive', label = 'level I')
# plt.scatter(new_prod_df['prod_I'], eps_newMAD_df['eps_LI'], color = 'darkgreen', edgecolor = 'olive', label = 'new level I')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Production [m^2/s^3]')
# plt.ylabel('Dissipation Rate [m^2/s^3]')
# plt.title('P vs. $\epsilon$')
# plt.legend()
# #%%
# # scatterplot of Production versus Dissipation (MAD Epsilon)
# fig = plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prod_df['prod_III'], eps_newMAD_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# plt.scatter(prod_df['prod_II'], eps_newMAD_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(prod_df['prod_I'], eps_newMAD_df['eps_LI'], color = 'darkgreen', edgecolor = 'olive', label = 'level I')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Production [m^2/s^3]')
# plt.ylabel('Dissipation Rate (MAD) [m^2/s^3]')
# plt.title('P vs. $\epsilon$ (New MAD)')
# plt.legend()


#%%
# # scatterplot of Production AND Buoyancy versus Dissipation
# fig = plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prodPLUSbuoy['P+B LIII'], eps_newMAD_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# plt.scatter(prodPLUSbuoy['P+B LII'], eps_newMAD_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(prodPLUSbuoy['P+B LI'], eps_newMAD_df['eps_LI'], color = 'darkgreen', edgecolor = 'olive', label = 'level I')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('P+B [m^2/s^3]')
# plt.ylabel('Dissipation Rate [m^2/s^3]')
# plt.title('P+B vs. $\epsilon$ (MAD)')

# #%%
# # scatterplot of Production AND Buoyancy versus Dissipation
# fig = plt.figure()
# plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.plot([10**-6,10], [10**-7,1], color = 'k', linestyle = '--', label = '+\- O10') #scale line by power of 10
# plt.plot([10**-6,1], [10**-5,10],color = 'k', linestyle = '--') #scale line by power of 10
# plt.scatter(prodPLUSbuoy['P+B LIII'], eps_df['eps_LIII'], color = 'skyblue', edgecolor = 'navy', label = 'level III')
# plt.scatter(prodPLUSbuoy['P+B LII'], eps_df['eps_LII'], color = 'darkorange', edgecolor = 'red',label = 'level II')
# plt.scatter(prodPLUSbuoy['P+B LI'], eps_df['eps_LI'], color = 'darkgreen', edgecolor = 'olive', label = 'level I')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('P+B [m^2/s^3]')
# plt.ylabel('Dissipation Rate [m^2/s^3]')
# plt.title('P+B vs. $\epsilon$ (fixed ISR)')

# #%%
# z1 = z_df['z_s1']
# z2 = z_df['z_s2']
# z3 = z_df['z_s3']
# z4 = z_df['z_s4']

# z_I = (np.array(z1)+np.array(z2))/2
# z_II = (np.array(z2)+np.array(z3))/2
# z_III = (np.array(z3)+np.array(z4))/2

# z_df = pd.DataFrame()
# z_df['z_LI'] = z_I
# z_df['z_LII'] = z_II
# z_df['z_LIII'] = z_III


# P_I = prod_df['prod_I']
# P_II = prod_df['prod_II']
# P_III = prod_df['prod_III']

# prod_df = pd.DataFrame()
# prod_df['prod_LI'] = P_I
# prod_df['prod_LII'] = P_II
# prod_df['prod_LIII'] = P_III

# eps_df = pd.DataFrame()
# eps_df['eps_LI'] = eps_LI
# eps_df['eps_LII'] = eps_LII
# eps_df['eps_LIII'] = eps_LIII

# buoy_df = pd.DataFrame()
# buoy_df['buoy_LI'] = buoy_LI
# buoy_df['buoy_LII'] = buoy_LII
# buoy_df['buoy_LIII'] = buoy_LIII


# # prodPLUSbuoy = prodPLUSbuoy.drop(columns=['date', 'index_num'])


#%% P = Diss
"""
def create_frame(t):
    file_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
    fig = plt.figure(figsize=(6,6))
    
    plt.scatter(eps_df.values[(t)], z_df.values[(t)], color = 'black', edgecolor='black', label = '$\epsilon$')
    plt.plot(eps_df.values[(t)], z_df.values[(t)], color = 'black', linewidth=3)
    plt.scatter(prod_df.values[(t)], z_df.values[(t)], color = 'crimson', edgecolor='crimson', label = 'P')    
    plt.plot(prod_df.values[(t)], z_df.values[(t)], color = 'crimson', linewidth = 3)
    plt.axvline(x=0.0, color = 'silver')
    
    plt.xlim(-0.05,0.45)
    plt.ylim(0,10)
    plt.legend()
    plt.xlabel('P, $\epsilon$ [m^2/s^3]')
    plt.ylabel('altitude (z) [m]')
    plt.title('Vertical Profile of P and $\epsilon$')
    
    plt.savefig(f'./vertProf_{t}.png',
                transparent = False,
                facecolor = 'white')
    plt.close()


print('started')
time = np.arange(0,len(z_df))
for t in time:
    create_frame(t)
frames = []
for t in time:
    image = imageio.v2.imread(f'./vertProf_{t}.png')
    frames.append(image)

file_save_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"    
imageio.mimsave(file_save_path+"vertProfiles_Oct2thru4.gif", #output gif
                frames,                                #array of input frames
                fps = 3,                               #optional: frames per second
                loop = 1)
print('done')

#%% P + B = Diss

def create_frame(t):
    file_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
    fig = plt.figure(figsize=(6,6))
    
    plt.scatter(buoy_df.values[(t)], z_df.values[(t)], color = 'goldenrod', edgecolor='goldenrod', label = 'B')    
    plt.plot(buoy_df.values[(t)], z_df.values[(t)], color = 'goldenrod',linewidth=3)
    plt.scatter(eps_df.values[(t)], z_df.values[(t)], color = 'black', edgecolor='black', label = '$\epsilon$')
    plt.plot(eps_df.values[(t)], z_df.values[(t)], color = 'black',linewidth=3)
    plt.scatter(prod_df.values[(t)], z_df.values[(t)], color = 'crimson', edgecolor='crimson', label = 'P')    
    plt.plot(prod_df.values[(t)], z_df.values[(t)], color = 'crimson',linewidth=3)
    plt.axvline(x=0.0, color = 'silver')
    
    plt.xlim(-0.2,0.45)
    plt.ylim(0,10)
    plt.legend()
    plt.xlabel('P, B, $\epsilon$ [m^2/s^3]')
    plt.ylabel('altitude (z) [m]')
    plt.title('Vertical Profile of P, B, and $\epsilon$')
    
    plt.savefig(f'./vertProf_PBE_{t}.png',
                transparent = False,
                facecolor = 'white')
    plt.close()


print('started')
time = np.arange(0,len(z_df))
for t in time:
    create_frame(t)
frames = []
for t in time:
    image = imageio.v2.imread(f'./vertProf_PBE_{t}.png')
    frames.append(image)

file_save_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"    
imageio.mimsave(file_save_path+"vertProfiles_PBE_Oct2thru4.gif", #output gif
                frames,                                #array of input frames
                fps = 3,                               #optional: frames per second
                loop = 1)
print('done')

"""
#%%
# fig = plt.figure()
# plt.plot([0, 1], [0, 1], 'k-')
# plt.scatter(prodPLUSbuoy['P+B LII'], prod_df['prod_LII'], color = 'green', label = 'level II')
# plt.scatter(prodPLUSbuoy['P+B LI'], prod_df['prod_LI'], color = 'limegreen', label = 'level I')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('P+B [m^2/s^3]')
# plt.ylabel('P only [m^2/s^3]')
# plt.title('P+B vs. P (Entire Fall Dataset)')
# plt.legend()

# #%%
# fig = plt.figure()
# # plt.plot([0, 1], [0, 1], 'k-')
# plt.plot(prodPLUSbuoy['P+B LI'], color = 'skyblue', label = 'P+B')
# plt.plot(prod_df['prod_LI'], color = 'blue', label = 'P')

# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel('P+B [m^2/s^3]')
# plt.ylim(-1,1)
# plt.ylabel('P, P+B [m^2/s^3]')
# plt.title('P versus (P+B) timeseries LI (Entire Fall Dataset)')
# plt.legend()

# #%%
# fig = plt.figure()
# # plt.plot([0, 1], [0, 1], 'k-')
# plt.plot(prodPLUSbuoy['P+B LII'], color = 'gray', label = 'P+B')
# plt.plot(prod_df['prod_LII'], color = 'black', label = 'P')

# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel('P+B [m^2/s^3]')
# plt.ylim(-1,1)
# plt.ylabel('P, P+B [m^2/s^3]')
# plt.title('P versus (P+B) timeseries LII (Entire Fall Dataset)')
# plt.legend()

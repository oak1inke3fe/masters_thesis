# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:57:53 2023

@author: oak
"""


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from hampel import hampel
print('done with imports')
#%%
file_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
plot_save_path = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/Plots/'

eps_UoverZ_Puu = pd.read_csv(file_path + "epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv")
eps_UoverZ_Pww = pd.read_csv(file_path + "epsW_terms_combinedAnalysis_MAD_k_UoverZbar.csv")
# eps_simpFreq_Pww = pd.read_csv(file_path + "eps_allFall_simpleFreqRange_alpha65_ISR_05_10.csv")
# eps_simpFreq_Puu = pd.read_csv(file_path + "eps_allFall_simpleFreqRange_alpha53_ISR_05_10_Puu.csv")

#%%
sonic1_df = pd.DataFrame()
sonic1_df['UoverZ_Pww'] = eps_UoverZ_Pww['epsW_sonic1_MAD']
sonic1_df['UoverZ_Puu'] = eps_UoverZ_Puu['epsU_sonic1_MAD']
# sonic1_df['simpFreq_Pww'] = eps_simpFreq_Pww['eps_sonic1']
# sonic1_df['simpFreq_Puu'] = eps_simpFreq_Puu['eps_sonic1']
r_s1 = sonic1_df.corr()
print(r_s1)

sonic2_df = pd.DataFrame()
sonic2_df['UoverZ_Pww'] = eps_UoverZ_Pww['epsW_sonic2_MAD']
sonic2_df['UoverZ_Puu'] = eps_UoverZ_Puu['epsU_sonic2_MAD']
# sonic2_df['simpFreq_Pww'] = eps_simpFreq_Pww['eps_sonic2']
# sonic2_df['simpFreq_Puu'] = eps_simpFreq_Puu['eps_sonic2']
r_s2 = sonic2_df.corr()
print(r_s2)

sonic3_df = pd.DataFrame()
sonic3_df['UoverZ_Pww'] = eps_UoverZ_Pww['epsW_sonic3_MAD']
sonic3_df['UoverZ_Puu'] = eps_UoverZ_Puu['epsU_sonic3_MAD']
# sonic3_df['simpFreq_Pww'] = eps_simpFreq_Pww['eps_sonic3']
# sonic3_df['simpFreq_Puu'] = eps_simpFreq_Puu['eps_sonic3']
r_s3 = sonic3_df.corr()
print(r_s3)

sonic4_df = pd.DataFrame()
sonic4_df['UoverZ_Pww'] = eps_UoverZ_Pww['epsW_sonic4_MAD']
sonic4_df['UoverZ_Puu'] = eps_UoverZ_Puu['epsU_sonic4_MAD']
# sonic4_df['simpFreq_Pww'] = eps_simpFreq_Pww['eps_sonic4']
# sonic4_df['simpFreq_Puu'] = eps_simpFreq_Puu['eps_sonic4']
r_s4 = sonic4_df.corr()
print(r_s4)

#%%
# sonic_arr = ['1','2','3','4']

# eps_simpFreq_Pww_despiked = pd.DataFrame()
# for sonic in sonic_arr:

#     L_array = eps_simpFreq_Pww['eps_sonic'+sonic]
    
#     # Just outlier detection
#     input_array = L_array
#     window_size = 10
#     n = 3
    
#     L_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
#     # Outlier Imputation with rolling median
#     L_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
#     L_despiked_1times = L_outlier_in_Ts
    
#     # plt.figure()
#     # plt.plot(L_despiked_once)

#     input_array2 = L_despiked_1times
#     L_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
#     # Outlier Imputation with rolling median
#     L_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)

#     eps_simpFreq_Pww_despiked['eps_sonic'+sonic] = L_outlier_in_Ts2
#     print("simp_freq: "+str(sonic))
#     # L_despiked_2times = L_outlier_in_Ts2
    
    
# eps_simpFreq_Puu_despiked = pd.DataFrame()
# for sonic in sonic_arr:

#     L_array = eps_simpFreq_Puu['eps_sonic'+sonic]
    
#     # Just outlier detection
#     input_array = L_array
#     window_size = 10
#     n = 3
    
#     L_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
#     # Outlier Imputation with rolling median
#     L_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
#     L_despiked_1times = L_outlier_in_Ts
    
#     # plt.figure()
#     # plt.plot(L_despiked_once)

#     input_array2 = L_despiked_1times
#     L_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
#     # Outlier Imputation with rolling median
#     L_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)

#     eps_simpFreq_Puu_despiked['eps_sonic'+sonic] = L_outlier_in_Ts2
#     print("simp_freq: "+str(sonic))
#     # L_despiked_2times = L_outlier_in_Ts2
    
#%%
sonic_arr = ['1','2','3','4']

eps_UoverZ_Pww_despiked = pd.DataFrame()
for sonic in sonic_arr:

    L_array = eps_UoverZ_Pww['epsW_sonic'+sonic+'_MAD']
    
    # Just outlier detection
    input_array = L_array
    window_size = 10
    n = 3
    
    L_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
    L_despiked_1times = L_outlier_in_Ts
    
    # plt.figure()
    # plt.plot(L_despiked_once)

    input_array2 = L_despiked_1times
    L_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)

    eps_UoverZ_Pww_despiked['epsW_sonic'+sonic+'_MAD'] = L_outlier_in_Ts2
    print("UoverZ: "+str(sonic))
    # L_despiked_2times = L_outlier_in_Ts2

eps_UoverZ_Puu_despiked = pd.DataFrame()
for sonic in sonic_arr:

    L_array = eps_UoverZ_Puu['epsU_sonic'+sonic+'_MAD']
    
    # Just outlier detection
    input_array = L_array
    window_size = 10
    n = 3
    
    L_outlier_indicies = hampel(input_array, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts = hampel(input_array, window_size, n, imputation=True)
    L_despiked_1times = L_outlier_in_Ts
    
    # plt.figure()
    # plt.plot(L_despiked_once)

    input_array2 = L_despiked_1times
    L_outlier_indicies2 = hampel(input_array2, window_size, n,imputation=False )
    # Outlier Imputation with rolling median
    L_outlier_in_Ts2 = hampel(input_array2, window_size, n, imputation=True)

    eps_UoverZ_Puu_despiked['epsU_sonic'+sonic+'_MAD'] = L_outlier_in_Ts2
    print("UoverZ: "+str(sonic))
    # L_despiked_2times = L_outlier_in_Ts2

#%%
sonic1_df_despiked = pd.DataFrame()
sonic1_df_despiked['mod_Pww'] = eps_UoverZ_Pww_despiked['epsW_sonic1_MAD']
sonic1_df_despiked['mod_Puu'] = eps_UoverZ_Puu_despiked['epsU_sonic1_MAD']
# sonic1_df_despiked['fix_Pww'] = eps_simpFreq_Pww_despiked['eps_sonic1']
# sonic1_df_despiked['fix_Puu'] = eps_simpFreq_Puu_despiked['eps_sonic1']
r_s1_despiked = sonic1_df_despiked.corr()
print(r_s1_despiked)

sonic2_df_despiked = pd.DataFrame()
sonic2_df_despiked['mod_Pww'] = eps_UoverZ_Pww_despiked['epsW_sonic2_MAD']
sonic2_df_despiked['mod_Puu'] = eps_UoverZ_Puu_despiked['epsU_sonic2_MAD']
# sonic2_df_despiked['fix_Pww'] = eps_simpFreq_Pww_despiked['eps_sonic2']
# sonic2_df_despiked['fix_Puu'] = eps_simpFreq_Puu_despiked['eps_sonic2']
r_s2_despiked = sonic2_df_despiked.corr()
print(r_s2_despiked)

sonic3_df_despiked = pd.DataFrame()
sonic3_df_despiked['mod_Pww'] = eps_UoverZ_Pww_despiked['epsW_sonic3_MAD']
sonic3_df_despiked['mod_Puu'] = eps_UoverZ_Puu_despiked['epsU_sonic3_MAD']
# sonic3_df_despiked['fix_Pww'] = eps_simpFreq_Pww_despiked['eps_sonic3']
# sonic3_df_despiked['fix_Puu'] = eps_simpFreq_Puu_despiked['eps_sonic3']
r_s3_despiked = sonic3_df_despiked.corr()
print(r_s3_despiked)

sonic4_df_despiked = pd.DataFrame()
sonic4_df_despiked['mod_Pww'] = eps_UoverZ_Pww_despiked['epsW_sonic4_MAD']
sonic4_df_despiked['mod_Puu'] = eps_UoverZ_Puu_despiked['epsU_sonic4_MAD']
# sonic4_df_despiked['fix_Pww'] = eps_simpFreq_Pww_despiked['eps_sonic4']
# sonic4_df_despiked['fix_Puu'] = eps_simpFreq_Puu_despiked['eps_sonic4']
r_s4_despiked = sonic4_df_despiked.corr()
print(r_s4_despiked)

#%%
LI_despiked = pd.DataFrame()
LI_despiked['mod_Pww'] = np.array((sonic1_df_despiked['mod_Pww']+sonic2_df_despiked['mod_Pww'])/2)
LI_despiked['mod_Puu'] = np.array((sonic1_df_despiked['mod_Puu']+sonic2_df_despiked['mod_Puu'])/2)
# LI_despiked['fix_Pww'] = np.array((sonic1_df_despiked['fix_Pww']+sonic2_df_despiked['fix_Pww'])/2)
# LI_despiked['fix_Puu'] = np.array((sonic1_df_despiked['fix_Puu']+sonic2_df_despiked['fix_Puu'])/2)

LII_despiked = pd.DataFrame()
LII_despiked['mod_Pww'] = np.array((sonic2_df_despiked['mod_Pww']+sonic3_df_despiked['mod_Pww'])/2)
LII_despiked['mod_Puu'] = np.array((sonic2_df_despiked['mod_Puu']+sonic3_df_despiked['mod_Puu'])/2)
# LII_despiked['fix_Pww'] = np.array((sonic2_df_despiked['fix_Pww']+sonic3_df_despiked['fix_Pww'])/2)
# LII_despiked['fix_Puu'] = np.array((sonic2_df_despiked['fix_Puu']+sonic3_df_despiked['fix_Puu'])/2)

print('done with making LI and LII dataframes')

#%%
L_I_mod_Pww_arr = np.array(LI_despiked['mod_Pww'])
percentile_95_I_mod_Pww = np.nanpercentile(np.abs(L_I_mod_Pww_arr), 95)
percentile_99_I_mod_Pww = np.nanpercentile(np.abs(L_I_mod_Pww_arr), 99)
L_I_mod_Pww_newArr_95 = np.where(np.abs(L_I_mod_Pww_arr) > percentile_95_I_mod_Pww, np.nan, L_I_mod_Pww_arr)
L_I_mod_Pww_newArr_99 = np.where(np.abs(L_I_mod_Pww_arr) > percentile_99_I_mod_Pww, np.nan, L_I_mod_Pww_arr)

L_I_mod_Puu_arr = np.array(LI_despiked['mod_Puu'])
percentile_95_I_mod_Puu = np.nanpercentile(np.abs(L_I_mod_Puu_arr), 95)
percentile_99_I_mod_Puu = np.nanpercentile(np.abs(L_I_mod_Puu_arr), 99)
L_I_mod_Puu_newArr_95 = np.where(np.abs(L_I_mod_Puu_arr) > percentile_95_I_mod_Puu, np.nan, L_I_mod_Puu_arr)
L_I_mod_Puu_newArr_99 = np.where(np.abs(L_I_mod_Puu_arr) > percentile_99_I_mod_Puu, np.nan, L_I_mod_Puu_arr)

# L_I_fix_Pww_arr = np.array(LI_despiked['fix_Pww'])
# percentile_95_I_fix_Pww = np.nanpercentile(np.abs(L_I_fix_Pww_arr), 95)
# percentile_99_I_fix_Pww = np.nanpercentile(np.abs(L_I_fix_Pww_arr), 99)
# L_I_fix_Pww_newArr_95 = np.where(np.abs(L_I_fix_Pww_arr) > percentile_95_I_fix_Pww, np.nan, L_I_fix_Pww_arr)
# L_I_fix_Pww_newArr_99 = np.where(np.abs(L_I_fix_Pww_arr) > percentile_99_I_fix_Pww, np.nan, L_I_fix_Pww_arr)

# L_I_fix_Puu_arr = np.array(LI_despiked['fix_Puu'])
# percentile_95_I_fix_Puu = np.nanpercentile(np.abs(L_I_fix_Puu_arr), 95)
# percentile_99_I_fix_Puu = np.nanpercentile(np.abs(L_I_fix_Puu_arr), 99)
# L_I_fix_Puu_newArr_95 = np.where(np.abs(L_I_fix_Puu_arr) > percentile_95_I_fix_Puu, np.nan, L_I_fix_Puu_arr)
# L_I_fix_Puu_newArr_99 = np.where(np.abs(L_I_fix_Puu_arr) > percentile_99_I_fix_Puu, np.nan, L_I_fix_Puu_arr)






L_II_mod_Pww_arr = np.array(LII_despiked['mod_Pww'])
percentile_95_II_mod_Pww = np.nanpercentile(np.abs(L_II_mod_Pww_arr), 95)
percentile_99_II_mod_Pww = np.nanpercentile(np.abs(L_II_mod_Pww_arr), 99)
L_II_mod_Pww_newArr_95 = np.where(np.abs(L_II_mod_Pww_arr) > percentile_95_II_mod_Pww, np.nan, L_II_mod_Pww_arr)
L_II_mod_Pww_newArr_99 = np.where(np.abs(L_II_mod_Pww_arr) > percentile_99_II_mod_Pww, np.nan, L_II_mod_Pww_arr)

L_II_mod_Puu_arr = np.array(LII_despiked['mod_Puu'])
percentile_95_II_mod_Puu = np.nanpercentile(np.abs(L_II_mod_Puu_arr), 95)
percentile_99_II_mod_Puu = np.nanpercentile(np.abs(L_II_mod_Puu_arr), 99)
L_II_mod_Puu_newArr_95 = np.where(np.abs(L_II_mod_Puu_arr) > percentile_95_II_mod_Puu, np.nan, L_II_mod_Puu_arr)
L_II_mod_Puu_newArr_99 = np.where(np.abs(L_II_mod_Puu_arr) > percentile_99_II_mod_Puu, np.nan, L_II_mod_Puu_arr)

# L_II_fix_Pww_arr = np.array(LII_despiked['fix_Pww'])
# percentile_95_II_fix_Pww = np.nanpercentile(np.abs(L_II_fix_Pww_arr), 95)
# percentile_99_II_fix_Pww = np.nanpercentile(np.abs(L_II_fix_Pww_arr), 99)
# L_II_fix_Pww_newArr_95 = np.where(np.abs(L_II_fix_Pww_arr) > percentile_95_II_fix_Pww, np.nan, L_II_fix_Pww_arr)
# L_II_fix_Pww_newArr_99 = np.where(np.abs(L_II_fix_Pww_arr) > percentile_99_II_fix_Pww, np.nan, L_II_fix_Pww_arr)

# L_II_fix_Puu_arr = np.array(LII_despiked['fix_Puu'])
# percentile_95_II_fix_Puu = np.nanpercentile(np.abs(L_II_fix_Puu_arr), 95)
# percentile_99_II_fix_Puu = np.nanpercentile(np.abs(L_II_fix_Puu_arr), 99)
# L_II_fix_Puu_newArr_95 = np.where(np.abs(L_II_fix_Puu_arr) > percentile_95_II_fix_Puu, np.nan, L_II_fix_Puu_arr)
# L_II_fix_Puu_newArr_99 = np.where(np.abs(L_II_fix_Puu_arr) > percentile_99_II_fix_Puu, np.nan, L_II_fix_Puu_arr)


#%%
perc99_LI_df = pd.DataFrame()
perc99_LI_df['mod_Pww'] = L_I_mod_Pww_newArr_99
perc99_LI_df['mod_Puu'] = L_I_mod_Puu_newArr_99
# perc99_LI_df['fix_Pww'] = L_I_fix_Pww_newArr_99
# perc99_LI_df['fix_Puu'] = L_I_fix_Puu_newArr_99
r_LI_99 = perc99_LI_df.corr()
print(r_LI_99)
r_LI_99_modComparison_str = round(r_LI_99['mod_Pww'][1],3)
# r_LI_99_fixComparison_str = round(r_LI_99['fix_Pww'][3],3)

perc99_LII_df = pd.DataFrame()
perc99_LII_df['mod_Pww'] = L_II_mod_Pww_newArr_99
perc99_LII_df['mod_Puu'] = L_II_mod_Puu_newArr_99
# perc99_LII_df['fix_Pww'] = L_II_fix_Pww_newArr_99
# perc99_LII_df['fix_Puu'] = L_II_fix_Puu_newArr_99
r_LII_99 = perc99_LII_df.corr()
print(r_LII_99)
r_LII_99_modComparison_str = round(r_LII_99['mod_Pww'][1],3)
# r_LII_99_fixComparison_str = round(r_LII_99['fix_Pww'][3],3)
#%%
test_mod = (0.890385+0.611871+0.873273)/3 
print('Mod variance = '+ str(test_mod))
test_fix = (0.933112+0.875262+0.839580)/3 
print('Fix variance = '+ str(test_fix))
test_U_mod_fix = (0.979066+0.913381+0.595852)/3 
print('Puu variance = '+ str(test_U_mod_fix))
test_W_mod_fix = (0.971048+0.765345+0.508209)/3 
print('Pww variance = '+ str(test_W_mod_fix))
test_modPw_fixPu = (0.847417+0.434470+0.418485)/3
print('ModPww v. FixPuu variance = '+ str(test_modPw_fixPu))
test_modPu_fixPw = (0.944603+0.892124+0.711380)/3
print('ModPuu v. FixPww variance = '+ str(test_modPu_fixPw))
#%% all values
plt.figure()
plt.scatter((eps_UoverZ_Pww_despiked['eps_sonic2_MAD'][27:]+eps_UoverZ_Pww_despiked['eps_sonic3_MAD'][27:])/2,(eps_UoverZ_Puu_despiked['eps_sonic2_MAD'][27:]+eps_UoverZ_Puu_despiked['eps_sonic3_MAD'][27:])/2, color = 'orange',edgecolor='red', label = 'L II')
plt.scatter((eps_UoverZ_Pww_despiked['eps_sonic1_MAD'][27:]+eps_UoverZ_Pww_despiked['eps_sonic2_MAD'][27:])/2,(eps_UoverZ_Puu_despiked['eps_sonic1_MAD'][27:]+eps_UoverZ_Puu_despiked['eps_sonic2_MAD'][27:])/2, color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title('Dissipation Rate Estimates (DC) ($\epsilon$ [$m^{2}s^{-3}$])', fontsize=12)
plt.xlabel('Modeled ISR $\epsilon_{Pww}$')
plt.ylabel('Modeled ISR $\epsilon_{Puu}$')
plt.axis('square')
plt.xlim(-0.1,1.0)
plt.ylim(-0.1,1.0)
plt.legend(loc='upper left')


plt.figure()
plt.scatter((eps_simpFreq_Pww_despiked['eps_sonic2'][27:]+eps_simpFreq_Pww_despiked['eps_sonic3'][27:])/2,(eps_UoverZ_Puu_despiked['eps_sonic2_MAD'][27:]+eps_UoverZ_Puu_despiked['eps_sonic3_MAD'][27:])/2, color = 'orange',edgecolor='red', label = 'L II')
plt.scatter((eps_simpFreq_Pww_despiked['eps_sonic1'][27:]+eps_simpFreq_Pww_despiked['eps_sonic2'][27:])/2,(eps_UoverZ_Puu_despiked['eps_sonic1_MAD'][27:]+eps_UoverZ_Puu_despiked['eps_sonic2_MAD'][27:])/2, color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title('Dissipation Rate Estimates (DC) ($\epsilon$ [$m^{2}s^{-3}$])', fontsize=12)
plt.xlabel('Fixed ISR $\epsilon_{Pww}$')
plt.ylabel('Modeled ISR $\epsilon_{Puu}$')
plt.axis('square')
plt.xlim(-0.1,1.0)
plt.ylim(-0.1,1.0)
plt.legend(loc='upper left')

plt.figure()
plt.scatter((eps_simpFreq_Pww_despiked['eps_sonic2'][27:]+eps_simpFreq_Pww_despiked['eps_sonic3'][27:])/2,(eps_UoverZ_Pww_despiked['eps_sonic2_MAD'][27:]+eps_UoverZ_Pww_despiked['eps_sonic3_MAD'][27:])/2, color = 'orange',edgecolor='red', label = 'L II')
plt.scatter((eps_simpFreq_Pww_despiked['eps_sonic1'][27:]+eps_simpFreq_Pww_despiked['eps_sonic2'][27:])/2,(eps_UoverZ_Pww_despiked['eps_sonic1_MAD'][27:]+eps_UoverZ_Pww_despiked['eps_sonic2_MAD'][27:])/2, color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title('Dissipation Rate Estimates (DC) ($\epsilon$ [$m^{2}s^{-3}$])', fontsize=12)
plt.xlabel('Fixed ISR $\epsilon_{Pww}$')
plt.ylabel('Modeled ISR $\epsilon_{Pww}$')
plt.axis('square')
plt.xlim(-0.1,1.0)
plt.ylim(-0.1,1.0)
plt.legend(loc='upper left')
#%% 99-th percentils
plt.figure()
plt.scatter(L_II_mod_Pww_newArr_99, L_II_mod_Puu_newArr_99, color = 'orange',edgecolor='red', label = 'L II')
plt.scatter(L_I_mod_Pww_newArr_99, L_I_mod_Puu_newArr_99, color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title('Dissipation Rate Estimates ($\epsilon$ [$m^{2}s^{-3}$]) \n Moving ISR; 99%-ile', fontsize=12)
plt.xlabel('$\epsilon_{Pww}$')
plt.ylabel('$\epsilon_{Puu}$')
plt.axis('square')
plt.xlim(-0.01,0.3)
plt.ylim(-0.01,0.3)
plt.legend(loc='lower right')
ax = plt.gca() 
plt.text(.05, .9, "Pearson's r L II ={:.3f}".format(r_LII_99_modComparison_str), transform=ax.transAxes)
plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_LI_99_modComparison_str), transform=ax.transAxes)


plt.figure()
plt.scatter(L_II_fix_Pww_newArr_99, L_II_fix_Puu_newArr_99, color = 'orange',edgecolor='red', label = 'L II')
plt.scatter(L_I_fix_Pww_newArr_99, L_I_fix_Puu_newArr_99, color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title('Dissipation Rate Estimates ($\epsilon$ [$m^{2}s^{-3}$]) \n Fixed ISR; 99%-ile', fontsize=12)
plt.xlabel('$\epsilon_{Pww}$')
plt.ylabel('$\epsilon_{Puu}$')
plt.axis('square')
plt.xlim(-0.01,0.3)
plt.ylim(-0.01,0.3)
plt.legend(loc='lower right')
ax = plt.gca()
plt.text(.05, .9, "Pearson's r L II ={:.3f}".format(r_LII_99_fixComparison_str), transform=ax.transAxes)
plt.text(.05, .8, "Pearson's r L I ={:.3f}".format(r_LI_99_fixComparison_str), transform=ax.transAxes)


#%%

plt.figure()
plt.scatter(L_II_fix_Pww_newArr_99, L_II_mod_Puu_newArr_99, color = 'orange',edgecolor='red', label = 'L II')
plt.scatter(L_I_fix_Pww_newArr_99, L_I_mod_Puu_newArr_99, color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title('Dissipation Rate Estimates ($\epsilon$ [$m^{2}s^{-3}$]) \n DC; 99%-ile', fontsize=12)
plt.xlabel('Fixed ISR $\epsilon_{Pww}$')
plt.ylabel('Modeled ISR $\epsilon_{Puu}$')
plt.axis('square')
plt.xlim(-0.01,0.3)
plt.ylim(-0.01,0.3)
plt.legend(loc='upper left')

plt.figure()
plt.scatter(L_II_fix_Pww_newArr_99, L_II_mod_Pww_newArr_99, color = 'orange',edgecolor='red', label = 'L II')
plt.scatter(L_I_fix_Pww_newArr_99, L_I_mod_Pww_newArr_99, color = 'green', edgecolor = 'darkgreen', label = 'L I')
plt.plot([-0.1, 1], [-0.1, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title('Dissipation Rate Estimates ($\epsilon$ [$m^{2}s^{-3}$]) \n DC; 99%-ile', fontsize=12)
plt.xlabel('Fixed ISR $\epsilon_{Pww}$')
plt.ylabel('Modeled ISR $\epsilon_{Pww}$')
plt.axis('square')
plt.xlim(-0.01,0.3)
plt.ylim(-0.01,0.3)
plt.legend(loc='upper left')

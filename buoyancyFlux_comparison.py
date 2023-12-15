# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:31:39 2023

@author: oak
"""

#buoyancy flux [W/m^2]=[kg/s^3] = rho*Cp*<w'Tv'>

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('done with imports')

#%%
file_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
rho_df = pd.read_csv(file_path+"rho_bar_allFall.csv")
rho_df = rho_df[27:]
WpTp_bar_df = pd.read_csv(file_path+"WpTp_bar_allSonics_allFall.csv")
WpTp_bar_df = WpTp_bar_df[27:]
Tbar_df = pd.read_csv(file_path+"Tbar_allSonics_allFall.csv")
Tbar_df = Tbar_df[27:]

# Tbar_df['Tbar_s1_C'] = Tbar_df['Tbar_s1']-273.15
z_df = pd.read_csv(file_path+"z_airSide_allFall.csv")
buoy_df = pd.read_csv(file_path+"New_buoy_terms_allFall.csv")
g = -9.81
#%%
BuoyancyFlux_coare_df = pd.DataFrame()
sonic_arr = ['1','2','3','4']
for sonic_num in sonic_arr:
# sonic_num = str(1)
    file_name = 'coare_outputs_s'+sonic_num+'_Warm_UbarGreaterThan2ms.txt'
    A_hdr = 'usr\ttau\thsb\thlb\thbb\thsbb\thlwebb\ttsr\tqsr\tzo\tzot\tzoq\tCd\t'
    A_hdr += 'Ch\tCe\tL\tzeta\tdT_skinx\tdq_skinx\tdz_skin\tUrf\tTrf\tQrf\t'
    A_hdr += 'RHrf\tUrfN\tTrfN\tQrfN\tlw_net\tsw_net\tLe\trhoa\tUN\tU10\tU10N\t'
    A_hdr += 'Cdn_10\tChn_10\tCen_10\thrain\tQs\tEvap\tT10\tT10N\tQ10\tQ10N\tRH10\t'
    A_hdr += 'P10\trhoa10\tgust\twc_frac\tEdis\tdT_warm\tdz_warm\tdT_warm_to_skin\tdu_warm'
    coare_warm = np.genfromtxt(file_path + file_name, delimiter='\t')
    BuoyancyFlux_coare_df['B_sonic'+sonic_num] = np.array(coare_warm[:,4])

    print('buoyancy coare: did this with sonic '+sonic_num)
#%%
Cp = 1004.67
BuoyancyFlux_dc_df = pd.DataFrame()
BuoyancyFlux_dc_df['new_index'] = np.arange(len(BuoyancyFlux_coare_df))
BuoyancyFlux_dc_df['B_sonic1'] = rho_df['rho_bar_1']*WpTp_bar_df['WpTp_bar_s1']*Cp
BuoyancyFlux_dc_df['B_sonic2'] = rho_df['rho_bar_2']*WpTp_bar_df['WpTp_bar_s2']*Cp
BuoyancyFlux_dc_df['B_sonic3'] = rho_df['rho_bar_3']*WpTp_bar_df['WpTp_bar_s3']*Cp
BuoyancyFlux_dc_df['B_sonic4'] = rho_df['rho_bar_3']*WpTp_bar_df['WpTp_bar_s4']*Cp

dc_buoyFlux_despiked = pd.DataFrame()
from hampel import hampel

sonic_arr = ['1','2','3','4']
for sonic in sonic_arr:

    L_array = (BuoyancyFlux_dc_df['B_sonic'+sonic])
    
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
    dc_buoyFlux_despiked['B_sonic'+sonic] = L_outlier_in_Ts2
    print("DC buoy despike sonic: "+str(sonic))

coare_buoyFlux_despiked = pd.DataFrame()
sonic_arr = ['1','2','3','4']
for sonic in sonic_arr:

    L_array = (BuoyancyFlux_coare_df['B_sonic'+sonic])
    
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
    coare_buoyFlux_despiked['B_sonic'+sonic] = L_outlier_in_Ts2
    print("COARE buoy despike sonic: "+str(sonic))



#%%
windDir_df = pd.read_csv(file_path + "windDir_withBadFlags_120to196.csv")
windDir_df = windDir_df.drop(['Unnamed: 0'], axis=1)
index_array = np.arange(len(windDir_df))
windDir_df['new_index_arr'] = np.where((windDir_df['good_wind_dir'])==True, np.nan, index_array)
mask_goodWindDir = np.isin(windDir_df['new_index_arr'],index_array)

windDir_df[mask_goodWindDir] = np.nan

BF_s1 = pd.DataFrame()
BF_s1['dc'] = dc_buoyFlux_despiked['B_sonic1']
BF_s1['coare'] = coare_buoyFlux_despiked['B_sonic1']
BF_s1[mask_goodWindDir] = np.nan
r_BF1 = BF_s1.corr()
print(r_BF1)

BF_s2 = pd.DataFrame()
BF_s2['dc'] = dc_buoyFlux_despiked['B_sonic2']
BF_s2['coare'] = coare_buoyFlux_despiked['B_sonic2']
BF_s2[mask_goodWindDir] = np.nan
r_BF2 = BF_s2.corr()
print(r_BF2)

BF_s3 = pd.DataFrame()
BF_s3['dc'] = dc_buoyFlux_despiked['B_sonic3']
BF_s3['coare'] = coare_buoyFlux_despiked['B_sonic3']
BF_s3[mask_goodWindDir] = np.nan
r_BF3 = BF_s3.corr()
print(r_BF3)

BF_s4 = pd.DataFrame()
BF_s4['dc'] = dc_buoyFlux_despiked['B_sonic4']
BF_s4['coare'] = coare_buoyFlux_despiked['B_sonic4']
BF_s4[mask_goodWindDir] = np.nan
r_BF4 = BF_s4.corr()
print(r_BF4)

#%%
BF_LI = pd.DataFrame()
BF_LI['dc'] = (dc_buoyFlux_despiked['B_sonic1']+dc_buoyFlux_despiked['B_sonic2'])/2
BF_LI['coare'] = (coare_buoyFlux_despiked['B_sonic1']+coare_buoyFlux_despiked['B_sonic2'])/2
BF_LI[mask_goodWindDir] = np.nan
r_BF_LI = BF_LI.corr()
print(r_BF_LI)
r_BF_LI_str = r_BF_LI['dc'][1]
r_BF_LI_str = round(r_BF_LI_str, 3)

BF_LII = pd.DataFrame()
BF_LII['dc'] = (dc_buoyFlux_despiked['B_sonic2']+dc_buoyFlux_despiked['B_sonic3'])/2
BF_LII['coare'] = (coare_buoyFlux_despiked['B_sonic2']+coare_buoyFlux_despiked['B_sonic3'])/2
BF_LII[mask_goodWindDir] = np.nan
r_BF_LII = BF_LII.corr()
print(r_BF_LII)
r_BF_LII_str = r_BF_LII['dc'][1]
r_BF_LII_str = round(r_BF_LII_str, 3)


#%%
BF_I_dc_arr = np.array(BF_LI['dc'])
percentile_95_I_dc = np.nanpercentile(np.abs(BF_I_dc_arr), 95)
percentile_99_I_dc = np.nanpercentile(np.abs(BF_I_dc_arr), 99)
print(percentile_95_I_dc)
BF_I_dc_newArr_95 = np.where(np.abs(BF_I_dc_arr) > percentile_95_I_dc, np.nan, BF_I_dc_arr)
BF_I_dc_newArr_99 = np.where(np.abs(BF_I_dc_arr) > percentile_99_I_dc, np.nan, BF_I_dc_arr)

BF_I_coare_arr = np.array(BF_LI['coare'])
percentile_95_I_coare = np.nanpercentile(np.abs(BF_I_coare_arr), 95)
percentile_99_I_coare = np.nanpercentile(np.abs(BF_I_coare_arr), 99)
print(percentile_95_I_coare)
BF_I_coare_newArr_95 = np.where(np.abs(BF_I_coare_arr) > percentile_95_I_coare, np.nan, BF_I_coare_arr)
BF_I_coare_newArr_99 = np.where(np.abs(BF_I_coare_arr) > percentile_99_I_coare, np.nan, BF_I_coare_arr)

plt.figure()
plt.plot(BF_I_dc_newArr_95, label = 'dc')
plt.plot(BF_I_coare_newArr_95, label = 'coare')
plt.legend()
plt.title('95% BF Level I')

BF_95p_I_df = pd.DataFrame()
BF_95p_I_df['dc'] = BF_I_dc_newArr_95
BF_95p_I_df['coare'] = BF_I_coare_newArr_95

r_BF_95p_I = BF_95p_I_df.corr()
print(r_BF_95p_I)
r_BF_95_I_str = r_BF_95p_I['dc'][1]
r_BF_95_I_str = round(r_BF_95_I_str, 3)

plt.figure()
plt.plot(BF_I_dc_newArr_99, label = 'dc')
plt.plot(BF_I_coare_newArr_99, label = 'coare')
plt.legend()
plt.title('99% BF Level I')

BF_99p_I_df = pd.DataFrame()
BF_99p_I_df['dc'] = BF_I_dc_newArr_99
BF_99p_I_df['coare'] = BF_I_coare_newArr_99

r_BF_99p_I = BF_99p_I_df.corr()
print(r_BF_99p_I)
r_BF_99_I_str = r_BF_99p_I['dc'][1]
r_BF_99_I_str = round(r_BF_99_I_str, 3)

#%%


BF_II_dc_arr = np.array(BF_LII['dc'])
percentile_95_II_dc = np.nanpercentile(np.abs(BF_II_dc_arr), 95)
percentile_99_II_dc = np.nanpercentile(np.abs(BF_II_dc_arr), 99)
print(percentile_95_II_dc)
BF_II_dc_newArr_95 = np.where(np.abs(BF_II_dc_arr) > percentile_95_II_dc, np.nan, BF_II_dc_arr)
BF_II_dc_newArr_99 = np.where(np.abs(BF_II_dc_arr) > percentile_99_II_dc, np.nan, BF_II_dc_arr)

BF_II_coare_arr = np.array(BF_LII['coare'])
percentile_95_II_coare = np.nanpercentile(np.abs(BF_II_coare_arr), 95)
percentile_99_II_coare = np.nanpercentile(np.abs(BF_II_coare_arr), 99)
print(percentile_95_II_coare)
BF_II_coare_newArr_95 = np.where(np.abs(BF_II_coare_arr) > percentile_95_II_coare, np.nan, BF_II_coare_arr)
BF_II_coare_newArr_99 = np.where(np.abs(BF_II_coare_arr) > percentile_99_II_coare, np.nan, BF_II_coare_arr)

plt.figure()
plt.plot(BF_II_dc_newArr_95, label = 'dc')
plt.plot(BF_II_coare_newArr_95, label = 'coare')
plt.legend()
plt.title('95% BF Level II')

BF_95p_II_df = pd.DataFrame()
BF_95p_II_df['dc'] = BF_II_dc_newArr_95
BF_95p_II_df['coare'] = BF_II_coare_newArr_95

r_BF_95p_II = BF_95p_II_df.corr()
print(r_BF_95p_II)
r_BF_95_II_str = r_BF_95p_II['dc'][1]
r_BF_95_II_str = round(r_BF_95_II_str, 3)

plt.figure()
plt.plot(BF_II_dc_newArr_99, label = 'dc')
plt.plot(BF_II_coare_newArr_99, label = 'coare')
plt.legend()
plt.title('99% BF Level II')

BF_99p_II_df = pd.DataFrame()
BF_99p_II_df['dc'] = BF_II_dc_newArr_99
BF_99p_II_df['coare'] = BF_II_coare_newArr_99

r_BF_99p_II = BF_99p_II_df.corr()
print(r_BF_99p_II)
r_BF_99_II_str = r_BF_99p_II['dc'][1]
r_BF_99_II_str = round(r_BF_99_II_str, 3)

#%%
plt.figure()
plt.plot(dc_buoyFlux_despiked['B_sonic1'], label = 'dc')
plt.plot(coare_buoyFlux_despiked['B_sonic1'], label = 'coare')
plt.ylim(-100,300)
plt.legend()
plt.title('Sonic 1: Buoyancy Flux DC bv. COARE')


plt.figure()
plt.plot(dc_buoyFlux_despiked['B_sonic2'], label = 'dc')
plt.plot(coare_buoyFlux_despiked['B_sonic2'], label = 'coare')
plt.ylim(-100,300)
plt.legend()
plt.title('Sonic 2: Buoyancy Flux DC bv. COARE')

plt.figure()
plt.plot(dc_buoyFlux_despiked['B_sonic3'], label = 'dc')
plt.plot(coare_buoyFlux_despiked['B_sonic3'], label = 'coare')
plt.ylim(-100,300)
plt.legend()
plt.title('Sonic 3: Buoyancy Flux DC bv. COARE')

plt.figure()
plt.plot(dc_buoyFlux_despiked['B_sonic4'], label = 'dc')
plt.plot(coare_buoyFlux_despiked['B_sonic4'], label = 'coare')
plt.ylim(-100,300)
plt.legend()
plt.title('Sonic 4: Buoyancy Flux DC bv. COARE')


#%%
plt.figure()
plt.plot(BF_LI['dc'], label = 'dc')
plt.plot(BF_LI['coare'], label = 'coare')
plt.ylim(-300,300)
plt.legend()
plt.title('LI: Buoyancy Flux DC bv. COARE')

plt.figure()
plt.plot(BF_LII['dc'], label = 'dc')
plt.plot(BF_LII['coare'], label = 'coare')
plt.ylim(-300,300)
plt.legend()
plt.title('LII: Buoyancy Flux DC bv. COARE')
#%%
plot_savePath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4\plots/"
plt.figure()
plt.scatter(BF_LII['dc'],BF_LII['coare'], color = 'orange', edgecolor = 'red', label = 'L II')
plt.scatter(BF_LI['dc'],BF_LI['coare'], color = 'dodgerblue', edgecolor = 'navy', label = 'L I')
plt.plot([-150, 250], [-150, 250], color = 'k', label = "1-to-1") #scale 1-to-1 line
# plt.title(r"Buoyancy Flux ($\rho*c_p*<w'T_s'>$ [$Wm^{-2}$])", fontsize=14)
plt.title(r"Buoyancy Flux [$Wm^{-2}$]: Full Dataset", fontsize=14)
plt.xlabel('DC')
plt.ylabel('COARE')
plt.legend(loc='lower right')
ax = plt.gca() 
plt.text(.03, .95, "r (L II) ={:.3f}".format(r_BF_LII_str), transform=ax.transAxes)
plt.text(.03, .9, "r (L I) ={:.3f}".format(r_BF_LI_str), transform=ax.transAxes)
# plt.axis('square')
plt.xlim(-1000,300)
# plt.ylim(-300,300)
plt.savefig(plot_savePath + "buoyancyFlux_scatterplot.png",dpi=300)



#%%
# Timeseries
import matplotlib.dates as mdates

date_df = pd.read_csv(file_path+'date_allFall.csv')
dates_arr = np.array(pd.to_datetime(date_df['datetime']))

fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15,8), sharex=True)
ax1.plot(dates_arr, BF_95p_I_df['dc'], color = 'black', label = 'DC')
ax1.plot(dates_arr, BF_95p_I_df['coare'], color = 'gray', label = 'COARE')
ax1.legend()
ax1.set_title('Level I')
ax1.set_ylim(-300,300)
ax1.set_ylabel('Buoyancy Flux [$Wm^{-2}$]')
ax2.plot(dates_arr, BF_95p_II_df['dc'], color = 'black', label = 'DC')
ax2.plot(dates_arr, BF_95p_II_df['coare'], color = 'gray', label = 'COARE')
ax2.legend()
ax2.set_title('Level II')
ax2.set_ylim(-300,300)
ax2.set_ylabel('Buoyancy Flux [$Wm^{-2}$]')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
# ax.xaxis.set_minor_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
fig.suptitle('Buoyancy Flux (95th %-ile) ($BF$)', fontsize=16)

fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15,8), sharex=True)
ax1.plot(dates_arr, BF_99p_I_df['dc'], color = 'black', label = 'DC')
ax1.plot(dates_arr, BF_99p_I_df['coare'], color = 'gray', label = 'COARE')
ax1.legend()
ax1.set_title('Level I')
ax1.set_ylim(-300,300)
ax1.set_ylabel('Buoyancy Flux [$Wm^{-2}$]')
ax2.plot(dates_arr, BF_99p_II_df['dc'], color = 'black', label = 'DC')
ax2.plot(dates_arr, BF_99p_II_df['coare'], color = 'gray', label = 'COARE')
ax2.legend()
ax2.set_title('Level II')
ax2.set_ylim(-300,300)
ax2.set_ylabel('Buoyancy Flux [$Wm^{-2}$]')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))
# ax.xaxis.set_minor_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')

fig.suptitle('Buoyancy Flux (99th %-ile) ($BF$)', fontsize=16)
#%%
plt.figure()
plt.scatter(BF_95p_II_df['dc'],BF_95p_II_df['coare'], color = 'orange', edgecolor = 'red', label = 'L II')
plt.scatter(BF_95p_I_df['dc'],BF_95p_I_df['coare'], color = 'dodgerblue', edgecolor = 'navy', label = 'L I')
plt.plot([-200, 200], [-200, 200], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title(r"Buoyancy Flux [$Wm^{-2}$]: 95%-ile", fontsize=14)
plt.xlabel('DC')
plt.ylabel('COARE')
plt.legend(loc='lower right')
plt.xlim(-300,300)
plt.ylim(-300,300)
ax = plt.gca() 
plt.text(.03, .95, "r (L II) ={:.3f}".format(r_BF_95_II_str), transform=ax.transAxes)
plt.text(.03, .9, "r (L I) ={:.3f}".format(r_BF_95_I_str), transform=ax.transAxes)
plt.axis('square')
plt.savefig(plot_savePath + "buoyancyFlux_scatterplot.png",dpi=300)

plt.figure()
plt.scatter(BF_99p_II_df['dc'],BF_99p_II_df['coare'], color = 'orange', edgecolor = 'red', label = 'L II')
plt.scatter(BF_99p_I_df['dc'],BF_99p_I_df['coare'], color = 'dodgerblue', edgecolor = 'navy', label = 'L I')
plt.plot([-200, 200], [-200, 200], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.title(r"Buoyancy Flux (99%-ile)", fontsize=16)
plt.xlabel('DC')
plt.ylabel('COARE')
plt.legend(loc='upper left')
plt.xlim(-300,300)
plt.ylim(-300,300)
plt.axis('square')

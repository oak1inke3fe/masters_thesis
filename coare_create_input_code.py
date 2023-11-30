# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:26:39 2023

@author: oak




This code combines all the variables we need for COARE into the one file that COARE takes as an input

"""
#%%
import pandas as pd
import numpy as np

#%%
sonic_arr = ['1','2','3','4']
# filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
filepath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
jd_file = "jd_combinedAnalysis.csv"
jd_df = pd.read_csv(filepath + jd_file)

sonic1_df = pd.read_csv(filepath+ 'despiked_s1_turbulenceTerms_andMore_combined.csv')
sonic2_df = pd.read_csv(filepath+ 'despiked_s2_turbulenceTerms_andMore_combined.csv')
sonic3_df = pd.read_csv(filepath+ 'despiked_s3_turbulenceTerms_andMore_combined.csv')
sonic4_df = pd.read_csv(filepath+ 'despiked_s4_turbulenceTerms_andMore_combined.csv')

u_df = pd.DataFrame()
u_df['Ubar_s1'] = np.array(sonic1_df['Ubar'])
u_df['Ubar_s2'] = np.array(sonic2_df['Ubar'])
u_df['Ubar_s3'] = np.array(sonic3_df['Ubar'])
u_df['Ubar_s4'] = np.array(sonic4_df['Ubar'])

z_file = "z_airSide_combinedAnalysis.csv"
z_df = pd.read_csv(filepath + z_file)

ta_rh_swDN_lwDN_file = "metAvg_combinedAnalysis.csv"
ta_rh_swDN_lwDN_df = pd.read_csv(filepath + ta_rh_swDN_lwDN_file)

ta_sonic_df = pd.DataFrame()
ta_sonic_df['Tbar_s1'] = np.array(sonic1_df['Tbar'])
ta_sonic_df['Tbar_s2'] = np.array(sonic2_df['Tbar'])
ta_sonic_df['Tbar_s3'] = np.array(sonic3_df['Tbar'])
ta_sonic_df['Tbar_s4'] = np.array(sonic4_df['Tbar'])

p_file = "parosAvg_combinedAnalysis.csv"
p_df = pd.read_csv(filepath + p_file)

ctd_spring_df = pd.read_csv(filepath + "ctd20mAvg_allSpring.csv")
ctd_fall_df = pd.read_csv(filepath + "ctd20mAvg_allFall.csv")
Tsea_df = pd.concat([ctd_spring_df, ctd_fall_df], axis=0)


# Tsea_temp_file = pd.read_csv(filepath + "thermistorsAvg_allFall.csv")

LatLon_df = pd.read_csv(filepath + "LatLon_combinedAnalysis.csv")
rain_df = pd.read_csv(filepath + "rain_rate_combinedAnalysis.csv")
# Ts_depth_df = pd.read_csv(filepath + "thermistorsAvg_allFall.csv")
waves_df = pd.read_csv(filepath + 'waveData_allFall.csv')
# Ss_file = none
# Cp_file = none
# sigH_file = none



#%%
sonic_arr = ['1','2','3','4']
for sonic in sonic_arr:
    blank_array = np.zeros(len(u_df))
    
    coare_df = pd.DataFrame()
    coare_df['jd'] = np.array(jd_df['jd'])                        
    # coare_df['u'] = np.array(u_file_new['Ubar_s1'] )  
    coare_df['u'] = np.array(np.where(u_df['Ubar_s'+sonic]<=2.0,np.nan,u_df['Ubar_s'+sonic]) )   #get rid of times when wind is too light                   
    coare_df['zu'] = np.array(z_df['z_sonic'+sonic]).flatten()               
    
    # coare_df['ta'] = np.array(ta_rh_swDN_lwDN_file_new['t1 [C]']).flatten() #this is using sonic T NOT metT
    coare_df['ta'] = np.array(ta_sonic_df['Tbar_s'+sonic]-273.15).flatten() #this is using sonic T NOT metT
    coare_df['zt'] = np.array(z_df['z_sonic'+sonic]).flatten()
    if int(sonic) <= 2:
        met_sensor = str(1)
    else:
        met_sensor = str(2)
    coare_df['rh'] = np.array(ta_rh_swDN_lwDN_df['rh'+met_sensor]).flatten()
    coare_df['zq'] = np.array(z_df['z_met'+met_sensor]).flatten()
    if int(sonic) > 3:
        sonicP = str(3)
    else:
        sonicP = sonic
    coare_df['P'] = np.array(p_df['p'+sonicP+'_avg [mb]']).flatten()
    # coare_df['Tsea'] = np.array(Tsea_file['sst']).flatten()
    coare_df['Tsea'] = np.array(Tsea_temp_df['temp_therm1']).flatten()
    coare_df['sw_dn'] = np.array(ta_rh_swDN_lwDN_df['sw_dn']).flatten()
    coare_df['lw_dn'] = np.array(ta_rh_swDN_lwDN_df['lw_dn']).flatten()
    coare_df['lat'] = np.array(LatLon_df['lat']).flatten()
    coare_df['lon'] = np.array(LatLon_df['lon']).flatten()
    coare_df['Zi'] = np.ones(len(u_df))*600 
    coare_df['rain'] = np.array(rain_df['rain_rate']).flatten()
    coare_df['Ss'] = np.array(Tsea_df['salinity']).flatten()
    # coare_df['Cp'] = blank_array
    coare_df['Cp'] = np.array(waves_df['Cp']).flatten()
    # coare_df['sigH'] = blank_array
    coare_df['sigH'] = np.array(waves_df['sigH']).flatten()
    # coare_df['tsg'] = np.array(Tsea_file['sst']).flatten()
    coare_df['tsg'] = np.array(Tsea_temp_df['temp_therm1']).flatten()
    # coare_df['Ts_depth'] = np.ones(len(u_file_new))*3.88         #this is just while we are using average depth of Ts sensor
    coare_df['Ts_depth'] = np.array(Tsea_temp_df['z_therm1']).flatten()
    # coare_df['tsnk'] = np.array(Tsea_file['sst']).flatten()
    coare_df['tsnk'] = np.array(Tsea_temp_df['temp_therm1']).flatten()
    # coare_df['ztsg'] = np.ones(len(u_file_new))*3.88 
    coare_df['ztsg'] = np.array(Tsea_temp_df['z_therm1']).flatten()           
    
    
    # coare_drop_arr = np.arange(4293,4329)
    # coare_df_new = coare_df.drop(index=coare_drop_arr)
    
    
    coare_df.to_csv(filepath+"coare_Inputs_s"+sonic+"_withWaves.csv")
    print("done with sonic "+ sonic)
    print("used met sensor "+ met_sensor)
    print('Used pressure from sonic/pressure head #'+sonicP)

 


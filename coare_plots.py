# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:43:05 2023

@author: oak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print('done with imports')
#%%
# filepath = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
filepath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/Fall_Deployment/OaklinCopyMNode/code_pipeline/Level4/'
jd_short = pd.read_csv(filepath+'jd_short.csv')
jd_full = pd.read_csv(filepath + "jd_allFall.csv")
UpWp_bar_df = pd.read_csv(filepath + "UpWp_bar_allSonics_allFall.csv")

rho_df = pd.read_csv(filepath+"rho_bar_allFall.csv")
rho_1 = rho_df['rho_bar_1']
rho_2 = rho_df['rho_bar_2']
rho_3 = rho_df['rho_bar_3']

buoy_df = pd.read_csv(filepath + "New_buoy_terms_allFall.csv")

#%% U star
usr_1 = np.sqrt(-1*UpWp_bar_df['UpWp_bar_s1'])
usr_df = pd.DataFrame()
usr_df['date'] = jd_full['jd']
usr_df['usr'] = usr_1
# usr_short = usr_df.loc[27:]
plt.figure()
plt.plot(usr_1)
plt.ylim(0,1)
plt.xlabel('time')
plt.ylabel("u* [m/s]")
plt.title('friction velocity bb-asit')
#%% Tau

tau_1 = rho_1*-1*UpWp_bar_df['UpWp_bar_s1']
tau_df = pd.DataFrame()
tau_df['date'] = jd_full['jd']
tau_df['tau'] = tau_1
usr_short = usr_df.loc[27:]
plt.figure()
plt.plot(tau_1)
plt.ylim(0,1)
plt.xlabel('time')
plt.ylabel("tau [N/s^2]")
plt.title('stress bb-asit')



#%% Buoyancy flux comparing to COARE (units W/m^2 = rho*cp*<w'T'>)
buoy_df_short = buoy_df.loc[27:]


plt.figure()
plt.plot(buoy_df.index, buoy_df['coare_buoy_1'])

# plt.ylim(0,1)
plt.xlabel('time')
plt.ylabel("buoyancy flux [W/m^2]")
plt.title('buoyancy flux bb-asit')
#%%
sonic = str(1)
# file_path = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level4/"
filepath = r'/Users/oaklinkeefe/documents/GitHub/masters_thesis/myAnalysisFiles/'
file_name = 'coare_outputs_s'+sonic+'_WindStressOnly.txt'
A_hdr = 'usr\ttau\thsb\thlb\thbb\thsbb\thlwebb\ttsr\tqsr\tzo\tzot\tzoq\tCd\t'
A_hdr += 'Ch\tCe\tL\tzeta\tdT_skinx\tdq_skinx\tdz_skin\tUrf\tTrf\tQrf\t'
A_hdr += 'RHrf\tUrfN\tTrfN\tQrfN\tlw_net\tsw_net\tLe\trhoa\tUN\tU10\tU10N\t'
A_hdr += 'Cdn_10\tChn_10\tCen_10\thrain\tQs\tEvap\tT10\tT10N\tQ10\tQ10N\tRH10\t'
A_hdr += 'P10\trhoa10\tgust\twc_frac\tEdis\tdT_warm\tdz_warm\tdT_warm_to_skin\tdu_warm'
coare_warm = np.genfromtxt(filepath + file_name, delimiter='\t')

#%%
hbl = np.array(coare_warm[:,3])
plt.figure()
plt.plot(hbl[0:100])
#%%
# time = np.array(jd_short['jd_short'])
usr_df = pd.read_csv(filepath + 'usr_combinedAnalysis.csv')
ustar_df = pd.DataFrame()
# ustar_df['time'] = time
ustar_df['usr_cov'] = usr_df['usr_s1']
ustar_df['usr_coare'] = coare_warm[:,0]
plt.figure()

plt.scatter(ustar_df['usr_cov'],ustar_df['usr_coare'])
plt.plot([0, 1], [0, 1], color = 'k', label = "1-to-1") #scale 1-to-1 line
plt.xlabel('u* dir. cov.')
plt.ylabel('u* bulk flux (COARE)')
plt.title("u* direct cov. vs. bulk flux")
plt.legend()


#%%
import binsreg
import seaborn as sns

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

# Estimate binsreg
df_binEstimate = binscatter(x='usr_cov', y='usr_coare', data=ustar_df, ci=(3,3))
#%%
# Plot binned scatterplot
sns.scatterplot(x='usr_cov', y='usr_coare', data=df_binEstimate);
plt.errorbar('usr_cov', 'usr_coare', yerr='ci', data=df_binEstimate, ls='', lw=2, alpha=0.2);
plt.title("u* direct cov. versus bulk flux (coare)")
#%%



plt.figure()
plt.plot(time,coare_warm[:,0], label = 'coare')
plt.plot(usr_df['date'], usr_df['usr'], label = 'not-coare')
plt.ylim(0,1)
plt.title('friction velocity  output')
plt.ylabel('friction velocity [m/s]')
plt.legend()
plt.xlabel('time')
#%%
plt.figure()
plt.plot(tau_df['date'], tau_df['tau'], label = 'not-coare')
plt.plot(time,coare_warm[:,1], label = 'coare')
plt.title('wind stress ')
plt.ylabel('wind stress [N/m^2]')
plt.legend()
plt.ylim(0,1)
plt.xlabel('time')

#%%
# plt.figure()
# plt.plot(time,coare_warm[:,5], label = 'coare')
# plt.plot(buoy_df_short['jd'], buoy_df_short['coare_buoy_1'], label = 'not-coare')
# plt.title('atmospheric buoyancy flux (from sonicT)')
# plt.ylabel('atmospheric buoyancy flux (from sonicT) [W/m^2]')
# plt.xlabel('time')
# plt.legend()
#%%

plt.figure()
plt.plot(time,coare_warm[:,2])
plt.title('sensible heat flux')
plt.ylabel('sensible heat flux [W/m^2]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,3])
plt.title('latent heat flux')
plt.ylabel('latent heat flux [W/m^2]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,4])
plt.title('atmospheric buoyancy flux COARE')
plt.ylabel('atmospheric buoyancy flux [W/m^2]')
plt.xlabel('time')

#%%
plt.figure()
plt.plot(time,coare_warm[:,6])
plt.title('Webb Factor to be added to hl')
plt.ylabel('Webb factor')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,7])
plt.title('temperature scaling parameter')
plt.ylabel('tsr [K]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,8])
plt.title('specific humidity scaling parameter')
plt.ylabel('qsr [kg/kg]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,9])
plt.title('momentum roughness length')
plt.ylabel('momentum roughness length [m]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,10])
plt.title('thermal roughness length')
plt.xlabel('thermal roughness length [m]')
plt.ylabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,11])
plt.title('moisture roughness length')
plt.ylabel('moisture roughness length [m]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,12])
plt.title('wind stress transfer (drag) coefficient')
plt.ylabel('Cd [unitless]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,13])
plt.title('sensible heat transfer coefficient')
plt.ylabel('Ch [unitless]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,14])
plt.title('latent heat transfer coefficient')
plt.ylabel('Ce [unitless]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,15])
plt.title('MO-Length')
plt.ylabel('L [m]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,16])
plt.title('MO stability parameter zu/L')
plt.ylabel('zu/L [dimensionless]')
plt.xlabel('time')
plt.xlim(280,290)

plt.figure()
plt.scatter(np.arange(len(time)),time)
plt.xlim(540,640)
#%%
plt.figure()
plt.plot(time,coare_warm[:,21])
plt.title('Wind Speed at U10')
plt.ylabel('Urf [m/s]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,22])
plt.title('Air temp at 10m')
plt.ylabel('Trf [C]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,23])
plt.title('air specific humidity at 10m')
plt.ylabel('Qrf')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,24])
plt.title('air relative humidity at 10m')
plt.ylabel('RHrf')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,25])
plt.title('Neutral value of wind speed at U10')
plt.ylabel('UrfN [m]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,26])
plt.title('Neutral air temp at 10m')
plt.ylabel('TrfN [C]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,27])
plt.title('Neutral air specific humidity at 10m')
plt.ylabel('QrfN')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,31])
plt.title('Air Density')
plt.ylabel('rhoa [kg/m^3]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,32])
plt.title('Neutral U @ zu')
plt.ylabel('UN [m/s]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,33])
plt.title('U10')
plt.ylabel('U10 [m/s]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,34])
plt.title('Neutral 10m wind stress transfer (drag) coefficient')
plt.ylabel('Cdn_10 [unitless]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,35])
plt.title('Neutral 10m sensible heat transfer coefficient')
plt.ylabel('Chn_10 [unitless]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,36])
plt.title('Neutral 10m latent heat transfer coefficient')
plt.ylabel('Cen_10 [unitless]')
plt.xlabel('time')
#%%
plt.figure()
plt.plot(time,coare_warm[:,49])
plt.title('Dissipation by wave breaking')
plt.ylabel('Edis [W/m^2]')
plt.xlabel('time')

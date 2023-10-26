# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:08:04 2023

@author: oaklin keefe

NOTE: this file needs to be run on the remote desktop.

This file is used to calculate the rate of TKE dissipation using the inertial subrange method.

INPUT files:
    sonic files from /Level1_align-despike-interp/ folder
    z_airSide_allSpring.csv
    z_airSide_allFall.csv
    
    
We also set:
    alpha = 0.53 (when working with U and not W)
    alpha = c1 = c1_prime

    
OUTPUT files:
    Puu_exampleSpectra.png
    epsU_terms_sonic1_MAD_k_UoverZbar.csv
    epsU_terms_sonic2_MAD_k_UoverZbar.csv
    epsU_terms_sonic3_MAD_k_UoverZbar.csv
    epsU_terms_sonic4_MAD_k_UoverZbar.csv
    epsU_terms_combinedAnalysis_MAD_k_UoverZbar
    
    
"""


#%%
import os
import natsort
import numpy as np
import pandas as pd
import math
import scipy.signal as signal
import matplotlib.pyplot as plt
import datetime

print('done with imports')

#%%
# plot_savePath = r"smb://zippel-nas.local/bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/plots/"
# plot_savePath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level4\plots/"
# filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level4/"
# test_filepath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
# test_df = pd.read_csv(test_filepath + 'meanQuantities_sonic1.csv')

filepath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'

plot_savePath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/plots/'
print('done with filepath')

#%%

#inputs:
    # n = number of observations (for freq 32 Hz, n = 38400)
    # phi_m = fit of spectrum as a function of k (wavenumber) from polyfit
    # phi_m_eqn5 = fit from eqn 5 of Bluteau (2016)
    # phi_w = the measured spectrum, as a function of k (wavenumber)
### function start
#######################################################################################
def MAD_epsilon(n, phi_m, phi_w):
    mad_arr = []
    # mad_eq5_arr = []
    for i in range(0,n):
        MAD_i = np.abs((phi_w[i]/phi_m[i]) - np.nanmean(phi_w/phi_m))
        mad_arr.append(MAD_i)
        
        # MAD_i_eq5 = (phi_w[i]/phi_m_eq5[i]) - np.mean(phi_w/phi_m_eq5)
        # mad_eq5_arr.append(MAD_i_eq5)
    
    MAD = 1/n*(np.sum(mad_arr))
    # MAD_eq5 = np.mean(mad_eq5_arr)
    
    return MAD
#######################################################################################
### function end
# returns: output_df
print('done with MAD_epsilon function')

### function start
#######################################################################################
def despikeThis(input_df,n_std):
    n = input_df.shape[1]
    output_df = pd.DataFrame()
    for i in range(0,n):
        elements_input = input_df.iloc[:,i]
        elements = elements_input
        mean = np.nanmean(elements)
        sd = np.nanstd(elements)
        extremes = np.abs(elements-mean)>(n_std*sd)
        elements[extremes]=np.NaN
        despiked = np.array(elements)
        colname = input_df.columns[i]
        output_df[str(colname)]=despiked

    return output_df
#######################################################################################
### function end
# returns: output_df
print('done with despike_this function')

### function start
#######################################################################################
# Function for interpolating the RMY sensor (freq = 32 Hz)
def interp_sonics123(df_sonics123):
    sonics123_xnew = np.arange(0, (32*60*20))   # 32 Hz * 60 sec/min * 20 min / file
    df_align_interp= df_sonics123.reindex(sonics123_xnew).interpolate(limit_direction='both')
    return df_align_interp
#######################################################################################
### function end
# returns: df_align_interp
print('done with interp_sonics123 simple function')

### function start
#######################################################################################
# Function for interpolating the Gill sensor (freq = 20 Hz)
def interp_sonics4(df_sonics4):
    sonics4_xnew = np.arange(0, (20*60*20))   # 20 Hz * 60 sec/min * 20 min / file
    df_align_interp_s4= df_sonics4.reindex(sonics4_xnew).interpolate(limit_direction='both')
    return df_align_interp_s4
#######################################################################################
### function end
# returns: df_align_interp_s4
print('done with interp_sonics4 function')
#%% testing on one file
filepath_PC = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-despike-interp/'
# filepath_PC = r"Z:\combinedAnalysis\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp/"
# my_file_level1_PC = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp\mNode_Port1_20221019_184000_1.csv"
# my_file_level1_PC = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp\mNode_Port1_20221021_110000_1.csv"
# my_file_level1_PC = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp\mNode_Port3_20221109_144000_1.csv" 
# my_file_level1_PC = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp\mNode_Port1_20221002_232000_1.csv"
my_file_level1_PC = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level1_align-despike-interp/mNode_Port1_20221002_232000_1.csv'
# my_file_level1_PC = r"Z:\Fall_Deployment\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp\mNode_Port1_20221025_094000_1.csv"
alpha = 0.53 #this is for U
# alpha = 0.65 #this is for W --> need to go back and fix this in the other code too!
# alpha = 18/55*C*4/3 for C = Kolmogorov's constant 1.5 +/- 0.1
s1_df = pd.read_csv(my_file_level1_PC, index_col=0, header=0)
s1_df_despiked = despikeThis(s1_df,2)
s1_df_despiked = interp_sonics123(s1_df_despiked)
# s1_df_interp = interp_sonics123(s1_df)
# w_prime = np.array(s1_df['Wr']-s1_df['Wr'].mean())
# u_prime = np.array(s1_df_despiked['Wr']-s1_df_despiked['Wr'].mean()) #w_prime but called uprime
u_prime = np.array(s1_df_despiked['Ur']-s1_df_despiked['Ur'].mean())
    
U = np.abs(np.array(s1_df_despiked['Ur']))
U_mean = np.nanmean(U)
fs = 32
N = 2048
N_s = fs*60*20
freq, Puu = signal.welch(u_prime,fs,nperseg=N,detrend=False) #pwelch function   16384
# freq_over_Umean = freq/U_mean            
# k = freq*(2*np.pi)/np.mean(U) #converting to wavenumber spectrum
k = freq*(2*math.pi)/U_mean #converting to wavenumber spectrum
dfreq = np.max(np.diff(freq,axis=0))
dk = np.max(np.diff(k,axis=0))
Suu = Puu*dfreq/dk
k_fit = np.polyfit(k,np.log(Suu),3)
trendpoly = np.poly1d(k_fit) 
phi_m = np.exp(trendpoly(k))

# isr = np.nonzero(k >= 1.11)[0] 
isr = np.nonzero(k >= (2*np.pi/2.288))[0]   #for 2.288 agv sonic 1 height # multiply by <u>/z, then convert to wavenumber by mult. 2pi/<u> 
                                            # so <u> cancels out and we are left with 2pi/z

b = slice(isr.item(0),isr.item(-1))
spec_isr = Suu[b]
k_isr = k[b]
phi_m_isr = phi_m[b]

# eps = <spectra_isr * (k_isr^(5/3)) / alpha > ^(3/2)
eps_wnoise = np.mean((spec_isr*(k_isr**(5/3)))/alpha)**(3/2)

#least-squares minimization
X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
X=X_t.T
B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
noise = B.item(0)
eps = (B.item(1)/alpha)**(3/2) #slope of regression has alpha and eps^2/3
real_imag = isinstance(eps, complex)
if real_imag == True:
    print('eps is complex')

model_absNoise = alpha*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
model_realNoise = alpha*eps**(2/3)*k_isr**(-5/3)+(noise)
model_raw = alpha*eps_wnoise**(2/3)*k_isr**(-5/3)

d = np.abs(1.89*(2*N/N_s-1))
MAD_limit = 2*(2/d)**(1/2)

MAD_absNoise = MAD_epsilon(len(model_absNoise), model_absNoise, spec_isr)
MAD = MAD_epsilon(len(model_realNoise), model_realNoise, spec_isr)
# plt.scatter(MAD,eps_wnoise)

# MAD_criteria_fit_1=[0]
# if MAD <= MAD_limit:
#     MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
# else:
#     MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
epsilon_string = np.round(eps,6)
#PLOT spectra
fig = plt.figure()
# plt.figure(1,figsize=(8,6))
plt.loglog(k,Suu, color = 'k', label='spectra')
plt.loglog(k_isr,model_absNoise,color='r',label='ISR accounting for noise')
# plt.loglog(k_isr,model_realNoise,color='g',label='ISR accounting for true noise')
# plt.loglog(k_isr, model_raw, color = 'silver',label='ISR with no noise')
# plt.loglog(k,phi_m, color = 'g', label = 'polyfit')
plt.xlabel('Wavenumber ($k$)')
plt.ylabel('$P_{uu}$')
# plt.ylabel('$P_{ww}$')
plt.legend(loc = 'lower left')
ax = plt.gca() 
plt.text(.03, .22, "$\epsilon$ = {:.4f}".format(epsilon_string)+" [$m^2s^{-3}$]", transform=ax.transAxes)
plt.title('$P_{uu}$ with Modeled Inertial Subrange (ISR)')
print('plot finished')
# plt.title('$P_{ww}$ with Modeled Inertial Subrange (ISR)')
plt.savefig(plot_savePath + "Puu_exampleSpectra.png",dpi=300)


# plt.savefig(plot_savePath + "Pww_exampleSpectra.png",dpi=300)
# plt.figure()
# plt.figure(2,figsize=(8,6))
# plt.loglog(k_isr,spec_isr,color = 'k', label='spectra')
# plt.loglog(k_isr,model,color='r',label='model accounting for noise')
# # plt.loglog(k_isr, model_raw, color = 'silver',label='simple model no noise')
# # plt.loglog(k_isr,phi_m_isr, color = 'g', label = 'polyfit')
# plt.legend()
# plt.xlabel('Wavenumber ($k$)')
# plt.ylabel('PSD')
# plt.legend()
# plt.title('Wavenumber Spectra with Modeled Inertial Subrange (ISR)')

#%%
## Comparing to the simple fixed frequency range
# c_primeW = 0.65 # W
c_primeU = 0.53 # U
c_prime = c_primeU
freq, Puu = signal.welch(u_prime,fs,nperseg=N,detrend=False) #pwelch function   
k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
dfreq = np.max(np.diff(freq,axis=0))
dk = np.max(np.diff(k,axis=0))
Suu = Puu*dfreq/dk


k_lim_lower_freq = 0.5
k_lim_upper_freq = 10
k_lim_lower_waveNum = k_lim_lower_freq*(2*math.pi)/U_mean
k_lim_upper_waveNum = k_lim_upper_freq*(2*math.pi)/U_mean

# k_lim_lower_waveNum = (np.pi/2.288)
# k_lim_upper_waveNum = np.max(k)*(2*math.pi)/U_mean

# k_lim_freq = (U_mean/z_avg)
# k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
# k_limit = 1.11
# isr = np.nonzero(k >= k_lim_waveNum)[0]
isr = np.nonzero((k >= k_lim_lower_waveNum)&(k <= k_lim_upper_waveNum))[0]
if len(isr)>2:                        
    b = slice(isr.item(0),isr.item(-1))
    spec_isr = Suu[b]
    k_isr = k[b]
    # eps = np.nanmean(spec_isr)
    eps_simplefreq = (np.nanmean(spec_isr*(k_isr**(5/3))/c_prime))**(3/2)
model_realNoise = alpha*eps_simplefreq**(2/3)*k_isr**(-5/3)
#PLOT spectra
fig = plt.figure()
# plt.figure(1,figsize=(8,6))
plt.loglog(k,Suu, color = 'k', label='spectra')
plt.loglog(k_isr,model_realNoise,color='g',label='ISR from Eps Simple Freq')
# plt.loglog(k_isr, model_raw, color = 'silver',label='ISR with no noise')
# plt.loglog(k,phi_m, color = 'g', label = 'polyfit')
plt.xlabel('Wavenumber ($k$)')
plt.ylabel('Puu')
plt.legend()
plt.title('Wavenumber Spectra (Puu) with Fixed Inertial Subrange (Fixed ISR)')
#%% For multiple files:
# filepath= r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_align-despike-interp/"
filepath = filepath_PC
# z_filepath = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level4/"
z_filepath = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'

z_avg_df_spring = pd.read_csv(z_filepath+"z_airSide_allSpring.csv")
z_avg_df_spring = z_avg_df_spring.drop('Unnamed: 0', axis=1)
z_avg_df_fall = pd.read_csv(z_filepath+"z_airSide_allFall.csv")
z_avg_df_fall = z_avg_df_fall.drop('Unnamed: 0', axis=1)
print(z_avg_df_fall.columns)
print(z_avg_df_spring.columns)

plt.figure()
z_avg_df_spring.plot()
plt.title('Spring z-heights')
print('done with spring plot')

plt.figure()
z_avg_df_fall.plot()
plt.title('Fall z-heights')
print('done with fall plot')
#%%
c_primeU = 0.53
# c_primeW = 0.65
c_prime = c_primeU #for simplicity so I don't have to change the code below that has c_prime not c_prime1
#C'1 = kolmogorov constant adjustment with dissipation perpendicular to ??
#C'1 = (18/55*C)*(4/3), where C is the original Kolmogorov constant 1.5 +/-0.1, and C1 = (18/55*C) for dissipation parallel to ??
B_all_1 = [0,0]
noise_all_1 =[0]
eps_all_1 =[0]
eps_wnoise_all_1 = [0]
MAD_criteria_fit_1 = [0]
MAD_all_1 = [0]





# file_save_path = r"Z:\combinedAnalysis\OaklinCopyMNode\code_pipeline\Level4/"
file_save_path = r'/run/user/1005/gvfs/smb-share:server=zippel-nas.local,share=bbasit/combined_analysis/OaklinCopyMNode/code_pipeline/Level4/'
# mNode_arr = ['1','2','3','4']
mNode_arr = ['4',]

start=datetime.datetime.now()

for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)

        for mNode in mNode_arr:
            #spring deployment = APRIL 04, MAY 05, JUNE 06 (2022)
            if filename.startswith("mNode_Port"+mNode+"_202204"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.747
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.537
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 1.842
                    u_prime = np.array(df_interp['Ur']-df_interp['Ur'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Puu = signal.welch(u_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Suu = Puu*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Suu[b]
                        k_isr = k[b]
                        # eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))*(2*np.pi*(np.mean(freq))/U_mean)
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            elif filename.startswith("mNode_Port"+mNode+"_202205"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.747
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.537
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 1.842
                    u_prime = np.array(df_interp['Ur']-df_interp['Ur'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Puu = signal.welch(u_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Suu = Puu*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Suu[b]
                        k_isr = k[b]
                        # eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))*(2*np.pi*(np.mean(freq))/U_mean)
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            elif filename.startswith("mNode_Port"+mNode+"_202206"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.747
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.537
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 1.842
                    u_prime = np.array(df_interp['Ur']-df_interp['Ur'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Puu = signal.welch(u_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Suu = Puu*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Suu[b]
                        k_isr = k[b]
                        # eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))*(2*np.pi*(np.mean(freq))/U_mean)
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            #fall deployment = SPETEMBER 09, OCTOBER 10, NOVEMBER 11 (2022)
            elif filename.startswith("mNode_Port"+mNode+"_202209"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.800
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.116
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 2.287
                    u_prime = np.array(df_interp['Ur']-df_interp['Ur'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Puu = signal.welch(u_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Suu = Puu*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Suu[b]
                        k_isr = k[b]
                        # eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))*(2*np.pi*(np.mean(freq))/U_mean)
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            elif filename.startswith("mNode_Port"+mNode+"_202210"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.800
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.116
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 2.287
                    u_prime = np.array(df_interp['Ur']-df_interp['Ur'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Puu = signal.welch(u_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Suu = Puu*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Suu[b]
                        k_isr = k[b]
                        # eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))*(2*np.pi*(np.mean(freq))/U_mean)
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            elif filename.startswith("mNode_Port"+mNode+"_202211"):
                filename_only = filename[:-4]
                print(filename_only)
                mNode_df = pd.read_csv(file, index_col=0, header=0)

                if len(mNode_df)>5:
                    df_despike = despikeThis(mNode_df,2)
                    if mNode == '4':
                        df_interp = interp_sonics4(df_despike)                        
                        fs = 20
                        z_avg = 9.800
                    if mNode == "3":
                        df_interp = interp_sonics123(df_despike)                    
                        fs = 32
                        z_avg = 7.332
                    if mNode == "2":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 4.116
                    if mNode == "1":
                        df_interp = interp_sonics123(df_despike)                        
                        fs = 32
                        z_avg = 2.287
                    u_prime = np.array(df_interp['Ur']-df_interp['Ur'].mean())                    
                    U = np.abs(np.array(df_interp['Ur']))
                    U_mean = np.nanmean(U)
                    U_median = np.nanmedian(U)
                    N = 2048
                    N_s = fs*60*20
                    d = np.abs(1.89*(2*N/N_s-1))
                    MAD_limit = 2*(2/d)**(1/2)
                    freq, Puu = signal.welch(u_prime,fs,nperseg=N,detrend=False) #pwelch function   
                    k = freq*(2*np.pi)/U_mean #converting to wavenumber spectrum
                    dfreq = np.max(np.diff(freq,axis=0))
                    dk = np.max(np.diff(k,axis=0))
                    Suu = Puu*dfreq/dk
                    k_lim_freq = (U_mean/z_avg)
                    k_lim_waveNum = (U_mean/z_avg)*(2*math.pi)/U_mean
                    # k_limit = 1.11
                    isr = np.nonzero(k >= k_lim_waveNum)[0]
                    if len(isr)>2:
                        b = slice(isr.item(0),isr.item(-1))
                        spec_isr = Suu[b]
                        k_isr = k[b]
                        # eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))*(2*np.pi*(np.mean(freq))/U_mean)
                        eps_wnoise = (np.mean((spec_isr*(k_isr**(5/3)))/c_prime)**(3/2))
                        #least-squares minimization
                        X_t = np.array(([np.ones(len(spec_isr)).T],[k_isr**(-5/3)])).reshape(2,len(spec_isr))
                        X=X_t.T
                        B = (np.matmul(np.matmul((np.linalg.inv(np.matmul(X.T,X))),X.T),spec_isr)).reshape(1,2)
                        noise = B.item(0)
                        eps = (B.item(1)/c_prime)**(3/2) #slope of regression has c_prime and eps^2/3
                        real_imag = isinstance(eps, float)
                        if real_imag == True: #this means epsilon is a real number
                            model = c_prime*eps**(2/3)*k_isr**(-5/3)+np.abs(noise)
                            model_raw = c_prime*eps_wnoise**(2/3)*k_isr**(-5/3)
                            MAD = MAD_epsilon(len(model), model, spec_isr)
                            if MAD <= MAD_limit:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'True'])
                            else:
                                MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
                    # plt.figure(1,figsize=(8,6))
                    # plt.loglog(k,Sww, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend()
                    # plt.title('Full Spectra')
        
                    # plt.figure()
                    # plt.loglog(k_isr,spec_isr, label='spectra')
                    # plt.loglog(k_isr,model,color='r',label='model accounting for noise')
                    # plt.loglog(k_isr, model_raw, color = 'k',label='simple model no noise')
                    # plt.legend(loc='lower left')
                    # plt.title('ISR only '+str(filename))
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            plt.scatter(MAD,eps_wnoise, color = 'k')
                            plt.xlabel('MAD value')
                            plt.ylabel('Dissipation Rate (with noise)')
                            plt.title('MAD vs. Dissipation Rate')
                            print(filename)
                        else: #this means epsilon is imaginary and we need to make it NaN
                            B = np.array([np.nan,np.nan])
                            B.reshape(1,2)
                            noise = np.nan
                            eps = np.nan
                            eps_wnoise = np.nan
                            MAD = np.nan
                            MAD_all_1 = np.vstack([MAD_all_1,MAD])
                            B_all_1 = np.vstack([B_all_1,B])
                            noise_all_1 = np.vstack([noise_all_1,noise])
                            eps_all_1 = np.vstack([eps_all_1,eps])
                            eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                            MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                    else: #this means epsilon is imaginary and we need to make it NaN
                        B = np.array([np.nan,np.nan])
                        B.reshape(1,2)
                        noise = np.nan
                        eps = np.nan
                        eps_wnoise = np.nan
                        MAD = np.nan
                        MAD_all_1 = np.vstack([MAD_all_1,MAD])
                        B_all_1 = np.vstack([B_all_1,B])
                        noise_all_1 = np.vstack([noise_all_1,noise])
                        eps_all_1 = np.vstack([eps_all_1,eps])
                        eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                        MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])

                else: #this means epsilon is imaginary and we need to make it NaN
                    B = np.array([np.nan,np.nan])
                    B.reshape(1,2)
                    noise = np.nan
                    eps = np.nan
                    eps_wnoise = np.nan
                    MAD = np.nan
                    MAD_all_1 = np.vstack([MAD_all_1,MAD])
                    B_all_1 = np.vstack([B_all_1,B])
                    noise_all_1 = np.vstack([noise_all_1,noise])
                    eps_all_1 = np.vstack([eps_all_1,eps])
                    eps_wnoise_all_1 = np.vstack([eps_wnoise_all_1,eps_wnoise])
                    MAD_criteria_fit_1 = np.vstack([MAD_criteria_fit_1,'False'])
            
            else:
                # error_files.append(filename[:-4])
                continue
end=datetime.datetime.now()
# import winsound
# duration = 3000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration)
print('done with this section') 
print(start)
print(end)        

#%%
noise_all_1 = np.delete(noise_all_1, 0,0)
eps_all_1 = np.delete(eps_all_1, 0,0)
eps_wnoise_all_1 = np.delete(eps_wnoise_all_1, 0,0)
MAD_all_1 = np.delete(MAD_all_1, 0,0)
MAD_criteria_fit_1 = np.delete(MAD_criteria_fit_1, 0,0)
MAD_goodFiles = np.where(MAD_criteria_fit_1 == 'False', np.nan,MAD_criteria_fit_1)

eps_df = pd.DataFrame()
eps_df['index']=np.arange(len(eps_all_1))
eps_df['eps_sonic'+mNode] = eps_all_1
eps_df['MAD_value_'+mNode] = MAD_all_1
eps_df['MAD_criteria_met_'+mNode] = MAD_goodFiles

eps_df['eps_sonic'+mNode] = np.where(eps_df['MAD_criteria_met_'+mNode].isnull(),np.nan,eps_df['eps_sonic'+mNode])

eps_df.to_csv(file_save_path+"epsU_terms_sonic"+mNode+"_MAD_k_UoverZbar.csv")


plt.figure()
plt.scatter(MAD_all_1,eps_all_1, color = 'b')
plt.axvline(x=MAD_limit, color = 'gray')
plt.yscale('log')
plt.xlabel('MAD value')
plt.ylabel('Dissipation Rate (with noise)')
plt.title('MAD vs. Dissipation Rate sonic '+mNode)


#%%
eps_1 = pd.read_csv(file_save_path+"epsU_terms_sonic1_MAD_k_UoverZbar.csv")
# eps_1 = pd.read_csv(file_save_path+"eps_terms_sonic1_MAD_k_UoverZ.csv")
# eps_1 = pd.read_csv(file_save_path+"eps_terms_sonic1_MAD_k1-11.csv")
eps_1 = eps_1.loc[:, ~eps_1.columns.str.contains('^Unnamed')]
eps_1 = eps_1.loc[:, ~eps_1.columns.str.contains('^index')]

eps_2 = pd.read_csv(file_save_path+"epsU_terms_sonic2_MAD_k_UoverZbar.csv")
# eps_2 = pd.read_csv(file_save_path+"eps_terms_sonic2_MAD_k_UoverZ.csv")
# eps_2 = pd.read_csv(file_save_path+"eps_terms_sonic2_MAD_k1-11.csv")
eps_2 = eps_2.loc[:, ~eps_2.columns.str.contains('^Unnamed')]
eps_2 = eps_2.loc[:, ~eps_2.columns.str.contains('^index')]

eps_3 = pd.read_csv(file_save_path+"epsU_terms_sonic3_MAD_k_UoverZbar.csv")
# eps_3 = pd.read_csv(file_save_path+"eps_terms_sonic3_MAD_k_UoverZ.csv")
# eps_3 = pd.read_csv(file_save_path+"eps_terms_sonic3_MAD_k1-11.csv")
eps_3 = eps_3.loc[:, ~eps_3.columns.str.contains('^Unnamed')]
eps_3 = eps_3.loc[:, ~eps_3.columns.str.contains('^index')]

eps_4 = pd.read_csv(file_save_path+"epsU_terms_sonic4_MAD_k_UoverZbar.csv")
# eps_4 = pd.read_csv(file_save_path+"eps_terms_sonic4_MAD_k_UoverZ.csv")
# eps_4 = pd.read_csv(file_save_path+"eps_terms_sonic4_MAD_k1-11.csv")
eps_4 = eps_4.loc[:, ~eps_4.columns.str.contains('^Unnamed')]
eps_4 = eps_4.loc[:, ~eps_4.columns.str.contains('^index')]

eps_combined = pd.concat([eps_1, eps_2, eps_3, eps_4], axis=1)
eps_combined.to_csv(file_save_path+'epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv')
# eps_combined.to_csv(file_save_path+'eps_allFall_MAD_k_UoverZ.csv')
# eps_combined.to_csv(file_save_path+'eps_allFall_MAD_k1-11.csv')

print('done')
#%%
eps_combined= pd.read_csv(file_save_path+'epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv')
eps_combined['epsU_sonic1_MAD'] = np.where(eps_combined['MAD_criteria_met_1']!=True,np.nan,eps_combined['eps_sonic1'])
eps_combined['epsU_sonic2_MAD'] = np.where(eps_combined['MAD_criteria_met_2']!=True,np.nan,eps_combined['eps_sonic2'])
eps_combined['epsU_sonic3_MAD'] = np.where(eps_combined['MAD_criteria_met_3']!=True,np.nan,eps_combined['eps_sonic3'])
eps_combined['epsU_sonic4_MAD'] = np.where(eps_combined['MAD_criteria_met_4']!=True,np.nan,eps_combined['eps_sonic4'])
# alpha_df = pd.read_csv(file_save_path+"windDir_withBadFlags.csv")
# eps_combined['good_wind_dir'] = alpha_df['good_wind_dir']
# eps_combined['potential_good_wind_dir'] = alpha_df['potential_good_wind_dir']
# eps_combined['date'] = alpha_df['date']
eps_combined.to_csv(file_save_path+'epsU_terms_combinedAnalysis_MAD_k_UoverZbar.csv')
print('done')

#%%
# mad1_bad_wind = np.where(eps_combined['good_wind_dir']==False,eps_combined['MAD_value_1'], np.nan)
# mad1_potential_bad_wind = np.where(eps_combined['potential_good_wind_dir']==False,eps_combined['MAD_value_1'], np.nan)

# mad1_bad_wind_dir_df = pd.DataFrame()
# mad1_bad_wind_dir_df['MAD_value_1'] = eps_combined['MAD_value_1']
# mad1_bad_wind_dir_df['eps_sonic1'] = eps_combined['eps_sonic1']
# mad1_bad_wind_dir_df['good_wind_dir'] = eps_combined['good_wind_dir']
# mad1_bad_wind_dir_df['MAD_values_for_bad_wind'] = mad1_bad_wind
# mad1_bad_wind_dir_df['MAD_values_for_potential_bad_wind'] = mad1_potential_bad_wind

# mad2_bad_wind = np.where(eps_combined['good_wind_dir']==False,eps_combined['MAD_value_2'], np.nan)
# mad2_potential_bad_wind = np.where(eps_combined['potential_good_wind_dir']==False,eps_combined['MAD_value_2'], np.nan)

# mad2_bad_wind_dir_df = pd.DataFrame()
# mad2_bad_wind_dir_df['MAD_value_2'] = eps_combined['MAD_value_2']
# mad2_bad_wind_dir_df['eps_sonic2'] = eps_combined['eps_sonic2']
# mad2_bad_wind_dir_df['good_wind_dir'] = eps_combined['good_wind_dir']
# mad2_bad_wind_dir_df['MAD_values_for_bad_wind'] = mad1_bad_wind
# mad2_bad_wind_dir_df['MAD_values_for_potential_bad_wind'] = mad1_potential_bad_wind

# mad3_bad_wind = np.where(eps_combined['good_wind_dir']==False,eps_combined['MAD_value_3'], np.nan)
# mad3_potential_bad_wind = np.where(eps_combined['potential_good_wind_dir']==False,eps_combined['MAD_value_3'], np.nan)

# mad3_bad_wind_dir_df = pd.DataFrame()
# mad3_bad_wind_dir_df['MAD_value_3'] = eps_combined['MAD_value_3']
# mad3_bad_wind_dir_df['eps_sonic3'] = eps_combined['eps_sonic3']
# mad3_bad_wind_dir_df['good_wind_dir'] = eps_combined['good_wind_dir']
# mad3_bad_wind_dir_df['MAD_values_for_bad_wind'] = mad1_bad_wind
# mad3_bad_wind_dir_df['MAD_values_for_potential_bad_wind'] = mad1_potential_bad_wind

# mad4_bad_wind = np.where(eps_combined['good_wind_dir']==False,eps_combined['MAD_value_4'], np.nan)
# mad4_potential_bad_wind = np.where(eps_combined['potential_good_wind_dir']==False,eps_combined['MAD_value_4'], np.nan)

# mad4_bad_wind_dir_df = pd.DataFrame()
# mad4_bad_wind_dir_df['MAD_value_4'] = eps_combined['MAD_value_4']
# mad4_bad_wind_dir_df['eps_sonic4'] = eps_combined['eps_sonic4']
# mad4_bad_wind_dir_df['good_wind_dir'] = eps_combined['good_wind_dir']
# mad4_bad_wind_dir_df['MAD_values_for_bad_wind'] = mad1_bad_wind
# mad4_bad_wind_dir_df['MAD_values_for_potential_bad_wind'] = mad1_potential_bad_wind

# #%%
# plt.figure()
# plt.scatter(mad4_bad_wind_dir_df['MAD_value_4'], mad4_bad_wind_dir_df['eps_sonic4'], color = 'y', label = 'sonic 4')
# plt.scatter(mad3_bad_wind_dir_df['MAD_value_3'], mad3_bad_wind_dir_df['eps_sonic3'], color = 'g', label = 'sonic 3')
# plt.scatter(mad2_bad_wind_dir_df['MAD_value_2'], mad2_bad_wind_dir_df['eps_sonic2'], color = 'b', label = 'sonic 2')
# plt.scatter(mad1_bad_wind_dir_df['MAD_value_1'], mad1_bad_wind_dir_df['eps_sonic1'], color = 'cyan', label = 'sonic 1')

# plt.axvline(x=2.17674,color = 'k', label = 'Mad_crit. sonics 1-3')
# plt.axvline(x=2.25917,color = 'gray', label = 'Mad_crit. sonic 4')

# plt.xscale('log')
# plt.yscale('log')
# plt.legend(prop={'size': 6})
# plt.xlim(0,20)
# plt.title('MAD vs. Epsilon (All Sonics)')
# plt.xlabel('MAD value')
# plt.ylabel('Epsilon [m^2/s^3]')

# #%%
# plt.figure()
# plt.scatter(mad1_bad_wind_dir_df['MAD_value_1'], mad1_bad_wind_dir_df['eps_sonic1'], color = 'k', label = 'fine wind direction')
# plt.scatter(mad1_bad_wind_dir_df['MAD_values_for_bad_wind'], mad1_bad_wind_dir_df['eps_sonic1'], color = 'r', label = 'bad wind direction')
# # plt.scatter(mad1_bad_wind_dir_df['MAD_values_for_potential_bad_wind'], mad1_bad_wind_dir_df['eps_sonic1'], color = 'b', label = 'potential bad wind direction')
# plt.axvline(x=MAD_limit,color = 'gray', label = 'Mad_crit.')
# plt.yscale('log')
# plt.legend()
# plt.xlim(0,20)
# plt.title('MAD vs. Epsilon (Sonic 1)')
# plt.xlabel('MAD value')
# plt.ylabel('Epsilon [m^2/s^3]')

# #%%
# plt.figure()
# plt.scatter(mad2_bad_wind_dir_df['MAD_value_2'], mad2_bad_wind_dir_df['eps_sonic2'], color = 'k', label = 'fine wind direction')
# plt.scatter(mad2_bad_wind_dir_df['MAD_values_for_bad_wind'], mad2_bad_wind_dir_df['eps_sonic2'], color = 'r', label = 'bad wind direction')
# # plt.scatter(mad1_bad_wind_dir_df['MAD_values_for_potential_bad_wind'], mad1_bad_wind_dir_df['eps_sonic1'], color = 'b', label = 'potential bad wind direction')
# plt.axvline(x=MAD_limit,color = 'gray', label = 'Mad_crit.')
# plt.yscale('log')
# plt.legend()
# plt.xlim(0,20)
# plt.title('MAD vs. Epsilon (Sonic 2)')
# plt.xlabel('MAD value')
# plt.ylabel('Epsilon [m^2/s^3]')

# #%%
# plt.figure()
# plt.scatter(mad3_bad_wind_dir_df['MAD_value_3'], mad3_bad_wind_dir_df['eps_sonic3'], color = 'k', label = 'fine wind direction')
# plt.scatter(mad3_bad_wind_dir_df['MAD_values_for_bad_wind'], mad3_bad_wind_dir_df['eps_sonic3'], color = 'r', label = 'bad wind direction')
# # plt.scatter(mad1_bad_wind_dir_df['MAD_values_for_potential_bad_wind'], mad1_bad_wind_dir_df['eps_sonic1'], color = 'b', label = 'potential bad wind direction')
# plt.axvline(x=MAD_limit,color = 'gray', label = 'Mad_crit.')
# plt.yscale('log')
# plt.legend()
# plt.xlim(0,20)
# plt.title('MAD vs. Epsilon (Sonic 3)')
# plt.xlabel('MAD value')
# plt.ylabel('Epsilon [m^2/s^3]')

# #%%
# plt.figure()
# plt.scatter(mad4_bad_wind_dir_df['MAD_value_4'], mad4_bad_wind_dir_df['eps_sonic4'], color = 'k', label = 'fine wind direction')
# plt.scatter(mad4_bad_wind_dir_df['MAD_values_for_bad_wind'], mad4_bad_wind_dir_df['eps_sonic4'], color = 'r', label = 'bad wind direction')
# # plt.scatter(mad1_bad_wind_dir_df['MAD_values_for_potential_bad_wind'], mad1_bad_wind_dir_df['eps_sonic1'], color = 'b', label = 'potential bad wind direction')
# plt.axvline(x=MAD_limit,color = 'gray', label = 'Mad_crit.')
# plt.yscale('log')
# plt.legend()
# plt.xlim(0,20)
# plt.title('MAD vs. Epsilon (Sonic 4)')
# plt.xlabel('MAD value')
# plt.ylabel('Epsilon [m^2/s^3]')
# #%%
# plt.hist(eps_all_1,bins=5)
# #%%
# merp = np.arange(len(MAD_all_1))
# plt.plot(merp, MAD_all_1)
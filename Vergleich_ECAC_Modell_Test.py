#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[208]:


#Laden der Dateien (acht Vorhersagen ECAC)
ecac_s1 = "/Users/michaelmerkel/Desktop/Alles/N_ECAC_Modellierung/S1ECAC.csv"
ecac_original_s1 = pd.read_csv(ecac_s1)
ecac_s1 = ecac_original_s1.copy()

ecac_s2 = "/Users/michaelmerkel/Desktop/Alles/N_ECAC_Modellierung/S2ECAC.csv"
ecac_original_s2 = pd.read_csv(ecac_s2)
ecac_s2 = ecac_original_s2.copy()

ecac_s3 = "/Users/michaelmerkel/Desktop/Alles/N_ECAC_Modellierung/S3ECAC.csv"
ecac_original_s3 = pd.read_csv(ecac_s3)
ecac_s3 = ecac_original_s3.copy()

ecac_s4 = "/Users/michaelmerkel/Desktop/Alles/N_ECAC_Modellierung/S4ECAC.csv"
ecac_original_s4 = pd.read_csv(ecac_s4)
ecac_s4 = ecac_original_s4.copy()

ecac_s5 = "/Users/michaelmerkel/Desktop/Alles/N_ECAC_Modellierung/S5ECAC.csv"
ecac_original_s5 = pd.read_csv(ecac_s5)
ecac_s5 = ecac_original_s5.copy()

ecac_s6 = "/Users/michaelmerkel/Desktop/Alles/N_ECAC_Modellierung/S6ECAC.csv"
ecac_original_s6 = pd.read_csv(ecac_s6)
ecac_s6 = ecac_original_s6.copy()

ecac_s7= "/Users/michaelmerkel/Desktop/Alles/N_ECAC_Modellierung/S7ECAC.csv"
ecac_original_s7 = pd.read_csv(ecac_s7)
ecac_s7 = ecac_original_s7.copy()

ecac_s8 = "/Users/michaelmerkel/Desktop/Alles/N_ECAC_Modellierung/S8ECAC.csv"
ecac_original_s8 = pd.read_csv(ecac_s8)
ecac_s8 = ecac_original_s8.copy()


# In[209]:


#Laden der Dateien (acht Szenarien)
szenario_s1 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_Meter/condition_1_A_warm.csv"
szenario_original_s1 = pd.read_csv(szenario_s1)
szenario_s1 = szenario_original_s1.copy()

szenario_s2 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_Meter/condition_2_A_cold.csv"
szenario_original_s2 = pd.read_csv(szenario_s2)
szenario_s2 = szenario_original_s2.copy()

szenario_s3 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_Meter/condition_3_A_wind.csv"
szenario_original_s3 = pd.read_csv(szenario_s3)
szenario_s3 = szenario_original_s3.copy()

szenario_s4 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_Meter/condition_4_A_no_wind.csv"
szenario_original_s4 = pd.read_csv(szenario_s4)
szenario_s4 = szenario_original_s4.copy()

szenario_s5 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_Meter/condition_5_D_warm.csv"
szenario_original_s5 = pd.read_csv(szenario_s5)
szenario_s5 = szenario_original_s5.copy()

szenario_s6 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_Meter/condition_6_D_cold.csv"
szenario_original_s6 = pd.read_csv(szenario_s6)
szenario_s6 = szenario_original_s6.copy()

szenario_s7= "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_Meter/condition_7_D_wind.csv"
szenario_original_s7 = pd.read_csv(szenario_s7)
szenario_s7 = szenario_original_s7.copy()

szenario_s8 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_Meter/condition_8_D_no_wind.csv"
szenario_original_s8 = pd.read_csv(szenario_s8)
szenario_s8 = szenario_original_s8.copy()


# In[210]:


#Laden der Dateien (acht Vorhersagen - Modell)
modell_s1 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/N_Vorhersagen_Modell/s1.csv"
modell_original_s1 = pd.read_csv(modell_s1)
modell_s1 = modell_original_s1.copy()

modell_s2 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/N_Vorhersagen_Modell/s2.csv"
modell_original_s2 = pd.read_csv(modell_s2)
modell_s2 = modell_original_s2.copy()

modell_s3 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/N_Vorhersagen_Modell/s3.csv"
modell_original_s3 = pd.read_csv(modell_s3)
modell_s3 = modell_original_s3.copy()

modell_s4 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/N_Vorhersagen_Modell/s4.csv"
modell_original_s4 = pd.read_csv(modell_s4)
modell_s4 = modell_original_s4.copy()

modell_s5 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/N_Vorhersagen_Modell/s5.csv"
modell_original_s5 = pd.read_csv(modell_s5)
modell_s5 = modell_original_s5.copy()

modell_s6 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/N_Vorhersagen_Modell/s6.csv"
modell_original_s6 = pd.read_csv(modell_s6)
modell_s6 = modell_original_s6.copy()

modell_s7= "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/N_Vorhersagen_Modell/s7.csv"
modell_original_s7 = pd.read_csv(modell_s7)
modell_s7 = modell_original_s7.copy()

modell_s8 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/N_Vorhersagen_Modell/s8.csv"
modell_original_s8 = pd.read_csv(modell_s8)
modell_s8 = modell_original_s8.copy()


# In[211]:


#modell_s1


# In[212]:


ecac_s1.drop('Unnamed: 0', axis=1, inplace=True)
ecac_s2.drop('Unnamed: 0', axis=1, inplace=True)
ecac_s3.drop('Unnamed: 0', axis=1, inplace=True)
ecac_s4.drop('Unnamed: 0', axis=1, inplace=True)
ecac_s5.drop('Unnamed: 0', axis=1, inplace=True)
ecac_s6.drop('Unnamed: 0', axis=1, inplace=True)
ecac_s7.drop('Unnamed: 0', axis=1, inplace=True)
ecac_s8.drop('Unnamed: 0', axis=1, inplace=True)


# In[213]:


ecac_1 = ecac_s1
ecac_2 = ecac_s2
ecac_3 = ecac_s3
ecac_4 = ecac_s4
ecac_5 = ecac_s5
ecac_6 = ecac_s6
ecac_7 = ecac_s7
ecac_8 = ecac_s8


# In[214]:


# Eine Liste mit allen Szenario-DataFrames
szenarios = [szenario_s1, szenario_s2, szenario_s3, szenario_s4, szenario_s5, szenario_s6, szenario_s7, szenario_s8]

reshaped_szenarios = {}

for i, szenario in enumerate(szenarios, 1):
    reshaped_array = szenario['Lärm'].values.reshape(-1, 40)
    reshaped_df = pd.DataFrame(reshaped_array)
    reshaped_df = reshaped_df.transpose()
    reshaped_df.columns = range(1, reshaped_df.shape[1] + 1)
    reshaped_szenarios[f'szenario_{i}'] = reshaped_df


# In[215]:


szenario_1 = reshaped_szenarios['szenario_1']
szenario_2 = reshaped_szenarios['szenario_2']
szenario_3 = reshaped_szenarios['szenario_3']
szenario_4 = reshaped_szenarios['szenario_4']
szenario_5 = reshaped_szenarios['szenario_5']
szenario_6 = reshaped_szenarios['szenario_6']
szenario_7 = reshaped_szenarios['szenario_7']
szenario_8 = reshaped_szenarios['szenario_8']


# In[216]:


#ecac_1


# In[217]:


#szenario_1


# In[218]:


# Fehlermetriken

def calculate_metrics(szenario, ecac):
    ecac = ecac.fillna(0).copy() 
    while ecac.shape[0] < szenario.shape[0]:
        ecac.loc[ecac.shape[0]] = [0] * ecac.shape[1]

    szenario.columns = szenario.columns.astype(str)
    ecac.columns = ecac.columns.astype(str)
    
    mae_list = []
    mse_list = []
    rmse_list = []
    
    if szenario.shape == ecac.shape:
        for col in szenario.columns:
            y_true = szenario[col]
            y_pred = ecac[col]
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0) & (y_pred != 0)
            
            if np.sum(mask) == 0:
                continue
            
            filtered_y_true = y_true[mask]
            filtered_y_pred = y_pred[mask]
            
            mae = mean_absolute_error(filtered_y_true, filtered_y_pred)
            mse = mean_squared_error(filtered_y_true, filtered_y_pred)
            rmse = np.sqrt(mse)
            
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            
        avg_mae = np.mean(mae_list)
        avg_mse = np.mean(mse_list)
        avg_rmse = np.mean(rmse_list)
        
        return avg_mae, avg_mse, avg_rmse, ecac 
    else:
        return np.nan, np.nan, np.nan, None
    

results = {'Szenario': [], 'MAE': [], 'MSE': [], 'RMSE': []}

ecac_dict = {}

for i in range(1, 9):
    szenario_name = f'szenario_{i}'
    ecac_name = f'ecac_{i}'
    
    szenario_df = locals()[szenario_name]
    ecac_df = locals()[ecac_name]
    
    avg_mae, avg_mse, avg_rmse, updated_ecac = calculate_metrics(szenario_df, ecac_df)

    if updated_ecac is not None:
        locals()[ecac_name] = updated_ecac 
        ecac_dict[ecac_name] = updated_ecac  
    
    results['Szenario'].append(f"Szenario {i}")
    results['MAE'].append(avg_mae)
    results['MSE'].append(avg_mse)
    results['RMSE'].append(avg_rmse)


metriken_ecac = pd.DataFrame(results)
metriken_ecac.set_index('Szenario', inplace=True)
metriken_ecac = metriken_ecac.transpose()

metriken_ecac = metriken_ecac.round(2)
metriken_ecac


# In[222]:


def calculate_metrics(szenario, ecac):
    ecac = ecac.fillna(0).copy() 
    while ecac.shape[0] < szenario.shape[0]:
        ecac.loc[ecac.shape[0]] = [0] * ecac.shape[1]

    szenario.columns = szenario.columns.astype(str)
    ecac.columns = ecac.columns.astype(str)
    
    mae_list = []
    mse_list = []
    rmse_list = []
    
    if szenario.shape == ecac.shape:
        for col in szenario.columns:
            y_true = szenario[col]
            y_pred = ecac[col]
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0) & (y_pred != 0)
            
            if np.sum(mask) == 0:
                continue
            
            filtered_y_true = y_true[mask]
            filtered_y_pred = y_pred[mask]
            
            mae = mean_absolute_error(filtered_y_true, filtered_y_pred)
            mse = mean_squared_error(filtered_y_true, filtered_y_pred)
            rmse = np.sqrt(mse)
            
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            
        avg_mae = np.mean(mae_list)
        avg_mse = np.mean(mse_list)
        avg_rmse = np.mean(rmse_list)
        
        return avg_mae, avg_mse, avg_rmse, ecac  # ecac zurückgeben
    else:
        return np.nan, np.nan, np.nan, None

# letzten auf Null setzen aus Originaldaten
def adjust_last_value(df):
    for col in df.columns:
        # Finde die Indizes, wo die Werte nicht Null sind
        non_zero_indices = df.index[df[col] != 0].tolist()
        if non_zero_indices:
            last_non_zero_index = non_zero_indices[-1]
            df.at[last_non_zero_index, col] = 0
    return df


results = {'Szenario': [], 'MAE': [], 'MSE': [], 'RMSE': []}
ecac_dict = {}

# Aktualisiere alle Szenario-DataFrames und berechne Metriken
for i in range(1, 9):
    szenario_name = f'szenario_{i}'
    ecac_name = f'ecac_{i}'
    
    szenario_df = locals()[szenario_name]
    ecac_df = locals()[ecac_name]

    adjusted_szenario_df = adjust_last_value(szenario_df)
    locals()[szenario_name] = adjusted_szenario_df
    avg_mae, avg_mse, avg_rmse, updated_ecac = calculate_metrics(adjusted_szenario_df, ecac_df)
    
    if updated_ecac is not None:
        locals()[ecac_name] = updated_ecac  
        ecac_dict[ecac_name] = updated_ecac 
    
    results['Szenario'].append(f"Szenario {i}")
    results['MAE'].append(avg_mae)
    results['MSE'].append(avg_mse)
    results['RMSE'].append(avg_rmse)

metriken_ecac = pd.DataFrame(results)
metriken_ecac.set_index('Szenario', inplace=True)
metriken_ecac = metriken_ecac.transpose()

metriken_ecac = metriken_ecac.round(2)
metriken_ecac


# In[ ]:





# In[170]:


#ecac_3


# In[224]:


#Abbildungen

import matplotlib.pyplot as plt

for i in range(1, 9): 
    szenario = locals()[f'szenario_{i}']
    ecac = locals()[f'ecac_{i}']

    for col in szenario.columns.astype(str): 
        plt.figure(figsize=(8, 5))
        
        plt.plot(szenario[col].dropna(), label='Realverlauf', color = "lightblue")
        plt.plot(ecac[col].dropna(), label=f'Vorhersage', color = 'darkred', linestyle='--')
        
        plt.title(f'Lärmberechnung nach ECAC Doc. 29 für Szenario {i} ({col})')
        plt.xlabel('Sekunde im Lärmereignis')
        plt.ylabel('Lärmpegel [dB(A)]')
        plt.ylim(-2,80)
        plt.legend()
        #plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_ECAC_Modellierung/Abbildungen/Neu_Szenario_{i}({col})', dpi = 300)
        plt.show()


# In[206]:


#Abbildungen

import matplotlib.pyplot as plt

for i in range(1, 9):  
    szenario = locals()[f'szenario_{i}']
    ecac = locals()[f'ecac_{i}']
    modell_vorhersage = locals()[f'modell_s{i}'] 

    for col_index in range(min(len(szenario.columns), len(ecac.columns), len(modell_vorhersage.columns))):
        plt.figure(figsize=(8, 5))
        plt.plot(szenario.iloc[:, col_index].dropna(), label='Realverlauf', color="lightblue")
        plt.plot(ecac.iloc[:, col_index].dropna(), label='ECAC Vorhersage', color='darkred', linestyle='--')
        plt.plot(modell_vorhersage.iloc[:, col_index].dropna(), label='Modell Vorhersage', color='#2E4E60', linestyle='--')

        plt.title(f'Vergleich der Lärmvorhersagen für Szenario {i}, Lärmereignis {col_index+1}')
        plt.xlabel('Sekunde im Lärmereignis')
        plt.ylabel('Lärmpegel [dB(A)]')
        plt.legend()
        plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_Szenarien/Beide_Vorhersagen_Vergleich/Szenario_{i}({col+1})', dpi=300)
        plt.show()


# In[221]:


# Abbildungen

import matplotlib.pyplot as plt

for i in range(1, 9):  
    szenario = locals()[f'szenario_{i}']
    ecac = locals()[f'ecac_{i}']
    modell_vorhersage = locals()[f'modell_s{i}'] 

    for col_index in range(len(szenario.columns)):  
        plt.figure(figsize=(8, 5))

        plt.plot(szenario.iloc[:, col_index].dropna(), label='Realverlauf', color="lightblue")
        plt.plot(ecac.iloc[:, col_index].dropna(), label='ECAC-Vorhersage', color='darkred', linestyle='--')
        plt.plot(modell_vorhersage.iloc[:, col_index].dropna(), label='Modellvorhersage', color='#2E4E60', linestyle='--')

        plt.title(f'Vergleich der Vorhersagen für Szenario {i} ({col_index+1})')
        plt.xlabel('Sekunde im Lärmereignis')
        plt.ylabel('Lärmpegel [dB(A)]')
        plt.ylim(-2,80)
        plt.legend()
        
        # Speichere die Abbildungen
        #plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_Szenarien/Beide_Vorhersagen_Vergleich/Szenario_{i}_Spalte_{col_index+1}.png', dpi=300)
        
        plt.show() 
        plt.close() 


# In[ ]:





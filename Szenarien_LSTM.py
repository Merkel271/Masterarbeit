#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint


# In[7]:


#Laden der Dateien (acht Szenarien und gesamter Datensatz ohne Szenarien)
datensatz_all = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/datensatz_ohne_szenarien.csv"
datensatz_original_all = pd.read_csv(datensatz_all)
datensatz_all = datensatz_original_all.copy()

datensatz_s1 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_1_A_warm.csv"
datensatz_original_s1 = pd.read_csv(datensatz_s1)
datensatz_s1 = datensatz_original_s1.copy()

datensatz_s2 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_2_A_cold.csv"
datensatz_original_s2 = pd.read_csv(datensatz_s2)
datensatz_s2 = datensatz_original_s2.copy()

datensatz_s3 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_3_A_wind.csv"
datensatz_original_s3 = pd.read_csv(datensatz_s3)
datensatz_s3 = datensatz_original_s3.copy()

datensatz_s4 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_4_A_no_wind.csv"
datensatz_original_s4 = pd.read_csv(datensatz_s4)
datensatz_s4 = datensatz_original_s4.copy()

datensatz_s5 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_5_D_warm.csv"
datensatz_original_s5 = pd.read_csv(datensatz_s5)
datensatz_s5 = datensatz_original_s5.copy()

datensatz_s6 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_6_D_cold.csv"
datensatz_original_s6 = pd.read_csv(datensatz_s6)
datensatz_s6 = datensatz_original_s6.copy()

datensatz_s7= "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_7_D_wind.csv"
datensatz_original_s7 = pd.read_csv(datensatz_s7)
datensatz_s7 = datensatz_original_s7.copy()

datensatz_s8 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_8_D_no_wind.csv"
datensatz_original_s8 = pd.read_csv(datensatz_s8)
datensatz_s8 = datensatz_original_s8.copy()


# In[37]:


#datensatz_s2


# In[9]:


# Erst Aufteilen, dann standardisieren!!!

unique_ids = datensatz_all['ID'].unique()
total_ids = len(unique_ids)

train_size = int(0.9 * total_ids)
val_size = total_ids - train_size

np.random.seed(42) 
np.random.shuffle(unique_ids)

train_ids = unique_ids[:train_size]
val_ids = unique_ids[train_size:]

train_daten = datensatz_all[datensatz_all['ID'].isin(train_ids)]
val_daten = datensatz_all[datensatz_all['ID'].isin(val_ids)]


# In[10]:


# Normalisierung
cols_to_normalize = ['Lärm', 'Temperatur', 'Windgeschwindigkeit', 'Windrichtung', 'Windgeschwindigkeit_vertikal', 'Höhe', 'Geschwindigkeit', 'Sekunde_im_Ereignis', 'Längengrad', 'Breitengrad','Uhrzeit']

scaler = MinMaxScaler()

train_daten = train_daten.copy()
val_daten = val_daten.copy()

test_datasets = [datensatz_s1, datensatz_s2, datensatz_s3, datensatz_s4, datensatz_s5, datensatz_s6, datensatz_s7, datensatz_s8]

# Normalisierung für Trainingsdatensatz
condition_train = (train_daten[cols_to_normalize] != 0).sum(axis=1) >= 2
scaler.fit(train_daten.loc[condition_train, cols_to_normalize])
train_daten.loc[condition_train, cols_to_normalize] = scaler.transform(train_daten.loc[condition_train, cols_to_normalize])

# Normalisierung für Validierungsdatensatz
condition_val = (val_daten[cols_to_normalize] != 0).sum(axis=1) >= 2
val_daten.loc[condition_val, cols_to_normalize] = scaler.transform(val_daten.loc[condition_val, cols_to_normalize])

normalized_test_datasets = {}

# Schleife für die Normalisierung der verschiedenen Testdatensätze
for i, test_data in enumerate(test_datasets):
    condition_test = (test_data[cols_to_normalize] != 0).sum(axis=1) >= 2
    test_data.loc[condition_test, cols_to_normalize] = scaler.transform(test_data.loc[condition_test, cols_to_normalize])
    
    # Speichern der normalisierten Testdatensätze im Dictionary
    normalized_test_datasets[f"datensatz_s{i+1}"] = test_data


# In[11]:


# Zugriff auf normalisierte Testdatensätze
datensatz_s1_n = normalized_test_datasets["datensatz_s1"]
datensatz_s2_n = normalized_test_datasets["datensatz_s2"]
datensatz_s3_n = normalized_test_datasets["datensatz_s3"]
datensatz_s4_n = normalized_test_datasets["datensatz_s4"]
datensatz_s5_n = normalized_test_datasets["datensatz_s5"]
datensatz_s6_n = normalized_test_datasets["datensatz_s6"]
datensatz_s7_n = normalized_test_datasets["datensatz_s7"]
datensatz_s8_n = normalized_test_datasets["datensatz_s8"]


# In[12]:


#datensatz_s1_n


# In[13]:


# Spalten selektieren
feature_set = [col for col in train_daten.columns if col not in ['ID', 'Sekunde_im_Ereignis', 'Lärm', 'Datum']]

num_features = len(feature_set)

X_train = train_daten[feature_set].values.astype('float32')
y_train = train_daten['Lärm'].values

num_samples = len(train_daten) // 40 

X_train = np.reshape(X_train, (num_samples, 40, num_features))
y_train = np.reshape(y_train, (num_samples, 40))

X_val = val_daten[feature_set].values.astype('float32')
y_val = val_daten['Lärm'].values
num_samples_val = len(val_daten) // 40
X_val = np.reshape(X_val, (num_samples_val, 40, num_features))
y_val = np.reshape(y_val, (num_samples_val, 40))

model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)

# Modell Definition
model_LSTM = Sequential() 
model_LSTM.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(40, num_features)))
model_LSTM.add(Dropout(0.2))
model_LSTM.add(Bidirectional(LSTM(256, return_sequences=True)))
model_LSTM.add(Bidirectional(LSTM(128)))
model_LSTM.add(Dense(512, activation='relu'))  
model_LSTM.add(Dropout(0.3))
model_LSTM.add(Dense(256, activation='relu'))  
model_LSTM.add(Dropout(0.2))
model_LSTM.add(Dense(128, activation='relu'))  
model_LSTM.add(Dropout(0.1))
model_LSTM.add(Dense(40, activation='linear')) 

model_LSTM.compile(optimizer='adam', loss='mean_squared_error')
history = model_LSTM.fit(X_train, y_train, epochs=200, batch_size=256, validation_data=(X_val, y_val), callbacks=[model_checkpoint,early_stopping])

# Testdatensätze
test_datasets = {
    's1': datensatz_s1_n,
    's2': datensatz_s2_n,
    's3': datensatz_s3_n,
    's4': datensatz_s4_n,
    's5': datensatz_s5_n,
    's6': datensatz_s6_n,
    's7': datensatz_s7_n,
    's8': datensatz_s8_n
}

predictions_LSTM = {}
original = {}

# Vorhersagen für Testdatensätze
for name, test_data in test_datasets.items():
    X_test = test_data[feature_set].values.astype('float32')
    y_test = test_data['Lärm'].values
    num_samples_test = len(test_data) // 40
    X_test = np.reshape(X_test, (num_samples_test, 40, num_features))
    y_test = np.reshape(y_test, (num_samples_test, 40))
    
    predictions_LSTM[name] = model_LSTM.predict(X_test)
    original[name] = y_test

history_dict = {'Feature Set 1': history}


# In[14]:


# Abbildung Test
preds_s1 = predictions_LSTM['s1']
true_s1 = original['s1']

for i in range(5):
    plt.figure(figsize=(14, 6))
    plt.plot(preds_s1[i], label=f'Vorhersage (Sequenz {i+1})')
    plt.plot(true_s1[i], label=f'Wahrheit (Sequenz {i+1})')
    plt.title(f'Vorhersage vs. Wahrheit für s1, Sequenz {i+1}')
    plt.xlabel('Zeitschritte')
    plt.ylabel('Lärm')
    plt.legend()
    plt.show()



# In[51]:


# Zurücktranformieren
y_test_transformed = {}
y_pred_transformed = {}

col_index = cols_to_normalize.index('Lärm')
min_val = scaler.data_min_[col_index]
max_val = scaler.data_max_[col_index]

for szenario in test_datasets.keys():
    y_test_original = original[szenario]
    y_pred_original = predictions_LSTM[szenario]
    
    non_zero_mask = y_test_original != 0
    
    scale_factor = max_val - min_val

    y_test_transformed[szenario] = np.where(non_zero_mask, y_test_original * scale_factor + min_val, 0)
    y_pred_transformed[szenario] = np.where(non_zero_mask, y_pred_original * scale_factor + min_val, 0)


# In[88]:


# Abbildung S1
y_test_s1 = y_test_transformed['s1']
y_pred_s1 = y_pred_transformed['s1']

# Für jeden der fünf Sequenzen
for i in range(5):
    plt.figure(figsize=(8,5))

    plt.plot(y_test_s1[i], label='Realverlauf', color = 'lightblue')

    plt.plot(y_pred_s1[i], label='Vorhersage', color = '#2E4E60', linestyle= '--')

    plt.title(f'Modellvorhersage für Szenario 1 ({i+1})')
    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel [dB(A)]')
    plt.ylim(-2,80)
    plt.legend()
    plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/S1/Lärmkurve{i+1}.png', dpi = 300)
    plt.show()



# In[89]:


# Abbildung S2
y_test_s1 = y_test_transformed['s2']
y_pred_s1 = y_pred_transformed['s2']

for i in range(5):
    plt.figure(figsize=(8,5))
    plt.plot(y_test_s1[i], label='Realverlauf', color = 'lightblue')
    plt.plot(y_pred_s1[i], label='Vorhersage', color = '#2E4E60', linestyle= '--')

    plt.title(f'Modellvorhersage für Szenario 2 ({i+1})')
    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel [dB(A)]')
    plt.ylim(-2,80)
    plt.legend()
    plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/S2/Lärmkurve{i+1}.png', dpi = 300)
    plt.show()



# In[90]:


# Abbildung S3
y_test_s1 = y_test_transformed['s3']
y_pred_s1 = y_pred_transformed['s3']

for i in range(5):
    plt.figure(figsize=(8,5))

    plt.plot(y_test_s1[i], label='Realverlauf', color = 'lightblue')

    plt.plot(y_pred_s1[i], label='Vorhersage', color = '#2E4E60', linestyle= '--')

    plt.title(f'Modellvorhersage für Szenario 3 ({i+1})')
    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel [dB(A)]')
    plt.ylim(-2,80)
    plt.legend()
    plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/S3/Lärmkurve{i+1}.png', dpi = 300)
    plt.show()



# In[91]:


# Abbildung S4
y_test_s1 = y_test_transformed['s4']
y_pred_s1 = y_pred_transformed['s4']

# Für jeden der fünf Sequenzen
for i in range(5):
    plt.figure(figsize=(8,5))

    plt.plot(y_test_s1[i], label='Realverlauf', color = 'lightblue')

    plt.plot(y_pred_s1[i], label='Vorhersage', color = '#2E4E60', linestyle= '--')

    plt.title(f'Modellvorhersage für Szenario 4 ({i+1})')
    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel [dB(A)]')
    plt.ylim(-2,80)
    plt.legend()
    plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/S4/Lärmkurve{i+1}.png', dpi = 300)
    plt.show()



# In[92]:


# Abbildung S5
y_test_s1 = y_test_transformed['s5']
y_pred_s1 = y_pred_transformed['s5']

for i in range(5):
    plt.figure(figsize=(8,5))

    plt.plot(y_test_s1[i], label='Realverlauf', color = 'lightblue')

    plt.plot(y_pred_s1[i], label='Vorhersage', color = '#2E4E60', linestyle= '--')

    plt.title(f'Modellvorhersage für Szenario 5 ({i+1})')
    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel [dB(A)]')
    plt.ylim(-2,80)
    plt.legend()
    plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/S5/Lärmkurve{i+1}.png', dpi = 300)
    plt.show()



# In[93]:


# Abbildung S6
y_test_s1 = y_test_transformed['s6']
y_pred_s1 = y_pred_transformed['s6']

for i in range(5):
    plt.figure(figsize=(8,5))

    plt.plot(y_test_s1[i], label='Realverlauf', color = 'lightblue')

    plt.plot(y_pred_s1[i], label='Vorhersage', color = '#2E4E60', linestyle= '--')

    plt.title(f'Modellvorhersage für Szenario 6 ({i+1})')
    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel [dB(A)]')
    plt.ylim(-2,80)
    plt.legend()
    plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/S6/Lärmkurve{i+1}.png', dpi = 300)
    plt.show()



# In[87]:


# Abbildung S7
y_test_s1 = y_test_transformed['s7']
y_pred_s1 = y_pred_transformed['s7']

for i in range(5):
    plt.figure(figsize=(8,5))

    plt.plot(y_test_s1[i], label='Realverlauf', color = 'lightblue')
    plt.plot(y_pred_s1[i], label='Vorhersage', color = '#2E4E60', linestyle= '--')

    plt.title(f'Modellvorhersage für Szenario 7 ({i+1})')
    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel [dB(A)]')
    plt.ylim(-2,80)
    plt.legend()
    plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/S7/Lärmkurve{i+1}.png', dpi = 300)
    plt.show()



# In[94]:


# Abbildung S8
y_test_s1 = y_test_transformed['s8']
y_pred_s1 = y_pred_transformed['s8']

for i in range(5):
    plt.figure(figsize=(8,5))
    plt.plot(y_test_s1[i], label='Realverlauf', color = 'lightblue')
    plt.plot(y_pred_s1[i], label='Vorhersage', color = '#2E4E60', linestyle= '--')

    plt.title(f'Modellvorhersage für Szenario 8 ({i+1})')
    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel [dB(A)]')
    plt.ylim(-2,80)
    plt.legend()
    plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/S8/Lärmkurve{i+1}.png', dpi = 300)
    plt.show()



# In[61]:


# Fehlermetrilen
metrics_dict = {}

for szenario in test_datasets.keys():
    y_test = y_test_transformed[szenario]
    y_pred = y_pred_transformed[szenario]

    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    
    metrics_dict[szenario] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

for szenario, metrics in metrics_dict.items():
    print(f"Metriken für {szenario}:")
    print(f"  MAE: {metrics['MAE']}")
    print(f"  MSE: {metrics['MSE']}")
    print(f"  RMSE: {metrics['RMSE']}")
    




# In[62]:


metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
new_columns = {f's{i+1}': f'Szenario {i+1}' for i in range(8)}
metrics_df.rename(columns=new_columns, inplace=True)
rounded_metrics_df = metrics_df.round(2)

rounded_metrics_df


# In[63]:


# Transponieren des DataFrames
transposed_metrics_df = rounded_metrics_df.transpose()
new_columns = {f's{i+1}': f'Szenario {i+1}' for i in range(8)}
transposed_metrics_df.rename(index=new_columns, inplace=True)
transposed_metrics_df


# In[64]:


# Manuelle Änderung der Spaltennamen
transposed_metrics_df.rename(index={
    's1': 'Szenario 1',
    's2': 'Szenario 2',
    's3': 'Szenario 3',
    's4': 'Szenario 4',
    's5': 'Szenario 5',
    's6': 'Szenario 6',
    's7': 'Szenario 7',
    's8': 'Szenario 8'
}, inplace=True)

transposed_metrics_df


# In[65]:


cloned_df = transposed_metrics_df.copy()
cloned_df.columns = ['Szenario 1', 'Szenario 2', 'Szenario 3', 'Szenario 4', 'Szenario 5', 'Szenario 6', 'Szenario 7', 'Szenario 8']
cloned_df


# In[33]:


szenarios = [datensatz_s1, datensatz_s2, datensatz_s3, datensatz_s4, datensatz_s5, datensatz_s6, datensatz_s7, datensatz_s8]

reshaped_szenarios = {}

for i, szenario in enumerate(szenarios, 1):
    reshaped_array = szenario['Lärm'].values.reshape(-1, 40)
    reshaped_df = pd.DataFrame(reshaped_array)
    reshaped_df = reshaped_df.transpose()
    reshaped_df.columns = range(1, reshaped_df.shape[1] + 1)
    reshaped_szenarios[f'datensatz_{i}'] = reshaped_df


# In[35]:


szenario_1 = reshaped_szenarios['datensatz_1']
szenario_2 = reshaped_szenarios['datensatz_2']
szenario_3 = reshaped_szenarios['datensatz_3']
szenario_4 = reshaped_szenarios['datensatz_4']
szenario_5 = reshaped_szenarios['datensatz_5']
szenario_6 = reshaped_szenarios['datensatz_6']
szenario_7 = reshaped_szenarios['datensatz_7']
szenario_8 = reshaped_szenarios['datensatz_8']


# In[36]:


szenario_1


# In[70]:


# Vorhersagen werden als CSV gespeichert
dfs = {}

for scenario in range(1, 9):  
    scenario_key = f's{scenario}' 
    dfs[scenario_key] = pd.DataFrame(y_pred_transformed[scenario_key].T, columns=[f'Lärmereignis_{i+1}' for i in range(y_pred_transformed[scenario_key].shape[0])])
#Kontrolle
#dfs['s1']


# In[71]:


import os

output_dir = '/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/N_Vorhersagen_Modell_'

for scenario, df in dfs.items():
    output_path = os.path.join(output_dir, f'{scenario}.csv')
    df.to_csv(output_path, index=False)
    print(f'DataFrame für {scenario} gespeichert unter: {output_path}')


# In[ ]:





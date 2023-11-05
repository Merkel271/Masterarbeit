#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


datensatz = "/Users/michaelmerkel/Desktop/Alles/Padded.csv"
datensatz_original = pd.read_csv(datensatz)
datensatz = datensatz_original.copy()


# In[3]:


#datensatz


# In[4]:


# Erst Aufteilen, dann standardisieren

unique_ids = datensatz['ID'].unique()
total_ids = len(unique_ids)

# zufällig Aufteilen
train_size = int(0.7 * total_ids)
val_size = int(0.1 * total_ids)
np.random.shuffle(unique_ids)
train_ids = unique_ids[:train_size]
val_ids = unique_ids[train_size:train_size + val_size]
test_ids = unique_ids[train_size + val_size:]

train_daten = datensatz[datensatz['ID'].isin(train_ids)]
val_daten = datensatz[datensatz['ID'].isin(val_ids)]
test_daten = datensatz[datensatz['ID'].isin(test_ids)]


# In[23]:


#train_daten


# In[24]:


#val_daten


# In[25]:


#test_daten


# In[5]:


# Normalisierung
cols_to_normalize = ['Lärm', 'Temperatur', 'Windgeschwindigkeit', 'Windrichtung', 'Windgeschwindigkeit_vertikal', 'Höhe', 'Geschwindigkeit', 'Sekunde_im_Ereignis', 'Längengrad', 'Breitengrad']

scaler = MinMaxScaler()

# Kopien
train_daten = train_daten.copy()
val_daten = val_daten.copy()
test_daten = test_daten.copy()

# Zero Padding beachten
condition_train = (train_daten[cols_to_normalize] != 0).sum(axis=1) >= 2
condition_val = (val_daten[cols_to_normalize] != 0).sum(axis=1) >= 2
condition_test = (test_daten[cols_to_normalize] != 0).sum(axis=1) >= 2

# Normalisierung für Trainingsdatensatz
scaler.fit(train_daten.loc[condition_train, cols_to_normalize])
train_daten.loc[condition_train, cols_to_normalize] = scaler.transform(train_daten.loc[condition_train, cols_to_normalize])

# Normalisierung für Validierungs- und Testdatensatz
val_daten.loc[condition_val, cols_to_normalize] = scaler.transform(val_daten.loc[condition_val, cols_to_normalize])
test_daten.loc[condition_test, cols_to_normalize] = scaler.transform(test_daten.loc[condition_test, cols_to_normalize])


# In[27]:


#print(test_daten.columns.tolist())


# In[53]:


from sklearn.model_selection import KFold

#mit allem
feature_columns1 = [col for col in train_daten.columns if col not in ['ID', 'Sekunde_im_Ereignis', 'Lärm', 'Datum','Uhrzeit']]

feature_sets = [feature_columns1]


kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_MLP = []
cv_scores_LSTM = []

predictions_MLP_cv = []
predictions_LSTM_cv = []

# Nur für das MLP Modell und das LSTM Modell
for e in feature_sets:
    num_features = len(e)

    X = train_daten[e].values
    y = train_daten['Lärm'].values

    fold = 1 

    for train_index, val_index in kf.split(X):
        print(f"--- Faltung {fold} ---")
        
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        y_train_cv, y_val_cv = y[train_index], y[val_index]
        num_samples_cv = len(X_train_cv) // 40
        remainder_train = len(X_train_cv) % 40
        if remainder_train != 0:
            X_train_cv = X_train_cv[:-remainder_train]
            y_train_cv = y_train_cv[:-remainder_train]

        remainder_val = len(X_val_cv) % 40
        if remainder_val != 0:
            X_val_cv = X_val_cv[:-remainder_val]
            y_val_cv = y_val_cv[:-remainder_val]

        X_train_cv = np.reshape(X_train_cv, (num_samples_cv, 40, num_features))
        y_train_cv = np.reshape(y_train_cv, (num_samples_cv, 40))
        X_train_cv = X_train_cv.astype('float32')

        num_samples_val_cv = len(X_val_cv) // 40
        X_val_cv = np.reshape(X_val_cv, (num_samples_val_cv, 40, num_features))
        y_val_cv = np.reshape(y_val_cv, (num_samples_val_cv, 40))
        X_val_cv = X_val_cv.astype('float32')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        
        #MLP
        model_MLP = Sequential()
        model_MLP.add(Dense(256, activation='sigmoid', input_shape=(40, num_features)))
        model_MLP.add(Dropout(0.3))
        model_MLP.add(Dense(1, activation='tanh'))
        model_MLP.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        history_MLP = model_MLP.fit(X_train_cv, y_train_cv, epochs=10, batch_size=512, validation_data=(X_val_cv, y_val_cv), callbacks=[early_stop])

        val_score_MLP = model_MLP.evaluate(X_val_cv, y_val_cv, verbose=0)
        cv_scores_MLP.append(val_score_MLP)
        print(f"MLP Validierungsscore für Faltung {fold}: {val_score_MLP}")

        #LSTM
        model_LSTM = Sequential()
        model_LSTM.add(LSTM(256, activation='relu', input_shape=(40, num_features)))
        model_LSTM.add(Dropout(0.3))
        model_LSTM.add(Dense(40, activation='linear'))  
        model_LSTM.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        history_LSTM = model_LSTM.fit(X_train_cv, y_train_cv, epochs=10, batch_size=512, validation_data=(X_val_cv, y_val_cv), callbacks=[early_stop])

        val_score_LSTM = model_LSTM.evaluate(X_val_cv, y_val_cv, verbose=0)
        cv_scores_LSTM.append(val_score_LSTM)
        print(f"LSTM Validierungsscore für Faltung {fold}: {val_score_LSTM}")

        fold += 1

# Durchschnittliche Validierungsscores und Standardabweichungen
avg_val_score_MLP = np.mean(cv_scores_MLP)
std_val_score_MLP = np.std(cv_scores_MLP)

avg_val_score_LSTM = np.mean(cv_scores_LSTM)
std_val_score_LSTM = np.std(cv_scores_LSTM)

print(f"Durchschnittlicher Validierungsscore für MLP: {avg_val_score_MLP} ± {std_val_score_MLP}")
print(f"Durchschnittlicher Validierungsscore für LSTM: {avg_val_score_LSTM} ± {std_val_score_LSTM}")


# In[20]:


import pandas as pd

mlp_values = [0.016135279089212418, 0.008655562065541744, 0.01341218501329422, 0.017622945830225945, 0.012570569291710854]
lstm_values = [0.05022292956709862, 0.05072935298085213, 0.049639992415905, 0.04894394427537918, 0.048850417137145996]

df = pd.DataFrame({
    'MLP': mlp_values,
    'LSTM': lstm_values
})

# Durchschnitt und Standardabweichung
mean_values = df.mean()
std_values = df.std()

summary_df = pd.DataFrame({
    'Durchschnittsverlust': mean_values,
    'Standardabweichung': std_values
})

summary_df


# In[78]:


# Boxplot

mlp_values = [0.016135279089212418, 0.008655562065541744, 0.01341218501329422, 0.017622945830225945, 0.012570569291710854]
lstm_values = [0.05022292956709862, 0.05072935298085213, 0.049639992415905, 0.04894394427537918, 0.048850417137145996]

data = [mlp_values, lstm_values]
labels = ['MLP', 'LSTM']

bp = plt.boxplot(data, labels=labels, patch_artist=True)
plt.title("Boxplot MLP und LSTM")
plt.ylabel("Validierungswert")
plt.xlabel("Modell")

for median in bp['medians']:
    median.set(color='lightblue', linewidth=2)
for box in bp['boxes']:
    box.set_facecolor('#2E4E60')
    
#plt.savefig('/Users/michaelmerkel/Desktop/Alles/Boxplot_MLP_LSTM.png', dpi = 300)
plt.show()


# In[79]:


# Spalten selektieren
#ohne alles 
#feature_columns1 = [col for col in train_daten.columns if col not in ['ID', 'Datum', 'Uhrzeit', 'Lärm', 'Temperatur', 'Windrichtung', 'Windgeschwindigkeit', 'Windgeschwindigkeit_vertikal', 'Geschwindigkeit', 'Höhe', 'Breitengrad', 'Längengrad', 'Sekunde_im_Ereignis', 'Flugzeugtyp_A139', 'Flugzeugtyp_A19N', 'Flugzeugtyp_A20N', 'Flugzeugtyp_A21N', 'Flugzeugtyp_A306', 'Flugzeugtyp_A310', 'Flugzeugtyp_A318', 'Flugzeugtyp_A319', 'Flugzeugtyp_A320', 'Flugzeugtyp_A321', 'Flugzeugtyp_A332', 'Flugzeugtyp_A333', 'Flugzeugtyp_A339', 'Flugzeugtyp_A342', 'Flugzeugtyp_A343', 'Flugzeugtyp_A345', 'Flugzeugtyp_A346', 'Flugzeugtyp_A359', 'Flugzeugtyp_A35K', 'Flugzeugtyp_A388', 'Flugzeugtyp_ASTR', 'Flugzeugtyp_AT75', 'Flugzeugtyp_AT76', 'Flugzeugtyp_B190', 'Flugzeugtyp_B350', 'Flugzeugtyp_B38M', 'Flugzeugtyp_B39M', 'Flugzeugtyp_B58T', 'Flugzeugtyp_B733', 'Flugzeugtyp_B734', 'Flugzeugtyp_B736', 'Flugzeugtyp_B737', 'Flugzeugtyp_B738', 'Flugzeugtyp_B739', 'Flugzeugtyp_B744', 'Flugzeugtyp_B748', 'Flugzeugtyp_B752', 'Flugzeugtyp_B753', 'Flugzeugtyp_B762', 'Flugzeugtyp_B763', 'Flugzeugtyp_B764', 'Flugzeugtyp_B772', 'Flugzeugtyp_B77L', 'Flugzeugtyp_B77W', 'Flugzeugtyp_B788', 'Flugzeugtyp_B789', 'Flugzeugtyp_B78X', 'Flugzeugtyp_BCS1', 'Flugzeugtyp_BCS3', 'Flugzeugtyp_BE20', 'Flugzeugtyp_BE40', 'Flugzeugtyp_BE58', 'Flugzeugtyp_C130', 'Flugzeugtyp_C17', 'Flugzeugtyp_C208', 'Flugzeugtyp_C25A', 'Flugzeugtyp_C25B', 'Flugzeugtyp_C25C', 'Flugzeugtyp_C25M', 'Flugzeugtyp_C421', 'Flugzeugtyp_C425', 'Flugzeugtyp_C500', 'Flugzeugtyp_C501', 'Flugzeugtyp_C510', 'Flugzeugtyp_C525', 'Flugzeugtyp_C550', 'Flugzeugtyp_C551', 'Flugzeugtyp_C55B', 'Flugzeugtyp_C560', 'Flugzeugtyp_C56X', 'Flugzeugtyp_C650', 'Flugzeugtyp_C680', 'Flugzeugtyp_C68A', 'Flugzeugtyp_C750', 'Flugzeugtyp_CL30', 'Flugzeugtyp_CL35', 'Flugzeugtyp_CL60', 'Flugzeugtyp_CRJ2', 'Flugzeugtyp_CRJ9', 'Flugzeugtyp_CRJX', 'Flugzeugtyp_D328', 'Flugzeugtyp_DA62', 'Flugzeugtyp_DH8D', 'Flugzeugtyp_E135', 'Flugzeugtyp_E145', 'Flugzeugtyp_E170', 'Flugzeugtyp_E190', 'Flugzeugtyp_E195', 'Flugzeugtyp_E290', 'Flugzeugtyp_E295', 'Flugzeugtyp_E35L', 'Flugzeugtyp_E50P', 'Flugzeugtyp_E545', 'Flugzeugtyp_E550', 'Flugzeugtyp_E55P', 'Flugzeugtyp_E75L', 'Flugzeugtyp_E75S', 'Flugzeugtyp_EC35', 'Flugzeugtyp_F100', 'Flugzeugtyp_F2TH', 'Flugzeugtyp_F900', 'Flugzeugtyp_FA7X', 'Flugzeugtyp_FA8X', 'Flugzeugtyp_G280', 'Flugzeugtyp_GA5C', 'Flugzeugtyp_GA6C', 'Flugzeugtyp_GALX', 'Flugzeugtyp_GL5T', 'Flugzeugtyp_GL7T', 'Flugzeugtyp_GLEX', 'Flugzeugtyp_GLF3', 'Flugzeugtyp_GLF4', 'Flugzeugtyp_GLF5', 'Flugzeugtyp_GLF6', 'Flugzeugtyp_H25B', 'Flugzeugtyp_H25C', 'Flugzeugtyp_H47', 'Flugzeugtyp_H60', 'Flugzeugtyp_HA4T', 'Flugzeugtyp_HDJT', 'Flugzeugtyp_IL76', 'Flugzeugtyp_J328', 'Flugzeugtyp_L410', 'Flugzeugtyp_LJ35', 'Flugzeugtyp_LJ45', 'Flugzeugtyp_LJ55', 'Flugzeugtyp_LJ60', 'Flugzeugtyp_LJ75', 'Flugzeugtyp_M600', 'Flugzeugtyp_MD82', 'Flugzeugtyp_MU2', 'Flugzeugtyp_P180', 'Flugzeugtyp_PAY3', 'Flugzeugtyp_PC12', 'Flugzeugtyp_PC24', 'Flugzeugtyp_PRM1', 'Flugzeugtyp_SB20', 'Flugzeugtyp_SF34', 'Flugzeugtyp_SW4', 'Flugzeugtyp_TBM7', 'Flugzeugtyp_TBM8', 'Flugzeugtyp_TBM9', 'Lärmstation_Achering', 'Lärmstation_Asenkofen', 'Lärmstation_Attaching', 'Lärmstation_Brandstadel', 'Lärmstation_Eitting', 'Lärmstation_Fahrenzhausen', 'Lärmstation_Glaslern', 'Lärmstation_Gremertshausen', 'Lärmstation_Hallbergmoos', 'Lärmstation_Marzling', 'Lärmstation_Massenhausen', 'Lärmstation_Mintraching', 'Lärmstation_Neufahrn', 'Lärmstation_Pallhausen', 'Lärmstation_Pulling', 'Lärmstation_Reisen', 'Lärmstation_Rudelzhofen', 'Lärmstation_Schwaig', 'Lärmstation_Unterschleißheim', 'Lärmstation_Viehlaßmoos', 'Lärmstation_Zieglberg', 'An-/Abflug_A', 'An-/Abflug_D', 'Jahreszeit_Frühling', 'Jahreszeit_Herbst', 'Jahreszeit_Sommer', 'Jahreszeit_Winter']]
#ohne Wetter
#feature_columns2 = [col for col in train_daten.columns if col not in ['ID', 'Sekunde_im_Ereignis', 'Lärm', 'Datum','Uhrzeit','Temperatur', 'Windrichtung', 'Windgeschwindigkeit', 'Windgeschwindigkeit_vertikal']]
#mit allem
feature_columns3 = [col for col in train_daten.columns if col not in ['ID', 'Sekunde_im_Ereignis', 'Lärm', 'Datum','Uhrzeit']]

feature_sets = [feature_columns3]

predictions_LSTM = []
original = []
predictions_MLP = []

for e in feature_sets:
    num_features = len(e)

    X = train_daten[e].values
    y = train_daten['Lärm'].values

    num_samples = len(train_daten) //40  # Länge der Sequenzen

    X_train = np.reshape(X, (num_samples,40, num_features))
    y_train = np.reshape(y, (num_samples,40))
    X_train = X_train.astype('float32')

    X_val = val_daten[e].values
    y_val = val_daten['Lärm'].values
    num_samples_val = len(val_daten) //40
    X_val = np.reshape(X_val, (num_samples_val, 40, num_features))
    y_val = np.reshape(y_val, (num_samples_val, 40))
    X_val = X_val.astype('float32')

    X_test = test_daten[e].values
    y_test = test_daten['Lärm'].values
    num_samples_test = len(test_daten) //40
    X_test = np.reshape(X_test, (num_samples_test, 40, num_features))
    y_test = np.reshape(y_test, (num_samples_test, 40))
    X_test = X_test.astype('float32')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # LSTM 
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(256, activation='relu', input_shape=(40, num_features)))  # Aktivierungsfunktion und input_shape angepasst
    model_LSTM.add(Dropout(0.3))  # Gleich wie im MLP
    model_LSTM.add(Dense(40, activation='linear'))  # Gleich wie im MLP 
    model_LSTM.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    history_LSTM = model_LSTM.fit(X_train, y_train, epochs=10, batch_size=512, validation_data=(X_val, y_val), callbacks=[early_stop])

    predictions_L = model_LSTM.predict(X_test)
    predictions_LSTM.append(predictions_L)
    
    # MLP 
    model_MLP = Sequential()
    model_MLP.add(Dense(256, activation='sigmoid', input_shape=(40, num_features)))
    model_MLP.add(Dropout(0.3))
    model_MLP.add(Dense(1, activation='tanh'))

    model_MLP.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    history_MLP = model_MLP.fit(X_train, y_train, epochs=10, batch_size=512, validation_data=(X_val, y_val), callbacks=[early_stop])
    
    predictions_M = model_MLP.predict(X_test)
    predictions_MLP.append(predictions_M)
    
    original.append(y_test)
    


# In[35]:


# Abbildung 
plt.figure(figsize=(10, 6))

# Für LSTM
plt.plot(history_LSTM.history['loss'], label='LSTM Trainingsverlust', color='#2E4E60')
plt.plot(history_LSTM.history['val_loss'], label='LSTM Validierungsverlust', color='#2E4E60', linestyle='--')

# Für MLP
plt.plot(history_MLP.history['loss'], label='MLP Trainingsverlust', color='lightblue')
plt.plot(history_MLP.history['val_loss'], label='MLP Validierungsverlust', color='lightblue', linestyle='--')

plt.xlabel('Epoche')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Trainings- und Validierungsverluste über zehn Epochen')
plt.legend()
plt.savefig('/Users/michaelmerkel/Desktop/Alles/val_loss_MLP_LSTM_neu_neu.png', dpi = 300)
plt.show()


# In[34]:


# Abbildung

for x in range(1):
    y_test = original[x]
    predictions = predictions_MLP[x]
    plt.figure(figsize=(10, 6))
    x_values = np.arange(0, 40)
    N = 10  # Number of data points
    colors = plt.cm.viridis(np.linspace(0, 1, N))  # Generate N colors, switched to 'viridis'

    for i in range(10):
        test = y_test[i]
        pred = predictions[i]
        plt.plot(x_values, test, label=f'Realverlauf {i+1}', color=colors[i])
        plt.plot(x_values, pred, label=f'Vorhersage {i+1}', color=colors[i], linestyle='--')

    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel (normalisiert)')
    plt.title(f'Testdatensatz vs. Vorhersage MLP')
    plt.legend(loc='upper right', fontsize= 6, ncol=2)
    plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/MLP_Set_neu_neu.png', dpi=300)
    plt.show()


# In[99]:


#Abbildung

for x in range(1):
    y_test = original[x]
    predictions = predictions_LSTM[x]
    plt.figure(figsize=(10, 6))
    x_values = np.arange(0, 40)
    N = 10  # Number of data points
    colors = plt.cm.viridis(np.linspace(0, 1, N))  # Generate N colors, switched to 'viridis'

    for i in range(10):
        test = y_test[i]
        pred = predictions[i]
        plt.plot(x_values, test, label=f'Realverlauf {i+1}', color=colors[i])
        plt.plot(x_values, pred, label=f'Vorhersage {i+1}', color=colors[i], linestyle='--')

    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel (normalisiert)')
    plt.title(f'Testdatensatz vs. Vorhersage LSTM')
    plt.legend(loc='upper right', fontsize= 6, ncol=2)
    plt.savefig(f'/Users/michaelmerkel/Desktop/Alles/LSTM_Set_neu.png', dpi=300)
    plt.show()


# In[103]:


from sklearn.model_selection import KFold

# k fache Kreuzvalidierung (5 fach) vor Modelltraining mit gesamten Trainingsdatensatz

# Spalten selektieren
#ohne Wetter
feature_columns2 = [col for col in train_daten.columns if col not in ['ID', 'Sekunde_im_Ereignis', 'Lärm', 'Datum','Uhrzeit','Temperatur', 'Windrichtung', 'Windgeschwindigkeit', 'Windgeschwindigkeit_vertikal']]
#mit allem
feature_columns3 = [col for col in train_daten.columns if col not in ['ID', 'Sekunde_im_Ereignis', 'Lärm', 'Datum','Uhrzeit']]
#best 20 ohne Wetter
feature_columns4 = ['Geschwindigkeit', 'Höhe', 'Breitengrad', 'Längengrad', 'Flugzeugtyp_A346', 'Flugzeugtyp_B738', 'Flugzeugtyp_B744', 'Lärmstation_Achering', 'Lärmstation_Brandstadel', 'Lärmstation_Eitting', 'Lärmstation_Glaslern', 'Lärmstation_Marzling', 'Lärmstation_Pulling', 'Lärmstation_Schwaig', 'An-/Abflug_A', 'An-/Abflug_D', 'Jahreszeit_Frühling', 'Jahreszeit_Herbst', 'Jahreszeit_Sommer', 'Jahreszeit_Winter']
#best 20 mit Wetter
feature_columns5 = ['Temperatur', 'Windrichtung', 'Windgeschwindigkeit', 'Windgeschwindigkeit_vertikal', 'Geschwindigkeit', 'Höhe', 'Breitengrad', 'Längengrad', 'Flugzeugtyp_B744', 'Lärmstation_Achering', 'Lärmstation_Brandstadel', 'Lärmstation_Glaslern', 'Lärmstation_Marzling', 'Lärmstation_Pulling', 'Lärmstation_Schwaig', 'An-/Abflug_A', 'An-/Abflug_D', 'Jahreszeit_Frühling', 'Jahreszeit_Herbst', 'Jahreszeit_Winter']

val_loss_list = []

feature_sets = [feature_columns2, feature_columns3, feature_columns4, feature_columns5]

for idx, e in enumerate(feature_sets):
    val_loss_per_fold = [] 
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    num_features = len(e)
    X = train_daten[e].values
    y = train_daten['Lärm'].values
    num_samples = len(train_daten) // 40
    X = np.reshape(X, (num_samples, 40, num_features))
    y = np.reshape(y, (num_samples, 40))

    for fold_num, (train, val) in enumerate(kfold.split(X)):
        X_train, X_val = X[train], X[val]
        y_train, y_val = y[train], y[val]

        # Modell Definition
        model_LSTM = Sequential() 

        model_LSTM.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(40, X_train.shape[2])))
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
        history = model_LSTM.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_val, y_val))
        
        val_loss = history.history['val_loss'][-1]
        val_loss_per_fold.append(val_loss)
        val_loss_list.append({'Feature_Set': idx+1, 'Fold': fold_num+1, 'Validierungsverlust': val_loss})


    avg_val_loss = np.mean(val_loss_per_fold)
    std_val_loss = np.std(val_loss_per_fold)

    print(f'Durchschnittliche Validierungsverlust für Feature-Set {idx+1} über {n_splits}-Folds: {avg_val_loss}')
    print(f'Standardabweichung des Validierungsverlusts für Feature-Set {idx+1} über {n_splits}-Folds: {std_val_loss}')

val_loss_df = pd.DataFrame(val_loss_list)

print("DataFrame der gespeicherten Validierungsverluste:")
print(val_loss_df)


# In[105]:


# Übersicht der Ergebnisse
reshaped_df = val_loss_df.pivot(index='Fold', columns='Feature_Set', values='Validierungsverlust')

reshaped_df.columns = [f'Feature_Set_{col}' for col in reshaped_df.columns]

reshaped_df


# In[110]:


# Boxplot

feature_sets = val_loss_df['Feature_Set'].unique()
data = [val_loss_df[val_loss_df['Feature_Set'] == f]['Validierungsverlust'].tolist() for f in feature_sets]
labels = [str(f) for f in feature_sets]

bp = plt.boxplot(data, labels=labels, patch_artist=True)

plt.title("Boxplot LSTM für verschiedene Feature Sets")
plt.ylabel("Validierungsverlust")
plt.xlabel("Feature Set")

for median in bp['medians']:
    median.set(color='lightblue', linewidth=2)
for box in bp['boxes']:
    box.set_facecolor('#2E4E60')

#plt.savefig('/Users/michaelmerkel/Desktop/Alles/Boxplot_LSTM_4Features.png', dpi = 300)
plt.show()



# In[100]:


# Best 20 Features herausfinden

from sklearn.feature_selection import SelectKBest, f_classif 
import numpy as np

# ohne Wetter
feature_columns2 = [col for col in train_daten.columns if col not in ['ID', 'Sekunde_im_Ereignis', 'Lärm', 'Datum', 'Uhrzeit', 'Temperatur', 'Windrichtung', 'Windgeschwindigkeit', 'Windgeschwindigkeit_vertikal']]
# mit allem
feature_columns3 = [col for col in train_daten.columns if col not in ['ID', 'Sekunde_im_Ereignis', 'Lärm', 'Datum', 'Uhrzeit']]

feature_sets = [feature_columns2, feature_columns3]

for idx, feature_set in enumerate(feature_sets):
    num_features = len(feature_set)

    X = train_daten[feature_set].values
    y = train_daten['Lärm'].values

    num_samples = len(train_daten) // 40  

    X_train = np.reshape(X, (num_samples, 40, num_features))
    y_train = np.reshape(y, (num_samples, 40))
    X_train = X_train.astype('float32')

    # Für Validierungsdaten
    X_val = val_daten[feature_set].values
    y_val = val_daten['Lärm'].values
    num_samples_val = len(val_daten) // 40
    X_val = np.reshape(X_val, (num_samples_val, 40, num_features))
    y_val = np.reshape(y_val, (num_samples_val, 40))
    X_val = X_val.astype('float32')

    # Für Testdaten
    X_test = test_daten[feature_set].values
    y_test = test_daten['Lärm'].values
    num_samples_test = len(test_daten) // 40
    X_test = np.reshape(X_test, (num_samples_test, 40, num_features))
    y_test = np.reshape(y_test, (num_samples_test, 40))
    X_test = X_test.astype('float32')

    # die besten 20 Features
    selector = SelectKBest(score_func=f_classif, k=20)
    X_new = selector.fit_transform(X, y)
    mask = selector.get_support()
    new_features = []

    for is_selected, feature in zip(mask, feature_set):
        if is_selected:
            new_features.append(feature)

    print(f"Die besten Features für den Satz {idx + 1} sind:", new_features)


# In[6]:


# Spalten selektieren
#ohne alles 
#feature_columns1 = [col for col in train_daten.columns if col not in ['ID', 'Datum', 'Uhrzeit', 'Lärm', 'Temperatur', 'Windrichtung', 'Windgeschwindigkeit', 'Windgeschwindigkeit_vertikal', 'Geschwindigkeit', 'Höhe', 'Breitengrad', 'Längengrad', 'Sekunde_im_Ereignis', 'Flugzeugtyp_A139', 'Flugzeugtyp_A19N', 'Flugzeugtyp_A20N', 'Flugzeugtyp_A21N', 'Flugzeugtyp_A306', 'Flugzeugtyp_A310', 'Flugzeugtyp_A318', 'Flugzeugtyp_A319', 'Flugzeugtyp_A320', 'Flugzeugtyp_A321', 'Flugzeugtyp_A332', 'Flugzeugtyp_A333', 'Flugzeugtyp_A339', 'Flugzeugtyp_A342', 'Flugzeugtyp_A343', 'Flugzeugtyp_A345', 'Flugzeugtyp_A346', 'Flugzeugtyp_A359', 'Flugzeugtyp_A35K', 'Flugzeugtyp_A388', 'Flugzeugtyp_ASTR', 'Flugzeugtyp_AT75', 'Flugzeugtyp_AT76', 'Flugzeugtyp_B190', 'Flugzeugtyp_B350', 'Flugzeugtyp_B38M', 'Flugzeugtyp_B39M', 'Flugzeugtyp_B58T', 'Flugzeugtyp_B733', 'Flugzeugtyp_B734', 'Flugzeugtyp_B736', 'Flugzeugtyp_B737', 'Flugzeugtyp_B738', 'Flugzeugtyp_B739', 'Flugzeugtyp_B744', 'Flugzeugtyp_B748', 'Flugzeugtyp_B752', 'Flugzeugtyp_B753', 'Flugzeugtyp_B762', 'Flugzeugtyp_B763', 'Flugzeugtyp_B764', 'Flugzeugtyp_B772', 'Flugzeugtyp_B77L', 'Flugzeugtyp_B77W', 'Flugzeugtyp_B788', 'Flugzeugtyp_B789', 'Flugzeugtyp_B78X', 'Flugzeugtyp_BCS1', 'Flugzeugtyp_BCS3', 'Flugzeugtyp_BE20', 'Flugzeugtyp_BE40', 'Flugzeugtyp_BE58', 'Flugzeugtyp_C130', 'Flugzeugtyp_C17', 'Flugzeugtyp_C208', 'Flugzeugtyp_C25A', 'Flugzeugtyp_C25B', 'Flugzeugtyp_C25C', 'Flugzeugtyp_C25M', 'Flugzeugtyp_C421', 'Flugzeugtyp_C425', 'Flugzeugtyp_C500', 'Flugzeugtyp_C501', 'Flugzeugtyp_C510', 'Flugzeugtyp_C525', 'Flugzeugtyp_C550', 'Flugzeugtyp_C551', 'Flugzeugtyp_C55B', 'Flugzeugtyp_C560', 'Flugzeugtyp_C56X', 'Flugzeugtyp_C650', 'Flugzeugtyp_C680', 'Flugzeugtyp_C68A', 'Flugzeugtyp_C750', 'Flugzeugtyp_CL30', 'Flugzeugtyp_CL35', 'Flugzeugtyp_CL60', 'Flugzeugtyp_CRJ2', 'Flugzeugtyp_CRJ9', 'Flugzeugtyp_CRJX', 'Flugzeugtyp_D328', 'Flugzeugtyp_DA62', 'Flugzeugtyp_DH8D', 'Flugzeugtyp_E135', 'Flugzeugtyp_E145', 'Flugzeugtyp_E170', 'Flugzeugtyp_E190', 'Flugzeugtyp_E195', 'Flugzeugtyp_E290', 'Flugzeugtyp_E295', 'Flugzeugtyp_E35L', 'Flugzeugtyp_E50P', 'Flugzeugtyp_E545', 'Flugzeugtyp_E550', 'Flugzeugtyp_E55P', 'Flugzeugtyp_E75L', 'Flugzeugtyp_E75S', 'Flugzeugtyp_EC35', 'Flugzeugtyp_F100', 'Flugzeugtyp_F2TH', 'Flugzeugtyp_F900', 'Flugzeugtyp_FA7X', 'Flugzeugtyp_FA8X', 'Flugzeugtyp_G280', 'Flugzeugtyp_GA5C', 'Flugzeugtyp_GA6C', 'Flugzeugtyp_GALX', 'Flugzeugtyp_GL5T', 'Flugzeugtyp_GL7T', 'Flugzeugtyp_GLEX', 'Flugzeugtyp_GLF3', 'Flugzeugtyp_GLF4', 'Flugzeugtyp_GLF5', 'Flugzeugtyp_GLF6', 'Flugzeugtyp_H25B', 'Flugzeugtyp_H25C', 'Flugzeugtyp_H47', 'Flugzeugtyp_H60', 'Flugzeugtyp_HA4T', 'Flugzeugtyp_HDJT', 'Flugzeugtyp_IL76', 'Flugzeugtyp_J328', 'Flugzeugtyp_L410', 'Flugzeugtyp_LJ35', 'Flugzeugtyp_LJ45', 'Flugzeugtyp_LJ55', 'Flugzeugtyp_LJ60', 'Flugzeugtyp_LJ75', 'Flugzeugtyp_M600', 'Flugzeugtyp_MD82', 'Flugzeugtyp_MU2', 'Flugzeugtyp_P180', 'Flugzeugtyp_PAY3', 'Flugzeugtyp_PC12', 'Flugzeugtyp_PC24', 'Flugzeugtyp_PRM1', 'Flugzeugtyp_SB20', 'Flugzeugtyp_SF34', 'Flugzeugtyp_SW4', 'Flugzeugtyp_TBM7', 'Flugzeugtyp_TBM8', 'Flugzeugtyp_TBM9', 'Lärmstation_Achering', 'Lärmstation_Asenkofen', 'Lärmstation_Attaching', 'Lärmstation_Brandstadel', 'Lärmstation_Eitting', 'Lärmstation_Fahrenzhausen', 'Lärmstation_Glaslern', 'Lärmstation_Gremertshausen', 'Lärmstation_Hallbergmoos', 'Lärmstation_Marzling', 'Lärmstation_Massenhausen', 'Lärmstation_Mintraching', 'Lärmstation_Neufahrn', 'Lärmstation_Pallhausen', 'Lärmstation_Pulling', 'Lärmstation_Reisen', 'Lärmstation_Rudelzhofen', 'Lärmstation_Schwaig', 'Lärmstation_Unterschleißheim', 'Lärmstation_Viehlaßmoos', 'Lärmstation_Zieglberg', 'An-/Abflug_A', 'An-/Abflug_D', 'Jahreszeit_Frühling', 'Jahreszeit_Herbst', 'Jahreszeit_Sommer', 'Jahreszeit_Winter']]
#ohne Wetter
feature_columns2 = [col for col in train_daten.columns if col not in ['ID', 'Sekunde_im_Ereignis', 'Lärm', 'Datum','Uhrzeit','Temperatur', 'Windrichtung', 'Windgeschwindigkeit', 'Windgeschwindigkeit_vertikal']]
#mit allem
feature_columns3 = [col for col in train_daten.columns if col not in ['ID', 'Sekunde_im_Ereignis', 'Lärm', 'Datum','Uhrzeit']]
#best 20 ohne Wetter
#feature_columns4 = ['Geschwindigkeit', 'Höhe', 'Breitengrad', 'Längengrad', 'Flugzeugtyp_A346', 'Flugzeugtyp_B738', 'Flugzeugtyp_B744', 'Lärmstation_Achering', 'Lärmstation_Brandstadel', 'Lärmstation_Eitting', 'Lärmstation_Glaslern', 'Lärmstation_Marzling', 'Lärmstation_Pulling', 'Lärmstation_Schwaig', 'An-/Abflug_A', 'An-/Abflug_D', 'Jahreszeit_Frühling', 'Jahreszeit_Herbst', 'Jahreszeit_Sommer', 'Jahreszeit_Winter']
#best 20 mit Wetter
#feature_columns5 = ['Temperatur', 'Windrichtung', 'Windgeschwindigkeit', 'Windgeschwindigkeit_vertikal', 'Geschwindigkeit', 'Höhe', 'Breitengrad', 'Längengrad', 'Flugzeugtyp_B744', 'Lärmstation_Achering', 'Lärmstation_Brandstadel', 'Lärmstation_Glaslern', 'Lärmstation_Marzling', 'Lärmstation_Pulling', 'Lärmstation_Schwaig', 'An-/Abflug_A', 'An-/Abflug_D', 'Jahreszeit_Frühling', 'Jahreszeit_Herbst', 'Jahreszeit_Winter']

feature_sets = [feature_columns2,feature_columns3]

predictions_LSTM = {}
original = {}
history_dict = {}

for idx, e in enumerate(feature_sets):
    num_features = len(e)

    X = train_daten[e].values
    y = train_daten['Lärm'].values

    num_samples = len(train_daten) //40 

    X_train = np.reshape(X, (num_samples,40, num_features))
    y_train = np.reshape(y, (num_samples,40))
    X_train = X_train.astype('float32')

    # Für Validierungsdaten
    X_val = val_daten[e].values
    y_val = val_daten['Lärm'].values
    num_samples_val = len(val_daten) //40
    X_val = np.reshape(X_val, (num_samples_val,40, num_features))
    y_val = np.reshape(y_val, (num_samples_val,40))
    X_val = X_val.astype('float32')

    # Für Testdaten
    X_test = test_daten[e].values
    y_test = test_daten['Lärm'].values
    num_samples_test = len(test_daten) //40
    X_test = np.reshape(X_test, (num_samples_test,40, num_features))
    y_test = np.reshape(y_test, (num_samples_test,40))
    X_test = X_test.astype('float32')
    
    # Callbacks ModelCheckpoint und EarlyStopping
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)
   
    # LSTM 
    model_LSTM = Sequential() 
    model_LSTM.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(40, X_train.shape[2])))
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
    history = model_LSTM.fit(X_train, y_train, epochs=200, batch_size=256, validation_data=(X_val, y_val), callbacks=[model_checkpoint, early_stopping])
    
    feature_set_name = f'Feature Set {idx + 1}'
    predictions_LSTM[feature_set_name] = model_LSTM.predict(X_test)
    original[feature_set_name] = y_test


# In[16]:


# Abbildung
for feature_set_name in predictions_LSTM.keys():
    y_test = original[feature_set_name]
    predictions = predictions_LSTM[feature_set_name]

    plt.figure(figsize=(10, 6))
    x_values = np.arange(0, 40)
    N = 10  
    colors = plt.cm.viridis(np.linspace(0, 1, N))

    for i in range(10):
        test = y_test[i]
        pred = predictions[i]
        plt.plot(x_values, test, label=f'Testwert {i+1}', color=colors[i])
        plt.plot(x_values, pred, label=f'Vorhersage {i+1}', color=colors[i], linestyle='--')

    plt.xlabel('Sekunde im Lärmereignis')
    plt.ylabel('Lärmpegel (normalisiert)')
    plt.title(f'Testdatensatz vs. Vorhersage {feature_set_name}')
    plt.legend(loc='upper right', fontsize= 6, ncol=2)
    plt.savefig(f'/Users/michaelmerkel/Desktop/LSTM_bidirectional_MSE_200Ep_{feature_set_name}.png', dpi = 300)
    plt.show()


# In[15]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

handles, labels = plt.gca().get_legend_handles_labels()

for idx, (feature_set_name, history) in enumerate(history_dict.items()):
    train_loss = history.history['loss'][1:]
    val_loss = history.history['val_loss'][1:]
    color = '#2E4E60' if idx == 0 else 'lightblue'
    train_label = f'{feature_set_name} - Trainingsverlust'
    if train_label not in labels:
        plt.plot(train_loss, label=train_label, color=color)
    val_label = f'{feature_set_name} - Validierungsverlust'
    if val_label not in labels:
        plt.plot(val_loss, label=val_label, linestyle='--', color=color)

    handles, labels = plt.gca().get_legend_handles_labels()

plt.title('Trainings- und Validierungsverluste nach 200 Epochen (mit Early-Stopping)')
plt.xlabel('Epochen')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.savefig('/Users/michaelmerkel/Desktop/Alles/LSTM_2_Features_Train_Val_200Epab2ep.png',dpi=300)
plt.show()


# In[9]:


# Rücktransformation der Lärmdaten (Test) und der vorhergesagten Lärmdaten

col_index = cols_to_normalize.index('Lärm')

min_val = scaler.data_min_[col_index]
max_val = scaler.data_max_[col_index]

y_test_transformed = {}
y_pred_transformed = {}

for feature_set_name in predictions_LSTM.keys():
    y_test = original[feature_set_name]
    predictions = predictions_LSTM[feature_set_name]
    y_test_original_scale = y_test * (max_val - min_val) + min_val
    y_pred_original_scale = predictions * (max_val - min_val) + min_val
    y_test_transformed[feature_set_name] = y_test_original_scale
    y_pred_transformed[feature_set_name] = y_pred_original_scale



# In[29]:


# Ein Dictionary für die Fehlermetriken
error_metrics = {'MAE': [], 'MSE': [], 'RMSE': []}

for feature_set_name in y_test_transformed.keys():
    y_test_original_scale = y_test_transformed[feature_set_name]
    y_pred_original_scale = y_pred_transformed[feature_set_name]

    # Fehlermetriken für jede Vorhersage
    mse_value = mean_squared_error(y_test_original_scale, y_pred_original_scale)
    rmse_value = np.sqrt(mse_value)
    mae_value = mean_absolute_error(y_test_original_scale, y_pred_original_scale)
    error_metrics['MAE'].append(mae_value)   
    error_metrics['MSE'].append(mse_value)
    error_metrics['RMSE'].append(rmse_value)

    print(f"Feature Set: {feature_set_name}")
    print(f"MAE: {mae_value}")
    print(f"MSE: {mse_value}")
    print(f"RMSE: {rmse_value}")
    print("---")
    
df_error_metrics = pd.DataFrame(error_metrics, index=list(y_test_transformed.keys()))
df_error_metrics = df_error_metrics.transpose()
df_error_metrics.rename(columns={
    'Feature Set 1': 'Ohne Wetterinformationen',
    'Feature Set 2': 'Mit Wetterinformationen'
}, inplace=True)

df_error_metric = df_error_metrics.round(2)
df_error_metric


# In[ ]:





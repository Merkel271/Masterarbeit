#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


#Laden der Dateien
datei1 = "/Users/michaelmerkel/Desktop/Alles/Datensatz_neu.csv"
datei2 = "/Users/michaelmerkel/Desktop/Alles/Datensatz_neu1.csv"
datei3 = "/Users/michaelmerkel/Desktop/Alles/Datensatz_neu2.csv"
datei4 = "/Users/michaelmerkel/Desktop/Alles/Datensatz_neu3.csv"

datei1_original = pd.read_csv(datei1)
datei2_original = pd.read_csv(datei2)
datei3_original = pd.read_csv(datei3)
datei4_original = pd.read_csv(datei4)

datei1 = datei1_original.copy()
datei2 = datei2_original.copy()
datei3 = datei3_original.copy()
datei4 = datei4_original.copy()


# In[4]:


#datei1


# In[9]:


# Dateien zu einem gesamten Datensatz zusammenfügen
id_mapping = {}
new_id_counter = 0

def assign_new_id(row):
    global new_id_counter
    old_id = row['ID']
    if old_id not in id_mapping:
        id_mapping[old_id] = new_id_counter
        new_id_counter += 1
    return id_mapping[old_id]

last_id_datei1 = int(datei1_original.iloc[-1]['ID'])

offset_datei2 = last_id_datei1
offset_datei3 = last_id_datei1 + int(datei2_original.iloc[-1]['ID']) + 1
offset_datei4 = offset_datei3 + int(datei3.iloc[-1]['ID']) + 1

datei2['ID'] += offset_datei2
datei3['ID'] += offset_datei3
datei4['ID'] += offset_datei4

ergebnisse = pd.concat([datei1, datei2, datei3, datei4], ignore_index=True)

# Liste der Spalten, anhand derer Duplikate erkannt werden sollen
columns_to_consider = [col for col in ergebnisse.columns if col != 'ID']

# Identische Zeilen finden
duplicates = ergebnisse[ergebnisse.duplicated(subset=columns_to_consider, keep=False)]

# Duplikate entfernen und DataFrame aktualisieren
ergebnisse = ergebnisse.drop_duplicates(subset=columns_to_consider, keep='first').reset_index(drop=True)

# Neue IDs
ergebnisse['New_ID'] = ergebnisse.apply(assign_new_id, axis=1)
ergebnisse.drop('ID', axis=1, inplace=True)
ergebnisse.rename(columns={'New_ID': 'ID'}, inplace=True)

ergebnisse_reorder = ergebnisse[["ID", "Datum", "Uhrzeit", "Flugzeugtyp", "Lärmstation", "Jahreszeit", "Lärm", "Temperatur", "Windrichtung", "Windgeschwindigkeit", "Windgeschwindigkeit_vertikal", "Geschwindigkeit", "Höhe", "Breitengrad", "Längengrad"]]
datensatz = ergebnisse_reorder


# In[11]:


#60601 Flugspuren herausgefiltert
datensatz


# In[14]:


#Hinzufügen der Spalte "An-oder Abflug". Wenn über 80% der Differenzen positiv oder 
#negativ sind, so wird der Flug einem An- oder Abflug zugeordnet.
#Ansonsten ist die Information unbekannt "U".

def calculate_trend(group):
    heights = group['Höhe'].values
    diffs = np.diff(heights)
    
    positive_diffs = np.sum(diffs > 0)
    negative_diffs = np.sum(diffs < 0)
    
    total_diffs = len(diffs)
    
    if total_diffs == 0:  
        return 'U'
    
    positive_ratio = positive_diffs / total_diffs
    negative_ratio = negative_diffs / total_diffs
    
    if positive_ratio >= 0.8:
        return 'D'  # Zunehmend
    elif negative_ratio >= 0.8:
        return 'A'  # Abnehmend
    else:
        return 'U'  # Unbestimmt / gemischt

grouped = datensatz.groupby('ID')
trends = grouped.apply(calculate_trend)

datensatz['An-/Abflug'] = 'U'

for id_, trend in trends.items():
    datensatz.loc[datensatz['ID'] == id_, 'An-/Abflug'] = trend

datensatz


# In[17]:


count_u = datensatz['An-/Abflug'].value_counts().get('U', 0)
print(f"Anzahl Zeilen mit U: {count_u}")


# In[22]:


filtered_datensatz = datensatz.groupby('ID').filter(lambda x: x['An-/Abflug'].iloc[0] != 'U')
filtered_datensatz


# In[23]:


# Nach Höhe filtern (zw. 1.500 und 6.000 ft)
def check_height_range(group):
    heights = group['Höhe']
    return all(1500 <= h <= 6000 for h in heights)

filtered_datensatz2 = filtered_datensatz.groupby('ID').filter(check_height_range)
filtered_datensatz2


# In[24]:


# IDs neu zählen, um Anzahl der Flugspuren einzusehen
new_id_counter = 0
id_mapping = {}

def assign_new_id(group):
    global new_id_counter
    old_id = group['ID'].iloc[0]
    if old_id not in id_mapping:
        id_mapping[old_id] = new_id_counter
        new_id_counter += 1
    group['ID'] = id_mapping[old_id]
    return group

filtered_datensatz3 = filtered_datensatz2.groupby('ID', group_keys=False).apply(assign_new_id)

filtered_datensatz3


# In[25]:


# Kontrolle
unique_id_count = filtered_datensatz2['ID'].nunique()
print(f"Anzahl der einzigartigen IDs: {unique_id_count}")


# In[26]:


filtered_datensatz3.to_csv("/Users/michaelmerkel/Desktop/Alles/Zusammengefuegt_AD_6000.csv", index=False)


# In[3]:


#Laden der Datei
datensatz = "/Users/michaelmerkel/Desktop/Alles/Zusammengefuegt_AD_6000.csv"
datensatz_original = pd.read_csv(datensatz)
datensatz = datensatz_original.copy()


# In[4]:


#datensatz


# In[5]:


# Umrechnungen

# Lärm durch 10 teilen
datensatz['Lärm'] = datensatz['Lärm'] / 10

# Windrichtung in meteorologische Windrichtung
datensatz['Windrichtung'] = (270 - datensatz['Windrichtung']) % 360

# Windgeschwindigkeit und Windgeschwindigkeit_vertikal von m/s in km/h umrechnen
datensatz[['Windgeschwindigkeit', 'Windgeschwindigkeit_vertikal']] *= 3.6

# Geschwindigkeit von Knoten in km/h (1 Knoten = 1.852 km/h)
datensatz['Geschwindigkeit'] *= 1.852

# Höhe von Fuß in Meter (1 Fuß = 0.3048 Meter)
datensatz['Höhe'] *= 0.3048


# In[7]:


# Sekunde im Ereignis, um Lärmereignislängen einzusehen
datensatz['Sekunde_im_Ereignis'] = 0
for name, group in datensatz.groupby('ID'):
    datensatz.loc[group.index, 'Sekunde_im_Ereignis'] = range(1, len(group) + 1)


# In[8]:


#datensatz


# In[9]:


datensatz.to_csv("/Users/michaelmerkel/Desktop/Alles/Zusammengefuegt_AD_6000_angepasst.csv", index=False)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[22]:


#Laden der Datei
datensatz = "/Users/michaelmerkel/Desktop/Alles/Padded.csv"
datensatz_original = pd.read_csv(datensatz)
datensatz = datensatz_original.copy()


# In[23]:


datensatz


# In[24]:


# Bedingungen
condition_1 = datensatz[
    (datensatz['An-/Abflug_A'] == 1) & 
    (datensatz['Temperatur'] > 15) &
    (datensatz['Windgeschwindigkeit'] < 20) & 
    (datensatz['Flugzeugtyp_A319'] == 1)
    ]
condition_2 = datensatz[
    (datensatz['An-/Abflug_A'] == 1) & 
    (datensatz['Temperatur'] < 0) & 
    (datensatz['Windgeschwindigkeit'] < 20) &
    (datensatz['Flugzeugtyp_A319'] == 1)
    ]
condition_3 = datensatz[
    (datensatz['An-/Abflug_A'] == 1) & 
    (datensatz['Temperatur'] < 10) & 
    (datensatz['Windgeschwindigkeit'] > 50) & 
    (datensatz['Flugzeugtyp_A319'] == 1)
    ]
condition_4 = datensatz[
    (datensatz['An-/Abflug_A'] == 1) & 
    (datensatz['Temperatur'] < 10) &
    (datensatz['Windgeschwindigkeit'] < 5) & 
    (datensatz['Flugzeugtyp_A319'] == 1)
    ]
condition_5 = datensatz[
    (datensatz['An-/Abflug_D'] == 1) & 
    (datensatz['Temperatur'] > 15) & 
    (datensatz['Windgeschwindigkeit'] < 20) & 
    (datensatz['Flugzeugtyp_A319'] == 1)
    ]
condition_6 = datensatz[
    (datensatz['An-/Abflug_D'] == 1) & 
    (datensatz['Temperatur'] < 0) & 
    (datensatz['Windgeschwindigkeit'] < 20) & 
    (datensatz['Flugzeugtyp_A319'] == 1)
    ]
condition_7 = datensatz[
    (datensatz['An-/Abflug_D'] == 1) & 
    (datensatz['Temperatur'] < 10) & 
    (datensatz['Windgeschwindigkeit'] > 50) & 
    (datensatz['Flugzeugtyp_A319'] == 1)
    ]
condition_8 = datensatz[
    (datensatz['An-/Abflug_D'] == 1) &  
    (datensatz['Temperatur'] < 10) & 
    (datensatz['Windgeschwindigkeit'] < 5) & 
    (datensatz['Flugzeugtyp_A319'] == 1)
    ]



# In[25]:


# Anzahl Bedingungen
all_conditions = [condition_1, condition_2, condition_3, condition_4, condition_5, condition_6, condition_7, condition_8]

for i, condition in enumerate(all_conditions):
    unique_ids = condition['ID'].drop_duplicates()
    print(f"Für condition_{i+1} gibt es {len(unique_ids)} Sequenzen.")


# In[26]:


# 5 Flugspuren extrahieren
extracted_dfs = []
processed_ids = []

for i, condition in enumerate(all_conditions):
    unique_ids = condition['ID'].drop_duplicates()
    selected_ids = unique_ids[~unique_ids.isin(processed_ids)].sample(min(5, len(unique_ids)))
    processed_ids.extend(selected_ids)
    extracted_df = datensatz[datensatz['ID'].isin(selected_ids)]
    datensatz = datensatz[~datensatz['ID'].isin(selected_ids)]
    extracted_dfs.append(extracted_df)
    
    print(f"Für condition_{i+1} wurden {len(selected_ids)} Flugspuren extrahiert.")



# In[27]:


for i, df in enumerate(extracted_dfs):
    print(f"DataFrame für condition_{i+1}:")
    print(df)


# In[31]:


for i, df in enumerate(extracted_dfs):
    filename = f"/Users/michaelmerkel/Desktop/Alles/Szenarien/8_Szenarien_neu/condition_{i+1}_data.csv"
    df.to_csv(filename, index=False)
    print(f"DataFrame für condition_{i+1} wurde als {filename} gespeichert.")


# In[32]:


datensatz.to_csv("/Users/michaelmerkel/Desktop/Alles/Szenarien/8_Szenarien_neu/datensatz_ohne_szenarien.csv", index = False)


# In[ ]:





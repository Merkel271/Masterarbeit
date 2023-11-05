#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# In[34]:


datensatz = "/Users/michaelmerkel/Desktop/Alles/Zusammengefuegt_AD_6000_angepasst.csv"
datensatz_original = pd.read_csv(datensatz)
datensatz = datensatz_original.copy()


# In[37]:


# Durchschnitt und Median der Flugspurlängen
grouped = datensatz.groupby('ID')
lengths = grouped.size()
average_length = lengths.mean()
median_length = lengths.median()
print(median_length)
print(average_length)


# In[38]:


grouped = datensatz_10.groupby('ID').size().reset_index(name='counts')
filtered_ids = grouped[(grouped['counts'] >= 25) & (grouped['counts'] <= 40)]['ID']
datensatz_filtered2 = datensatz_10[datensatz_10['ID'].isin(filtered_ids)]
datensatz_10 = datensatz_filtered2
#datensatz_10


# In[39]:


grouped = datensatz.groupby('ID').size().reset_index(name='counts')
filtered_ids = grouped[(grouped['counts'] >= 25) & (grouped['counts'] <= 40)]['ID']
datensatz = datensatz[datensatz['ID'].isin(filtered_ids)]


# In[41]:


new_id_counter = 0

id_mapping = {}

# Funktion zum Zuweisen der neuen IDs
def assign_new_id(group):
    global new_id_counter
    old_id = group['ID'].iloc[0]
    if old_id not in id_mapping:
        id_mapping[old_id] = new_id_counter
        new_id_counter += 1
    group['ID'] = id_mapping[old_id]
    return group

datensatz = datensatz.groupby('ID', group_keys=False).apply(assign_new_id)

#Kontrolle
datensatz


# In[43]:


# One-Hot-Encoding
datensatz_one_hot = pd.get_dummies(datensatz, columns=['Flugzeugtyp', 'Lärmstation', 'An-/Abflug', 'Jahreszeit'], drop_first=False)


# In[45]:


def apply_padding(data, max_seq_length, padding_value=0):
    padded_data_list = []  
    new_id = 0 
    
    for _, group in data.groupby(['ID']):
        group['ID'] = [new_id] * len(group)
        group.sort_index(inplace=True)
        
        current_seq_length = len(group)
        if current_seq_length < max_seq_length:
            padding_length = max_seq_length - current_seq_length
            padding_data = pd.DataFrame([[padding_value] * len(data.columns)] * padding_length, columns=data.columns)
            padding_data['ID'] = [new_id] * padding_length  
            group = pd.concat([group, padding_data])
            
        padded_data_list.append(group)  
        new_id += 1 
    return pd.concat(padded_data_list)  


# Maximale Sequenzlänge
max_seq_length = 40
padded_daten = apply_padding(datensatz_one_hot, max_seq_length)


# In[46]:


#padded_daten


# In[26]:


#padded_daten10.to_csv("/Users/michaelmerkel/Desktop/Alles/Padded_10.csv")


# In[32]:


#padded_daten.to_csv("/Users/michaelmerkel/Desktop/Alles/Padded.csv", index = False)


# In[ ]:





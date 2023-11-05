#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#Laden der Dateien (acht Szenarien)
szenario_s1 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_1_A_warm.csv"
szenario_original_s1 = pd.read_csv(szenario_s1)
szenario_s1 = szenario_original_s1.copy()

szenario_s2 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_2_A_cold.csv"
szenario_original_s2 = pd.read_csv(szenario_s2)
szenario_s2 = szenario_original_s2.copy()

szenario_s3 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_3_A_wind.csv"
szenario_original_s3 = pd.read_csv(szenario_s3)
szenario_s3 = szenario_original_s3.copy()

szenario_s4 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_4_A_no_wind.csv"
szenario_original_s4 = pd.read_csv(szenario_s4)
szenario_s4 = szenario_original_s4.copy()

szenario_s5 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_5_D_warm.csv"
szenario_original_s5 = pd.read_csv(szenario_s5)
szenario_s5 = szenario_original_s5.copy()

szenario_s6 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_6_D_cold.csv"
szenario_original_s6 = pd.read_csv(szenario_s6)
szenario_s6 = szenario_original_s6.copy()

szenario_s7= "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_7_D_wind.csv"
szenario_original_s7 = pd.read_csv(szenario_s7)
szenario_s7 = szenario_original_s7.copy()

szenario_s8 = "/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_neu/condition_8_D_no_wind.csv"
szenario_original_s8 = pd.read_csv(szenario_s8)
szenario_s8 = szenario_original_s8.copy()


# In[4]:


#szenario_s8


# In[11]:


dataset_names = [f'szenario_s{i}' for i in range(1, 9)]
# Umrechnung für jeden Datensatz
for name in dataset_names:
    df = globals()[name]
    df['Geschwindigkeit'] = df['Geschwindigkeit'] / 3.6
    df.to_csv(f"/Users/michaelmerkel/Desktop/Alles/N_Szenarien/{name}_converted.csv", index=False)


# In[12]:


#Kontrolle
#szenario_s8


# In[ ]:





# In[ ]:





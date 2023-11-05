#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr


# In[2]:


#Laden der Datei
datensatz = "/Users/michaelmerkel/Desktop/Alles/Zusammengefuegt_AD_6000_angepasst.csv"
datensatz_original = pd.read_csv(datensatz)
datensatz = datensatz_original.copy()
datensatz


# In[3]:


gruppiert = datensatz.groupby(['Lärmstation', 'Flugzeugtyp', 'An-/Abflug'])
for name, group in gruppiert:
    print(name)
    print(group)


# In[4]:


filtered_groups = {}
for name, group in gruppiert:
    laermstation, flugzeugtyp, an_abflug = name
    if an_abflug != 'U':
        print(name)
        print(group)

        
        filtered_groups[name] = group
        


# In[5]:


filtered_dataframe = pd.concat(filtered_groups.values())
filtered_dataframe


# In[6]:


# Die größten Gruppen für A und D werden ausgewählt
group_sizes = {}
for name, group in gruppiert:
    group_sizes[name] = len(group)
sorted_names = sorted(group_sizes.keys(), key=lambda x: group_sizes[x], reverse=True)
for name in sorted_names:
    group = gruppiert.get_group(name)
    print(name)
    print(group)


# In[7]:


gruppe_glaslern_A20N_A = gruppiert.get_group(('Glaslern', 'A20N', 'A'))
gruppe_pulling_A319_D = gruppiert.get_group(('Pulling', 'A319', 'D'))


# In[8]:


datensatz_grouped_list1 = gruppe_glaslern_A20N_A.groupby('ID').agg({
    'Temperatur': 'mean',
    'Windrichtung': 'mean',
    'Windgeschwindigkeit': 'mean',
    'Windgeschwindigkeit_vertikal': 'mean',
    'Höhe': 'mean',
    'Lärm': 'mean'
}).reset_index()

datensatz_grouped_list1.drop('ID', axis=1, inplace=True)

#datensatz_grouped_list1


# In[9]:


datensatz_grouped_list1


# In[10]:


datensatz_grouped_list2 = gruppe_pulling_A319_D.groupby('ID').agg({
    'Temperatur': 'mean',
    'Windrichtung': 'mean',
    'Windgeschwindigkeit': 'mean',
    'Windgeschwindigkeit_vertikal': 'mean',
    'Höhe': 'mean',
    'Lärm': 'mean'
}).reset_index()

datensatz_grouped_list2.drop('ID', axis=1, inplace=True)
#datensatz_grouped_list2


# In[11]:


datensatz_grouped_list2


# In[12]:


# Pearson-Korrelationsmatrix
pearson_corr_matrix = datensatz_grouped_list1.corr(method='pearson')
print("Pearson-Korrelationsmatrix:")
print(pearson_corr_matrix)

# Spearman-Korrelationsmatrix
spearman_corr_matrix = datensatz_grouped_list1.corr(method='spearman')
print("Spearman-Korrelationsmatrix:")
print(spearman_corr_matrix)


# In[13]:


# Heatmap für Pearson-Korrelation
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(pearson_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title("Pearson-Korrelationsmatrix")
heatmap.tick_params(axis='both', labelsize=6.5, labelrotation = 0)
plt.savefig("/Users/michaelmerkel/Desktop/Alles/Pearson_Gruppe_1.png")
plt.show()


# Heatmap für Spearman-Korrelation
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title("Spearman-Korrelationsmatrix")
heatmap.tick_params(axis='both', labelsize=6.5, labelrotation = 0)
plt.savefig("/Users/michaelmerkel/Desktop/Alles/Spearman_Gruppe_1.png")
plt.show()


# In[14]:


# Pearson-Korrelationsmatrix
pearson_corr_matrix = datensatz_grouped_list2.corr(method='pearson')
print("Pearson-Korrelationsmatrix:")
print(pearson_corr_matrix)

# Spearman-Korrelationsmatrix
spearman_corr_matrix = datensatz_grouped_list2.corr(method='spearman')
print("Spearman-Korrelationsmatrix:")
print(spearman_corr_matrix)


# In[15]:


# Heatmap für Pearson-Korrelation
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(pearson_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title("Pearson-Korrelationsmatrix - Gruppierung")
heatmap.tick_params(axis='both', labelsize=6.5, labelrotation = 0)
plt.savefig("/Users/michaelmerkel/Desktop/Alles/Pearson_Gruppe_2.png")
plt.show()


# Heatmap für Spearman-Korrelation
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title("Spearman-Korrelationsmatrix - Gruppierung")
heatmap.tick_params(axis='both', labelsize=6.5, labelrotation = 0)
plt.savefig("/Users/michaelmerkel/Desktop/Spearman_Gruppe_2.png")
plt.show()


# In[18]:


# Daten
hoehe = datensatz_grouped_list2['Höhe']
laerm = datensatz_grouped_list2['Lärm']

# Scatterplot erstellen mit Trendlinie
plt.figure(figsize=(4, 3), dpi = 450)
sns.regplot(x=hoehe, y=laerm, scatter_kws={'s': 0.1}, color='midnightblue',line_kws={'color': 'red','linewidth': 1})
plt.xlabel('Höhe [m]', fontsize = 6.5)
plt.ylabel('Lärm [dB(A)]', fontsize = 6.5)
plt.title('Scatterplot von Höhe und Lärm', fontsize = 8)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5) 

plt.savefig("/Users/michaelmerkel/Desktop/Alles/HoeheLaerm_Gruppe2")
plt.show()


# Daten
temp = datensatz_grouped_list2['Temperatur']
laerm = datensatz_grouped_list2['Lärm']

# Scatterplot erstellen mit Trendlinie
plt.figure(figsize=(4, 3), dpi = 450)
sns.regplot(x=temp, y=laerm, scatter_kws={'s': 0.1}, color='midnightblue',line_kws={'color': 'red','linewidth': 1})
plt.xlabel('Temperatur [°C]', fontsize = 6.5)
plt.ylabel('Lärm [dB(A)]', fontsize = 6.5)
plt.title('Scatterplot von Temperatur und Lärm', fontsize = 8)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5)
plt.savefig("/Users/michaelmerkel/Desktop/Alles/TempLaerm_Gruppe2")
plt.show()

# Daten
wind = datensatz_grouped_list2['Windgeschwindigkeit']
laerm = datensatz_grouped_list2['Lärm']

# Scatterplot erstellen mit Trendlinie
plt.figure(figsize=(4, 3), dpi = 450)
sns.regplot(x=wind, y=laerm, scatter_kws={'s': 0.1}, color='midnightblue',line_kws={'color': 'red','linewidth': 1})
plt.xlabel('Windgeschwindigkeit [km/h]', fontsize = 6.5)
plt.ylabel('Lärm [dB(A)]', fontsize = 6.5)
plt.title('Scatterplot von Windgeschwindigkeit und Lärm', fontsize = 8)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5)
plt.savefig("/Users/michaelmerkel/Desktop/Alles/WindgeschwindigkeitLaerm_Gruppe2")
plt.show()

# Daten
wind_wz = datensatz_grouped_list2['Windgeschwindigkeit_vertikal']
laerm = datensatz_grouped_list2['Lärm']

# Scatterplot erstellen mit Trendlinie
plt.figure(figsize=(4, 3), dpi = 450)
sns.regplot(x=wind_wz, y=laerm, scatter_kws={'s': 0.1}, color='midnightblue',line_kws={'color': 'red','linewidth': 1})
plt.xlabel('vertikale Windgeschwindigkeit [km/h]', fontsize = 6.5)
plt.ylabel('Lärm [dB(A)]', fontsize = 6.5)
plt.title('Scatterplot von vertikaler Windgeschwindigkeit und Lärm', fontsize = 8)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5)
plt.savefig("/Users/michaelmerkel/Desktop/Alles/wind_wzLaerm_Gruppe2")
plt.show()

# Daten
windrichtung = datensatz_grouped_list2['Windrichtung']
laerm = datensatz_grouped_list2['Lärm']

# Scatterplot erstellen mit Trendlinie
plt.figure(figsize=(4, 3), dpi = 450)
sns.regplot(x=windrichtung, y=laerm, scatter_kws={'s': 0.1}, color='midnightblue',line_kws={'color': 'red','linewidth': 1})
plt.xlabel('Windrichtung [°]', fontsize = 6.5)
plt.ylabel('Lärm [dB(A)]', fontsize = 6.5)
plt.title('Scatterplot von Windrichtung und Lärm', fontsize = 8)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5)
plt.savefig("/Users/michaelmerkel/Desktop/Alles/WindrichtungLaerm_Gruppe2")
plt.show()


# In[26]:


import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Daten erstellen
data = {
    'Temperatur': [datensatz_grouped_list1['Temperatur'].corr(datensatz_grouped_list1['Lärm']),
                   pearsonr(datensatz_grouped_list1['Temperatur'], datensatz_grouped_list1['Lärm'])[1],
                   datensatz_grouped_list1['Temperatur'].corr(datensatz_grouped_list1['Lärm'], method='spearman'),
                   spearmanr(datensatz_grouped_list1['Temperatur'], datensatz_grouped_list1['Lärm'])[1]],
    'Windrichtung': [datensatz_grouped_list1['Windrichtung'].corr(datensatz_grouped_list1['Lärm']),
                     pearsonr(datensatz_grouped_list1['Windrichtung'], datensatz_grouped_list1['Lärm'])[1],
                     datensatz_grouped_list1['Windrichtung'].corr(datensatz_grouped_list1['Lärm'], method='spearman'),
                     spearmanr(datensatz_grouped_list1['Windrichtung'], datensatz_grouped_list1['Lärm'])[1]],
    'Windgeschwindigkeit': [datensatz_grouped_list1['Windgeschwindigkeit'].corr(datensatz_grouped_list1['Lärm']),
                            pearsonr(datensatz_grouped_list1['Windgeschwindigkeit'], datensatz_grouped_list1['Lärm'])[1],
                            datensatz_grouped_list1['Windgeschwindigkeit'].corr(datensatz_grouped_list1['Lärm'], method='spearman'),
                            spearmanr(datensatz_grouped_list1['Windgeschwindigkeit'], datensatz_grouped_list1['Lärm'])[1]],
    'vertikale Windgeschwindigkeit': [datensatz_grouped_list1['Windgeschwindigkeit_vertikal'].corr(datensatz_grouped_list1['Lärm']),
                                      pearsonr(datensatz_grouped_list1['Windgeschwindigkeit_vertikal'], datensatz_grouped_list1['Lärm'])[1],
                                      datensatz_grouped_list1['Windgeschwindigkeit_vertikal'].corr(datensatz_grouped_list1['Lärm'], method='spearman'),
                                      spearmanr(datensatz_grouped_list1['Windgeschwindigkeit_vertikal'], datensatz_grouped_list1['Lärm'])[1]],
    'Höhe': [datensatz_grouped_list1['Höhe'].corr(datensatz_grouped_list1['Lärm']),
            pearsonr(datensatz_grouped_list1['Höhe'], datensatz_grouped_list1['Lärm'])[1],
            datensatz_grouped_list1['Höhe'].corr(datensatz_grouped_list1['Lärm'], method='spearman'),
            spearmanr(datensatz_grouped_list1['Höhe'], datensatz_grouped_list1['Lärm'])[1]]
}


index = ['Pearson', 'p-Wert (Pearson)', 'Spearman', 'p-Wert (Spearman)']
df = pd.DataFrame(data, index=index)

# Funktion zum Formatieren von Fließkommazahlen
def format_float(x):
    if abs(x) < 1e-2:
        return f"{x:.2e}"
    return f"{x:.2f}"

pd.options.display.float_format = format_float
df


# In[ ]:





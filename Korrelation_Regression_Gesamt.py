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


# In[4]:


#Laden der Datei
datensatz = "/Users/michaelmerkel/Desktop/Alles/Zusammengefuegt_AD_6000_angepasst.csv"
datensatz_original = pd.read_csv(datensatz)
datensatz = datensatz_original.copy()


# In[5]:


#datensatz


# In[9]:


# Mittelwerte für Korrelation / Regression
datensatz_grouped_list = datensatz.groupby('ID').agg({
    'Temperatur': 'mean',
    'Windrichtung': 'mean',
    'Windgeschwindigkeit': 'mean',
    'Windgeschwindigkeit_vertikal': 'mean',
    'Höhe': 'mean',
    'Lärm': 'mean'
}).reset_index()

datensatz_grouped_list.drop('ID', axis=1, inplace=True)
datensatz_grouped_list


# In[10]:


# Pearson-Korrelationsmatrix
pearson_corr_matrix = datensatz_grouped_list.corr(method='pearson')
print("Pearson-Korrelationsmatrix:")
print(pearson_corr_matrix)

# Spearman-Korrelationsmatrix
spearman_corr_matrix = datensatz_grouped_list.corr(method='spearman')
print("Spearman-Korrelationsmatrix:")
print(spearman_corr_matrix)


# In[11]:


# Heatmap für Pearson-Korrelation
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(pearson_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title("Pearson-Korrelationsmatrix")
heatmap.tick_params(axis='both', labelsize=6.5, labelrotation = 0)
plt.savefig("/Users/michaelmerkel/Desktop/Pearson.png")
plt.show()


# Heatmap für Spearman-Korrelation
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title("Spearman-Korrelationsmatrix")
heatmap.tick_params(axis='both', labelsize=6.5, labelrotation = 0)
plt.savefig("/Users/michaelmerkel/Desktop/Spearman.png")
plt.show()


# In[12]:


temperatur = datensatz_grouped_list['Temperatur']
laerm = datensatz_grouped_list['Lärm']

# Pearson-Korrelation und p-Wert berechnen
corr, p_value = pearsonr(temperatur, laerm)

print("Temperatur: Pearson-Korrelationskoeffizient:", corr)
print("p-Wert:", p_value)

windrichtung = datensatz_grouped_list['Windrichtung']
laerm = datensatz_grouped_list['Lärm']

# Pearson-Korrelation und p-Wert berechnen
corr, p_value = pearsonr(windrichtung, laerm)

print("Windrichtung: Pearson-Korrelationskoeffizient:", corr)
print("p-Wert:", p_value)

windgeschwindigkeit = datensatz_grouped_list['Windgeschwindigkeit']
laerm = datensatz_grouped_list['Lärm']

# Pearson-Korrelation und p-Wert berechnen
corr, p_value = pearsonr(windgeschwindigkeit, laerm)

print("Windgeschwindigkeit: Pearson-Korrelationskoeffizient:", corr)
print("p-Wert:", p_value)

wind_v = datensatz_grouped_list['Windgeschwindigkeit_vertikal']
laerm = datensatz_grouped_list['Lärm']

# Pearson-Korrelation und p-Wert berechnen
corr, p_value = pearsonr(wind_v, laerm)

print("Windgeschwindigkeit_vertikal: Pearson-Korrelationskoeffizient:", corr)
print("p-Wert:", p_value)

hoehe = datensatz_grouped_list['Höhe']
laerm = datensatz_grouped_list['Lärm']

# Pearson-Korrelation und p-Wert berechnen
corr, p_value = pearsonr(hoehe, laerm)

print("Höhe: Pearson-Korrelationskoeffizient:", corr)
print("p-Wert:", p_value)



# In[13]:


temperatur = datensatz_grouped_list['Temperatur']
laerm = datensatz_grouped_list['Lärm']

# Spearman-Korrelation und p-Wert berechnen
corr, p_value = spearmanr(temperatur, laerm)

print("Temperatur: Spearman-Korrelationskoeffizient:", corr)
print("p-Wert:", p_value)

windrichtung = datensatz_grouped_list['Windrichtung']
laerm = datensatz_grouped_list['Lärm']

# Spearman-Korrelation und p-Wert berechnen
corr, p_value = spearmanr(windrichtung, laerm)

print("Windrichtung: Spearman-Korrelationskoeffizient:", corr)
print("p-Wert:", p_value)

windgeschwindigkeit = datensatz_grouped_list['Windgeschwindigkeit']
laerm = datensatz_grouped_list['Lärm']

# Spearman-Korrelation und p-Wert berechnen
corr, p_value = spearmanr(windgeschwindigkeit, laerm)

print("Windgeschwindigkeit: Spearman-Korrelationskoeffizient:", corr)
print("p-Wert:", p_value)

wind_v = datensatz_grouped_list['Windgeschwindigkeit_vertikal']
laerm = datensatz_grouped_list['Lärm']

# Spearman-Korrelation und p-Wert berechnen
corr, p_value = spearmanr(wind_v, laerm)

print("Windgeschwindigkeit_vertikal: Spearman-Korrelationskoeffizient:", corr)
print("p-Wert:", p_value)

hoehe = datensatz_grouped_list['Höhe']
laerm = datensatz_grouped_list['Lärm']

# Spearman-Korrelation und p-Wert berechnen
corr, p_value = spearmanr(hoehe, laerm)

print("Höhe: Spearman-Korrelationskoeffizient:", corr)
print("p-Wert:", p_value)



# In[14]:


import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Daten erstellen
data = {
    'Temperatur': [datensatz_grouped_list['Temperatur'].corr(datensatz_grouped_list['Lärm']),
                   pearsonr(datensatz_grouped_list['Temperatur'], datensatz_grouped_list['Lärm'])[1],
                   datensatz_grouped_list['Temperatur'].corr(datensatz_grouped_list['Lärm'], method='spearman'),
                   spearmanr(datensatz_grouped_list['Temperatur'], datensatz_grouped_list['Lärm'])[1]],
    'Windrichtung': [datensatz_grouped_list['Windrichtung'].corr(datensatz_grouped_list['Lärm']),
                     pearsonr(datensatz_grouped_list['Windrichtung'], datensatz_grouped_list['Lärm'])[1],
                     datensatz_grouped_list['Windrichtung'].corr(datensatz_grouped_list['Lärm'], method='spearman'),
                     spearmanr(datensatz_grouped_list['Windrichtung'], datensatz_grouped_list['Lärm'])[1]],
    'Windgeschwindigkeit': [datensatz_grouped_list['Windgeschwindigkeit'].corr(datensatz_grouped_list['Lärm']),
                            pearsonr(datensatz_grouped_list['Windgeschwindigkeit'], datensatz_grouped_list['Lärm'])[1],
                            datensatz_grouped_list['Windgeschwindigkeit'].corr(datensatz_grouped_list['Lärm'], method='spearman'),
                            spearmanr(datensatz_grouped_list['Windgeschwindigkeit'], datensatz_grouped_list['Lärm'])[1]],
    'vertikale Windgeschwindigkeit': [datensatz_grouped_list['Windgeschwindigkeit_vertikal'].corr(datensatz_grouped_list['Lärm']),
                                      pearsonr(datensatz_grouped_list['Windgeschwindigkeit_vertikal'], datensatz_grouped_list['Lärm'])[1],
                                      datensatz_grouped_list['Windgeschwindigkeit_vertikal'].corr(datensatz_grouped_list['Lärm'], method='spearman'),
                                      spearmanr(datensatz_grouped_list['Windgeschwindigkeit_vertikal'], datensatz_grouped_list['Lärm'])[1]],
    'Höhe': [datensatz_grouped_list['Höhe'].corr(datensatz_grouped_list['Lärm']),
            pearsonr(datensatz_grouped_list['Höhe'], datensatz_grouped_list['Lärm'])[1],
            datensatz_grouped_list['Höhe'].corr(datensatz_grouped_list['Lärm'], method='spearman'),
            spearmanr(datensatz_grouped_list['Höhe'], datensatz_grouped_list['Lärm'])[1]]
}

index = ['Pearson', 'p-Wert (Pearson)', 'Spearman', 'p-Wert (Spearman)']
df = pd.DataFrame(data, index=index)
def format_float(x):
    if abs(x) < 1e-2:
        return f"{x:.2e}"
    return f"{x:.2f}"

pd.options.display.float_format = format_float

df


# In[16]:


# Daten
hoehe = datensatz_grouped_list['Höhe']
laerm = datensatz_grouped_list['Lärm']

# Scatterplot erstellen mit Trendlinie
plt.figure(figsize=(4, 3), dpi = 450)
sns.regplot(x=hoehe, y=laerm, scatter_kws={'s': 0.1}, color='midnightblue',line_kws={'color': 'red','linewidth': 1})
plt.xlabel('Höhe [m]', fontsize = 6.5)
plt.ylabel('Lärm [dB(A)]', fontsize = 6.5)
plt.title('Scatterplot von Höhe und Lärm', fontsize = 8)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5) 

plt.savefig("/Users/michaelmerkel/Desktop/HoeheLaerm_alle")
plt.show()


# Daten
temp = datensatz_grouped_list['Temperatur']
laerm = datensatz_grouped_list['Lärm']

# Scatterplot erstellen mit Trendlinie
plt.figure(figsize=(4, 3), dpi = 450)
sns.regplot(x=temp, y=laerm, scatter_kws={'s': 0.1}, color='midnightblue',line_kws={'color': 'red','linewidth': 1})
plt.xlabel('Temperatur [°C]', fontsize = 6.5)
plt.ylabel('Lärm [dB(A)]', fontsize = 6.5)
plt.title('Scatterplot von Temperatur und Lärm', fontsize = 8)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5)
plt.savefig("/Users/michaelmerkel/Desktop/TempLaerm_alle")
plt.show()

# Daten
wind = datensatz_grouped_list['Windgeschwindigkeit']
laerm = datensatz_grouped_list['Lärm']

# Scatterplot erstellen mit Trendlinie
plt.figure(figsize=(4, 3), dpi = 450)
sns.regplot(x=wind, y=laerm, scatter_kws={'s': 0.1}, color='midnightblue',line_kws={'color': 'red','linewidth': 1})
plt.xlabel('Windgeschwindigkeit [km/h]', fontsize = 6.5)
plt.ylabel('Lärm [dB(A)]', fontsize = 6.5)
plt.title('Scatterplot von Windgeschwindigkeit und Lärm', fontsize = 8)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5)
plt.savefig("/Users/michaelmerkel/Desktop/WindgeschwindigkeitLaerm_alle")
plt.show()

# Daten
wind_wz = datensatz_grouped_list['Windgeschwindigkeit_vertikal']
laerm = datensatz_grouped_list['Lärm']

# Scatterplot erstellen mit Trendlinie
plt.figure(figsize=(4, 3), dpi = 450)
sns.regplot(x=wind_wz, y=laerm, scatter_kws={'s': 0.1}, color='midnightblue',line_kws={'color': 'red','linewidth': 1})
plt.xlabel('vertikale Windgeschwindigkeit [km/h]', fontsize = 6.5)
plt.ylabel('Lärm [dB(A)]', fontsize = 6.5)
plt.title('Scatterplot von vertikaler Windgeschwindigkeit und Lärm', fontsize = 8)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5)
plt.savefig("/Users/michaelmerkel/Desktop/wind_wzLaerm_alle")
plt.show()

# Daten
windrichtung = datensatz_grouped_list['Windrichtung']
laerm = datensatz_grouped_list['Lärm']

# Scatterplot erstellen mit Trendlinie
plt.figure(figsize=(4, 3), dpi = 450)
sns.regplot(x=windrichtung, y=laerm, scatter_kws={'s': 0.1}, color='midnightblue',line_kws={'color': 'red','linewidth': 1})
plt.xlabel('Windrichtung [°]', fontsize = 6.5)
plt.ylabel('Lärm [dB(A)]', fontsize = 6.5)
plt.title('Scatterplot von Windrichtung und Lärm', fontsize = 8)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5)
plt.savefig("/Users/michaelmerkel/Desktop/WindrichtungLaerm_alle")
plt.show()


# In[ ]:





# In[ ]:





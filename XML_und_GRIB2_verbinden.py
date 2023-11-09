#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xml.etree.ElementTree as ET
import pandas as pd
import xarray as xr
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter
from metpy.units import units
from fractions import Fraction
from mpl_toolkits.mplot3d import Axes3D
from decimal import Decimal


# In[3]:


# auf Datensatz zugreifen und importieren

Jan514 = '/Users/michaelmerkel/Desktop/Datensatz/Lärm/2023-01/muc05-01-2023_14.xml'
tree = ET.parse(Jan514) #parse = einlesen
root = tree.getroot()

Jan0523 = "/Users/michaelmerkel/Desktop/Datensatz/Wetter/2023-01/munich_20230105_15-66_p-qv-t-u-v-w.grib2"
ds = xr.load_dataset(Jan0523, engine='cfgrib')

munich = "/Users/michaelmerkel/Desktop/Datensatz/Wetter/hhl_munich.grib2" #an München angepasste Höhen(stufen)
munich_grid = xr.load_dataset(munich, engine='cfgrib')


# In[4]:


# füge die Variable h nur mit Layern 15 - 66 dem ds hinzu
relevante_h = munich_grid['h']
relevante_h = relevante_h[14:]
meter_in_fuß = 3.28084
relevante_h = relevante_h * meter_in_fuß
ds['half_level'] = relevante_h

# Werte von half_level holen
half_level_values = ds['half_level'].values

# Hinzufügen und Berechnung von full_levels
generalVerticalLayer_values = ds.coords['generalVerticalLayer'].values
latitude_values = ds.coords['latitude'].values
longitude_values = ds.coords['longitude'].values

# Erstellen Sie eine leere Variable full_level mit den gewünschten Dimensionen
full_level_values = np.empty((len(generalVerticalLayer_values), 
                      len(latitude_values), 
                      len(longitude_values)), dtype=np.float32)

# Mittelwert für die ersten 51 Layer von half_level berechnen und in full_level hinzufügen
for i, gv in enumerate(generalVerticalLayer_values[:51]):
        full_level_values[i, :, :] = (half_level_values[i, :, :] + half_level_values[i+1, :, :]) / 2.0

# neue Variable "full_level" mit den Werten von full_level
ds['full_level'] = (('generalVerticalLayer', 'latitude', 'longitude'), full_level_values)

# Liste mit Zeitpunkten und Vorhersageschritten
time_values = ds['time'].values
step_values = ds['step'].values

# Füge den "half_level" und "full_level" die Dimensionen "time" und "step" hinzu
ds_half_level = ds['half_level'].expand_dims(time=time_values, step=step_values)
ds_full_level = ds['full_level'].expand_dims(time=time_values, step=step_values)
ds = ds.assign(half_level=ds_half_level, full_level=ds_full_level)

ds


# In[5]:


# Code zur Ermittlung eindeutiger Flugspuren

# alle Zeitpunkte aller Flugspuren in der jeweiligen Stunde im UNIX-Format
flugspuren_unix = []
for trk in root.iter('trk'):
    startzeitpunkt = int(trk.find('p').attrib['t'])  # Startzeitpunkt der Flugspur
    flugspur_unix = []
    for i, p in enumerate(trk.iter('p')):
        relative_zeit = int(p.attrib['t'])
        if i == 0:
            flugspur_unix.append(startzeitpunkt)
        else:
            unix_timestamp = startzeitpunkt + relative_zeit
            flugspur_unix.append(unix_timestamp)
    flugspuren_unix.append(flugspur_unix)
    
# nur Start- Endzeitpunkt für jede Flugspur
flugspuren_unix = []
for trk in root.iter('trk'):
    startzeitpunkt = int(trk.find('p').attrib['t'])  
    endzeitpunkt = int(trk.find('p[last()]').attrib['t']) + startzeitpunkt  
    act = (trk.attrib['act'])
    flugspuren_unix.append((act, startzeitpunkt, endzeitpunkt))

# Start- Endzeitpunkt für jedes Lärmereignis
larmereignisse = []
for ne in root.iter('ne'):
    startzeitpunkt = int(ne.attrib['tstart']) 
    endzeitpunkt = int(ne.attrib['tstop']) 
    actype = (ne.attrib['actype'])
    larm_max = (ne.attrib['las'])
    larmereignisse.append((actype, startzeitpunkt, endzeitpunkt, larm_max))
    

flight_paths_df = pd.DataFrame(flugspuren_unix, columns=['Flugzeugtyp', 'Startzeitpunkt', 'Endzeitpunkt'])
noise_events_df = pd.DataFrame(larmereignisse, columns=['Flugzeugtyp', 'Startzeitpunkt', 'Endzeitpunkt', 'Maximaler Lärm'])

# Überschneidungen zwischen Flugspuren und Lärmereignissen
uberschneidungen = []
for i, (act_flugspur, start_flugspur, ende_flugspur) in enumerate(flugspuren_unix):
    for j, (actype_larm, start_larm, ende_larm, larm_max_larm) in enumerate(larmereignisse):
        if start_larm >= start_flugspur and start_larm <= ende_flugspur and ende_larm >= start_flugspur and ende_larm <= ende_flugspur and act_flugspur == actype_larm:
            uberschneidungen.append((i+1, j+1))
            
# Anzahl der Überschneidungen und Ausgabe der Überschneidungen
anzahl_uberschneidungen = len(uberschneidungen)
max_lärmereignis_idx = 0
lärmereignis_häufigkeit = {}
for flugspur_idx, larmereignis_idx in uberschneidungen:
    if larmereignis_idx > max_lärmereignis_idx:
        max_lärmereignis_idx = larmereignis_idx
    # print(f"Flugspur {flugspur_idx} und Lärmereignis {larmereignis_idx} überschneiden sich.")


# Liste für die eindeutig zugeordneten Flugspuren und Flugzeugtypen erstellen
eindeutige_flugspuren = []
max_larmpegel = []  # Liste für die maximalen Lärmpegel
luftfahrzeugtypen = []  # Liste für die Luftfahrzeugtypen

# Schleife über die Lärmereignisse
for larmereignis_idx, (actype_larm, start_larm, ende_larm, larm_max_larm) in enumerate(larmereignisse):
    zugeordnete_flugspuren = []
    
    # Schleife über die Flugspuren
    for flugspur_idx, (act_flugspur, start_flugspur, ende_flugspur) in enumerate(flugspuren_unix):
        if start_larm >= start_flugspur and ende_larm <= ende_flugspur and actype_larm == act_flugspur:
            zugeordnete_flugspuren.append(flugspur_idx+1)
    
    # Prüfen, ob nur eine Flugspur zugeordnet wurde
    if len(zugeordnete_flugspuren) == 1:
        eindeutige_flugspuren.append((larmereignis_idx+1, zugeordnete_flugspuren[0]))
        max_larmpegel.append(larm_max_larm)  
        
        # Luftfahrzeugtyp basierend auf Lärmereignis hinzufügen
        luftfahrzeugtypen.append(actype_larm)

# DataFrame für die eindeutig zugeordneten Flugspuren und Lärmereignisse
df_eindeutige_zuordnung = pd.DataFrame(eindeutige_flugspuren, columns=['Lärmereignis', 'Flugspur'])
df_eindeutige_zuordnung['Maximaler Lärmpegel'] = max_larmpegel  # Max Lärmpegel als neue Spalte hinzufügen
df_eindeutige_zuordnung['Flugzeugtyp'] = luftfahrzeugtypen  # Flugzeugtypen hinzufügen

# Konvertiere die Spalte 'Maximaler Lärmpegel' in numerischen Datentyp
df_eindeutige_zuordnung['Maximaler Lärmpegel'] = pd.to_numeric(df_eindeutige_zuordnung['Maximaler Lärmpegel'])

# DataFrame nach "Flugspur" gruppieren und filtere den maximalen Lärmpegel für jede Flugspur
flugspur_eindeutige_zuordnung = df_eindeutige_zuordnung.groupby('Flugspur')['Maximaler Lärmpegel'].idxmax()
df_eindeutige_zuordnung = df_eindeutige_zuordnung.loc[flugspur_eindeutige_zuordnung]

# Sortiere nach Flugspur
df_eindeutige_zuordnung = df_eindeutige_zuordnung.sort_values(by='Flugspur')
df_eindeutige_zuordnung.reset_index(drop=True, inplace=True)

print(df_eindeutige_zuordnung)


# In[6]:


# Koordinaten zwei Nachkommastellen und in 0,02er Schritten

# Funktion zur Extraktion der Dezimalstellen und Rundung auf gerade Zahlen
def round_to_even_decimal(x):
    decimal_part = Decimal(str(x)) % 1
    rounded_decimal = round(decimal_part * 100)  
    if rounded_decimal % 2 != 0:
        rounded_decimal += 1
    return int(x) + rounded_decimal / 100

# Datum und Uhrzeit aus der .XML filtern
if 't' in root[0].attrib:
    t_value = root[0].attrib['t']
    t_value = int(t_value)  

date_time = datetime.datetime.fromtimestamp(t_value)
datum = date_time.date() 
uhrzeit = date_time.time() 

# Datum und Uhrzeit der XML
datum_str = date_time.strftime("%Y-%m-%d") # Datum als String im Format "YYYY-MM-DD"
uhrzeit_str = date_time.strftime("%H:%M Uhr")
uhrzeit_int = int(date_time.strftime("%H"))  # Uhrzeit als Integer im Format "HH"
#print(uhrzeit_int)

# von Uhrzeit der .XML auf Uhrzeit der .GRIB2 zugreifen
def fix_times(uhrzeit_int):
    i = uhrzeit_int // 3
    j = uhrzeit_int % 3
    return i, j
tupel_grib = fix_times(uhrzeit_int) # Tupel herausfinden
zeitpunkt_idx, vorhersage_idx = tupel_grib # Zugriff auf die Grib-Datei mit Hilfe des berechneten Tupels (i, j)
zeitpunkt = ds.isel(time=zeitpunkt_idx) # time = i
uhrzeit_grib = zeitpunkt.isel(step=vorhersage_idx) # step = j
print(uhrzeit_grib)


# In[137]:


# Zuordnung Jahreszeit
monat = int(datum_str[5:7])

jahreszeiten = {
    1: "Winter",
    2: "Winter",
    3: "Frühling",
    4: "Frühling",
    5: "Frühling",
    6: "Sommer",
    7: "Sommer",
    8: "Sommer",
    9: "Herbst",
    10: "Herbst",
    11: "Herbst",
    12: "Winter"
}

jahreszeit = jahreszeiten.get(monat)
#jahreszeit


# In[138]:


print(datum_str, uhrzeit_str)
print(jahreszeit)
print(df_eindeutige_zuordnung)


# In[139]:


anzahl_flugzeugtypen = df_eindeutige_zuordnung["Flugzeugtyp"].value_counts()
print(anzahl_flugzeugtypen)


# In[140]:


# Zuordnung Flugspur - Temperatur

def get_grib_point_value(ds, lat, lon, lev, time_idx, step_idx):
    lat_idx = abs(ds.latitude - lat).argmin()
    lon_idx = abs(ds.longitude - lon).argmin()
    lev_idx = abs(ds.full_level[time_idx, step_idx, :, lat_idx, lon_idx] - lev).argmin()
    temp = round(ds.t[time_idx, step_idx, lev_idx, lat_idx, lon_idx].values.item() - 273.15, 2)
    return (lat, lon, lev), temp

# Konvertiere den DataFrame in ein Dictionary
dict_maximaler_larmpegel = df_eindeutige_zuordnung.set_index('Flugspur')['Maximaler Lärmpegel'].to_dict()
dict_flugzeugtyp = df_eindeutige_zuordnung.set_index('Flugspur')['Flugzeugtyp'].to_dict()

flugspur_temperatur = {}
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    coordinates = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    (float(p.attrib['l']))) for p in flugspur_element.iter('p')]
    unique_coordinates = list(dict.fromkeys(coordinates)) # entferne doppelte Werte
    max_larmpegel = dict_maximaler_larmpegel.get(flugspur_index)  # hole den maximalen Lärmpegel
    flugzeugtyp = dict_flugzeugtyp.get(flugspur_index) # hole den Flugzeugtypen
    flugspur_key = f"Flugspur {flugspur_index} (max. Lärmpegel: {max_larmpegel}, Flugzeugtyp: {flugzeugtyp})"  # Erstelle den neuen Schlüssel
    flugspur_temperatur[flugspur_key] = []  # Erstelle die Liste für diese Flugspur
    for coord in unique_coordinates:
        coord, value = get_grib_point_value(ds, *coord, zeitpunkt_idx, vorhersage_idx)
        flugspur_temperatur[flugspur_key].append((coord, value))  # Füge die Daten zur Liste der Flugspur hinzu
        
flugspur_temperatur


# In[141]:


# Zuordnung Flugspur - wind_u
def get_grib_point_value(ds, lat, lon, lev, time_idx, step_idx):
    lat_idx = abs(ds.latitude - lat).argmin()
    lon_idx = abs(ds.longitude - lon).argmin()
    lev_idx = abs(ds.full_level[time_idx, step_idx, :, lat_idx, lon_idx] - lev).argmin()
    wind_u = round(ds.u[time_idx, step_idx, lev_idx, lat_idx, lon_idx].values.item(), 2)
    return (lat, lon, lev), wind_u

flugspur_wind_u = {}
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    coordinates = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    (float(p.attrib['l']))) for p in flugspur_element.iter('p')]
    unique_coordinates = list(dict.fromkeys(coordinates)) # entferne doppelte Werte
    max_larmpegel = dict_maximaler_larmpegel.get(flugspur_index)  # hole den maximalen Lärmpegel
    flugzeugtyp = dict_flugzeugtyp.get(flugspur_index)  # hole den Flugzeugtypen
    flugspur_key = f"Flugspur {flugspur_index} (max. Lärmpegel: {max_larmpegel}, Flugzeugtyp: {flugzeugtyp})"  # erstelle den neuen Schlüssel
    flugspur_wind_u[flugspur_key] = []  # erstelle die Liste für diese Flugspur
    for coord in unique_coordinates:
        coord, value = get_grib_point_value(ds, *coord, zeitpunkt_idx, vorhersage_idx)
        flugspur_wind_u[flugspur_key].append((coord, value))  # füge die Daten zur Liste der Flugspur hinzu

#flugspur_wind_u


# In[142]:


# Zuordnung Flugspur - wind_v
def get_grib_point_value(ds, lat, lon, lev, time_idx, step_idx):
    lat_idx = abs(ds.latitude - lat).argmin()
    lon_idx = abs(ds.longitude - lon).argmin()
    lev_idx = abs(ds.full_level[time_idx, step_idx, :, lat_idx, lon_idx] - lev).argmin()
    wind_v = round(ds.v[time_idx, step_idx, lev_idx, lat_idx, lon_idx].values.item(), 2)
    return (lat, lon, lev), wind_v

flugspur_wind_v = {}
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    coordinates = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    (float(p.attrib['l']))) for p in flugspur_element.iter('p')]
    unique_coordinates = list(dict.fromkeys(coordinates)) # entferne doppelte Werte
    max_larmpegel = dict_maximaler_larmpegel.get(flugspur_index)  # hole den maximalen Lärmpegel
    flugzeugtyp = dict_flugzeugtyp.get(flugspur_index)  # hole den Flugzeugtypen
    flugspur_key = f"Flugspur {flugspur_index} (max. Lärmpegel: {max_larmpegel}, Flugzeugtyp: {flugzeugtyp})"  # erstelle den neuen Schlüssel
    flugspur_wind_v[flugspur_key] = []  # erstelle die Liste für diese Flugspur
    for coord in unique_coordinates:
        coord, value = get_grib_point_value(ds, *coord, zeitpunkt_idx, vorhersage_idx)
        flugspur_wind_v[flugspur_key].append((coord, value))  # füge die Daten zur Liste der Flugspur hinzu

#flugspur_wind_v


# In[143]:


# Zuordnung Flugspur - wind_wz
def get_grib_point_value(ds, lat, lon, lev, time_idx, step_idx):
    lat_idx = abs(ds.latitude - lat).argmin()
    lon_idx = abs(ds.longitude - lon).argmin()
    lev_idx = abs(ds.half_level[time_idx, step_idx, :, lat_idx, lon_idx] - lev).argmin()
    wind_wz = round(ds.wz[time_idx, step_idx, lev_idx, lat_idx, lon_idx].values.item(), 2)
    return (lat, lon, lev), wind_wz

flugspur_wind_wz = {}
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    coordinates = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    (float(p.attrib['l']))) for p in flugspur_element.iter('p')]
    unique_coordinates = list(dict.fromkeys(coordinates)) # entferne doppelte Werte
    max_larmpegel = dict_maximaler_larmpegel.get(flugspur_index)  # hole den maximalen Lärmpegel
    flugzeugtyp = dict_flugzeugtyp.get(flugspur_index)  # hole den Flugzeugtypen
    flugspur_key = f"Flugspur {flugspur_index} (max. Lärmpegel: {max_larmpegel}, Flugzeugtyp: {flugzeugtyp})"  # erstelle den neuen Schlüssel
    flugspur_wind_wz[flugspur_key] = []  # erstelle die Liste für diese Flugspur
    for coord in unique_coordinates:
        coord, value = get_grib_point_value(ds, *coord, zeitpunkt_idx, vorhersage_idx)
        flugspur_wind_wz[flugspur_key].append((coord, value))  # füge die Daten zur Liste der Flugspur hinzu

flugspur_wind_wz


# In[144]:


# Zuordnung Flugspur und Windrichtung und -geschwindigkeit
def get_wind_components(ds, lat, lon, lev, time_idx, step_idx):
    lat_idx = abs(ds.latitude - lat).argmin()
    lon_idx = abs(ds.longitude - lon).argmin()
    lev_idx = abs(ds.full_level[time_idx, step_idx, :, lat_idx, lon_idx] - lev).argmin()
    u = ds.u[time_idx, step_idx, lev_idx, lat_idx, lon_idx].values.item()
    v = ds.v[time_idx, step_idx, lev_idx, lat_idx, lon_idx].values.item()

    windrichtung = np.arctan2(v, u) * 180 / np.pi  # Umrechnung von Radiant in Grad
    windrichtung = (windrichtung + 360) % 360  # Umwandlung in den Bereich von 0 bis 360 Grad
    windrichtung = round(windrichtung, 2)

    windgeschwindigkeit = np.sqrt(u**2 + v**2)
    windgeschwindigkeit = round(windgeschwindigkeit, 2)

    return (lat, lon, lev), windrichtung, windgeschwindigkeit

flugspur_wind = {}
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    coordinates = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    (float(p.attrib['l']))) for p in flugspur_element.iter('p')]
    unique_coordinates = list(dict.fromkeys(coordinates)) # entferne doppelte Werte
    max_larmpegel = dict_maximaler_larmpegel.get(flugspur_index)  # hole den maximalen Lärmpegel
    flugzeugtyp = dict_flugzeugtyp.get(flugspur_index)  # hole den Flugzeugtypen
    flugspur_key = f"Flugspur {flugspur_index} (max. Lärmpegel: {max_larmpegel}, Flugzeugtyp: {flugzeugtyp})"  # erstelle den neuen Schlüssel
    flugspur_wind[flugspur_key] = []  # erstelle die Liste für diese Flugspur
    for coord in unique_coordinates:
        coord, windrichtung, windgeschwindigkeit = get_wind_components(ds, *coord, zeitpunkt_idx, vorhersage_idx)
        flugspur_wind[flugspur_key].append((coord, windrichtung, windgeschwindigkeit))  # füge die Daten zur Liste der Flugspur hinzu

flugspur_wind


# In[145]:


from matplotlib.cm import get_cmap

# Darstellung der zweidimensionalen Flugspuren
# Überlagerung der genauen Flugspuren mit den gerundeten
plt.figure(figsize=(15, 10))
ungerundete_koordinaten = []

# Anzahl der Flugspuren
num_flugspuren = len(df_eindeutige_zuordnung['Flugspur'])

# Generiere eine Colormap mit ausreichend vielen Farben
cmap = get_cmap('plasma')
colors = [cmap(i / num_flugspuren) for i in range(num_flugspuren)]

for i, flugspur_index in enumerate(df_eindeutige_zuordnung['Flugspur']):
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(float(p.attrib['a']), 
                    float(p.attrib['n'])) for p in flugspur_element.iter('p')]
    ungerundete_koordinaten.append(koordinaten)   
    # Aufteilen der Koordinaten in separate Listen für x- und y-Koordinaten
    x_coords, y_coords = zip(*koordinaten)
    # Plot der Flugspur als Linie
    plt.plot(y_coords, x_coords, label=f"Flugspur {flugspur_index}", color=colors[i])
    
for i, flugspur_index in enumerate(flugspur_eindeutige_zuordnung.index):
    # Extrahiere Flugspur-Koordinaten aus dem 'trk'-Element in der XML-Datei
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(round_to_even_decimal(float(p.attrib['a'])), 
                    round_to_even_decimal(float(p.attrib['n']))) for p in flugspur_element.iter('p')]
    x_coords, y_coords = zip(*koordinaten)
    plt.plot(y_coords, x_coords, label=f"Gerundete Flugspur {flugspur_index}", color=colors[i])

plt.xlabel('Längengrad')
plt.ylabel('Breitengrad')
plt.title('2D-Flugspuren - ungerundet und gerundet')
plt.legend()
plt.grid(True)
plt.show()


# In[146]:


# Konvertiere den DataFrame in ein Dictionary für schnelleren Zugriff
dict_maximaler_larmpegel = df_eindeutige_zuordnung.set_index('Flugspur')['Maximaler Lärmpegel'].to_dict()

flugspur_temperatur = {}
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    coordinates = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    (float(p.attrib['l']))) for p in flugspur_element.iter('p')]
    unique_coordinates = list(dict.fromkeys(coordinates)) # entferne doppelte Werte
    max_larmpegel = dict_maximaler_larmpegel.get(flugspur_index)  # Hole den maximalen Lärmpegel
    flugspur_key = f"Flugspur {flugspur_index} (Maximaler Lärmpegel: {max_larmpegel})"  # Erstelle den neuen Schlüssel
    flugspur_temperatur[flugspur_key] = []  # Erstelle die Liste für diese Flugspur
    for coord in unique_coordinates:
        coord, value = get_grib_point_value(ds, *coord, zeitpunkt_idx, vorhersage_idx)
        flugspur_temperatur[flugspur_key].append((coord, value))  # Füge die Daten zur Liste der Flugspur hinzu
        
#flugspur_temperatur


# In[147]:


# Koordinaten zwei Nachkommastellen und in 0,02er Schritten + Höhe in 100er Schritten

# Funktion zur Extraktion der Dezimalstellen und Rundung auf gerade Zahlen
def round_to_even_decimal(x):
    decimal_part = Decimal(str(x)) % 1
    rounded_decimal = round(decimal_part * 100)  # Multipliziere mit 100, um zwei Dezimalstellen zu erhalten
    if rounded_decimal % 2 != 0:
        # Wenn die gerundete Dezimalstelle ungerade ist, füge 1 hinzu, um auf eine gerade Zahl zu runden
        rounded_decimal += 1
    return int(x) + rounded_decimal / 100  # Füge die gerundete Dezimalstelle zur Ganzzahl hinzu

# Funktion zur Rundung auf 100er Schritte
def round_to_nearest_100(x):
    return round(x / 100) * 100

# Schleife über die Indizes der Serie 'flugspur_eindeutige_zuordnung'
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    # Extrahiere Flugspur-Koordinaten aus 'trk' Element in der XML-Datei
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    round_to_nearest_100(float(p.attrib['l']))) for p in flugspur_element.iter('p')]
    
    # Ausgabe der gerundeten Flugspur-Koordinaten
    print(f"Flugspur {flugspur_index} angepasste Koordinaten: {koordinaten}")


# In[ ]:


# Schleife über die Indizes der Serie 'flugspur_eindeutige_zuordnung'
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    # Extrahiere Flugspur-Koordinaten aus 'trk' Element in der XML-Datei
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    round_to_nearest_100(float(p.attrib['l']))) for p in flugspur_element.iter('p')]
    
    # Ausgabe der gerundeten Flugspur-Koordinaten
    print(f"Flugspur {flugspur_index} angepasste Koordinaten: {koordinaten}")


# In[13]:


# Datum und Uhrzeit aus der .XML filtern
if 't' in root[0].attrib:
    t_value = root[0].attrib['t']
    t_value = int(t_value)  

date_time = datetime.datetime.fromtimestamp(t_value)
datum = date_time.date() 
uhrzeit = date_time.time() 

# Datum und Uhrzeit der XML
datum_str = date_time.strftime("%Y-%m-%d")  # Das Datum als String im Format "YYYY-MM-DD"
uhrzeit_int = int(date_time.strftime("%H"))  # Die Uhrzeit als Integer im Format "HH"
#print(uhrzeit_int)

# von Uhrzeit der .XML auf Uhrzeit der .GRIB2 zugreifen
def fix_times(uhrzeit_int):
    i = uhrzeit_int // 3
    j = uhrzeit_int % 3
    return i, j
tupel_grib = fix_times(uhrzeit_int) # Tupel herausfinden

zeitpunkt_idx, vorhersage_idx = tupel_grib # Zugriff auf die Grib-Datei mit Hilfe des berechneten Tupels (i, j)
zeitpunkt = ds.isel(time=zeitpunkt_idx) # time = i
uhrzeit_grib = zeitpunkt.isel(step=vorhersage_idx) # step = j
#print(uhrzeit_grib)

def get_grib_point_value(ds, lat, lon, lev, time_idx, step_idx):
    lat_idx = abs(ds.latitude - lat).argmin()
    lon_idx = abs(ds.longitude - lon).argmin()
    lev_idx = abs(ds.full_level[time_idx, step_idx, :, lat_idx, lon_idx] - lev).argmin()
    temp = round(ds.t[time_idx, step_idx, lev_idx, lat_idx, lon_idx].values.item() - 273.15, 2)
    return (lat, lon, lev), temp

flugspur_temperatur = {}
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    coordinates = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    (float(p.attrib['l']))) for p in flugspur_element.iter('p')]
    unique_coordinates = list(dict.fromkeys(coordinates)) # entferne doppelte Werte
    flugspur_temperatur[flugspur_index] = []
    for coord in unique_coordinates:
        coord, value = get_grib_point_value(ds, *coord, zeitpunkt_idx, vorhersage_idx)
        flugspur_temperatur[flugspur_index].append((coord, value))
        
#flugspur_temperatur


# In[14]:


def get_grib_point_value(ds, lat, lon, lev, time_idx, step_idx):
    lat_idx = abs(ds.latitude - lat).argmin()
    lon_idx = abs(ds.longitude - lon).argmin()
    lev_idx = abs(ds.full_level[time_idx, step_idx, :, lat_idx, lon_idx] - lev).argmin()
    time_idx = zeitpunkt_idx
    step_idx = vorhersage_idx
    temp = round(ds.t[time_idx, step_idx, lev_idx, lat_idx, lon_idx].values.item() - 273.15, 2)
    return (lat, lon, lev), temp

flugspur_temperatur = {}
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    coordinates = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    (float(p.attrib['l']))) for p in flugspur_element.iter('p')]
    flugspur_temperatur[flugspur_index] = []
    for coord in coordinates:
        coord, value = get_grib_point_value(ds, *coord, zeitpunkt_idx, vorhersage_idx)
        flugspur_temperatur[flugspur_index].append((coord, value))
        
#flugspur_temperatur


# In[339]:


# Runden der Koordinaten der zugeordneten Flugspuren und der Koordinaten der GRIB Datei

# lat und lon auf zwei Nachkommastellen runden, Höhe auf Hunderter runden
latitude_values = ds.latitude.round(2)
longitude_values = ds.longitude.round(2)
half_level_values = ds.half_level.round(-2)
full_level_values = ds.full_level.round(-2)



# Schleife über die Indizes der Serie 'flugspur_eindeutige_zuordnung'
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    # Extrahiere Flugspur-Koordinaten aus 'trk' Element in der XML-Datei
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    round_to_nearest_100(float(p.attrib['l']))) for p in flugspur_element.iter('p')]


# In[15]:


# Funktion zur Bestimmung des nächstgelegenen Punkts in den GRIB-Daten
def find_nearest_grib_point(ds, lat, lon, lev):
    lat_idx = abs(ds.latitude - lat).argmin()
    lon_idx = abs(ds.longitude - lon).argmin()
    lev_idx = abs(ds.full_level - lev).argmin()
    return lat_idx, lon_idx, lev_idx

# Liste zur Speicherung der entsprechenden GRIB-Indizes
grib_points = []

def feet_to_meters(feet):
    return feet * 0.3048

# Schleife über die Flugspuren
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(round_to_even_decimal(float(p.attrib['a'])),
                    round_to_even_decimal(float(p.attrib['n'])),
                    feet_to_meters(round_to_nearest_100(float(p.attrib['l'])))) for p in flugspur_element.iter('p')]

    # Schleife über die Koordinaten jeder Flugspur
    for coord in koordinaten:
        lat_idx, lon_idx, lev_idx = find_nearest_grib_point(ds, *coord)
        grib_points.append((lat_idx, lon_idx, lev_idx))

# Zugriff auf einen spezifischen Datenpunkt
specific_point = ds.isel(latitude=grib_points[0][0], longitude=grib_points[0][1])
specific_point
#grib_point


# In[151]:


# Koordinaten zwei Nachkommastellen und Höhe

# Schleife über die Indizes der Serie 'flugspur_eindeutige_zuordnung'
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(round(float(p.attrib['a']), 2), 
                    round(float(p.attrib['n']), 2), 
                    float(p.attrib['l'])) for p in flugspur_element.iter('p')]
    
    # Ausgabe der Flugspur-Koordinaten
    #print(f"Flugspur {flugspur_index} Koordinaten: {koordinaten}")


# In[152]:


# Komplette 3D-Koordinaten

# Schleife über die Indizes der Serie 'flugspur_eindeutige_zuordnung'
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(float(p.attrib['a']), 
                    float(p.attrib['n']), 
                    float(p.attrib['l'])) for p in flugspur_element.iter('p')]
    # Ausgabe der Flugspur-Koordinaten
#    print(f"Flugspur {flugspur_index} Koordinaten: {koordinaten}")


# In[ ]:


# Funktion zum Extrahieren von Datum und Uhrzeit aus der XML-Datei
def extract_date_and_time_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ts = root.find(".//sysSet").attrib['ts']
    date, time = ts.split(' ')
    # Das Datum im Format 'dd.mm.yyyy' in 'yyyy-mm-dd' umwandeln
    date = '-'.join(date.split('.')[::-1])
    return date, time

date, time = extract_date_and_time_from_xml(Jan514)
print(f"Datum: {date}")
print(f"Uhrzeit: {time}")


# In[153]:


# Schleife über die Indizes der Serie 'flugspur_eindeutige_zuordnung'
plt.figure(figsize=(15, 10))
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(round(float(p.attrib['a']), 2), round(float(p.attrib['n']), 2)) for p in flugspur_element.iter('p')]

    # Aufteilen der Koordinaten in separate Listen für x- und y-Koordinaten
    x_coords, y_coords = zip(*koordinaten)
    
    # Plot der Flugspur als Linie   
    plt.plot(y_coords, x_coords, label=f"Flugspur {flugspur_index}")

    
# Achsenbeschriftungen und Titel
plt.xlabel('Längengrad')
plt.ylabel('Breitengrad')
plt.title('2D-Flugspuren')
plt.legend()
plt.grid(True)

# Diagramm anzeigen
plt.show()


# In[154]:


# Schleife über die Indizes der Serie 'flugspur_eindeutige_zuordnung'
plt.figure(figsize=(15, 10))
for flugspur_index in df_eindeutige_zuordnung['Flugspur']:
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(float(p.attrib['a']), float(p.attrib['n'])) for p in flugspur_element.iter('p')]
    ungerundete_koordinaten.append(koordinaten)   
    # Aufteilen der Koordinaten in separate Listen für x- und y-Koordinaten
    x_coords, y_coords = zip(*koordinaten)
    # Plot der Flugspur als Linie
    plt.plot(y_coords, x_coords, label=f"Ungerundete Flugspur {flugspur_index}")
    
for flugspur_index in flugspur_eindeutige_zuordnung.index:
    # Extrahiere Flugspur-Koordinaten aus dem 'trk'-Element in der XML-Datei
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(round_to_even_decimal(float(p.attrib['a'])), round_to_even_decimal(float(p.attrib['n']))) for p in flugspur_element.iter('p')]
    x_coords, y_coords = zip(*koordinaten)
    plt.plot(y_coords, x_coords, label=f"Flugspur {flugspur_index}")

plt.xlabel('Längengrad')
plt.ylabel('Breitengrad')
plt.title('2D-Flugspuren')
plt.legend()
plt.grid(True)
plt.show()


# In[155]:


# Extrahiere ungerundete Koordinaten aus 'trk' Element in der XML-Datei
ungerundete_koordinaten = []
for flugspur_index in df_eindeutige_zuordnung['Flugspur']:
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(float(p.attrib['a']), float(p.attrib['n'])) for p in flugspur_element.iter('p')]
    ungerundete_koordinaten.append(koordinaten)

# Extrahiere gerundete Koordinaten (zwei Nachkommastellen) aus 'trk' Element in der XML-Datei
gerundete_koordinaten = []
for flugspur_index in df_eindeutige_zuordnung['Flugspur']:
    flugspur_element = root[flugspur_index + 1]
    koordinaten = [(round(float(p.attrib['a']), 2), round(float(p.attrib['n']), 2)) for p in flugspur_element.iter('p')]
    gerundete_koordinaten.append(koordinaten)

# 2D-Plot für Flugspuren mit ungerundeten Koordinaten
plt.figure(figsize=(10, 6))
for koordinaten in ungerundete_koordinaten:
    lon, lat = zip(*koordinaten)
    plt.plot(lat, lon, markersize=0.5)
    
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flugspuren mit ungerundeten Koordinaten')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for koordinaten in gerundete_koordinaten:
    lon, lat = zip(*koordinaten)
    plt.plot(lat, lon, markersize=0.5)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flugspuren mit gerundeten Koordinaten')
plt.grid(True)
plt.show()


# In[156]:


# Liste für die Koordinaten der einzelnen Flugspuren erstellen
flight_paths = []

# Flugspuren aus der XML-Datei extrahieren und zur Liste hinzufügen
for track in root.findall('.//trk'):
    coordinates = []
    for point in track.findall('.//p'):
        lat = float(point.attrib['a'])
        lon = float(point.attrib['n'])
        coordinates.append((lat, lon))
    flight_paths.append(coordinates)

# 2D-Plot erstellen für jede Flugspur separat
plt.figure(figsize=(10, 8))
for i, path in enumerate(flight_paths):
    latitudes = [lat for lat, _ in path]
    longitudes = [lon for _, lon in path]
    plt.plot(longitudes, latitudes, linewidth=0.5)

plt.xlabel('Längengrad')
plt.ylabel('Breitengrad')
plt.title('2D-Flugspuren')
plt.grid(True)
plt.show()


# In[157]:


# Liste für die Koordinaten der einzelnen Flugspuren erstellen
flight_paths = []

# Flugspuren aus der XML-Datei extrahieren und zur Liste hinzufügen
for track in root.findall('.//trk'):
    coordinates = []
    for point in track.findall('.//p'):
        lat = float(point.attrib['a'])
        lon = float(point.attrib['n'])
        alt = float(point.attrib['l'])
        coordinates.append((lat, lon, alt))
    flight_paths.append(coordinates)

# 3D-Plot erstellen für jede Flugspur separat
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
for i, path in enumerate(flight_paths):
    latitudes = [lat for lat, _, _ in path]
    longitudes = [lon for _, lon, _ in path]
    altitudes = [alt for _, _, alt in path]
    ax.plot(longitudes, latitudes, altitudes, linewidth=0.5)
ax.set_xlabel('Längengrad')
ax.set_ylabel('Breitengrad')
ax.set_zlabel('Höhe')
ax.set_title('3D-Flugspuren')
ax.view_init(elev=40, azim=-45)  # Startansicht
plt.show()


# In[158]:


#Spielerei

import plotly.graph_objects as go

# Liste für die Koordinaten der einzelnen Flugspuren erstellen
flight_paths = []

# Flugspuren aus der XML-Datei extrahieren und zur Liste hinzufügen
for track in root.findall('.//trk'):
    coordinates = []
    for point in track.findall('.//p'):
        lat = float(point.attrib['a'])
        lon = float(point.attrib['n'])
        alt = float(point.attrib['l'])
        coordinates.append((lat, lon, alt))
    flight_paths.append(coordinates)

# 3D-Plot mit Plotly erstellen für jede Flugspur separat
fig = go.Figure()
for i, path in enumerate(flight_paths):
    latitudes = [lat for lat, _, _ in path]
    longitudes = [lon for _, lon, _ in path]
    altitudes = [alt for _, _, alt in path]
    fig.add_trace(go.Scatter3d(
        x=longitudes,
        y=latitudes,
        z=altitudes,
        mode='lines',
        name=f'Flugspur {i+1}',
        line=dict(color='blue', width=2)
    ))

# Layout des 3D-Plots anpassen
fig.update_layout(
    scene=dict(
        xaxis=dict(title='Längengrad'),
        yaxis=dict(title='Breitengrad'),
        zaxis=dict(title='Höhe'),
        aspectratio=dict(x=1, y=1, z=0.3),
        camera=dict(
            eye=dict(x=0.7, y=-0.7, z=0.7),  # Startposition der Kamera
            up=dict(x=0, y=0, z=1)  # Ausrichtung der Kamera
        ),
        dragmode='orbit'  # Aktiviert die interaktive Rotation des Plots
    )
)

# Plot anzeigen
fig.show()


# In[ ]:





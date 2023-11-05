#%%
import xml.etree.ElementTree as ET
import pandas as pd
import xarray as xr
import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from decimal import Decimal
import os
import glob
from scipy import interpolate
import matplotlib.ticker as ticker


#%%


# Funktionsdefinitionen
def round_to_even_decimal(x):
    multiplied = int(x * 100)
    rounded = multiplied + 1 if multiplied % 2 != 0 else multiplied
    return rounded / 100


def process_xml_file(xml_file_path):
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Fehler beim Parsen der XML-Datei: {xml_file_path}")
        return None  # None zurückgeben, um einen Fehler anzugeben

    flugspuren_unix = []
    for trk in root.iter('trk'):
        startzeitpunkt_element = trk.find('p')
        if startzeitpunkt_element is not None:
            startzeitpunkt = int(startzeitpunkt_element.attrib.get('t', 0))  # Startzeitpunkt der Flugspur
            endzeitpunkt = int(trk.find('p[last()]').attrib['t']) + startzeitpunkt  # Endzeitpunkt der Flugspur
            act = (trk.attrib['act'])
            flugspuren_unix.append((act, startzeitpunkt, endzeitpunkt))

    larmereignisse = []
    for ne in root.iter('ne'):
        startzeitpunkt = int(ne.attrib['tstart'])  # Startzeitpunkt des Lärmereignisses
        endzeitpunkt = int(ne.attrib['tstop'])  # Endzeitpunkt des Lärmereignisses
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
                uberschneidungen.append((i + 1, j + 1))

    eindeutige_flugspuren = []
    max_larmpegel = []
    luftfahrzeugtypen = []

    for larmereignis_idx, (actype_larm, start_larm, ende_larm, larm_max_larm) in enumerate(larmereignisse):
        zugeordnete_flugspuren = []

        for flugspur_idx, (act_flugspur, start_flugspur, ende_flugspur) in enumerate(flugspuren_unix):
            if start_larm >= start_flugspur and ende_larm <= ende_flugspur and actype_larm == act_flugspur:
                zugeordnete_flugspuren.append(flugspur_idx + 1)

        if len(zugeordnete_flugspuren) == 1:
            eindeutige_flugspuren.append((larmereignis_idx + 1, zugeordnete_flugspuren[0]))
            max_larmpegel.append(larm_max_larm)
            luftfahrzeugtypen.append(actype_larm)

    df_eindeutige_zuordnung = pd.DataFrame(eindeutige_flugspuren, columns=['Lärmereignis', 'Flugspur'])
    df_eindeutige_zuordnung['Maximaler Lärmpegel'] = max_larmpegel
    df_eindeutige_zuordnung['Flugzeugtyp'] = luftfahrzeugtypen

    df_eindeutige_zuordnung['Maximaler Lärmpegel'] = pd.to_numeric(df_eindeutige_zuordnung['Maximaler Lärmpegel'])

    flugspur_eindeutige_zuordnung = df_eindeutige_zuordnung.groupby('Flugspur')['Maximaler Lärmpegel'].idxmax()
    df_eindeutige_zuordnung = df_eindeutige_zuordnung.loc[flugspur_eindeutige_zuordnung]

    df_eindeutige_zuordnung = df_eindeutige_zuordnung.sort_values(by='Flugspur')
    df_eindeutige_zuordnung.reset_index(drop=True, inplace=True)

    return df_eindeutige_zuordnung, larmereignisse, root

# Durchgehen jeder Datei in den Ordnern
ordner_lärm = '/Users/michaelmerkel/Desktop/Datensatz/Lärm/'
monate_ordner = [ordner for ordner in os.listdir(ordner_lärm) if os.path.isdir(os.path.join(ordner_lärm, ordner))]
mehrere_monate = pd.DataFrame()

# Verarbeite jede XML-Datei für die ersten beiden Monate und füge die Ergebnisse zum DataFrame hinzu
for monat_ordnername in monate_ordner[:13]: 
    monat_ordnerpfad = os.path.join(ordner_lärm, monat_ordnername)
    xml_files = [file for file in os.listdir(monat_ordnerpfad) if file.endswith('.xml')]
    for xml_file in xml_files:
        xml_file_path = os.path.join(monat_ordnerpfad, xml_file)
        result = process_xml_file(xml_file_path)
        if result is None:
            print(f"Überspringe ungültige Datei: {xml_file_path}")
            continue
        ergebnisse_xml, larmereignisse, root = result
        mehrere_monate = pd.concat([mehrere_monate, ergebnisse_xml])
        
        # Extrahiere Messungen für jedes Lärmereignis, das eindeutig zugeordnet wurde
        for index, row in ergebnisse_xml.iterrows():
            larmereignis_idx = row['Lärmereignis'] - 1
            start_larm = int(larmereignisse[larmereignis_idx][1])
            ende_larm = int(larmereignisse[larmereignis_idx][2])
    
            actype_larm = larmereignisse[larmereignis_idx][0]
            larm_max_larm = larmereignisse[larmereignis_idx][3]  # Maximaler Lärmpegel für das Lärmereignis
    
             # nmt
            nmt_element = None
            for nmt in root.iter('nmt'):
                first_time = int(nmt.attrib['firstTime'])
                if first_time <= start_larm:
                    nmt_element = nmt
                    break
                    
             # ne
            ne_element = None
            for ne in root.iter('ne'):
                tstart_ne = int(ne.attrib['tstart'])
                if tstart_ne == start_larm:
                    ne_element = ne
                    break
    
            # Extrahiere das übergeordnete <nmt>-Element von <ne>
            if ne_element is not None:
                nmt_element = None
                for parent in root.iter():
                    if ne_element in list(parent):
                        nmt_element = parent
                        break
            else:
                continue  # Falls das <ne>-Element nicht gefunden wurde, überspringe den Rest der Schleife
    
            # Lärmpegel v
            v_werte = []
            if nmt_element is not None:
                v = nmt_element.find('l').attrib['v'].split(';')
                v_werte.extend(v[start_larm - first_time : ende_larm - first_time])
            
            # nmt - Stationname
            station_name = nmt_element.attrib['name']

            # Kontrolle
            #print("Lärmstation:", station_name)
            #print("Start Lärm:", start_larm)
            #print("Ende Lärm:", ende_larm)
            #print("Anzahl Lärmpegel(soll):", ende_larm - start_larm)
            #print("Flugzeugtyp:", actype_larm)
            #print("Maximaler Lärmpegel:", larm_max_larm)  # Maximaler Lärmpegel für das Lärmereignis
            #print("Anzahl Lärmpegel(ist):", len(v_werte))
            #print("Lärmpegel:", v_werte)
        
    
# LFZ-Typen
anzahl_flugzeugtyp = mehrere_monate["Flugzeugtyp"].value_counts()
print(anzahl_flugzeugtyp)


# In[3]:


# Speichern der gesamten Liste in einem separaten DataFrame
#df_anzahl_flugzeugtyp = pd.DataFrame(anzahl_flugzeugtyp)
#print(df_anzahl_flugzeugtyp)



#anzahl_gefiltert = 'anzahl_flugzeugtyp_gefiltert.csv'
#df_anzahl_flugzeugtyp.to_csv(anzahl_gefiltert, index=False)
# Abbildung zu Lärmpegelverteilung
plt.hist(mehrere_monate["Maximaler Lärmpegel"] /10, bins=20, color='lightgrey', edgecolor='black', linewidth= 0.5)
plt.xlabel('Lärmpegel')
plt.ylabel('Häufigkeit')
plt.title('Häufigkeiten des maximalen Lärmpegels', fontname='Arial')
plt.xticks(fontname='Arial')
plt.yticks(fontname='Arial')
plt.xlim(55, 90)
formatter = ticker.FuncFormatter(lambda x, pos: format(int(x), ','))
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()


# In[4]:


# GRIB
wetter_ordner = '/Users/michaelmerkel/Desktop/Datensatz/Wetter/'
# hhl_munich
munich = "/Users/michaelmerkel/Desktop/Datensatz/Wetter/hhl_munich.grib2"
munich_grid = xr.load_dataset(munich, engine='cfgrib')

# Funktion zur Hinzufügung der Variablen "half_level" und "full_level" zu einem Dataset
def add_half_and_full_levels(ds):
    relevante_h = munich_grid['h']
    relevante_h = relevante_h[14:]
    meter_in_fuß = 3.28084
    relevante_h = relevante_h * meter_in_fuß
    ds['half_level'] = relevante_h
    half_level_values = ds['half_level'].values
    generalVerticalLayer_values = ds.coords['generalVerticalLayer'].values
    latitude_values = ds.coords['latitude'].values
    longitude_values = ds.coords['longitude'].values

    # full_level
    full_level_values = np.empty((len(generalVerticalLayer_values), 
                          len(latitude_values), 
                          len(longitude_values)), dtype=np.float32)
    # Mittelwertberechnung
    for i, gv in enumerate(generalVerticalLayer_values[:51]):
        full_level_values[i, :, :] = (half_level_values[i, :, :] + half_level_values[i+1, :, :]) / 2.0
    ds['full_level'] = (('generalVerticalLayer', 'latitude', 'longitude'), full_level_values)
     # Füge den DataArrays "half_level" und "full_level" die Dimensionen "time" und "step" hinzu
    time_values = ds['time'].values
    step_values = ds['step'].values
    ds_half_level = ds['half_level'].expand_dims(time=time_values, step=step_values)
    ds_full_level = ds['full_level'].expand_dims(time=time_values, step=step_values)
    ds = ds.assign(half_level=ds_half_level, full_level=ds_full_level)
    
    return ds


wetter_monate = [ordner for ordner in os.listdir(wetter_ordner) if os.path.isdir(os.path.join(wetter_ordner, ordner))]
# Schleife über die Monate
for wetter_monat in wetter_monate[:]:
    wetter_monat_pfad = os.path.join(wetter_ordner, wetter_monat)
    grib_files = glob.glob(os.path.join(wetter_monat_pfad, '*.grib2'))

    # Schleife über alle GRIB-Dateien im aktuellen Monat
    for grib_file in grib_files:
        ds = xr.load_dataset(grib_file, engine='cfgrib')
        ds = add_half_and_full_levels(ds)
       #print(ds)


# In[19]:

wetter_ordner = '/Users/michaelmerkel/Desktop/Datensatz/Wetter/'
munich = "/Users/michaelmerkel/Desktop/Datensatz/Wetter/hhl_munich.grib2"
munich_grid = xr.load_dataset(munich, engine='cfgrib')


# Funktion zur Hinzufügung der Variablen "half_level" und "full_level" zu einem Dataset
def add_half_and_full_levels(ds):
    relevante_h = munich_grid['h']
    relevante_h = relevante_h[14:]
    meter_in_fuß = 3.28084
    relevante_h = relevante_h * meter_in_fuß
    ds['half_level'] = relevante_h
    half_level_values = ds['half_level'].values
    generalVerticalLayer_values = ds.coords['generalVerticalLayer'].values
    latitude_values = ds.coords['latitude'].values
    longitude_values = ds.coords['longitude'].values
    full_level_values = np.empty((len(generalVerticalLayer_values), 
                          len(latitude_values), 
                          len(longitude_values)), dtype=np.float32)
    for i, gv in enumerate(generalVerticalLayer_values[:51]):
        full_level_values[i, :, :] = (half_level_values[i, :, :] + half_level_values[i+1, :, :]) / 2.0
    # neue Variable
    ds['full_level'] = (('generalVerticalLayer', 'latitude', 'longitude'), full_level_values)
    # Füge den DataArrays "half_level" und "full_level" die Dimensionen "time" und "step" hinzu  
    time_values = ds['time'].values
    step_values = ds['step'].values
    ds_half_level = ds['half_level'].expand_dims(time=time_values, step=step_values)
    ds_full_level = ds['full_level'].expand_dims(time=time_values, step=step_values)
    ds = ds.assign(half_level=ds_half_level, full_level=ds_full_level)
    
    return ds

# Funktionsdefinitionen
def fix_times(uhrzeit_int):
    i = uhrzeit_int // 3
    j = uhrzeit_int % 3
    return i, j
def sort_ordner_by_date(ordner_name):
    return datetime.datetime.strptime(ordner_name, '%Y-%m')
def sort_grib_files(grib_file):
    date_part = grib_file.split('_')[1][:8]
    return datetime.datetime.strptime(date_part, '%Y%m%d').date()
sorted_grib_files = sorted(grib_files, key=sort_grib_files)
def sort_xml_files(xml_file):
    split_parts = xml_file.split('_')
    date_part = split_parts[0][3:]
    time_part = split_parts[1]
    day, month, year = [int(x) for x in date_part.split('-')]
    hour = int(time_part.split('.')[0])
    return datetime.datetime(year, month, day, hour)
sorted_xml_files = sorted(xml_files, key=sort_xml_files)


wetter_monate = [ordner for ordner in os.listdir(wetter_ordner) if os.path.isdir(os.path.join(wetter_ordner, ordner))]
ordner_lärm = '/Users/michaelmerkel/Desktop/Datensatz/Lärm/'
monate_ordner = [ordner for ordner in os.listdir(ordner_lärm) if os.path.isdir(os.path.join(ordner_lärm, ordner))]
mehrere_monate = pd.DataFrame()
gemeinsame_monate = set(wetter_monate) & set(monate_ordner)
sorted_wetter_monate = sorted(wetter_monate, key=sort_ordner_by_date)
sorted_monate_ordner = sorted(monate_ordner, key=sort_ordner_by_date)

# Schleife über die gemeinsamen Monate
gemeinsame_monate_list = list(gemeinsame_monate)
for monat_ordnername in sorted(gemeinsame_monate, key=sort_ordner_by_date):
    wetter_monat_pfad = os.path.join(wetter_ordner, monat_ordnername)
    monate_ordner_pfad = os.path.join(ordner_lärm, monat_ordnername)
    grib_files = glob.glob(os.path.join(wetter_monat_pfad, '*.grib2'))
    sorted_grib_files = sorted(grib_files, key=sort_grib_files)

    xml_files = [file for file in os.listdir(monate_ordner_pfad) if file.endswith('.xml')]  # Hinzugefügt: XML-Dateien im aktuellen Monat
    sorted_xml_files = sorted(xml_files, key=sort_xml_files)

    for xml_file in sorted_xml_files:
        xml_file_path = os.path.join(monate_ordner_pfad, xml_file)  # Geändert: Pfad zur XML-Datei im aktuellen Monat
        ergebnisse_xml, larmereignisse, root = process_xml_file(xml_file_path)
        if ergebnisse_xml is None:
            continue

        # Datum und Uhrzeit
        if 'ts' in root[0].attrib:
            ts_value = root[0].attrib['ts']
            date_time_xml = datetime.datetime.strptime(ts_value, '%d.%m.%Y %H:%M:%S')
        else:
            continue 

        datum_xml = date_time_xml.date()
        uhrzeit_xml_int = date_time_xml.hour 

        # Von der Uhrzeit der XML auf den nächsten Zeitpunkt der GRIB-Datei zugreifen
        tupel_grib = fix_times(uhrzeit_xml_int)
        zeitpunkt_idx, vorhersage_idx = tupel_grib

        # Schleife über die GRIB-Dateien, um das entsprechende Datum zu finden
        matching_grib_file_path = None
        for grib_file_path in grib_files:
            grib_file_name = os.path.basename(grib_file_path)
            datum_grib_str = grib_file_name.split('_')[1][:8] 
            datum_grib = datetime.datetime.strptime(datum_grib_str, '%Y%m%d').date()
            # Vergleiche das Datum der XML-Datei mit dem Datum der GRIB-Datei
            if datum_xml == datum_grib:
                matching_grib_file_path = grib_file_path
                break

        if matching_grib_file_path is not None:
            print(f"XML-Datei '{xml_file}' wurde der GRIB-Datei '{os.path.basename(matching_grib_file_path)}' zugeordnet.")
            print(f"Zugeordnetes Tupel: (zeitpunkt_idx={zeitpunkt_idx}, vorhersage_idx={vorhersage_idx})")
            ds_grib = xr.load_dataset(matching_grib_file_path, engine='cfgrib')
            ds_grib = add_half_and_full_levels(ds_grib)
            ds_grib = ds_grib.isel(time=zeitpunkt_idx, step=vorhersage_idx)
        else:
            print(f"Keine passende GRIB-Datei gefunden für die XML-Datei '{xml_file}'.")


# In[93]:


# Definition von Funktionen
def fix_times(uhrzeit_int):
    i = uhrzeit_int // 3
    j = uhrzeit_int % 3
    return i, j
def get_absolute_time(start_time, relative_time):
    return start_time + relative_time
def interpolate_coordinates(koordinaten, start_larm, ende_larm):
    latitudes = [coord[0] for coord in koordinaten]
    longitudes = [coord[1] for coord in koordinaten]
    altitudes = [coord[2] for coord in koordinaten]
    time_points = [coord[3] for coord in koordinaten]
    speeds = [coord[4] for coord in koordinaten]

    interp_latitude = interpolate.interp1d(time_points, latitudes, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_longitude = interpolate.interp1d(time_points, longitudes, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_altitude = interpolate.interp1d(time_points, altitudes, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_speed = interpolate.interp1d(time_points, speeds, kind='linear', bounds_error=False, fill_value='extrapolate')
    time_range = list(range(start_larm, ende_larm + 1))
    interpolated_coords = []
    for i in time_range:
        lat = interp_latitude(i)
        lon = interp_longitude(i)
        alt = interp_altitude(i)
        spd = interp_speed(i)
        interpolated_coords.append((lat, lon, alt, spd, i))

    return interpolated_coords
def get_grib_point_value(ds_grib, lat, lon, alt, time_idx, step_idx):
    try:
        lat_idx = abs(ds_grib.latitude - lat).argmin()
        lon_idx = abs(ds_grib.longitude - lon).argmin()
        lev_idx = abs(ds_grib.full_level[ :, lat_idx, lon_idx] - alt).argmin()
        temp = round(ds_grib.t[lev_idx, lat_idx, lon_idx].values.item() - 273.15, 2)
        u = ds_grib.u[lev_idx, lat_idx, lon_idx].values.item()
        v = ds_grib.v[lev_idx, lat_idx, lon_idx].values.item()    
        wind_wz = round(ds_grib.wz[lev_idx, lat_idx, lon_idx].values.item(), 2)  
    except (IndexError, TypeError):
        return None     
    windrichtung = np.arctan2(v, u) * 180 / np.pi 
    windrichtung = (windrichtung + 360) % 360 
    windrichtung = round(windrichtung, 2)
    windgeschwindigkeit = np.sqrt(u**2 + v**2)
    windgeschwindigkeit = round(windgeschwindigkeit, 2)
    return (lat, lon, alt), temp, windrichtung, windgeschwindigkeit, wind_wz
def sort_ordner_by_date(ordner_name):
    return datetime.datetime.strptime(ordner_name, '%Y-%m')
def sort_grib_files(grib_file):
    date_part = grib_file.split('_')[1][:8]  # Datum aus dem Dateinamen extrahieren
    return datetime.datetime.strptime(date_part, '%Y%m%d').date()  # Datum in ein Datumsobjekt konvertieren
sorted_grib_files = sorted(grib_files, key=sort_grib_files)
def sort_xml_files(xml_file):
    # Datum und Uhrzeit aus dem Dateinamen extrahieren
    split_parts = xml_file.split('_')
    date_part = split_parts[0][3:]  # Dies schneidet "muc" ab und behält den Rest für das Datum
    time_part = split_parts[1]
    day, month, year = [int(x) for x in date_part.split('-')]
    hour = int(time_part.split('.')[0])  # Falls die Dateiendung nicht berücksichtigt wird
    return datetime.datetime(year, month, day, hour)
sorted_xml_files = sorted(xml_files, key=sort_xml_files)

# Ordner und Dateien durchgehen
wetter_monate = [ordner for ordner in os.listdir(wetter_ordner) if os.path.isdir(os.path.join(wetter_ordner, ordner))]
ordner_lärm = '/Users/michaelmerkel/Desktop/Datensatz/Lärm/'
monate_ordner = [ordner for ordner in os.listdir(ordner_lärm) if os.path.isdir(os.path.join(ordner_lärm, ordner))]
gemeinsame_monate = set(wetter_monate) & set(monate_ordner)
sorted_wetter_monate = sorted(wetter_monate, key=sort_ordner_by_date)
sorted_monate_ordner = sorted(monate_ordner, key=sort_ordner_by_date)
Tabelle = pd.DataFrame(columns=["ID","Datum","Uhrzeit","Flugzeugtyp","Lärmstation","Jahreszeit","Lärm","Temperatur","Windrichtung","Windgeschwindigkeit","Windgeschwindigkeit_vertikal","Geschwindigkeit", "Höhe","Breitengrad","Längengrad"])
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

# Schleife über die gemeinsamen Monate
gemeinsame_monate_list = list(gemeinsame_monate)
gemeinsame_monate_list_sorted = sorted(gemeinsame_monate, key=sort_ordner_by_date)
counter = 0
for monat_ordnername in gemeinsame_monate_list_sorted:
    
    wetter_monat_pfad = os.path.join(wetter_ordner, monat_ordnername)
    monate_ordner_pfad = os.path.join(ordner_lärm, monat_ordnername)
    grib_files = glob.glob(os.path.join(wetter_monat_pfad, '*.grib2'))
    sorted_grib_files = sorted(grib_files, key=sort_grib_files)

    xml_files = [file for file in os.listdir(monate_ordner_pfad) if file.endswith('.xml')]  
    sorted_xml_files = sorted(xml_files, key=sort_xml_files)

    for xml_file in sorted_xml_files:
        xml_file_path = os.path.join(monate_ordner_pfad, xml_file) 
        result = process_xml_file(xml_file_path)
        if result is None:
            print(f"Fehler bei der Verarbeitung der XML-Datei: {xml_file_path}")
            continue
        ergebnisse_xml, larmereignisse, root = result

        # Datum und Uhrzeit
        if 'ts' in root[0].attrib:
            ts_value = root[0].attrib['ts']
            date_time_xml = datetime.datetime.strptime(ts_value, '%d.%m.%Y %H:%M:%S')
        else:
            continue 
        # Zeiverschiebung
        if 'utc' in root[0].attrib:
            utc_offset_seconds = int(root[0].attrib['utc'])
        else:
            continue 
        utc_offset_hours = utc_offset_seconds // 3600
        date_time_xml_utc = date_time_xml - timedelta(hours=utc_offset_hours)
        datum_xml = date_time_xml_utc.date() 
        uhrzeit_xml_int = date_time_xml_utc.hour
        #Kontrolle
        #print(datum_xml)
        #print(uhrzeit_xml_int)

        monat = datum_xml.month
        jahreszeit = jahreszeiten.get(monat)

        # Von der Uhrzeit der XML auf den nächsten Zeitpunkt der GRIB-Datei zugreifen
        tupel_grib = fix_times(uhrzeit_xml_int) 
        zeitpunkt_idx, vorhersage_idx = tupel_grib  

        # Schleife über die GRIB-Dateien, um das entsprechende Datum zu finden
        matching_grib_file_path = None
        for grib_file_path in grib_files:
            grib_file_name = os.path.basename(grib_file_path)
            datum_grib_str = grib_file_name.split('_')[1] 
            datum_grib = datetime.datetime.strptime(datum_grib_str, '%Y%m%d').date() 
            if datum_xml == datum_grib:
                matching_grib_file_path = grib_file_path
                break

        if matching_grib_file_path is not None:
            try:
                #print(f"XML-Datei '{xml_file}' wurde der GRIB-Datei '{os.path.basename(matching_grib_file_path)}' zugeordnet.")
                #print(f"Zugeordnetes Tupel: (zeitpunkt_idx={zeitpunkt_idx}, vorhersage_idx={vorhersage_idx})")
                ds_grib = xr.load_dataset(matching_grib_file_path, engine='cfgrib')
                ds_grib = add_half_and_full_levels(ds_grib)
                ds_grib = ds_grib.isel(time=zeitpunkt_idx, step=vorhersage_idx)
            except (IndexError, TypeError):
                pass
        else:
            #print(f"Keine passende GRIB-Datei gefunden für die XML-Datei '{xml_file}'.")
            continue

        if ergebnisse_xml is not None:
            # Extrahiere Messungen für jedes Lärmereignis, das eindeutig zugeordnet wurde
            for index, row in ergebnisse_xml.iterrows():
                larmereignis_idx = row['Lärmereignis'] - 1
                start_larm = int(larmereignisse[larmereignis_idx][1])
                ende_larm = int(larmereignisse[larmereignis_idx][2])
                actype_larm = larmereignisse[larmereignis_idx][0]
                larm_max_larm = larmereignisse[larmereignis_idx][3]
                # nmt
                nmt_element = None
                for nmt in root.iter('nmt'):
                    first_time = int(nmt.attrib['firstTime'])
                    if first_time <= start_larm:
                        nmt_element = nmt
                        break
                # ne
                ne_element = None
                for ne in root.iter('ne'):
                    tstart_ne = int(ne.attrib['tstart'])
                    if tstart_ne == start_larm:
                        ne_element = ne
                        break
                # Extrahiere das übergeordnete <nmt>-Element von <ne>
                if ne_element is not None:
                    nmt_element = None
                    for parent in root.iter():
                        if ne_element in list(parent):
                            nmt_element = parent
                            break
                else:
                    continue 
                # Lärmpegel
                v_werte = []
                if nmt_element is not None:
                    v = nmt_element.find('l').attrib['v'].split(';')
                    v_werte.extend(v[start_larm - first_time : ende_larm - first_time + 1])
                # Stationsname
                station_name = nmt_element.attrib['name']
                # Koordinaten
                if ne_element is not None:
                    flugspur_index = int(row['Flugspur']) + 1
                    flugspur_element = root[flugspur_index]
                    try:
                        start_time = int(flugspur_element.find('p').attrib['t'])
                    except AttributeError:
                        continue
                    koordinaten = [(round_to_even_decimal(float(p.attrib['a'])),
                                    round_to_even_decimal(float(p.attrib['n'])),
                                    round(float(p.attrib['l'])),
                                    get_absolute_time(start_time, int(p.attrib['t'])),
                                    float(p.attrib['s'])) for p in flugspur_element.iter('p')]      
                    original_koordinaten = [(float(p.attrib['a']),
                                            float(p.attrib['n']),
                                            float(p.attrib['l']),
                                            get_absolute_time(start_time, int(p.attrib['t'])),
                                            float(p.attrib['s'])) for p in flugspur_element.iter('p')]

                    # Kontrolle
                    koordinaten_larmereignis = [coord for coord in koordinaten if start_larm <= coord[3] <= ende_larm]
                    if len(koordinaten_larmereignis) == 0:
                        #print(f"Keine Koordinaten für Lärmereignis {index} im Zeitraum {start_larm}-{ende_larm}. Überspringe dieses Lärmereignis.")
                        continue
                    interpolated_coords = interpolate_coordinates(koordinaten_larmereignis, start_larm, ende_larm)
                    interpolated_original_coords = interpolate_coordinates(original_koordinaten, start_larm, ende_larm)
                    # Kontrolle
                    if len(v_werte) != len(interpolated_coords) or len(v_werte) != len(interpolated_original_coords):
                        #print(f"Anzahl der v-Werte und Koordinaten stimmt nicht überein für Lärmereignis {index}. Überspringe dieses Lärmereignis.")
                        continue

                    #print("Flugzeugtyp:", actype_larm)
                    #print("Lärmstation:", station_name)
                    #print("Maximaler Lärmpegel:", larm_max_larm)
                    #print("Anzahl der v-Werte:", len(v_werte))
                    #print("Anzahl der Koordinaten:", len(interpolated_coords))
                    #print("Start der Messung:", first_time)
                    #print("Start Lärm:", start_larm)
                    #print("Ende Lärm:", ende_larm)
                    #print("Jahreszeit:", jahreszeit)

                    for i, (coord, original_coord) in enumerate(zip(interpolated_coords, interpolated_original_coords)):
                        if all(np.isfinite(coord)):  # Überprüfe, ob alle Werte der Koordinate finite (kein NaN) sind
                            lat_rounded = round_to_even_decimal(coord[0])
                            lon_rounded = round_to_even_decimal(coord[1])
                            alt_rounded = round(coord[2].item())
                            noise = int(v_werte[i])
                            geschwindigkeit = coord[3]
                            
                            lat_original = original_coord[0]
                            lon_original = original_coord[1]
                            alt_original = original_coord[2]

                            result = get_grib_point_value(ds_grib, lat_rounded, lon_rounded, alt_rounded, zeitpunkt_idx, vorhersage_idx)
                        if result is None or len(result) !=5:
                            #print("Überspringe Datei, weil GRIB Werte ungültig sind")
                            continue
                        #print(f"  Geschwindigkeit: {geschwindigkeit}, Zeit: {coord[4]}, Längengrad: {lon_rounded}, Breitengrad: {lat_rounded}, Höhe: {alt_rounded}, Lärmpegel: {noise}, Längengrad_o: {lon_original}, Breitengrad_o: {lat_original}, Höhe_o: {alt_original}")
                        print(f" Datum: {datum_xml}")

                        # Wetterinformationen
                        coord, value, windrichtung, windgeschwindigkeit, wind_wz = result
                        #print(f"  Temperatur: {value}, Windrichtung: {windrichtung}, Windgeschwindigkeit: {windgeschwindigkeit}, Windgeschwindigkeit (vertikal): {wind_wz}")             
                        data_list = []
                        dictionary = {"ID":counter,
                                      "Datum": datum_xml,
                                      "Uhrzeit": uhrzeit_xml_int,
                                      "Flugzeugtyp":actype_larm,
                                      "Lärmstation":station_name,
                                      "Jahreszeit":jahreszeit,
                                      "Lärm":noise,
                                      "Temperatur":value,
                                      "Windrichtung":windrichtung,
                                      "Windgeschwindigkeit":windgeschwindigkeit,
                                      "Windgeschwindigkeit_vertikal":wind_wz,
                                      "Geschwindigkeit": geschwindigkeit,
                                      "Höhe":alt_original,
                                      "Breitengrad":lat_original,
                                      "Längengrad":lon_original}
                        data_list.append(dictionary) 
                        Tabelle = pd.concat([Tabelle, pd.DataFrame(data_list)], ignore_index=True) 
                    counter+=1


# In[94]:


Tabelle

#%%
Tabelle.to_csv('/Users/michaelmerkel/Desktop/Alles/Datensatz_neu3.csv',index=False)


# In[ ]:





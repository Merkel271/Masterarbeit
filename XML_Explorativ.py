#!/usr/bin/env python
# coding: utf-8

# In[16]:


import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import datetime


# In[17]:


#auf Datensatz zugreifen und importieren

Jan514 = '/Users/michaelmerkel/Desktop/Datensatz/Lärm/2023-01/muc05-01-2023_14.xml'
tree = ET.parse(Jan514) #parse = einlesen
root = tree.getroot()


# In[18]:


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
    startzeitpunkt = int(trk.find('p').attrib['t'])  # Startzeitpunkt der Flugspur
    endzeitpunkt = int(trk.find('p[last()]').attrib['t']) + startzeitpunkt  # Endzeitpunkt der Flugspur
    act = (trk.attrib['act']) #Flugzeugtyp
    a_d = (trk.attrib['ad']) #An-/Abflug
    flugspuren_unix.append((act, startzeitpunkt, endzeitpunkt))

larmereignisse = []
for ne in root.iter('ne'):
    startzeitpunkt = int(ne.attrib['tstart'])  # Startzeitpunkt des Lärmereignisses
    endzeitpunkt = int(ne.attrib['tstop'])  # Endzeitpunkt des Lärmereignisses
    actype = (ne.attrib['actype']) # Flugzeugtyp
    larm_max = (ne.attrib['las']) # Maximaler Lärmpegel
    larmereignisse.append((actype, startzeitpunkt, endzeitpunkt, larm_max))

    
# Erstellen des DataFrames für die Flugspuren und Lärmereignisse
flight_paths_df = pd.DataFrame(flugspuren_unix, columns=['Flugzeugtyp', 'Startzeitpunkt', 'Endzeitpunkt'])
noise_events_df = pd.DataFrame(larmereignisse, columns=['Flugzeugtyp', 'Startzeitpunkt', 'Endzeitpunkt', 'Maximaler Lärm'])

# Anzeigen der Tabelle für die Flugspuren
#print("Flugspuren:")
#print(flight_paths_df)

# Anzeigen der Tabelle für die Lärmereignisse
#print("Lärmereignisse:")
#print(noise_events_df)
    
uberschneidungen = []
for i, (act_flugspur, start_flugspur, ende_flugspur) in enumerate(flugspuren_unix):
    for j, (actype_larm, start_larm, ende_larm, larm_max_larm) in enumerate(larmereignisse):
        if start_larm >= start_flugspur and start_larm <= ende_flugspur and ende_larm >= start_flugspur and ende_larm <= ende_flugspur and act_flugspur == actype_larm:
            uberschneidungen.append((i+1, j+1))


# Ausgabe der Anzahl der Überschneidungen
anzahl_uberschneidungen = len(uberschneidungen)
#print("Anzahl der Überschneidungen:", anzahl_uberschneidungen)
            
# Ausgabe der Überschneidungen
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
abflug_anflug = [] #Liste für Ab-/Anflug

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
        max_larmpegel.append(larm_max_larm)  # Max Lärmpegel hinzufügen
        
        # Den Luftfahrzeugtyp (Flugzeugtyp) basierend auf dem Lärmereignis hinzufügen
        luftfahrzeugtypen.append(actype_larm)
        
        # Zuordnung Ab-/Anflug
        abflug_anflug.append(a_d)
        
    
        
# Erstellen des DataFrames für die eindeutig zugeordneten Flugspuren und Lärmereignisse
df_eindeutige_zuordnung = pd.DataFrame(eindeutige_flugspuren, columns=['Lärmereignis', 'Flugspur'])
df_eindeutige_zuordnung['Maximaler Lärmpegel'] = max_larmpegel  # Max Lärmpegel als neue Spalte hinzufügen
df_eindeutige_zuordnung['Flugzeugtyp'] = luftfahrzeugtypen  # Flugzeugtypen hinzufügen


# Konvertiere die Spalte 'Maximaler Lärmpegel' in numerischen Datentyp, falls nicht bereits erfolgt
df_eindeutige_zuordnung['Maximaler Lärmpegel'] = pd.to_numeric(df_eindeutige_zuordnung['Maximaler Lärmpegel'], errors='coerce')

# Gruppiere den DataFrame nach "Flugspur" und ermittele den maximalen Lärmpegel für jede Flugspur
max_larmpegel_indices = df_eindeutige_zuordnung.groupby('Flugspur')['Maximaler Lärmpegel'].idxmax()

# Filtere den DataFrame, um nur die Zeilen mit den maximalen Lärmpegeln zu behalten
df_eindeutige_zuordnung = df_eindeutige_zuordnung.loc[max_larmpegel_indices]

# Sortiere nach Flugspur
df_eindeutige_zuordnung = df_eindeutige_zuordnung.sort_values(by='Flugspur')
df_eindeutige_zuordnung.reset_index(drop=True, inplace=True)
# Spaltenreihenfolge ändern
column_order = ['Flugspur', 'Lärmereignis', 'Maximaler Lärmpegel', 'Flugzeugtyp']

# DataFrame mit neuer Spaltenreihenfolge erstellen
df_eindeutige_zuordnung = df_eindeutige_zuordnung[column_order]

# Zeige den aktualisierten DataFrame an
df_eindeutige_zuordnung


# In[19]:


#bekannte Parameter werden herausgefiltert

parameters_track_records= ["a","n","h","l","s","t"]
parameters_tracks = ["act", "ad", "airport", "airportInt", "ap", "azbGroup", "carr", "cs", "fNo", "id", "reg", "rwy", "ssr"]
parameters_nmt = ["firstTime", "height", "id", "lat", "lon", "lstart", "lstartnight", "mob", "name", "short", "status"]
parameters_l = ["v"]
parameters_ne = ["actype","las","tlas","tstart","tstop"]

#wichtige Parameter aus Track-Daten
parameters_ad = ["ad","act","azbGroup"]

#Parameter Name der Lärmstation
parameters_ls = ["name"]


# In[20]:


#Zähler - wie viele Flüge wurden im Datensatz erfasst?

counter=0
for neighbor in root.iter('trk'):
    #+print(neighbor.attrib)
    counter = counter + 1
#print(counter)

#Fehlermeldung sagt nur, dass bald .append veraltet ist und in neueren Versionen mit .concat gearbeitet werden muss
#df_ne = pd.DataFrame(columns=parameters_ne)
#counter = 1
#for neighbor in root.iter('ne'):
    #series = pd.Series(neighbor.attrib,index=df_ne.columns, name=str(counter))
    #df_ne = df_ne.append(series)  
    #counter+=1
#df_ne


# In[21]:


#Tabelle mit allen Lärmereignissen (ne) von allen Lärmmessstationen im Datensatz

df_ne = pd.DataFrame(columns=parameters_ne)

for neighbor in root.iter('ne'):
    series = pd.Series(neighbor.attrib, index=df_ne.columns)
    df_ne = pd.concat([df_ne, series.to_frame().T], ignore_index=True)
    df_ne.index = df_ne.index + 1

df_ne


# In[22]:


#Tabelle mit den Informationen zu den An- oder Abflügen aus den trk-Daten

df_ad = pd.DataFrame(columns=parameters_ad)

for neighbor in root.iter('trk'):
    series = pd.Series(neighbor.attrib,index=df_ad.columns)
    df_ad = pd.concat([df_ad, series.to_frame().T], ignore_index=True)
    df_ad.index = df_ad.index + 1
    
df_ad


# In[23]:


df_ls = pd.DataFrame(columns=parameters_ls)

for neighbor in root.iter('nmt'):
    series = pd.Series(neighbor.attrib,index=df_ls.columns)
    df_ls = pd.concat([df_ls, series.to_frame().T], ignore_index=True)
    df_ls.index = df_ls.index + 1
#df_ls


# In[24]:


#Versuch, mit pd.merge zu arbeiten
#Die Flugzeugtypen wurden zusammengefasst, allerdings sind nun über 1400 Zeilen entstanden
#hier ist wahrscheinlich jede Station drin, sodass sich die Flugspuren überschneiden

merged_df = pd.merge(df_ne, df_ad, left_on='actype', right_on='act')
merged_df = merged_df.drop(['act'], axis=1)
merged_df

#Sortieren nach tstart
#pd.set_option('display.max_rows', 20) #zeigt alle Zeilen an, die Zahl auf None setzen

df_sortiert = merged_df.sort_values('tstart')
df_sortiert


# In[25]:


#Gruppierung der Daten
#hier ist nun der maximale Pegel für jeden Flugzeugtypen, sortiert nach Ab- und Abflug
#Problem: es ist nur der maximal gemessene Pegel pro Stunde, nicht für jeden Flug
#Problem: die Zuordnung funktioniert nicht ganz, da A und D immer denselben las haben

df_gruppe = merged_df.groupby(['actype', 'ad'])['las'].max().reset_index()
df_gruppe


# In[58]:


#Die Daten für den maximalen Lärmpegel, den Start- und Endzeitpunkt eines Lärmereignisses waren Objekte und mussten erst in Zahlenwerte transformiert werden

y = [x for x in range(len(start))]

start = df_ne["tstart"]
start = list(map(int, start))

stop = df_ne["tstop"]
stop = list(map(int, stop))

#y = df_ne["las"]
#y = list(map(int, y))


#Darstellung der Lärmereignisse im zeitlichen (horizontalen) Verlauf. 

fig, ax = plt.subplots()
for i in range(len(start)):
    e = start[i]
    ax.hlines(y=y[i], xmin=start[i], xmax=stop[i], linewidth=2, color='darkblue')

plt.title("Zeitspanne einzelner Lärmereignisse")    
plt.xlabel('Zeitpunkt(Unixzeit)')  # Beschriftung der X-Achse
plt.ylabel('Lärmereignis')

    
#plt.xlim(0,22)
#plt.ylim(0,10)
plt.savefig('/Users/michaelmerkel/Desktop/Lärmereignisse_PLOT', dpi = 200)
plt.show()


# In[57]:


y = [x for x in range(len(start_stop))]


#Darstellung der Flugspuren im zeitlichen (horizontalen) Verlauf. 

fig, ax = plt.subplots()
for i in range(len(start_stop)):
    e = start_stop[i]
    ax.hlines(y=y[i], xmin=int(e[0]), xmax=(int(e[0])+int(e[1])), linewidth=2, color='darkblue')

plt.title("Zeitspanne einzelner Flugspuren")
plt.xlabel('Zeitpunkt(Unixzeit)')  # Beschriftung der X-Achse
plt.ylabel('Flugspur')
plt.ylim(-2,72) 
plt.savefig('/Users/michaelmerkel/Desktop/Flugspuren_PLOT', dpi=200)
plt.show()


# In[29]:


ls = [type(item) for item in start]


# In[30]:


#df_ne["actype"].tolist()


# In[31]:


df_nmt = pd.DataFrame(columns=parameters_nmt)

for neighbor in root.iter('nmt'):
    series = pd.Series(neighbor.attrib,index=df_nmt.columns)
    df_nmt = pd.concat([df_nmt, series.to_frame().T], ignore_index=True)
    df_nmt.index = df_nmt.index + 1
df_nmt


# In[32]:


df_tracks = pd.DataFrame(columns=parameters_tracks)
for neighbor in root.iter('trk'):
    series = pd.Series(neighbor.attrib,index=df_tracks.columns)
    df_tracks = pd.concat([df_tracks, series.to_frame().T], ignore_index=True)
    df_tracks.index = df_tracks.index + 1

df_tracks


# In[33]:


#Start-Zeitpunkt für jede Flugspur und Dauer in s

liste = []
for neighbor in root.iter('p'):
    liste.append(neighbor.attrib['t'])

start_stop = []
start = liste[0]
for i in range(1,len(liste)):
    e = liste[i]
    if len(e)>=6:
        start_new = e
        stop =  liste[i-1]
        #print(stop)
        start_stop.append([start,stop])
        start = start_new
    if i==(len(liste)-1):
        stop = liste[i]
        #print(start)
        #print(stop)
        start_stop.append([start,stop])
start_stop


# In[34]:


y = [x for x in range(len(start_stop))]


#Darstellung der Flugspuren im zeitlichen (horizontalen) Verlauf. 

fig, ax = plt.subplots()
for i in range(len(start_stop)):
    e = start_stop[i]
    ax.hlines(y=y[i], xmin=int(e[0]), xmax=(int(e[0])+int(e[1])), linewidth=2, color='blue')

plt.ylim(-2,72) 
plt.show()


# In[22]:


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

# Ausgabe der Start-Stop-Zeitstempel im UNIX-Code-Format für jede Flugspur und jeden Zeitstempel
#for i, flugspur in enumerate(flugspuren_unix):
#     print(f"Flugspur {i+1}:")
#     for unix_timestamp in flugspur:
#     print(unix_timestamp)
        

# nur Start- Endzeitpunkt für jede Flugspur
flugspuren_unix = []
for trk in root.iter('trk'):
    startzeitpunkt = int(trk.find('p').attrib['t'])  # Startzeitpunkt der Flugspur
    endzeitpunkt = int(trk.find('p[last()]').attrib['t']) + startzeitpunkt  # Endzeitpunkt der Flugspur
    act = (trk.attrib['act'])
    flugspuren_unix.append((act, startzeitpunkt, endzeitpunkt))

# Ausgabe des Start- und Endzeitpunkts für jede Flugspur
for i, (act, startzeitpunkt, endzeitpunkt) in enumerate(flugspuren_unix):
    print(f"Flugspur {i+1}:")
    print("Flugzeugtyp:", act)
    print("Start:", startzeitpunkt)
    print("Ende:", endzeitpunkt)
    print("---")


    


# In[23]:


larmereignisse = []
for ne in root.iter('ne'):
    startzeitpunkt = int(ne.attrib['tstart'])  # Startzeitpunkt des Lärmereignisses
    endzeitpunkt = int(ne.attrib['tstop'])  # Endzeitpunkt des Lärmereignisses
    actype = (ne.attrib['actype'])
    larmereignisse.append((actype, startzeitpunkt, endzeitpunkt))

# Ausgabe des Start- und Endzeitpunkts für jedes Lärmereignis
for i, (actype, startzeitpunkt, endzeitpunkt) in enumerate(larmereignisse):
    print(f"Lärmereignis {i+1}:")
    print("Flugzeugtyp:", actype)
    print("Startzeitpunkt:", startzeitpunkt)
    print("Endzeitpunkt:", endzeitpunkt)
    print("---")


# In[24]:


uberschneidungen = []
for i, (act_flugspur, start_flugspur, ende_flugspur) in enumerate(flugspuren_unix):
    for j, (actype_larm, start_larm, ende_larm) in enumerate(larmereignisse):
        if start_larm >= start_larm and start_larm <= ende_flugspur and ende_larm >= start_flugspur and ende_larm <= ende_flugspur and act_flugspur == actype_larm:
            uberschneidungen.append((i+1, j+1))

# Ausgabe der Anzahl der Überschneidungen
anzahl_uberschneidungen = len(uberschneidungen)
print("Anzahl der Überschneidungen:", anzahl_uberschneidungen)
            
# Ausgabe der Überschneidungen
max_lärmereignis_idx = 0
lärmereignis_häufigkeit = {}
for flugspur_idx, larmereignis_idx in uberschneidungen:
    if larmereignis_idx > max_lärmereignis_idx:
        max_lärmereignis_idx = larmereignis_idx
    print(f"Flugspur {flugspur_idx} und Lärmereignis {larmereignis_idx} überschneiden sich.")



# In[25]:


wert_dict = {}
for i in range(max_lärmereignis_idx+1):
    wert_dict[i] = 0
for flugspur_idx, larmereignis_idx in uberschneidungen:
    wert_dict[larmereignis_idx] += 1
schluessel = list(wert_dict.keys())
werte = list(wert_dict.values())
plt.figure(figsize=(15, 6))
plt.bar(schluessel, werte)
# Beschrifte die Achsen und das Diagramm
plt.xlabel('Schlüssel')
plt.ylabel('Werte')
plt.title('Balkendiagramm')

# Zeige das Diagramm an
plt.show()


# In[26]:


plt.hist(werte)

# Beschrifte die Achsen und das Diagramm
plt.xlabel('Y-Werte')
plt.ylabel('Häufigkeit')
plt.title('Histogramm der Y-Werte')

# Zeige das Histogramm an
plt.show()


# In[27]:


#for track in root:
    #print(track.tag, track.attrib)


# In[28]:


timestamp = 1641387300
date = datetime.datetime.fromtimestamp(timestamp)
print(date)


# In[ ]:





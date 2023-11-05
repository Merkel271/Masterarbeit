# %%
from pathlib import Path

import pandas as pd
import pyproj
from ifl.noise.ecac import ECAC
from ifl.noise.receptor import Receptor
from ifl.noise.segment import Segment

# %%
transformer = pyproj.Transformer.from_crs(  # TODO: for Munich, Mercator may be better
    "EPSG:4326",
    "EPSG:32632",  # UTM 32N lon [0.0, 6.0] / lat [12.0, 84.0]
    always_xy=True
)

input_file = Path("/Users/michaelmerkel/Desktop/Alles/N_Szenarien/8_Szenarien_Meter/condition_8_D_no_wind.csv")
df = (
    pd.read_csv(input_file)
    .rename(
        {
            'Breitengrad': 'latitude',
            'Längengrad': 'longitude',
        },
        axis="columns",
    )
    .assign(
        x=lambda df: df['longitude'].where(
            df['longitude'] == 0,
            transformer.transform(df['longitude'], df['latitude'])[0]
        ),
        y=lambda df: df['latitude'].where(
            df['latitude'] == 0,
            transformer.transform(df['longitude'], df['latitude'])[1]
        ),
    )
)

# %%
# UTM(32) Koordinaten der Lärmmessstationen Flughafen München
r = {
    'Achering': Receptor(
        700396.626,
        5357824.477,
        456.0
    ),
    'Asenkofen': Receptor(
        711197.881,
        5367253.491,
        486.0
    ),
    'Attaching': Receptor(
        705795.940,
        5361387.161,
        442.0
    ),
    'Brandstadel': Receptor(
        700981.526,
        5354742.414,
        462.0
    ),
    'Eitting': Receptor(
        713719.351,
        5360345.216,
        447.0
    ),
    'Fahrenzhausen': Receptor(
        689915.871,
        5358952.435,
        456.0
    ),
    'Glaslern': Receptor(
        717419.410,
        5362246.992,
        438.0
    ),
    'Hallbergmoos': Receptor(
        704147.116,
        5356935.295,
        459.0
    ),
    'Massenhausen': Receptor(
        695241.725,
        5358866.087,
        487.0
    ),
    'Mintraching': Receptor(
        698831.970,
        5355284.494,
        461.0
    ),
    'Neufahrn': Receptor(
        696796.193,
        5355486.357,
        461.0
    ),
    'Pulling': Receptor(
        700798.301,
        5359880.272,
        452.0
    ),
    'Reisen': Receptor(
        713365.950,
        5358391.292,
        456.0
    ),
    'Schwaig': Receptor(
        709610.351,
        5358365.912,
        447.0
    ),
    'Viehlaßmoos': Receptor(
        712894.629,
        5363767.089,
        434.0
    ),
    'Pallhausen': Receptor(
        696773.241,
        5361423.052,
        451.0
    ),
    
    
}

def process_flight(df):
    ac_type = "A319-131"  # TODO: fetch from df
    ecac = ECAC(
        Path("..") / "resources" / "noise" / ac_type / "NPD_data.csv",
        101325.0,  # TODO: this may be improved using METAR data
        288.15
    )
    noise = []
    for i in range(len(df) - 1):
        begin = df.iloc[i, :]
        end = df.iloc[i + 1, :]
        
        # Bestimme operation_type basierend auf den Werten in deinem DataFrame
        operation_type = "A" if end['An-/Abflug_A'] == 1 else ("D" if end['An-/Abflug_D'] == 1 else None)

        # wegen Padding werden Sequenzen übersprungen, wenn sowohl A als auch D gleich Null sind
        if operation_type is None:
            continue

        thrust = 125000 if operation_type == "D" else 9000  # TODO: select IDLE / MTOW Thrust depending on flight phase
        
        s = Segment(
            begin['x'],
            begin['y'],
            begin['Höhe'],
            begin['Geschwindigkeit'],
            thrust,
            end['x'],
            end['y'],
            end['Höhe'],
            end['Geschwindigkeit'],
            thrust,
            operation_type,
            "CLIMB" if operation_type == "D" else "DESCENT",
        )
        # Finde den richtigen Schlüssel für die Lärmstation
        lärmstation_key = None
        for key in r.keys():
            one_hot_key = f"Lärmstation_{key}"
            if begin[one_hot_key] == 1:
                lärmstation_key = key
                break
        noise.append(
            ecac.calculateSegmentSEL(s, r[lärmstation_key])
        )
    return noise


noise = df.groupby("ID").apply(process_flight)

# %%
noise
# %%


# %%
# Umwandlung in einen DataFrame
noise_e = pd.DataFrame(noise.tolist(), index=noise.index).transpose()

# Spaltennamen ersetzen
noise_e.columns = range(1, len(noise_e.columns) + 1)

noise_e

# %%
noise_e.to_csv("/Users/michaelmerkel/Desktop/Alles/N_ECAC_Modellierung/S8ECAC.csv")

# %%

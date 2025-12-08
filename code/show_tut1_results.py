import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

# ##########################################################################
# 1. KONFIGURATION
# Definiere den relativen Pfad zur NetCDF-Datei VOM PYPSA-EUR HAUPTVERZEICHNIS aus.
RELATIVER_PFAD_ZUR_DATEI = "pypsa-eur/results/test-elec/networks/base_s_6_elec_.nc"
# ##########################################################################

# Ermittle den absoluten Pfad zur Netzwerkdatei, basierend auf dem Skript-Ort.
# Das Skript geht davon aus, dass das PyPSA-EUR-Repo im Verzeichnis 'store2hydro-tpa' liegt.
SKRIPT_DIR = Path(__file__).parent
NETZWERK_PFAD = SKRIPT_DIR / RELATIVER_PFAD_ZUR_DATEI

# Lade das PyPSA-Netzwerk
try:
    net = pypsa.Network(NETZWERK_PFAD)
    print(f"Netzwerk erfolgreich geladen von: {NETZWERK_PFAD}")
except FileNotFoundError:
    # Der zweite Fehler tritt auf. Nutze den korrekten Pfad.
    print(f"FEHLER: Netzwerkdatei nicht gefunden unter {NETZWERK_PFAD}")
    print("Bitte Pfad in RELATIVER_PFAD_ZUR_DATEI korrigieren.")
    exit()
except Exception as e:
    print(f"FEHLER beim Laden des Netzwerks: {e}")
    exit()

# Erstelle die Karte
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([-10, 35, 30, 70], crs=ccrs.PlateCarree()) # Fokus auf Europa

# Korrektur für Fehler A: Nutze add_feature für Ländergrenzen
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='lightgray', alpha=0.5)

# Karten-Basiselemente
ax.coastlines(resolution='50m', color='black')
# ax.gridlines(draw_labels=True) # Gridlines sind optional

# 1. Leitungen (Lines) zeichnen
for line_name, line in net.lines.iterrows():
    bus0 = net.buses.loc[line.bus0]
    bus1 = net.buses.loc[line.bus1]

    ax.plot(
        [bus0.x, bus1.x],
        [bus0.y, bus1.y],
        color='darkblue',
        linewidth=0.5,
        alpha=0.6,
        transform=ccrs.Geodetic()
    )

# 2. Busse (Knoten) zeichnen
ax.scatter(
    net.buses.x,
    net.buses.y,
    marker='o',
    color='red',
    s=5,
    alpha=0.8,
    transform=ccrs.PlateCarree()
)

plt.title(f"PyPSA-Netzwerk: {len(net.buses)} Busse, {len(net.lines)} Leitungen")
plt.show()
"""
utils.py – Gemeinsame Hilfsfunktionen für Store2Hydro Analyse
=============================================================
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Kein GUI auf HPC – muss VOR pyplot import stehen
import matplotlib.pyplot as plt
import numpy as np
import pypsa


# ─────────────────────────────────────────────────────────────
# Farb-Schema (konsistent über alle Plots)
# ─────────────────────────────────────────────────────────────
CARRIER_COLORS = {
    "PHS_retrofit": "#1f77b4",
    "PHS":          "#aec7e8",
    "hydro":        "#17becf",
    "onwind":       "#2ca02c",
    "offwind":      "#98df8a",
    "solar":        "#ffbb78",
    "nuclear":      "#9467bd",
    "gas":          "#d62728",
    "coal":         "#8c564b",
    "lignite":      "#c49c94",
    "load":         "#e377c2",
}

SCENARIO_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def carrier_color(carrier: str) -> str:
    return CARRIER_COLORS.get(carrier, "#333333")


def scenario_color(idx: int) -> str:
    return SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]


# ─────────────────────────────────────────────────────────────
# Netzwerke laden
# ─────────────────────────────────────────────────────────────
def load_scenario_networks(scenario_path: str, filter_years=None) -> dict:
    """
    Liest alle Sektor-Netzwerke eines Szenario-Ordners.

    Erwartet Struktur:
        <scenario_path>/networks/base_s_X___YYYY.nc

    Gibt zurück: { 2025: pypsa.Network, 2030: pypsa.Network, ... }
    """
    networks_dir = Path(scenario_path) / "networks"
    if not networks_dir.exists():
        print(f"  WARNUNG: Kein 'networks'-Unterordner gefunden in {scenario_path}")
        return {}

    result = {}
    # Sektor-Netzwerke erkennen: base_s_X___YYYY.nc (drei Unterstriche vor Jahr)
    # Strom-Netzwerk (elec): base_s_X_elec__YYYY.nc – wird hier nicht geladen
    pattern = re.compile(r"base_s_\d+___(\d{4})\.nc$")

    for nc_file in sorted(networks_dir.glob("*.nc")):
        m = pattern.match(nc_file.name)
        if not m:
            continue
        year = int(m.group(1))
        if filter_years and year not in filter_years:
            continue
        try:
            n = pypsa.Network(str(nc_file))
            result[year] = n
            print(f"    OK {nc_file.name}  ({len(n.buses)} Busse, "
                  f"{len(n.storage_units)} StorageUnits)")
        except Exception as e:
            print(f"    FEHLER beim Laden von {nc_file.name}: {e}")

    return result


# ─────────────────────────────────────────────────────────────
# Komponenten-Filter
# ─────────────────────────────────────────────────────────────
def get_phs_retrofit(n: pypsa.Network, carrier: str = "PHS_retrofit"):
    """Gibt alle StorageUnits mit dem Retrofit-Carrier zurück."""
    mask = n.storage_units["carrier"] == carrier
    return n.storage_units[mask]


def get_original_hydro(n: pypsa.Network):
    """Gibt alle StorageUnits mit carrier 'hydro' oder 'PHS' zurück."""
    mask = n.storage_units["carrier"].isin(["hydro", "PHS"])
    return n.storage_units[mask]


def get_generators_by_carrier(n: pypsa.Network, carriers: list):
    """Gibt alle Generatoren eines bestimmten Carriers zurück."""
    mask = n.generators["carrier"].isin(carriers)
    return n.generators[mask]


def get_retrofit_investment(n: pypsa.Network, carrier: str = "PHS_retrofit"):
    """
    Bestimmt ob der Retrofit investiert wurde anhand von p_nom_opt.

    In PyPSA-Eur myopic mit p_nom_extendable=True:
    - p_nom_opt > 0  ->  Retrofit wurde gebaut (z=1)
    - p_nom_opt == 0 ->  kein Retrofit (z=0)

    Gibt DataFrame zurück mit: bus, p_nom, p_nom_opt, invested (bool)
    """
    units = get_phs_retrofit(n, carrier)
    if units.empty:
        return units
    result = units[["bus", "p_nom"]].copy()
    if "p_nom_opt" in units.columns:
        result["p_nom_opt"] = units["p_nom_opt"]
    else:
        result["p_nom_opt"] = 0.0
    result["invested"] = result["p_nom_opt"] > 0.01
    return result


# ─────────────────────────────────────────────────────────────
# n.statistics() Hilfsfunktionen
# ─────────────────────────────────────────────────────────────
def get_statistics(n: pypsa.Network):
    """
    Ruft n.statistics() auf und gibt einen bereinigten DataFrame zurück.

    n.statistics() liefert MultiIndex-DataFrame (component, carrier) mit Spalten:
      'Capital Expenditure'     – annualisierte Investitionskosten [EUR/a]
      'Operational Expenditure' – Betriebskosten [EUR/a]
      'Revenue'                 – Erlöse [EUR/a]
      'Curtailment'             – Abregelung [MWh]
      'Capacity'                – installierte Kapazität [MW oder MWh]
      'Optimal Capacity'        – optimierte Kapazität [MW oder MWh]
      'Supply'                  – erzeugte Energie [MWh]
      'Withdrawal'              – verbrauchte Energie [MWh]

    Beispiel-Zugriff:
        stats = get_statistics(n)
        # Capex für PHS_retrofit:
        capex = stats.loc[("StorageUnit", "PHS_retrofit"), "Capital Expenditure"]
        # Alle StorageUnit-Carrier:
        su = stats.xs("StorageUnit", level="component")
    """
    try:
        return n.statistics()
    except Exception as e:
        print(f"  WARNUNG: n.statistics() fehlgeschlagen: {e}")
        return None


def get_capex(n: pypsa.Network, component: str, carrier: str) -> float:
    """Annualisierte Kapitalkosten für (component, carrier). Gibt 0.0 zurück wenn fehlt."""
    stats = get_statistics(n)
    if stats is None:
        return 0.0
    try:
        return float(stats.loc[(component, carrier), "Capital Expenditure"])
    except (KeyError, TypeError):
        return 0.0


def get_system_cost(n: pypsa.Network) -> float:
    """
    Gesamtsystemkosten aus n.objective [EUR].
    Fallback: Summe Capex + Opex aus statistics().
    """
    if hasattr(n, "objective") and n.objective is not None:
        return float(n.objective)
    stats = get_statistics(n)
    if stats is not None:
        capex = stats.get("Capital Expenditure", 0)
        opex  = stats.get("Operational Expenditure", 0)
        total = 0.0
        if hasattr(capex, "sum"):
            total += capex.sum()
        if hasattr(opex, "sum"):
            total += opex.sum()
        return total
    return float("nan")


# ─────────────────────────────────────────────────────────────
# Speichern und Ordnerstruktur
# ─────────────────────────────────────────────────────────────
def save_fig(fig, outdir: Path, name: str, dpi: int = 300):
    """Speichert Matplotlib-Figure als PNG mit 300 dpi."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> gespeichert: {path}")


def setup_output_dirs(base: Path):
    for sub in ["maps", "dispatch", "grid", "cost"]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def print_summary(label: str, networks: dict, carrier: str):
    print(f"\n  Zusammenfassung '{label}':")
    for year, n in sorted(networks.items()):
        inv   = get_retrofit_investment(n, carrier)
        invested_count = inv["invested"].sum() if not inv.empty else 0
        total_cap = inv["p_nom_opt"].sum() if not inv.empty else 0.0
        cost = get_system_cost(n)
        cost_str = f"{cost/1e9:.3f} Mrd EUR" if not np.isnan(cost) else "n/a"
        print(f"    {year}: {len(inv)} Retrofit-Kandidaten | "
              f"{invested_count} investiert | "
              f"Kapazitaet: {total_cap:.1f} MW | "
              f"Systemkosten: {cost_str}")

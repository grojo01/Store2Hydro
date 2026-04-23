#!/usr/bin/env python3
"""
main.py – Store2Hydro Analyse-Orchestrierung
=============================================
Startet alle Analyse-Module und erzeugt alle Grafiken.

Verwendung auf dem HPC:
    python main.py \\
        --pypsa-results /path/to/pypsa-eur/results \\
        --scenarios FR_10_3h_myopic_25_30_35 FR_10_3h_myopic_25_30_35_noretro \\
        --outdir /workdir/bt713593/Store2Hydro/results \\
        --baseline FR_10_3h_myopic_25_30_35_noretro

    # Oder: alle Szenarien automatisch erkennen
    python main.py \\
        --pypsa-results /path/to/pypsa-eur/results \\
        --auto-discover \\
        --outdir /workdir/bt713593/Store2Hydro/results

    # Nur bestimmte Module ausführen:
    python main.py ... --modules maps dispatch

Verfügbare Module: maps, dispatch, grid, cost (Standard: alle)
"""

import argparse
import os
import sys
import time
import traceback

import pypsa

# Lokale Module importieren (gleiches Verzeichnis)
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    find_networks, load_network, discover_scenarios, SCENARIO_PALETTE
)
import maps     as m_maps
import dispatch as m_dispatch
import grid     as m_grid
import cost     as m_cost


# ---------------------------------------------------------------------------
# Ausgabeverzeichnisse
# ---------------------------------------------------------------------------
SUBDIRS = {
    "maps":     "01_maps",
    "dispatch": "02_dispatch",
    "grid":     "03_grid",
    "cost":     "04_cost",
}

ALL_MODULES = list(SUBDIRS.keys())


# ---------------------------------------------------------------------------
# Netzwerke laden
# ---------------------------------------------------------------------------
def load_all_networks(pypsa_results: str, scenario_names: list) -> dict:
    """
    Lädt alle Netzwerke für alle Szenarien und Perioden.

    Gibt zurück:
        {
            'szenario_name': {
                '2025': pypsa.Network,
                '2030': pypsa.Network,
                '2035': pypsa.Network,
            },
            ...
        }
    """
    result = {}
    for scen in scenario_names:
        scen_dir = os.path.join(pypsa_results, scen)
        print(f"\n[main] Lade Szenario: {scen}")
        try:
            nc_files = find_networks(scen_dir)
        except FileNotFoundError as e:
            print(f"  WARNUNG: {e}")
            continue

        period_networks = {}
        for period, nc_path in sorted(nc_files.items()):
            print(f"  Periode {period}: {os.path.basename(nc_path)} ... ", end="")
            t0 = time.time()
            try:
                n = load_network(nc_path)
                print(f"OK ({time.time()-t0:.1f}s, "
                      f"{len(n.buses)} Busse, "
                      f"{len(n.storage_units)} StorageUnits)")
                period_networks[period] = n
            except Exception as e:
                print(f"FEHLER: {e}")

        if period_networks:
            result[scen] = period_networks
        else:
            print(f"  WARNUNG: Keine Netzwerke geladen für {scen}")

    return result


# ---------------------------------------------------------------------------
# Analyse-Module ausführen
# ---------------------------------------------------------------------------
def run_module(module_name: str, networks: dict, outdir: str,
               scenario_dirs: dict, baseline: str, verbose: bool):
    """Führt ein einzelnes Analyse-Modul aus."""
    sub = os.path.join(outdir, SUBDIRS[module_name])
    os.makedirs(sub, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Modul: {module_name.upper()}  →  {sub}")
    print(f"{'='*60}")

    try:
        if module_name == "maps":
            m_maps.plot_retrofit_decisions(networks, sub)
            m_maps.plot_storage_capacity_map(networks, sub)
            m_maps.plot_myopic_investment_timeline(networks, sub)

        elif module_name == "dispatch":
            m_dispatch.plot_soc_profiles(networks, sub)
            m_dispatch.plot_soc_heatmap(networks, sub)
            m_dispatch.plot_pump_vs_renewables(networks, sub)
            m_dispatch.plot_inflow_handling(networks, sub)
            m_dispatch.plot_roundtrip_losses(networks, sub)

        elif module_name == "grid":
            m_grid.plot_curtailment(networks, sub)
            m_grid.plot_lmp_distribution(networks, sub)
            m_grid.plot_price_spike_distribution(networks, sub)
            m_grid.plot_res_capacity_evolution(networks, sub)
            m_grid.plot_local_line_loading(networks, sub)

        elif module_name == "cost":
            m_cost.plot_system_costs(networks, sub)
            m_cost.plot_capex_retrofit(networks, sub)
            m_cost.plot_net_savings(networks, sub, baseline_key=baseline)
            m_cost.plot_breakeven(networks, sub, baseline_key=baseline)
            if scenario_dirs:
                m_cost.plot_cost_assumptions(scenario_dirs, sub)

    except Exception as e:
        print(f"\n  [FEHLER] Modul {module_name}: {e}")
        if verbose:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Argument-Parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Store2Hydro – PyPSA Ergebnisanalyse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--pypsa-results",
        default="/home/bt713593/pypsa-eur/results",
        help="Pfad zum pypsa-eur/results Ordner"
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        help="Namen der Szenario-Unterordner (z.B. FR_10_3h_myopic_25_30_35)"
    )
    parser.add_argument(
        "--auto-discover", action="store_true",
        help="Alle Szenarien in --pypsa-results automatisch erkennen"
    )
    parser.add_argument(
        "--outdir",
        default="/workdir/bt713593/Store2Hydro/results",
        help="Ausgabepfad für Grafiken"
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Name des Baseline-Szenarios (ohne Retrofit) für Kostenvergleich"
    )
    parser.add_argument(
        "--modules", nargs="+", choices=ALL_MODULES, default=ALL_MODULES,
        help=f"Welche Module ausführen (Standard: alle). Wählbar: {ALL_MODULES}"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Ausführliche Fehlermeldungen (Traceback)"
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="Verfügbare Szenarien in --pypsa-results auflisten und beenden"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Verfügbare Szenarien auflisten
    if args.list_scenarios:
        found = discover_scenarios(args.pypsa_results)
        print(f"\nVerfügbare Szenarien in: {args.pypsa_results}")
        for s in found:
            print(f"  {s}")
        return

    # Szenarien bestimmen
    if args.auto_discover:
        scenario_names = discover_scenarios(args.pypsa_results)
        print(f"[main] Auto-discover: {len(scenario_names)} Szenarien gefunden")
    elif args.scenarios:
        scenario_names = args.scenarios
    else:
        print("FEHLER: Bitte --scenarios oder --auto-discover angeben.")
        sys.exit(1)

    if not scenario_names:
        print("FEHLER: Keine Szenarien gefunden/angegeben.")
        sys.exit(1)

    print(f"\n[main] Szenarien ({len(scenario_names)}):")
    for s in scenario_names:
        print(f"  {s}")
    print(f"[main] Module: {args.modules}")
    print(f"[main] Ausgabe: {args.outdir}")
    if args.baseline:
        print(f"[main] Baseline: {args.baseline}")

    # Ausgabeverzeichnis anlegen
    os.makedirs(args.outdir, exist_ok=True)

    # Netzwerke laden
    t_start = time.time()
    networks = load_all_networks(args.pypsa_results, scenario_names)

    if not networks:
        print("\nFEHLER: Keine Netzwerke geladen. Bitte Pfade prüfen.")
        sys.exit(1)

    print(f"\n[main] Netzwerke geladen in {time.time()-t_start:.1f}s")
    print(f"[main] {len(networks)} Szenarien × "
          f"{len(next(iter(networks.values())))} Perioden")

    # Szenario-Verzeichnisse für Config-Plots
    scenario_dirs = {
        s: os.path.join(args.pypsa_results, s)
        for s in scenario_names
    }

    # Module ausführen
    t_all = time.time()
    for module in args.modules:
        run_module(
            module_name=module,
            networks=networks,
            outdir=args.outdir,
            scenario_dirs=scenario_dirs,
            baseline=args.baseline,
            verbose=args.verbose,
        )

    # Zusammenfassung
    print(f"\n{'='*60}")
    print(f"  FERTIG in {time.time()-t_all:.1f}s")
    print(f"  Alle Grafiken in: {args.outdir}")
    for mod in args.modules:
        sub = os.path.join(args.outdir, SUBDIRS[mod])
        if os.path.isdir(sub):
            pngs = [f for f in os.listdir(sub) if f.endswith(".png")]
            print(f"  {SUBDIRS[mod]}/  →  {len(pngs)} Grafiken")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
analyse.py – Store2Hydro Hauptskript
=====================================
Aufruf:
    python analyse.py \
        --scenarios /path/to/results/Scenario_A /path/to/results/Scenario_B \
        --labels "Baseline" "Retrofit_500" \
        --outdir /workdir/bt713593/Store2Hydro/results/plots \
        --carrier PHS_retrofit \
        --modules all

    # Nur bestimmte Module:
    python analyse.py --scenarios ... --labels ... --modules dispatch cost

Verfügbare Module: maps, dispatch, grid, cost
"""

import argparse
import sys
from pathlib import Path

import pypsa

from utils import load_scenario_networks, setup_output_dirs, print_summary
from plot_maps import run_maps
from plot_dispatch import run_dispatch
from plot_grid import run_grid
from plot_cost import run_cost


def parse_args():
    parser = argparse.ArgumentParser(
        description="Store2Hydro – Ergebnisanalyse für PyPSA-Eur Retrofit-Szenarien"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        help="Pfade zu den Szenario-Ordnern (je ein Ordner pro Szenario, "
             "z.B. pypsa-eur/results/FR_10_3h_myopic_25_30_35)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Kurznamen für die Szenarien (gleiche Reihenfolge wie --scenarios)"
    )
    parser.add_argument(
        "--outdir",
        default="/workdir/bt713593/Store2Hydro/results/plots",
        help="Ausgabeordner für alle Grafiken"
    )
    parser.add_argument(
        "--carrier",
        default="PHS_retrofit",
        help="Carrier-Name der retrofitted StorageUnit im Netzwerk (default: PHS_retrofit)"
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["all"],
        choices=["all", "maps", "dispatch", "grid", "cost"],
        help="Welche Analyse-Module ausgeführt werden sollen (default: all)"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Nur bestimmte Planungsperioden analysieren (z.B. --years 2025 2030)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validierung
    if len(args.scenarios) != len(args.labels):
        print("FEHLER: Anzahl --scenarios muss gleich Anzahl --labels sein.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Store2Hydro Ergebnisanalyse")
    print(f"{'='*60}")
    print(f"  Szenarien : {args.labels}")
    print(f"  Carrier   : {args.carrier}")
    print(f"  Ausgabe   : {args.outdir}")
    print(f"  Module    : {args.modules}")
    print(f"{'='*60}\n")

    # Ausgabeordner anlegen
    outdir = Path(args.outdir)
    setup_output_dirs(outdir)

    # Netzwerke laden: Dict { label: { year: pypsa.Network } }
    all_networks = {}
    for scenario_path, label in zip(args.scenarios, args.labels):
        print(f"  Lade Netzwerke für Szenario '{label}' aus: {scenario_path}")
        networks = load_scenario_networks(
            scenario_path=scenario_path,
            filter_years=args.years
        )
        if not networks:
            print(f"  WARNUNG: Keine Netzwerke gefunden für '{label}' – überspringe.")
            continue
        all_networks[label] = networks
        print_summary(label, networks, args.carrier)

    if not all_networks:
        print("FEHLER: Keine Netzwerke geladen. Abbruch.")
        sys.exit(1)

    run_all = "all" in args.modules
    modules = args.modules

    # ── 1. Karten ──────────────────────────────────────────────
    if run_all or "maps" in modules:
        print("\n[1/4] Karten (Retrofit-Entscheidung, Speicherkapazitäten)...")
        run_maps(all_networks, args.carrier, outdir / "maps")

    # ── 2. Dispatch ────────────────────────────────────────────
    if run_all or "dispatch" in modules:
        print("\n[2/4] Dispatch (SoC, Pump-Events, Inflow, Roundtrip)...")
        run_dispatch(all_networks, args.carrier, outdir / "dispatch")

    # ── 3. Grid ────────────────────────────────────────────────
    if run_all or "grid" in modules:
        print("\n[3/4] Grid (Curtailment, LMP, RES-Kapazität, Leitungen)...")
        run_grid(all_networks, args.carrier, outdir / "grid")

    # ── 4. Kosten ──────────────────────────────────────────────
    if run_all or "cost" in modules:
        print("\n[4/4] Kosten (Capex, Ersparnis, Break-even, Myopic)...")
        run_cost(all_networks, args.carrier, outdir / "cost")

    print(f"\n{'='*60}")
    print(f"  Fertig. Grafiken gespeichert in: {args.outdir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

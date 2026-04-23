# Store2Hydro – Analyse-Skripte

Analyse-Paket für PyPSA-Eur Ergebnisse des Store2Hydro Projekts.

## Dateistruktur

```
store2hydro_analysis/
├── main.py        # Einstiegspunkt, CLI, Orchestrierung
├── utils.py       # Hilfsfunktionen (laden, filtern, speichern)
├── maps.py        # Geografische Karten (Retrofit-Entscheidung, Kapazitäten)
├── dispatch.py    # Dispatch-Analyse (SoC, Pumpen, Inflow, Verluste)
├── grid.py        # Systemeffekte (Curtailment, LMPs, RES, Netz)
└── cost.py        # Kosten-Nutzen (Capex, Ersparnis, Break-even)
```

## Verwendung

### Einfachster Aufruf (alle Module, alle Szenarien auto-erkennen)
```bash
cd /workdir/bt713593/Store2Hydro/scripts
python main.py \
    --pypsa-results /home/bt713593/pypsa-eur/results \
    --auto-discover \
    --outdir /workdir/bt713593/Store2Hydro/results
```

### Mit expliziten Szenarien und Baseline
```bash
python main.py \
    --pypsa-results /home/bt713593/pypsa-eur/results \
    --scenarios FR_10_3h_myopic_25_30_35 FR_10_3h_myopic_noretro \
    --baseline FR_10_3h_myopic_noretro \
    --outdir /workdir/bt713593/Store2Hydro/results
```

### Nur bestimmte Module
```bash
python main.py ... --modules dispatch cost
```

### Verfügbare Szenarien anzeigen
```bash
python main.py \
    --pypsa-results /home/bt713593/pypsa-eur/results \
    --list-scenarios
```

## Ausgabestruktur

```
/workdir/bt713593/Store2Hydro/results/
├── 01_maps/
│   ├── map_retrofit_decisions_FR_10_3h_myopic_25_30_35.png
│   ├── map_storage_capacity_FR_10_3h_myopic_25_30_35.png
│   └── map_myopic_investment_timeline.png
├── 02_dispatch/
│   ├── dispatch_soc_profiles_2025.png
│   ├── dispatch_soc_heatmap_..._2025_...unit....png
│   ├── dispatch_pump_vs_res_..._2025.png
│   ├── dispatch_inflow_..._2025_...unit....png
│   └── dispatch_roundtrip_losses.png
├── 03_grid/
│   ├── grid_curtailment_comparison.png
│   ├── grid_lmp_distribution_2025.png
│   ├── grid_price_spikes.png
│   ├── grid_res_capacity_evolution.png
│   └── grid_local_line_loading_2025.png
└── 04_cost/
    ├── cost_system_costs.png
    ├── cost_capex_retrofit.png
    ├── cost_net_savings.png
    ├── cost_breakeven.png
    └── cost_assumptions_from_config.png
```

## Voraussetzungen

```bash
pip install pypsa matplotlib cartopy seaborn pyyaml
```

Cartopy ist optional – ohne Cartopy werden einfache Scatter-Karten gezeichnet.

## Hinweise zur Analyse-Reihenfolge

1. **maps** – Investitionsentscheidung validieren: wurde z=1 gesetzt?
2. **dispatch** – SoC-Profile & Pump-Events verstehen
3. **grid** – Systemeffekte quantifizieren (Curtailment, LMPs, RES)
4. **cost** – Kosten-Nutzen zusammenführen & Break-even

## Bekannte Einschränkungen

- `n.statistics()` erfordert PyPSA ≥ 0.25. Bei älteren Versionen wird
  auf `p_nom_opt * capital_cost` zurückgefallen.
- Leitungsauslastung (`n.lines_t.p0`) ist nur bei Netzwerken mit
  aktiviertem Leitungsfluss verfügbar (lopf/linopt).
- `cartopy` muss separat installiert werden und ist auf manchen
  HPC-Clustern nicht standardmäßig verfügbar.

"""
cost.py – Kosten-Nutzen-Analyse für Store2Hydro
================================================
Plots:
  1. Gesamtsystemkosten (n.objective) – Vergleich Szenarien × Perioden
  2. Capex Retrofit (aus n.statistics)
  3. Netto-Systemkostenersparnis: ΔSystemkosten − Capex_Retrofit
  4. Break-even: Maximaler capital_cost bei dem Retrofit gewählt wird
  5. Angenommene Preise & Investitionskosten aus Config
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils import (
    COLORS, SCENARIO_PALETTE, save_fig,
    get_retrofit_units, get_statistics,
)


# ---------------------------------------------------------------------------
# Plot 1: Gesamtsystemkosten
# ---------------------------------------------------------------------------
def plot_system_costs(networks_per_scenario: dict, outdir: str):
    """
    Vergleicht n.objective (Gesamtsystemkosten [€/a]) über Szenarien und Perioden.

    n.objective ist die optimierte Zielfunktion, d.h. annualisierte
    Gesamtkosten (Capex + Opex) des Systems.

    Zeigt:
    - Balkendiagramm pro Periode, gruppiert nach Szenario
    - Differenzlinie: Δ = Kosten_Baseline − Kosten_Retrofit (Ersparnis)
    """
    records = []
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            obj = getattr(n, "objective", None)
            if obj is None or np.isnan(obj):
                print(f"  [cost] n.objective nicht verfügbar: {scen} {period}")
                continue
            records.append({
                "scenario": scen,
                "period":   int(period),
                "objective_bn": obj / 1e9,  # € → Mrd. €
            })

    if not records:
        print("[cost] Keine Systemkosten-Daten.")
        return

    df = pd.DataFrame(records)
    periods   = sorted(df["period"].unique())
    scenarios = sorted(df["scenario"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # -- Linkes Panel: absolut --
    ax = axes[0]
    x = np.arange(len(scenarios))
    width = 0.8 / len(periods)

    for j, period in enumerate(periods):
        vals = []
        for scen in scenarios:
            row = df[(df["scenario"] == scen) & (df["period"] == period)]
            vals.append(row["objective_bn"].values[0] if not row.empty else np.nan)
        offset = (j - len(periods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=str(period), alpha=0.85,
               color=SCENARIO_PALETTE[j % len(SCENARIO_PALETTE)])

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, fontsize=9)
    ax.set_ylabel("Gesamtsystemkosten [Mrd. €/a]", fontsize=11)
    ax.set_title("Absolute Systemkosten", fontsize=12)
    ax.legend(title="Periode", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # -- Rechtes Panel: Entwicklung über Perioden --
    ax2 = axes[1]
    for i, scen in enumerate(scenarios):
        sub = df[df["scenario"] == scen].sort_values("period")
        ax2.plot(sub["period"], sub["objective_bn"],
                 marker="o", lw=2,
                 color=SCENARIO_PALETTE[i % len(SCENARIO_PALETTE)],
                 label=scen)

    ax2.set_xlabel("Planungsperiode", fontsize=11)
    ax2.set_ylabel("Gesamtsystemkosten [Mrd. €/a]", fontsize=11)
    ax2.set_title("Kostenentwicklung über Perioden", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Gesamtsystemkosten (n.objective)", fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, outdir, "cost_system_costs")


# ---------------------------------------------------------------------------
# Plot 2: Capex Retrofit aus n.statistics()
# ---------------------------------------------------------------------------
def plot_capex_retrofit(networks_per_scenario: dict, outdir: str):
    """
    Investitionskosten des Retrofits aus n.statistics().

    n.statistics() liefert einen MultiIndex-DataFrame.
    Relevante Spalten: 'Capital Expenditure', 'Capacity'

    Zugriff:
        stats = n.statistics()
        # Alle StorageUnits:
        su_stats = stats.loc["StorageUnit"]
        # Nur PHS_retrofit:
        retrofit_stats = su_stats[
            su_stats.index.str.contains("retrofit", case=False)
        ]

    Zeigt:
    - Balkendiagramm: annualisierter Capex [M€/a] pro Szenario × Periode
    - Gestapelt nach Retrofit-Unit (falls mehrere)
    """
    records = []
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            try:
                stats = get_statistics(n)
            except Exception as e:
                print(f"  [cost] n.statistics() fehlgeschlagen: {e}")
                continue

            # Zugriff auf StorageUnit-Zeilen
            try:
                su_stats = stats.loc["StorageUnit"]
            except KeyError:
                print(f"  [cost] 'StorageUnit' nicht in statistics: {scen} {period}")
                continue

            # Filter: nur retrofit carrier
            if hasattr(su_stats.index, "str"):
                retrofit_mask = su_stats.index.str.contains(
                    "retrofit", case=False, na=False)
            else:
                # MultiIndex: carrier-Level
                retrofit_mask = su_stats.index.get_level_values(-1).str.contains(
                    "retrofit", case=False, na=False)

            retrofit_stats = su_stats[retrofit_mask]

            if retrofit_stats.empty:
                # Fallback: direkt aus p_nom_opt * capital_cost berechnen
                retrofit_units = get_retrofit_units(n)
                for unit in retrofit_units:
                    p_nom_opt    = n.storage_units.at[unit, "p_nom_opt"]
                    capital_cost = n.storage_units.at[unit, "capital_cost"]
                    capex = p_nom_opt * capital_cost / 1e6  # M€
                    records.append({
                        "scenario": scen, "period": period,
                        "unit": unit, "capex_m_eur": capex,
                    })
            else:
                cap_col = "Capital Expenditure" if "Capital Expenditure" \
                          in retrofit_stats.columns else \
                    retrofit_stats.columns[0]
                for idx, row in retrofit_stats.iterrows():
                    capex = row.get(cap_col, 0) / 1e6
                    records.append({
                        "scenario": scen, "period": period,
                        "unit": str(idx), "capex_m_eur": capex,
                    })

    if not records:
        print("[cost] Keine Capex-Daten.")
        return

    df = pd.DataFrame(records)
    periods   = sorted(df["period"].unique())
    scenarios = sorted(df["scenario"].unique())
    units     = sorted(df["unit"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scenarios))
    width = 0.8 / len(periods)

    for j, period in enumerate(sorted(periods)):
        sub_p = df[df["period"] == period]
        bottoms = np.zeros(len(scenarios))
        for k, unit in enumerate(units):
            vals = []
            for scen in scenarios:
                row = sub_p[(sub_p["scenario"] == scen) & (sub_p["unit"] == unit)]
                vals.append(row["capex_m_eur"].values[0] if not row.empty else 0)
            offset = (j - len(periods) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9,
                   bottom=bottoms,
                   color=SCENARIO_PALETTE[k % len(SCENARIO_PALETTE)],
                   alpha=0.8,
                   label=f"{unit[:15]} ({period})" if j == 0 else "")
            bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, fontsize=9)
    ax.set_ylabel("Capex Retrofit [M€/a annualisiert]", fontsize=11)
    ax.set_title("Investitionskosten PHS_retrofit",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, outdir, "cost_capex_retrofit")


# ---------------------------------------------------------------------------
# Plot 3: Netto-Systemkostenersparnis
# ---------------------------------------------------------------------------
def plot_net_savings(networks_per_scenario: dict, outdir: str,
                     baseline_key: str = None):
    """
    Netto-Ersparnis = ΔSystemkosten − Capex_Retrofit

    ΔSystemkosten = Kosten_Baseline − Kosten_Retrofit_Szenario

    Falls baseline_key nicht angegeben: das Szenario mit dem niedrigsten
    capital_cost bzw. das erste alphabetisch als Baseline verwenden.

    Zeigt:
    - Grouped Bar: Brutto-Ersparnis, Capex, Netto-Ersparnis
    - Positiv = Retrofit rechnet sich, Negativ = kein wirtschaftlicher Retrofit
    """
    # Systemkosten sammeln
    obj = {}
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            o = getattr(n, "objective", None)
            if o is not None and not np.isnan(o):
                obj[(scen, period)] = o

    if not obj:
        print("[cost] Keine n.objective Daten für Ersparnis-Plot.")
        return

    # Capex sammeln (vereinfacht: p_nom_opt * capital_cost)
    capex = {}
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            total = 0
            for unit in get_retrofit_units(n):
                pnom = n.storage_units.at[unit, "p_nom_opt"]
                ccost = n.storage_units.at[unit, "capital_cost"]
                total += pnom * ccost
            capex[(scen, period)] = total

    periods   = sorted({p for _, p in obj})
    scenarios = sorted({s for s, _ in obj})

    # Baseline bestimmen
    if baseline_key is None or baseline_key not in scenarios:
        baseline_key = scenarios[0]
        print(f"  [cost] Baseline-Szenario: '{baseline_key}'")

    records = []
    for scen in scenarios:
        if scen == baseline_key:
            continue
        for period in periods:
            base_cost   = obj.get((baseline_key, period), np.nan)
            scen_cost   = obj.get((scen, period), np.nan)
            scen_capex  = capex.get((scen, period), 0)

            gross_saving = base_cost - scen_cost      # positiv = günstiger
            net_saving   = gross_saving - scen_capex  # nach Abzug Investition

            records.append({
                "scenario":     scen,
                "period":       period,
                "gross_saving_m": gross_saving / 1e6,
                "capex_m":        scen_capex  / 1e6,
                "net_saving_m":   net_saving  / 1e6,
            })

    if not records:
        print("[cost] Nur ein Szenario – kein Ersparnis-Vergleich möglich.")
        return

    df = pd.DataFrame(records)
    compare_scenarios = sorted(df["scenario"].unique())

    fig, axes = plt.subplots(1, len(periods),
                              figsize=(6 * len(periods), 6), sharey=True)
    if len(periods) == 1:
        axes = [axes]

    for ax, period in zip(axes, periods):
        sub = df[df["period"] == period]
        x = np.arange(len(sub))
        width = 0.25

        ax.bar(x - width, sub["gross_saving_m"], width,
               color="#2ca02c", alpha=0.8, label="Brutto-Ersparnis")
        ax.bar(x,          sub["capex_m"],        width,
               color="#d62728", alpha=0.8, label="Capex Retrofit")
        ax.bar(x + width,  sub["net_saving_m"],   width,
               color="#1f77b4", alpha=0.8, label="Netto-Ersparnis")

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["scenario"], rotation=15, fontsize=9)
        ax.set_ylabel("M€/a", fontsize=11)
        ax.set_title(f"Periode {period}\n(Baseline: {baseline_key})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Netto-Systemkostenersparnis durch Retrofit",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, outdir, "cost_net_savings")


# ---------------------------------------------------------------------------
# Plot 4: Break-even Analyse
# ---------------------------------------------------------------------------
def plot_breakeven(networks_per_scenario: dict, outdir: str,
                   capital_cost_range: np.ndarray = None,
                   baseline_key: str = None):
    """
    Break-even: Ab welchem capital_cost lohnt sich Retrofit nicht mehr?

    Methode:
    - Systemkostenersparnis (ohne Capex) = ΔSystemkosten
    - Break-even capital_cost = ΔSystemkosten / p_nom_opt  [€/MW]
    - Eingezeichnet: tatsächlich angenommener capital_cost aus n.storage_units

    Zeigt:
    - Horizontale Linie: ΔSystemkosten (Maximal tolerierbarer Capex)
    - Vertikale Linie: tatsächlicher capital_cost
    - Gefüllter Bereich: Retrofit lohnt sich

    Für Sensitivitäts-Sweep: capital_cost_range kann manuell übergeben werden.
    """
    obj = {}
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            o = getattr(n, "objective", None)
            if o is not None and not np.isnan(o):
                obj[(scen, period)] = o

    if not obj:
        return

    periods   = sorted({p for _, p in obj})
    scenarios = sorted({s for s, _ in obj})

    if baseline_key is None or baseline_key not in scenarios:
        baseline_key = scenarios[0]

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, scen in enumerate(scenarios):
        if scen == baseline_key:
            continue
        color = SCENARIO_PALETTE[i % len(SCENARIO_PALETTE)]
        for period in periods:
            base_cost = obj.get((baseline_key, period), np.nan)
            scen_cost = obj.get((scen, period), np.nan)
            if np.isnan(base_cost) or np.isnan(scen_cost):
                continue

            delta_sys = base_cost - scen_cost  # € brutto

            # Gesamte Retrofit-Kapazität um €/MW Threshold zu berechnen
            n = networks_per_scenario[scen][period]
            total_p_nom = sum(
                n.storage_units.at[u, "p_nom_opt"]
                for u in get_retrofit_units(n)
            )
            if total_p_nom < 1e-3:
                continue

            be_cost = delta_sys / total_p_nom  # €/MW

            # Aktuell angenommener capital_cost (erste Retrofit-Unit)
            actual_cc = None
            for u in get_retrofit_units(n):
                actual_cc = n.storage_units.at[u, "capital_cost"]
                break

            label = f"{scen} – {period}"
            ax.axhline(be_cost / 1e3, color=color, ls="-", lw=1.5,
                       alpha=0.8, label=f"Break-even {label}")
            if actual_cc is not None:
                ax.axvline(actual_cc / 1e3, color=color, ls="--", lw=1.0,
                           alpha=0.6)
                ax.annotate(
                    f"{actual_cc/1e3:.0f}k\n(angenommen)",
                    xy=(actual_cc / 1e3, be_cost / 1e3),
                    fontsize=7, ha="left",
                    color=color
                )

    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(
        ax.get_xlim() if ax.get_xlim() != (0, 1) else [0, 500],
        0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1000,
        alpha=0.05, color="green", label="Retrofit lohnt sich"
    )
    ax.set_xlabel("capital_cost [k€/MW]", fontsize=11)
    ax.set_ylabel("Break-even capital_cost [k€/MW]", fontsize=11)
    ax.set_title(
        f"Break-even-Analyse (Baseline: {baseline_key})",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, outdir, "cost_breakeven")


# ---------------------------------------------------------------------------
# Plot 5: Investitionskosten & Preisannahmen aus Config
# ---------------------------------------------------------------------------
def plot_cost_assumptions(scenario_dirs: dict, outdir: str):
    """
    Liest Kostenannahmen direkt aus den YAML-Config-Dateien.

    Zeigt:
    - capital_cost für PHS_retrofit aus config.base_s_X___YYYY.yaml
    - marginal_cost (falls gesetzt)
    - Tabellarische Darstellung als Heatmap

    scenario_dirs: {'szenario_name': '/pfad/zu/FR_10_3h_myopic_25_30_35'}
    """
    try:
        import yaml
    except ImportError:
        print("[cost] PyYAML nicht installiert – Config-Plot übersprungen.")
        return

    records = []
    for scen, sdir in scenario_dirs.items():
        cfg_dir = os.path.join(sdir, "configs")
        if not os.path.isdir(cfg_dir):
            continue
        for fname in sorted(os.listdir(cfg_dir)):
            if not fname.endswith(".yaml"):
                continue
            m = re.search(r"_(\d{4})\.yaml$", fname)
            if not m:
                continue
            period = m.group(1)
            with open(os.path.join(cfg_dir, fname)) as f:
                try:
                    cfg = yaml.safe_load(f)
                except Exception:
                    continue

            # Suche nach PHS_retrofit Kosten in typischen PyPSA-Eur Pfaden
            cost_val = None
            for key_path in [
                ["costs", "PHS_retrofit", "capital_cost"],
                ["costs", "capital_cost", "PHS_retrofit"],
                ["override_component_attrs", "StorageUnit", "capital_cost"],
            ]:
                try:
                    val = cfg
                    for k in key_path:
                        val = val[k]
                    cost_val = val
                    break
                except (KeyError, TypeError):
                    continue

            records.append({
                "scenario": scen,
                "period":   period,
                "capital_cost": cost_val,
            })

    if not records or all(r["capital_cost"] is None for r in records):
        print("[cost] Keine Kostenannahmen in Config-Dateien gefunden.")
        return

    df = pd.DataFrame(records)
    pivot = df.pivot_table(
        values="capital_cost",
        index="scenario", columns="period",
        aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.8)))
    im = ax.imshow(pivot.values.astype(float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="capital_cost [€/MW]")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                ax.text(j, i, f"{float(val):,.0f}",
                        ha="center", va="center", fontsize=9)

    ax.set_title("Kostenannahmen PHS_retrofit aus Config",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, outdir, "cost_assumptions_from_config")

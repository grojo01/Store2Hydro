"""
grid.py – Systemeffekte & Netzanalyse für Store2Hydro
======================================================
Plots:
  1. Curtailment – Vergleich mit/ohne Retrofit pro Periode
  2. LMP-Verteilung (Marginalpreise) – Violin-Plot
  3. RES-Kapazitätsentwicklung (Wind + Solar über Perioden)
  4. Netzauslastung lokal am Hydro-Standort
  5. Preisspitzen-Verteilung
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils import (
    COLORS, SCENARIO_PALETTE, save_fig,
    get_retrofit_units, get_generators_by_carrier,
)


# ---------------------------------------------------------------------------
# Plot 1: Curtailment
# ---------------------------------------------------------------------------
def plot_curtailment(networks_per_scenario: dict, outdir: str):
    """
    Curtailment = p_nom_opt * capacity_factor - tatsächlicher Dispatch.

    Berechnung je Generator mit p_nom_opt und p_max_pu Zeitreihe:
        curtailment = (p_nom_opt * p_max_pu - p_dispatch) * dt  [MWh]

    Zeigt:
    - Gestapeltes Balkendiagramm: Curtailment nach Carrier (Wind, Solar) pro
      Szenario und Periode
    - Prozentualer Anteil am potentiellen Ertrag

    Damit sieht man direkt ob und wie viel Curtailment der Retrofit reduziert.
    """
    records = []
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            dt = _timestep_hours(n)
            gen_t_p     = n.generators_t.p
            gen_t_p_max = n.generators_t.p_max_pu

            for carrier_key in ("wind", "solar"):
                units = get_generators_by_carrier(n, carrier_key)
                for unit in units:
                    if unit not in gen_t_p.columns:
                        continue
                    p_nom_opt = n.generators.at[unit, "p_nom_opt"]
                    if unit in gen_t_p_max.columns:
                        potential = (p_nom_opt * gen_t_p_max[unit]).sum() * dt
                    else:
                        cf = n.generators.at[unit, "p_max_pu"] \
                             if "p_max_pu" in n.generators.columns else 1.0
                        potential = p_nom_opt * cf * len(n.snapshots) * dt
                    actual    = gen_t_p[unit].sum() * dt
                    curtailed = max(0, potential - actual)

                    records.append({
                        "scenario": scen, "period": period,
                        "carrier": carrier_key,
                        "potential_GWh": potential / 1e3,
                        "actual_GWh":    actual    / 1e3,
                        "curtailed_GWh": curtailed / 1e3,
                    })

    if not records:
        print("[grid] Keine Curtailment-Daten.")
        return

    df = pd.DataFrame(records)
    df_agg = df.groupby(["scenario", "period", "carrier"])[
        ["potential_GWh", "actual_GWh", "curtailed_GWh"]
    ].sum().reset_index()

    periods   = sorted(df_agg["period"].unique())
    scenarios = sorted(df_agg["scenario"].unique())

    fig, axes = plt.subplots(
        1, len(periods),
        figsize=(6 * len(periods), 6),
        sharey=True
    )
    if len(periods) == 1:
        axes = [axes]

    carriers = ["wind", "solar"]
    carrier_colors = [COLORS["wind"], COLORS["solar"]]

    for ax, period in zip(axes, periods):
        sub = df_agg[df_agg["period"] == period]
        x = np.arange(len(scenarios))
        width = 0.35
        bottom = np.zeros(len(scenarios))

        for carrier, c_color in zip(carriers, carrier_colors):
            vals = []
            for scen in scenarios:
                row = sub[(sub["scenario"] == scen) & (sub["carrier"] == carrier)]
                vals.append(row["curtailed_GWh"].sum() if not row.empty else 0)
            ax.bar(x, vals, width, bottom=bottom,
                   color=c_color, alpha=0.8, label=carrier)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=15, fontsize=9)
        ax.set_title(period, fontsize=11)
        ax.set_ylabel("Curtailment [GWh]", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Curtailment – Vergleich Szenarien und Perioden",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, outdir, "grid_curtailment_comparison")


# ---------------------------------------------------------------------------
# Plot 2: LMP-Verteilung (Marginalpreise)
# ---------------------------------------------------------------------------
def plot_lmp_distribution(networks_per_scenario: dict, outdir: str,
                           country_filter: str = None):
    """
    Verteilung der Locational Marginal Prices (LMPs = n.buses_t.marginal_price).

    Zeigt:
    - Violin-Plot pro Szenario und Periode
    - Optional: nur Busse eines Landes (country_filter, z.B. 'FR')
    - Preisspitzen werden explizit markiert (> 95. Perzentil)
    """
    for period in _all_periods(networks_per_scenario):
        fig, ax = plt.subplots(figsize=(10, 6))
        data_list  = []
        tick_labels = []

        for i, (scen, periods) in enumerate(
                sorted(networks_per_scenario.items())):
            if period not in periods:
                continue
            n = periods[period]

            if not hasattr(n.buses_t, "marginal_price") or \
               n.buses_t.marginal_price.empty:
                print(f"  [grid] Keine LMP-Daten in {scen} {period}")
                continue

            prices = n.buses_t.marginal_price
            # Optionaler Länder-Filter
            if country_filter:
                cols = [c for c in prices.columns
                        if str(c).startswith(country_filter)]
                prices = prices[cols]

            flat_prices = prices.values.flatten()
            flat_prices = flat_prices[~np.isnan(flat_prices)]
            # Extremwerte kappen für Darstellung (99. Pz.)
            p99 = np.percentile(flat_prices, 99)
            flat_prices_clipped = np.clip(flat_prices, -500, p99 * 1.2)

            data_list.append(flat_prices_clipped)
            tick_labels.append(f"{scen}\n{period}")

        if not data_list:
            plt.close(fig)
            continue

        parts = ax.violinplot(data_list, positions=range(len(data_list)),
                              showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], SCENARIO_PALETTE):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel("Marginalpreis [€/MWh]", fontsize=11)
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
        loc_str = f"({country_filter})" if country_filter else "(alle Busse)"
        ax.set_title(f"LMP-Verteilung {loc_str} – Periode {period}",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, outdir, f"grid_lmp_distribution_{period}")


# ---------------------------------------------------------------------------
# Plot 3: Preisspitzen-Häufigkeit
# ---------------------------------------------------------------------------
def plot_price_spike_distribution(networks_per_scenario: dict, outdir: str,
                                   spike_threshold: float = 150.0):
    """
    Stunden mit Preisspitzen > threshold [€/MWh] pro Szenario und Periode.

    Zeigt:
    - Balkendiagramm: Anzahl Stunden mit Preis > threshold
    - Zeitliche Verteilung der Spitzen (Monat)
    """
    records = []
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            if not hasattr(n.buses_t, "marginal_price") or \
               n.buses_t.marginal_price.empty:
                continue
            prices = n.buses_t.marginal_price
            # Maximaler Preis pro Zeitstempel über alle Busse
            max_price = prices.max(axis=1)
            n_spikes  = (max_price > spike_threshold).sum()
            records.append({
                "scenario": scen, "period": period,
                "n_spikes": int(n_spikes),
                "max_price": float(max_price.max()),
            })

    if not records:
        print("[grid] Keine Preisspitzen-Daten.")
        return

    df = pd.DataFrame(records)
    periods   = sorted(df["period"].unique())
    scenarios = sorted(df["scenario"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(scenarios))
    width = 0.8 / len(periods)

    for j, period in enumerate(periods):
        vals = []
        for scen in scenarios:
            row = df[(df["scenario"] == scen) & (df["period"] == period)]
            vals.append(row["n_spikes"].values[0] if not row.empty else 0)
        offset = (j - len(periods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=period, alpha=0.8,
               color=SCENARIO_PALETTE[j % len(SCENARIO_PALETTE)])

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, fontsize=9)
    ax.set_ylabel(f"Stunden mit Preis > {spike_threshold} €/MWh", fontsize=10)
    ax.set_title(f"Preisspitzen-Häufigkeit (Threshold: {spike_threshold} €/MWh)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Periode", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, outdir, "grid_price_spikes")


# ---------------------------------------------------------------------------
# Plot 4: RES-Kapazitätsentwicklung
# ---------------------------------------------------------------------------
def plot_res_capacity_evolution(networks_per_scenario: dict, outdir: str):
    """
    Wie viel Wind und Solar wird mit/ohne Retrofit ausgebaut?

    Zeigt:
    - Liniendiagramm: p_nom_opt [GW] über Perioden
    - Je Szenario eine Linie, Carrier (Wind/Solar) als Linientyp
    - Direkter Vergleich ob Retrofit den RES-Ausbau beeinflusst
    """
    records = []
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            for carrier_key in ("wind", "solar"):
                units = get_generators_by_carrier(n, carrier_key)
                cap_gw = sum(
                    n.generators.at[u, "p_nom_opt"]
                    for u in units
                    if u in n.generators.index
                ) / 1e3  # MW → GW
                records.append({
                    "scenario": scen,
                    "period": int(period),
                    "carrier": carrier_key,
                    "capacity_GW": cap_gw,
                })

    if not records:
        print("[grid] Keine RES-Kapazitätsdaten.")
        return

    df = pd.DataFrame(records)
    carriers  = ["wind", "solar"]
    linestyles = ["-", "--"]

    fig, axes = plt.subplots(1, len(carriers),
                              figsize=(7 * len(carriers), 5),
                              sharey=False)
    if len(carriers) == 1:
        axes = [axes]

    for ax, carrier, ls in zip(axes, carriers, linestyles):
        sub = df[df["carrier"] == carrier]
        for i, scen in enumerate(sorted(sub["scenario"].unique())):
            s_sub = sub[sub["scenario"] == scen].sort_values("period")
            ax.plot(s_sub["period"], s_sub["capacity_GW"],
                    marker="o", lw=2, ls=ls,
                    color=SCENARIO_PALETTE[i % len(SCENARIO_PALETTE)],
                    label=scen)
        ax.set_xlabel("Planungsperiode", fontsize=11)
        ax.set_ylabel(f"{carrier.capitalize()}-Kapazität [GW]", fontsize=11)
        ax.set_title(carrier.capitalize(), fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("RES-Kapazitätsentwicklung – Wind & Solar",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, outdir, "grid_res_capacity_evolution")


# ---------------------------------------------------------------------------
# Plot 5: Leitungsauslastung am Hydro-Standort
# ---------------------------------------------------------------------------
def plot_local_line_loading(networks_per_scenario: dict, outdir: str,
                             max_lines: int = 10):
    """
    Netzauslastung der Leitungen, die direkt an Retrofit-Busse angebunden sind.

    Auslastung = |p0| / s_nom_opt [%]

    Zeigt:
    - Box-Plot der Auslastungsverteilung pro Leitung und Szenario
    - Hervorhebung von Leitungen mit > 80% Auslastung (Überlastrisiko)
    """
    for period in _all_periods(networks_per_scenario):
        fig, ax = plt.subplots(figsize=(12, 6))
        data_list  = []
        tick_labels = []

        for scen, periods_dict in sorted(networks_per_scenario.items()):
            if period not in periods_dict:
                continue
            n = periods_dict[period]

            # Busse der Retrofit-Units
            retrofit_buses = set()
            for unit in get_retrofit_units(n):
                bus = n.storage_units.at[unit, "bus"]
                retrofit_buses.add(bus)

            if not retrofit_buses:
                continue

            if not hasattr(n.lines_t, "p0") or n.lines_t.p0.empty:
                print(f"  [grid] Keine Leitungsdaten in {scen} {period}")
                continue

            # Leitungen, die an Retrofit-Busse angebunden sind
            local_lines = n.lines[
                n.lines["bus0"].isin(retrofit_buses) |
                n.lines["bus1"].isin(retrofit_buses)
            ].index.tolist()[:max_lines]

            for line in local_lines:
                if line not in n.lines_t.p0.columns:
                    continue
                s_nom_opt = n.lines.at[line, "s_nom_opt"]
                if s_nom_opt < 1e-3:
                    continue
                loading = (n.lines_t.p0[line].abs() / s_nom_opt * 100)
                data_list.append(loading.values)
                tick_labels.append(f"{scen}\n{line[:12]}")

        if not data_list:
            plt.close(fig)
            continue

        bp = ax.boxplot(data_list, labels=tick_labels,
                        patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"],
                                 SCENARIO_PALETTE * (len(data_list) // len(SCENARIO_PALETTE) + 1)):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(80, ls="--", color="#d62728", lw=1.2,
                   label="80% Grenze (Überlastrisiko)")
        ax.axhline(100, ls="-", color="#d62728", lw=1.5, alpha=0.5)
        ax.set_ylabel("Leitungsauslastung [%]", fontsize=11)
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=30)
        ax.set_title(
            f"Leitungsauslastung am Retrofit-Standort – Periode {period}",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_fig(fig, outdir, f"grid_local_line_loading_{period}")


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _all_periods(networks_per_scenario: dict) -> list:
    return sorted({p for s in networks_per_scenario.values() for p in s})


def _timestep_hours(n) -> float:
    if len(n.snapshots) < 2:
        return 1.0
    return (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600

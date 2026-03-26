"""
dispatch.py – Dispatch-Analyse für Store2Hydro
===============================================
Plots:
  1. SoC-Zeitreihen der PHS_retrofit Units (mit Kapazitätsauslastung)
  2. SoC-Heatmap (Tag × Stunde)
  3. Pump-Events vs. Wind/PV-Einspeisung (Korrelation)
  4. Inflow-Handling: natürlicher Zufluss vs. Dispatch
  5. Round-trip-Verluste: Energie rein vs. raus
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import (
    COLORS, SCENARIO_PALETTE, save_fig,
    get_retrofit_units, get_units_by_carrier, get_generators_by_carrier,
)


# ---------------------------------------------------------------------------
# Plot 1: SoC-Zeitreihen
# ---------------------------------------------------------------------------
def plot_soc_profiles(networks_per_scenario: dict, outdir: str,
                      sample_weeks: int = 4):
    """
    State of Charge der PHS_retrofit Units über die Zeit.

    Zeigt:
    - SoC in MWh (absolut) oder % der max. Kapazität
    - Strichlinie bei 100% und 0% (Kapazitätsauslastung)
    - Vergleich mehrere Szenarien übereinander

    sample_weeks: Anzahl Wochen für den Detail-Plot (übersichtlicher als ganzes Jahr)
    """
    for period in _all_periods(networks_per_scenario):
        fig, axes = plt.subplots(
            len(networks_per_scenario), 1,
            figsize=(14, 3.5 * len(networks_per_scenario)),
            sharex=True
        )
        if len(networks_per_scenario) == 1:
            axes = [axes]

        for ax, (scen, color) in zip(
                axes, zip(networks_per_scenario, SCENARIO_PALETTE)):
            if period not in networks_per_scenario[scen]:
                ax.set_title(f"{scen} – {period} nicht vorhanden", fontsize=10)
                continue
            n = networks_per_scenario[scen][period]
            retrofit_units = get_retrofit_units(n)

            if not retrofit_units:
                ax.set_title(f"{scen} – keine Retrofit-Units", fontsize=10)
                continue

            soc_t = n.storage_units_t.state_of_charge

            for i, unit in enumerate(retrofit_units):
                if unit not in soc_t.columns:
                    continue
                soc = soc_t[unit]
                max_cap = n.storage_units.at[unit, "p_nom_opt"] * \
                          n.storage_units.at[unit, "max_hours"]
                if max_cap < 1e-3:
                    max_cap = soc.max() or 1

                # Auf % normieren
                soc_pct = 100 * soc / max_cap
                unit_color = list(COLORS.values())[i % len(COLORS)]
                ax.plot(soc.index, soc_pct, lw=0.8,
                        color=unit_color, label=unit[:20], alpha=0.85)

            ax.axhline(100, ls="--", color="gray", lw=0.8, alpha=0.7,
                       label="100% Kapazität")
            ax.axhline(0, ls=":", color="gray", lw=0.8, alpha=0.7)
            ax.set_ylabel("SoC [%]", fontsize=10)
            ax.set_ylim(-5, 110)
            ax.legend(fontsize=7, loc="upper right", ncol=3)
            ax.set_title(f"{scen} – {period}", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Zeitstempel", fontsize=10)
        fig.suptitle(f"State of Charge – Planungsperiode {period}",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        save_fig(fig, outdir, f"dispatch_soc_profiles_{period}")


# ---------------------------------------------------------------------------
# Plot 2: SoC-Heatmap (Tag × Stunde)
# ---------------------------------------------------------------------------
def plot_soc_heatmap(networks_per_scenario: dict, outdir: str):
    """
    2D-Heatmap: x=Stunde des Tages, y=Tag des Jahres, Farbe=SoC [%].

    Zeigt saisionale Muster und Tageszyklen der PHS.
    Je eine Figur pro Szenario × Periode × Retrofit-Unit.
    """
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            retrofit_units = get_retrofit_units(n)
            soc_t = n.storage_units_t.state_of_charge

            for unit in retrofit_units:
                if unit not in soc_t.columns:
                    continue

                soc = soc_t[unit]
                max_cap = n.storage_units.at[unit, "p_nom_opt"] * \
                          n.storage_units.at[unit, "max_hours"]
                if max_cap < 1e-3:
                    max_cap = soc.max() or 1
                soc_pct = (100 * soc / max_cap).rename("soc_pct")

                # Pivot: Zeilen=Tag, Spalten=Stunde
                df = soc_pct.to_frame()
                df["hour"] = df.index.hour
                df["day"]  = df.index.dayofyear
                pivot = df.pivot_table(
                    values="soc_pct", index="day", columns="hour", aggfunc="mean"
                )

                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.pcolormesh(
                    pivot.columns, pivot.index, pivot.values,
                    cmap="RdYlGn", vmin=0, vmax=100
                )
                cb = fig.colorbar(im, ax=ax)
                cb.set_label("SoC [%]", fontsize=11)
                ax.set_xlabel("Stunde des Tages", fontsize=11)
                ax.set_ylabel("Tag des Jahres", fontsize=11)
                ax.set_title(
                    f"SoC-Heatmap – {unit} | {scen} | {period}",
                    fontsize=12, fontweight="bold"
                )
                plt.tight_layout()
                safe_unit = unit.replace("/", "_").replace(" ", "_")
                save_fig(fig, outdir,
                         f"dispatch_soc_heatmap_{scen}_{period}_{safe_unit}")


# ---------------------------------------------------------------------------
# Plot 3: Pump-Events vs. Wind/PV
# ---------------------------------------------------------------------------
def plot_pump_vs_renewables(networks_per_scenario: dict, outdir: str,
                             top_n_weeks: int = 2):
    """
    Korrelation: Wann pumpen die PHS? → Vergleich mit Wind + Solar Einspeisung.

    Zeigt für die N Wochen mit höchster Windeinspeisung:
    - Balken: Pump-Leistung (negativ = Laden)
    - Linie: Wind-Einspeisung
    - Linie: Solar-Einspeisung
    - Überlagert für Baseline vs. Retrofit-Szenarien
    """
    for period in _all_periods(networks_per_scenario):
        for scen, periods in networks_per_scenario.items():
            if period not in periods:
                continue
            n = periods[period]
            retrofit_units = get_retrofit_units(n)
            if not retrofit_units:
                continue

            # Dispatch der Retrofit-Units (negativ = Pumpen/Laden)
            p_t = n.storage_units_t.p
            retrofit_p = p_t[[u for u in retrofit_units if u in p_t.columns]]
            total_pump = retrofit_p.clip(upper=0).sum(axis=1)  # Lade-Leistung

            # Wind + Solar Einspeisung
            wind_units  = get_generators_by_carrier(n, "wind")
            solar_units = get_generators_by_carrier(n, "solar")
            gen_t = n.generators_t.p

            wind_total  = gen_t[[u for u in wind_units  if u in gen_t.columns]].sum(axis=1)
            solar_total = gen_t[[u for u in solar_units if u in gen_t.columns]].sum(axis=1)

            if wind_total.empty and solar_total.empty:
                print(f"  [dispatch] Keine Wind/Solar Daten in {scen} {period}")
                continue

            # Top-N Wochen nach höchster Wind-Einspeisung finden
            res_total = wind_total.add(solar_total, fill_value=0)
            weekly_res = res_total.resample("W").mean()
            top_weeks  = weekly_res.nlargest(top_n_weeks).index

            fig, axes = plt.subplots(
                top_n_weeks, 1,
                figsize=(14, 4 * top_n_weeks),
                sharex=False
            )
            if top_n_weeks == 1:
                axes = [axes]

            for ax, week_start in zip(axes, top_weeks):
                week_end = week_start + pd.Timedelta("7D")
                mask = (res_total.index >= week_start) & \
                       (res_total.index <  week_end)
                t_idx = res_total.index[mask]

                ax2 = ax.twinx()
                # Pump als gefüllten Bereich (Laden = negativ)
                ax.fill_between(t_idx, total_pump.reindex(t_idx, fill_value=0),
                                0, alpha=0.6, color=COLORS["PHS_retrofit"],
                                label="PHS Pump-Leistung [MW]")
                ax2.plot(t_idx, wind_total.reindex(t_idx, fill_value=0),
                         color=COLORS["wind"], lw=1.2, label="Wind [MW]")
                ax2.plot(t_idx, solar_total.reindex(t_idx, fill_value=0),
                         color=COLORS["solar"], lw=1.2, ls="--",
                         label="Solar [MW]")

                ax.set_ylabel("Pump-Leistung [MW]", fontsize=9,
                              color=COLORS["PHS_retrofit"])
                ax2.set_ylabel("Erzeugung [MW]", fontsize=9)
                ax.set_title(
                    f"Woche ab {week_start.strftime('%d.%m.%Y')} "
                    f"(Top-{top_n_weeks} Wind)", fontsize=10
                )
                ax.grid(True, alpha=0.3)

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                          fontsize=8, loc="upper right")

            fig.suptitle(
                f"Pump-Events vs. Erneuerbare – {scen} | {period}",
                fontsize=13, y=1.01
            )
            plt.tight_layout()
            save_fig(fig, outdir,
                     f"dispatch_pump_vs_res_{scen}_{period}")


# ---------------------------------------------------------------------------
# Plot 4: Inflow-Handling
# ---------------------------------------------------------------------------
def plot_inflow_handling(networks_per_scenario: dict, outdir: str):
    """
    Vergleich: natürlicher Zufluss (inflow) vs. tatsächlicher Dispatch der
    retrofitted PHS.

    Zeigt:
    - Inflow als graue Fläche (fest, nicht steuerbar)
    - Erzeugung (positiver Dispatch) als blaue Linie
    - Pump-Leistung (negativer Dispatch) als rote Fläche
    - SoC als grüne Linie (rechte Achse)

    Zugriff auf inflow:
        n.storage_units_t.inflow  (falls zeitvariabel)
        n.storage_units.inflow    (falls konstant)
    """
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            retrofit_units = get_retrofit_units(n)

            for unit in retrofit_units:
                # Inflow auslesen (zeitvariabel oder konstant)
                if (hasattr(n.storage_units_t, "inflow") and
                        unit in n.storage_units_t.inflow.columns):
                    inflow = n.storage_units_t.inflow[unit]
                elif "inflow" in n.storage_units.columns:
                    val = n.storage_units.at[unit, "inflow"]
                    idx = n.snapshots
                    inflow = pd.Series(val, index=idx)
                else:
                    print(f"  [dispatch] Kein inflow für {unit} gefunden.")
                    continue

                p_t = n.storage_units_t.p
                soc_t = n.storage_units_t.state_of_charge
                if unit not in p_t.columns:
                    continue

                dispatch  = p_t[unit]
                generation = dispatch.clip(lower=0)
                pumping    = dispatch.clip(upper=0)
                soc = soc_t[unit] if unit in soc_t.columns else None

                fig, ax = plt.subplots(figsize=(14, 5))
                ax2 = ax.twinx()

                ax.fill_between(inflow.index, inflow.values, 0,
                                alpha=0.3, color="gray", label="Inflow [MW]")
                ax.fill_between(generation.index, generation.values, 0,
                                alpha=0.7, color=COLORS["PHS_retrofit"],
                                label="Turbinierung [MW]")
                ax.fill_between(pumping.index, pumping.values, 0,
                                alpha=0.6, color="#d62728",
                                label="Pumpen [MW]")
                if soc is not None:
                    ax2.plot(soc.index, soc.values, color="#2ca02c",
                             lw=1.0, alpha=0.8, label="SoC [MWh]")
                    ax2.set_ylabel("SoC [MWh]", fontsize=10, color="#2ca02c")

                ax.set_ylabel("Leistung [MW]", fontsize=10)
                ax.set_xlabel("Zeitstempel", fontsize=10)
                ax.grid(True, alpha=0.3)

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                          fontsize=9, loc="upper right")
                ax.set_title(
                    f"Inflow-Handling – {unit} | {scen} | {period}",
                    fontsize=12, fontweight="bold"
                )
                plt.tight_layout()
                safe_unit = unit.replace("/", "_").replace(" ", "_")
                save_fig(fig, outdir,
                         f"dispatch_inflow_{scen}_{period}_{safe_unit}")


# ---------------------------------------------------------------------------
# Plot 5: Round-trip-Verluste
# ---------------------------------------------------------------------------
def plot_roundtrip_losses(networks_per_scenario: dict, outdir: str):
    """
    Round-trip-Effizienz der Retrofit-Units im Vergleich.

    Berechnung:
        - Energie rein (Pumpen): sum(dispatch < 0) * (-1) * Δt  [MWh]
        - Energie raus (Turbinieren): sum(dispatch > 0) * Δt    [MWh]
        - η_roundtrip = Energie_raus / Energie_rein
        - Verlust = Energie_rein - Energie_raus

    Achtung: η entspricht auch efficiency_store * efficiency_dispatch aus
    n.storage_units. Diese Werte werden als Referenz eingezeichnet.

    Zeigt Balkendiagramm über Szenarien und Perioden.
    """
    records = []
    for scen, periods in networks_per_scenario.items():
        for period, n in periods.items():
            retrofit_units = get_retrofit_units(n)
            p_t = n.storage_units_t.p
            # Zeitschritt in Stunden
            dt = _timestep_hours(n)

            for unit in retrofit_units:
                if unit not in p_t.columns:
                    continue
                d = p_t[unit]
                e_in  = (-d.clip(upper=0)).sum() * dt   # MWh ins Speicher
                e_out = d.clip(lower=0).sum() * dt       # MWh aus Speicher

                if e_in < 1:
                    rt_eff = np.nan
                else:
                    rt_eff = e_out / e_in

                # Theoretische Effizienz aus Parametern
                eta_store    = n.storage_units.at[unit, "efficiency_store"]
                eta_dispatch = n.storage_units.at[unit, "efficiency_dispatch"]
                rt_theoretical = eta_store * eta_dispatch

                records.append({
                    "scenario": scen, "period": period,
                    "unit": unit,
                    "e_in_GWh":  e_in / 1e3,
                    "e_out_GWh": e_out / 1e3,
                    "rt_eff_actual":      rt_eff,
                    "rt_eff_theoretical": rt_theoretical,
                })

    if not records:
        print("[dispatch] Keine Daten für Round-trip-Verluste.")
        return

    df = pd.DataFrame(records)
    periods = sorted(df["period"].unique())
    scenarios = df["scenario"].unique()

    fig, axes = plt.subplots(
        1, len(periods),
        figsize=(5 * len(periods), 5),
        sharey=True
    )
    if len(periods) == 1:
        axes = [axes]

    for ax, period in zip(axes, periods):
        sub = df[df["period"] == period]
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["rt_eff_actual"] * 100, color=SCENARIO_PALETTE[:len(sub)],
                      alpha=0.8, label="Tatsächlich")
        ax.scatter(x, sub["rt_eff_theoretical"] * 100, color="black",
                   zorder=5, marker="D", s=60, label="Theoretisch (η_store×η_disp)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{row.scenario}\n{row.unit[:10]}" for _, row in sub.iterrows()],
            fontsize=8, rotation=15
        )
        ax.set_ylabel("Round-trip-Effizienz [%]", fontsize=10)
        ax.set_ylim(0, 110)
        ax.axhline(100, ls="--", color="gray", lw=0.8)
        ax.set_title(period, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Round-trip-Verluste der PHS_retrofit Units",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, outdir, "dispatch_roundtrip_losses")


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _all_periods(networks_per_scenario: dict) -> list:
    """Gibt sortierte Liste aller Perioden über alle Szenarien zurück."""
    return sorted({p for s in networks_per_scenario.values() for p in s})


def _timestep_hours(n) -> float:
    """Berechnet den Zeitschritt des Netzwerks in Stunden."""
    if len(n.snapshots) < 2:
        return 1.0
    delta = n.snapshots[1] - n.snapshots[0]
    return delta.total_seconds() / 3600

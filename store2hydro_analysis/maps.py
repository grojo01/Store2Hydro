"""
maps.py – Geografische Karten für Store2Hydro
=============================================
Plots:
  1. Karte der Busse + Retrofit-Entscheidung pro Periode (z=0/1)
  2. Karte der StorageUnit-Kapazitäten (Bubble-Map)
  3. Myopischer Retrofit-Zeitverlauf (wann wird wo investiert?)

Abhängigkeiten: matplotlib, cartopy (optional), utils.py
Cartopy wird nur für geografische Hintergründe benötigt.
Wenn cartopy fehlt, wird auf einfache Scatter-Karte zurückgefallen.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    COLORS, SCENARIO_PALETTE, save_fig,
    get_retrofit_units, get_investment_status,
    get_units_by_carrier,
)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("[maps.py] Cartopy nicht verfügbar – einfache Scatter-Karten werden verwendet.")


# ---------------------------------------------------------------------------
# Hilfsfunktion: Achse mit Kartenhintergrund
# ---------------------------------------------------------------------------
def _make_map_axes(extent=None):
    """Erstellt eine Axes mit optionalem Cartopy-Hintergrund."""
    if HAS_CARTOPY:
        fig, ax = plt.subplots(
            figsize=(10, 8),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f0")
        ax.add_feature(cfeature.OCEAN, facecolor="#d6eaf8")
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("#f5f5f0")
        ax.grid(True, linestyle="--", alpha=0.4)
    return fig, ax


def _scatter_kwargs():
    """Gibt kwargs für cartopy-kompatibles scatter zurück."""
    if HAS_CARTOPY:
        return {"transform": ccrs.PlateCarree(), "zorder": 5}
    return {"zorder": 5}


# ---------------------------------------------------------------------------
# Plot 1: Retrofit-Entscheidung pro Periode
# ---------------------------------------------------------------------------
def plot_retrofit_decisions(networks_per_scenario: dict, outdir: str):
    """
    Karte: Welche Units wurden in welcher Periode retrofittet (z=1)?

    Parameter:
        networks_per_scenario: {
            'scenario_name': {
                '2025': pypsa.Network,
                '2030': pypsa.Network,
                '2035': pypsa.Network
            }
        }
        outdir: Ausgabepfad

    Für jedes Szenario wird eine Figur mit einem Subplot pro Periode erstellt.
    Buses: graue Punkte; Retrofit-Units: farbige Marker (grün=investiert, rot=nicht)
    """
    for scen_name, periods in networks_per_scenario.items():
        sorted_periods = sorted(periods.keys())
        n_periods = len(sorted_periods)

        fig, axes = plt.subplots(
            1, n_periods,
            figsize=(6 * n_periods, 7),
            subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        )
        if n_periods == 1:
            axes = [axes]

        for ax, period in zip(axes, sorted_periods):
            n = periods[period]

            # Kartenhintergrund
            if HAS_CARTOPY:
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.LAND, facecolor="#f5f5f0")
                ax.add_feature(cfeature.OCEAN, facecolor="#d6eaf8")

            # Alle Busse als graue Punkte
            bus_x = n.buses["x"]
            bus_y = n.buses["y"]
            ax.scatter(bus_x, bus_y, s=10, color="lightgray",
                       alpha=0.6, **_scatter_kwargs())

            # Retrofit-Units
            investment = get_investment_status(n)
            retrofit_units = get_retrofit_units(n)

            for unit in retrofit_units:
                bus = n.storage_units.at[unit, "bus"]
                if bus not in n.buses.index:
                    continue
                x = n.buses.at[bus, "x"]
                y = n.buses.at[bus, "y"]
                invested = investment.get(unit, False)
                color = "#2ca02c" if invested else "#d62728"
                marker = "^" if invested else "x"
                ax.scatter(x, y, s=120, color=color, marker=marker,
                           edgecolors="black", linewidths=0.5,
                           **_scatter_kwargs())
                ax.annotate(
                    unit[:15],  # Name kürzen
                    (x, y), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points"
                )

            ax.set_title(f"{period}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Längengrad")
            ax.set_ylabel("Breitengrad")

        # Legende
        legend_elements = [
            mpatches.Patch(color="#2ca02c", label="Retrofit investiert (z=1)"),
            mpatches.Patch(color="#d62728", label="Kein Retrofit (z=0)"),
            mpatches.Patch(color="lightgray", label="Sonstige Busse"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=3, fontsize=10, frameon=True,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(f"Retrofit-Entscheidung – {scen_name}", fontsize=14, y=1.01)
        plt.tight_layout()
        save_fig(fig, outdir, f"map_retrofit_decisions_{scen_name}")


# ---------------------------------------------------------------------------
# Plot 2: StorageUnit-Kapazitäten als Bubble-Map
# ---------------------------------------------------------------------------
def plot_storage_capacity_map(networks_per_scenario: dict, outdir: str,
                               carriers=("PHS_retrofit", "PHS", "hydro")):
    """
    Bubble-Map: Optimierte Kapazitäten (p_nom_opt) der StorageUnits.

    Bubblengröße ∝ p_nom_opt [MW].
    Farbe nach carrier.
    Je Periode ein Subplot, je Szenario eine Figur.
    """
    for scen_name, periods in networks_per_scenario.items():
        sorted_periods = sorted(periods.keys())
        n_periods = len(sorted_periods)

        fig, axes = plt.subplots(
            1, n_periods,
            figsize=(6 * n_periods, 7),
            subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        )
        if n_periods == 1:
            axes = [axes]

        # Maximale Kapazität über alle Perioden bestimmen (für einheitliche Skalierung)
        max_cap = 0
        for n in periods.values():
            max_cap = max(max_cap, n.storage_units["p_nom_opt"].max())
        if max_cap < 1:
            max_cap = 1  # Vermeidet Division durch 0

        for ax, period in zip(axes, sorted_periods):
            n = periods[period]

            if HAS_CARTOPY:
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.LAND, facecolor="#f5f5f0")
                ax.add_feature(cfeature.OCEAN, facecolor="#d6eaf8")

            # Alle Busse
            ax.scatter(n.buses["x"], n.buses["y"],
                       s=8, color="lightgray", alpha=0.4, **_scatter_kwargs())

            for carrier in carriers:
                units = get_units_by_carrier(n, carrier)
                if not units:
                    continue
                color = COLORS.get(carrier, "gray")
                for unit in units:
                    bus = n.storage_units.at[unit, "bus"]
                    if bus not in n.buses.index:
                        continue
                    cap = n.storage_units.at[unit, "p_nom_opt"]
                    if cap < 1e-3:
                        continue
                    x = n.buses.at[bus, "x"]
                    y = n.buses.at[bus, "y"]
                    size = 50 + 800 * (cap / max_cap)   # Skalierung: 50–850 pt²
                    ax.scatter(x, y, s=size, color=color, alpha=0.7,
                               edgecolors="black", linewidths=0.4,
                               **_scatter_kwargs())

            ax.set_title(f"{period}  (max={max_cap:.0f} MW)", fontsize=11)

        # Legende Carrier
        legend_elements = [
            mpatches.Patch(color=COLORS.get(c, "gray"), label=c)
            for c in carriers
        ]
        # Referenzgröße
        legend_elements.append(
            plt.scatter([], [], s=50 + 800 * 0.5, color="gray",
                        alpha=0.5, label=f"~{max_cap/2:.0f} MW")
        )
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=len(carriers) + 1, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(f"StorageUnit-Kapazitäten (p_nom_opt) – {scen_name}",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        save_fig(fig, outdir, f"map_storage_capacity_{scen_name}")


# ---------------------------------------------------------------------------
# Plot 3: Myopischer Retrofit-Verlauf – Balkendiagramm (kein Map)
# ---------------------------------------------------------------------------
def plot_myopic_investment_timeline(networks_per_scenario: dict, outdir: str):
    """
    Zeigt für jedes Szenario, welche Retrofit-Units in welcher Periode
    investiert wurden. Balkendiagramm: x=Periode, Farbe=investiert/nicht.

    Vergleicht mehrere Szenarien nebeneinander.
    """
    all_units_set = set()
    for periods in networks_per_scenario.values():
        for n in periods.values():
            all_units_set.update(get_retrofit_units(n))
    all_units = sorted(all_units_set)

    if not all_units:
        print("[maps] Keine Retrofit-Units gefunden – myopic timeline übersprungen.")
        return

    sorted_scenarios = sorted(networks_per_scenario.keys())
    all_periods = sorted({p for s in networks_per_scenario.values() for p in s})

    fig, axes = plt.subplots(
        1, len(sorted_scenarios),
        figsize=(5 * len(sorted_scenarios), max(3, 0.6 * len(all_units))),
        sharey=True
    )
    if len(sorted_scenarios) == 1:
        axes = [axes]

    for ax, scen in zip(axes, sorted_scenarios):
        periods = networks_per_scenario[scen]
        data = np.zeros((len(all_units), len(all_periods)))

        for j, period in enumerate(all_periods):
            if period not in periods:
                continue
            inv = get_investment_status(periods[period])
            for i, unit in enumerate(all_units):
                data[i, j] = 1 if inv.get(unit, False) else 0

        # Heatmap: grün=1 (investiert), rot=0 (nicht)
        cmap = matplotlib.colors.ListedColormap(["#ffcccc", "#ccffcc"])
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(all_periods)))
        ax.set_xticklabels(all_periods, fontsize=10)
        ax.set_yticks(range(len(all_units)))
        ax.set_yticklabels(all_units, fontsize=8)
        ax.set_title(scen, fontsize=11, fontweight="bold")
        ax.set_xlabel("Planungsperiode")

        # Werte in Zellen
        for i in range(len(all_units)):
            for j in range(len(all_periods)):
                label = "z=1" if data[i, j] == 1 else "z=0"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=9, color="black")

    axes[0].set_ylabel("Retrofit-Unit")
    legend_elements = [
        mpatches.Patch(color="#ccffcc", label="Investiert (z=1)"),
        mpatches.Patch(color="#ffcccc", label="Nicht investiert (z=0)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Myopischer Investitionsverlauf – Retrofit-Entscheidungen",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, "map_myopic_investment_timeline")

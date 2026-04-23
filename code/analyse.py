"""
Store2Hydro – Result Analysis Script
Usage:
    python analyse.py --results /path/to/pypsa-eur/results \
                      --scenarios scen_a scen_b \
                      --baseline scen_a \
                      --outdir /workdir/bt713593/Store2Hydro/results
"""
import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypsa

# ── Config ────────────────────────────────────────────────────────────────────
COLORS = {"PHS_retrofit": "#1f77b4", "wind": "#2ca02c", "solar": "#ffbb78"}
PAL    = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
DPI    = 300

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def load_networks(results_root, scenario_names):
    """Returns {scenario: {year: Network}}"""
    data = {}
    for scen in scenario_names:
        ndir = os.path.join(results_root, scen, "networks")
        if not os.path.isdir(ndir):
            print(f"  warning: missing networks dir for scenario '{scen}' at {ndir}", flush=True)
            continue
        nets = {}
        for f in sorted(os.listdir(ndir)):
            m = re.search(r"_(\d{4})\.nc$", f)
            if m:
                print(f"  loading {scen}/{f} ...", flush=True)
                nets[m.group(1)] = pypsa.Network(os.path.join(ndir, f))
        if nets:
            data[scen] = nets
    return data

def retrofit_units(n):
    """StorageUnits with 'retrofit' in their carrier name."""
    mask = n.storage_units["carrier"].str.contains("retrofit", case=False, na=False)
    return n.storage_units.index[mask].tolist()

def gens_by_carrier(n, substr):
    mask = n.generators["carrier"].str.contains(substr, case=False, na=False)
    return n.generators.index[mask].tolist()

def dt_h(n):
    """Snapshot timestep in hours."""
    return (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600 if len(n.snapshots) > 1 else 1.0

def all_years(data):
    return sorted({y for s in data.values() for y in s})

# ── 1. Maps ───────────────────────────────────────────────────────────────────
def plot_investment_map(data, outdir):
    """One subplot per year: buses (grey dots) + retrofit units coloured by z=0/1."""
    years = all_years(data)
    for scen, nets in data.items():
        fig, axes = plt.subplots(1, len(years), figsize=(5*len(years), 5))
        if len(years) == 1: axes = [axes]
        for ax, yr in zip(axes, years):
            if yr not in nets: continue
            n = nets[yr]
            ax.scatter(n.buses["x"], n.buses["y"], s=8, c="lightgrey", zorder=1)
            for u in retrofit_units(n):
                bus = n.storage_units.at[u, "bus"]
                invested = n.storage_units.at[u, "p_nom_opt"] > 1e-3
                ax.scatter(n.buses.at[bus, "x"], n.buses.at[bus, "y"],
                           s=120, c="#2ca02c" if invested else "#d62728",
                           marker="^" if invested else "x", zorder=3,
                           edgecolors="k", linewidths=0.4)
            ax.set_title(yr); ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.suptitle(f"Retrofit investment (green=z=1) – {scen}", y=1.01)
        plt.tight_layout()
        save(fig, outdir, f"map_investment_{scen}")

def plot_capacity_bubbles(data, outdir):
    """Bubble map: p_nom_opt per StorageUnit."""
    years = all_years(data)
    carriers = ("PHS_retrofit", "PHS", "hydro")
    for scen, nets in data.items():
        max_cap = max(n.storage_units["p_nom_opt"].max() for n in nets.values()) or 1
        fig, axes = plt.subplots(1, len(years), figsize=(5*len(years), 5))
        if len(years) == 1: axes = [axes]
        for ax, yr in zip(axes, years):
            if yr not in nets: continue
            n = nets[yr]
            ax.scatter(n.buses["x"], n.buses["y"], s=6, c="lightgrey")
            for carrier in carriers:
                units = [u for u in n.storage_units.index
                         if carrier.lower() in n.storage_units.at[u,"carrier"].lower()]
                for u in units:
                    bus = n.storage_units.at[u, "bus"]
                    cap = n.storage_units.at[u, "p_nom_opt"]
                    if cap < 1e-3: continue
                    ax.scatter(n.buses.at[bus,"x"], n.buses.at[bus,"y"],
                               s=50+600*(cap/max_cap),
                               c=COLORS.get(carrier,"grey"), alpha=0.7,
                               edgecolors="k", linewidths=0.3, label=carrier)
            ax.set_title(yr)
        # deduplicate legend entries from all axes (robust if first axis has no data)
        all_handles, all_labels = [], []
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)
        by_label = dict(zip(all_labels, all_handles))
        if by_label:
            fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3)
        fig.suptitle(f"Storage capacity (p_nom_opt) – {scen}", y=1.01)
        plt.tight_layout()
        save(fig, outdir, f"map_capacity_{scen}")

# ── 2. Dispatch ───────────────────────────────────────────────────────────────
def plot_soc(data, outdir):
    """SoC [%] time series per year, scenarios overlaid."""
    for yr in all_years(data):
        fig, axes = plt.subplots(len(data), 1, figsize=(14, 3.5*len(data)), sharex=True)
        if len(data) == 1: axes = [axes]
        for ax, (scen, nets) in zip(axes, data.items()):
            if yr not in nets: continue
            n = nets[yr]
            for i, u in enumerate(retrofit_units(n)):
                if u not in n.storage_units_t.state_of_charge.columns: continue
                soc   = n.storage_units_t.state_of_charge[u]
                maxc  = n.storage_units.at[u,"p_nom_opt"] * n.storage_units.at[u,"max_hours"] or soc.max() or 1
                ax.plot(soc.index, 100*soc/maxc, lw=0.7, label=u[:20], color=PAL[i%len(PAL)])
            ax.axhline(100, ls="--", c="grey", lw=0.7)
            ax.set_ylim(-3, 108); ax.set_ylabel("SoC [%]")
            ax.set_title(scen); ax.legend(fontsize=7, ncol=3)
        fig.suptitle(f"State of Charge – {yr}", y=1.01)
        plt.tight_layout()
        save(fig, outdir, f"dispatch_soc_{yr}")

def plot_soc_heatmap(data, outdir):
    """Day×hour SoC heatmap per retrofit unit."""
    for scen, nets in data.items():
        for yr, n in nets.items():
            for u in retrofit_units(n):
                if u not in n.storage_units_t.state_of_charge.columns: continue
                soc  = n.storage_units_t.state_of_charge[u]
                maxc = n.storage_units.at[u,"p_nom_opt"] * n.storage_units.at[u,"max_hours"] or soc.max() or 1
                df   = pd.DataFrame({"v": 100*soc/maxc, "h": soc.index.hour, "d": soc.index.dayofyear})
                piv  = df.pivot_table("v","d","h","mean")
                fig, ax = plt.subplots(figsize=(11,5))
                im = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap="RdYlGn", vmin=0, vmax=100)
                plt.colorbar(im, ax=ax, label="SoC [%]")
                ax.set(xlabel="Hour of day", ylabel="Day of year",
                       title=f"SoC heatmap – {u} | {scen} | {yr}")
                plt.tight_layout()
                save(fig, outdir, f"dispatch_soc_heatmap_{scen}_{yr}_{u[:15].replace('/','_')}")

def plot_pump_vs_res(data, outdir):
    """Pump events vs wind+solar for top-2 wind weeks."""
    for scen, nets in data.items():
        for yr, n in nets.items():
            units = [u for u in retrofit_units(n) if u in n.storage_units_t.p.columns]
            if not units: continue
            pump  = n.storage_units_t.p[units].clip(upper=0).sum(axis=1)
            wind  = n.generators_t.p[gens_by_carrier(n,"wind")].sum(axis=1)  if gens_by_carrier(n,"wind")  else pd.Series(0,index=n.snapshots)
            solar = n.generators_t.p[gens_by_carrier(n,"solar")].sum(axis=1) if gens_by_carrier(n,"solar") else pd.Series(0,index=n.snapshots)
            res   = wind + solar
            top_weeks = res.resample("W").mean().nlargest(2).index
            fig, axes = plt.subplots(2, 1, figsize=(13, 7))
            for ax, ws in zip(axes, top_weeks):
                idx = res.index[(res.index >= ws) & (res.index < ws + pd.Timedelta("7D"))]
                ax2 = ax.twinx()
                ax.fill_between(idx, pump.reindex(idx,fill_value=0), 0, alpha=0.6,
                                color=COLORS["PHS_retrofit"], label="Pump [MW]")
                ax2.plot(idx, wind.reindex(idx,fill_value=0),  c=COLORS["wind"],  lw=1, label="Wind")
                ax2.plot(idx, solar.reindex(idx,fill_value=0), c=COLORS["solar"], lw=1, ls="--", label="Solar")
                ax.set_ylabel("Pump [MW]"); ax2.set_ylabel("Generation [MW]")
                ax.set_title(f"Week from {ws.strftime('%d.%m.%Y')}")
                lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
                labs  = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
                ax.legend(lines, labs, fontsize=8)
            fig.suptitle(f"Pump events vs renewables – {scen} | {yr}", y=1.01)
            plt.tight_layout()
            save(fig, outdir, f"dispatch_pump_vs_res_{scen}_{yr}")

def plot_inflow(data, outdir):
    """Natural inflow vs dispatch for each retrofit unit."""
    for scen, nets in data.items():
        for yr, n in nets.items():
            for u in retrofit_units(n):
                if u not in n.storage_units_t.p.columns: continue
                # get inflow (time-varying or constant)
                if hasattr(n.storage_units_t,"inflow") and u in n.storage_units_t.inflow.columns:
                    inflow = n.storage_units_t.inflow[u]
                elif "inflow" in n.storage_units.columns:
                    inflow = pd.Series(n.storage_units.at[u,"inflow"], index=n.snapshots)
                else:
                    continue
                d  = n.storage_units_t.p[u]
                fig, ax = plt.subplots(figsize=(13,4))
                ax2 = ax.twinx()
                ax.fill_between(inflow.index, inflow, 0, alpha=0.3, color="grey",  label="Inflow")
                ax.fill_between(d.index, d.clip(lower=0), 0, alpha=0.7, color=COLORS["PHS_retrofit"], label="Turbine")
                ax.fill_between(d.index, d.clip(upper=0), 0, alpha=0.5, color="#d62728", label="Pump")
                if u in n.storage_units_t.state_of_charge.columns:
                    ax2.plot(n.snapshots, n.storage_units_t.state_of_charge[u], c="#2ca02c", lw=0.8, label="SoC")
                    ax2.set_ylabel("SoC [MWh]")
                ax.set(ylabel="Power [MW]", title=f"Inflow handling – {u} | {scen} | {yr}")
                lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
                labs  = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
                ax.legend(lines, labs, fontsize=8)
                plt.tight_layout()
                save(fig, outdir, f"dispatch_inflow_{scen}_{yr}_{u[:15].replace('/','_')}")

def plot_roundtrip(data, outdir):
    """Round-trip efficiency: actual (e_out/e_in) vs theoretical (η_s * η_d)."""
    rows = []
    for scen, nets in data.items():
        for yr, n in nets.items():
            for u in retrofit_units(n):
                if u not in n.storage_units_t.p.columns: continue
                d    = n.storage_units_t.p[u]; h = dt_h(n)
                e_in  = (-d.clip(upper=0)).sum() * h
                e_out =  d.clip(lower=0).sum()  * h
                rows.append(dict(scen=scen, yr=yr, unit=u,
                                 actual=e_out/e_in if e_in>1 else np.nan,
                                 theory=n.storage_units.at[u,"efficiency_store"]*n.storage_units.at[u,"efficiency_dispatch"]))
    if not rows: return
    df  = pd.DataFrame(rows)
    yrs = sorted(df["yr"].unique())
    fig, axes = plt.subplots(1, len(yrs), figsize=(5*len(yrs), 4), sharey=True)
    if len(yrs)==1: axes=[axes]
    for ax, yr in zip(axes, yrs):
        sub = df[df["yr"]==yr].reset_index(drop=True)
        ax.bar(sub.index, sub["actual"]*100, color=PAL[:len(sub)], alpha=0.8, label="Actual")
        ax.scatter(sub.index, sub["theory"]*100, c="k", marker="D", zorder=5, label="Theoretical")
        ax.set_xticks(sub.index)
        ax.set_xticklabels([f"{r.scen[:10]}\n{r.unit[:10]}" for _,r in sub.iterrows()], fontsize=7, rotation=15)
        ax.set_ylabel("Round-trip eff. [%]"); ax.set_title(yr)
        ax.axhline(100, ls="--", c="grey", lw=0.7); ax.legend(fontsize=8)
    fig.suptitle("Round-trip losses", y=1.01)
    plt.tight_layout()
    save(fig, outdir, "dispatch_roundtrip")

# ── 3. Grid ───────────────────────────────────────────────────────────────────
def plot_curtailment(data, outdir):
    """Curtailment [GWh] by carrier, grouped by scenario and year."""
    rows = []
    for scen, nets in data.items():
        for yr, n in nets.items():
            h = dt_h(n)
            for carrier in ("wind","solar"):
                for u in gens_by_carrier(n, carrier):
                    if u not in n.generators_t.p.columns: continue
                    pnom = n.generators.at[u,"p_nom_opt"]
                    pmpu = n.generators_t.p_max_pu[u] if u in n.generators_t.p_max_pu.columns \
                           else n.generators.get("p_max_pu", pd.Series(1.0, [u]))[u]
                    potential = (pnom * pmpu).sum() * h if hasattr(pmpu,"sum") else pnom*pmpu*len(n.snapshots)*h
                    curtailed = max(0, potential - n.generators_t.p[u].sum()*h)
                    rows.append(dict(scen=scen, yr=yr, carrier=carrier, curt_GWh=curtailed/1e3))
    if not rows: return
    df  = pd.DataFrame(rows).groupby(["scen","yr","carrier"])["curt_GWh"].sum().reset_index()
    yrs = sorted(df["yr"].unique()); scens = sorted(df["scen"].unique())
    fig, axes = plt.subplots(1, len(yrs), figsize=(5*len(yrs), 5), sharey=True)
    if len(yrs)==1: axes=[axes]
    for ax, yr in zip(axes, yrs):
        sub = df[df["yr"]==yr]; x = np.arange(len(scens)); bot = np.zeros(len(scens))
        for carrier, c in [("wind",COLORS["wind"]),("solar",COLORS["solar"])]:
            vals = [sub[(sub["scen"]==s)&(sub["carrier"]==carrier)]["curt_GWh"].sum() for s in scens]
            ax.bar(x, vals, bottom=bot, color=c, alpha=0.8, label=carrier); bot+=np.array(vals)
        ax.set_xticks(x); ax.set_xticklabels(scens, rotation=15, fontsize=8)
        ax.set_ylabel("Curtailment [GWh]"); ax.set_title(yr); ax.legend(fontsize=8)
    fig.suptitle("Curtailment comparison", y=1.01)
    plt.tight_layout(); save(fig, outdir, "grid_curtailment")

def plot_lmp(data, outdir):
    """Violin plot of bus marginal prices per year."""
    for yr in all_years(data):
        fig, ax = plt.subplots(figsize=(9,5))
        vals, labels = [], []
        for scen, nets in data.items():
            if yr not in nets: continue
            n = nets[yr]
            if not hasattr(n.buses_t,"marginal_price") or n.buses_t.marginal_price.empty: continue
            flat = n.buses_t.marginal_price.values.flatten()
            flat = flat[~np.isnan(flat)]
            vals.append(np.clip(flat, -200, np.percentile(flat,99)*1.2))
            labels.append(scen)
        if not vals: continue
        parts = ax.violinplot(vals, showmedians=True)
        for pc, c in zip(parts["bodies"], PAL): pc.set_facecolor(c); pc.set_alpha(0.7)
        ax.set_xticks(range(1,len(labels)+1)); ax.set_xticklabels(labels, rotation=15, fontsize=8)
        ax.set_ylabel("LMP [€/MWh]"); ax.axhline(0, ls="--", c="grey", lw=0.7)
        ax.set_title(f"Marginal price distribution – {yr}")
        plt.tight_layout(); save(fig, outdir, f"grid_lmp_{yr}")

def plot_res_capacity(data, outdir):
    """Wind and solar p_nom_opt [GW] over planning years."""
    rows = []
    for scen, nets in data.items():
        for yr, n in nets.items():
            for carrier in ("wind","solar"):
                cap = sum(n.generators.at[u,"p_nom_opt"] for u in gens_by_carrier(n,carrier))
                rows.append(dict(scen=scen, yr=int(yr), carrier=carrier, cap_GW=cap/1e3))
    if not rows: return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for ax, carrier in zip(axes, ("wind","solar")):
        for i, scen in enumerate(sorted(df["scen"].unique())):
            sub = df[(df["scen"]==scen)&(df["carrier"]==carrier)].sort_values("yr")
            ax.plot(sub["yr"], sub["cap_GW"], marker="o", lw=2, color=PAL[i%len(PAL)], label=scen)
        ax.set(xlabel="Year", ylabel="Capacity [GW]", title=carrier.capitalize())
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("RES capacity evolution", y=1.01)
    plt.tight_layout(); save(fig, outdir, "grid_res_capacity")

# ── 4. Cost ───────────────────────────────────────────────────────────────────
def plot_system_costs(data, outdir):
    """n.objective [Bn €/a] per scenario and year."""
    rows = []
    for scen, nets in data.items():
        for yr, n in nets.items():
            obj = getattr(n,"objective",None)
            if obj is not None and not np.isnan(obj):
                rows.append(dict(scen=scen, yr=int(yr), cost_bn=obj/1e9))
    if not rows: return
    df = pd.DataFrame(rows); yrs = sorted(df["yr"].unique()); scens = sorted(df["scen"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    # bar chart
    ax = axes[0]; x = np.arange(len(scens)); w = 0.8/len(yrs)
    for j, yr in enumerate(yrs):
        vals = [df[(df["scen"]==s)&(df["yr"]==yr)]["cost_bn"].values[0]
                if not df[(df["scen"]==s)&(df["yr"]==yr)].empty else np.nan for s in scens]
        ax.bar(x+(j-len(yrs)/2+.5)*w, vals, w*.9, label=str(yr), alpha=0.85, color=PAL[j%len(PAL)])
    ax.set_xticks(x); ax.set_xticklabels(scens, rotation=15, fontsize=8)
    ax.set_ylabel("System cost [Bn €/a]"); ax.legend(title="Year"); ax.grid(axis="y",alpha=0.3)
    # line chart
    ax2 = axes[1]
    for i, scen in enumerate(scens):
        sub = df[df["scen"]==scen].sort_values("yr")
        ax2.plot(sub["yr"], sub["cost_bn"], marker="o", lw=2, color=PAL[i%len(PAL)], label=scen)
    ax2.set(xlabel="Year", ylabel="System cost [Bn €/a]"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.suptitle("Total system cost (n.objective)", y=1.01)
    plt.tight_layout(); save(fig, outdir, "cost_system_costs")

def plot_net_savings(data, outdir, baseline):
    """Gross saving, capex, and net saving vs baseline scenario."""
    obj   = {(s,y): getattr(n,"objective",np.nan) for s,nets in data.items() for y,n in nets.items()}
    capex = {}
    for s, nets in data.items():
        for y, n in nets.items():
            capex[(s,y)] = sum(n.storage_units.at[u,"p_nom_opt"]*n.storage_units.at[u,"capital_cost"]
                               for u in retrofit_units(n))
    rows = []
    for scen in data:
        if scen == baseline: continue
        for yr in data[scen]:
            b, c = obj.get((baseline,yr),np.nan), obj.get((scen,yr),np.nan)
            if np.isnan(b) or np.isnan(c): continue
            gross = b - c
            rows.append(dict(scen=scen, yr=yr, gross=gross/1e6,
                             capex=capex.get((scen,yr),0)/1e6,
                             net=(gross-capex.get((scen,yr),0))/1e6))
    if not rows: print("[cost] Need at least 2 scenarios for savings plot."); return
    df = pd.DataFrame(rows); yrs = sorted(df["yr"].unique())
    fig, axes = plt.subplots(1, len(yrs), figsize=(5*len(yrs),5), sharey=True)
    if len(yrs)==1: axes=[axes]
    for ax, yr in zip(axes, yrs):
        sub = df[df["yr"]==yr].reset_index(drop=True); x = np.arange(len(sub)); w=0.25
        ax.bar(x-w, sub["gross"], w, color="#2ca02c", alpha=0.8, label="Gross saving")
        ax.bar(x,   sub["capex"], w, color="#d62728", alpha=0.8, label="Capex")
        ax.bar(x+w, sub["net"],   w, color="#1f77b4", alpha=0.8, label="Net saving")
        ax.axhline(0, c="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(sub["scen"], rotation=15, fontsize=8)
        ax.set_ylabel("M€/a"); ax.set_title(f"{yr}  (base: {baseline})"); ax.legend(fontsize=8)
    fig.suptitle("Net system cost savings", y=1.01)
    plt.tight_layout(); save(fig, outdir, "cost_net_savings")

def plot_breakeven(data, outdir, baseline):
    """Break-even capital_cost = ΔSystemCost / total_p_nom_opt [k€/MW]."""
    obj = {(s,y): getattr(n,"objective",np.nan) for s,nets in data.items() for y,n in nets.items()}
    fig, ax = plt.subplots(figsize=(8,5))
    for i, (scen, nets) in enumerate(data.items()):
        if scen == baseline: continue
        for yr, n in nets.items():
            b, c = obj.get((baseline,yr),np.nan), obj.get((scen,yr),np.nan)
            if np.isnan(b) or np.isnan(c): continue
            p_tot = sum(n.storage_units.at[u,"p_nom_opt"] for u in retrofit_units(n))
            if p_tot < 1: continue
            be = (b-c)/p_tot/1e3  # k€/MW
            retrofit = retrofit_units(n)
            actual_cc = np.mean([n.storage_units.at[u,"capital_cost"]/1e3 for u in retrofit]) if retrofit else None
            ax.scatter(int(yr), be, s=100, color=PAL[i%len(PAL)], zorder=5,
                       label=f"{scen} ({yr})")
            if actual_cc is not None and not np.isnan(actual_cc):
                ax.scatter(int(yr), actual_cc, s=80, color=PAL[i%len(PAL)],
                           marker="D", zorder=5)
                ax.annotate("actual", (int(yr), actual_cc), fontsize=7, xytext=(3,3),
                            textcoords="offset points", color=PAL[i%len(PAL)])
    ax.axhline(0, c="k", lw=0.8)
    ax.set(xlabel="Year", ylabel="Break-even capital_cost [k€/MW]",
           title=f"Break-even analysis (base: {baseline})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); save(fig, outdir, "cost_breakeven")

# ── CLI & main ────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser(description="Store2Hydro analysis")
    p.add_argument("--results",   required=True, help="Path to pypsa-eur/results")
    p.add_argument("--scenarios", nargs="+",     help="Scenario folder names (default: auto-discover)")
    p.add_argument("--baseline",  default=None,  help="Baseline scenario name for cost comparison")
    p.add_argument("--outdir",    default="./results")
    p.add_argument("--modules",   nargs="+", default=["maps","dispatch","grid","cost"],
                   choices=["maps","dispatch","grid","cost"])
    return p.parse_args()

def main():
    args = parse()

    # auto-discover if no scenarios given
    if not args.scenarios:
        args.scenarios = [d for d in sorted(os.listdir(args.results))
                          if os.path.isdir(os.path.join(args.results, d, "networks"))]
        print(f"Auto-discovered: {args.scenarios}")

    baseline = args.baseline or args.scenarios[0]
    data = load_all(args.results, args.scenarios)  # typo fix below

    sub = lambda name: os.path.join(args.outdir, f"{['maps','dispatch','grid','cost'].index(name)+1:02d}_{name}")

    if "maps" in args.modules:
        print("\n── maps ─────────────────────────────────")
        plot_investment_map(data, sub("maps"))
        plot_capacity_bubbles(data, sub("maps"))

    if "dispatch" in args.modules:
        print("\n── dispatch ─────────────────────────────")
        plot_soc(data, sub("dispatch"))
        plot_soc_heatmap(data, sub("dispatch"))
        plot_pump_vs_res(data, sub("dispatch"))
        plot_inflow(data, sub("dispatch"))
        plot_roundtrip(data, sub("dispatch"))

    if "grid" in args.modules:
        print("\n── grid ─────────────────────────────────")
        plot_curtailment(data, sub("grid"))
        plot_lmp(data, sub("grid"))
        plot_res_capacity(data, sub("grid"))

    if "cost" in args.modules:
        print("\n── cost ─────────────────────────────────")
        plot_system_costs(data, sub("cost"))
        plot_net_savings(data, sub("cost"), baseline)
        plot_breakeven(data, sub("cost"), baseline)

    print(f"\nDone. Output: {args.outdir}")

def load_all(results_root, scenario_names):
    """Wrapper kept outside main() so it can be imported."""
    return load_networks(results_root, scenario_names)

if __name__ == "__main__":
    main()

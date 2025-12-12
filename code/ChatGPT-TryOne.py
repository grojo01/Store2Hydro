import pypsa
import pandas as pd
import numpy as np

# === Einstellungen / Annahmen ===
EFF_TURBINE = 0.9       # Turbine Wirkungsgrad
EFF_PUMP = 0.85         # Pumen Wirkungsgrad (Wirkungsgrad beim Aufpumpen)
RESERVOIR_HOURS = 24*7  # anfängliche Speicherkapazität als Stunden × p_nom (z.B. 1 Woche)
CAPEX_PHS_PER_MW = 1e6  # Beispiel-Investkosten EUR/MW (nur Platzhalter!)
CAPEX_RES_PER_MWH = 200e3  # EUR/MWh Speicherkapazität (Platzhalter)

# === Hilfsfunktionen ===
def identify_hydro_generators(n: pypsa.Network):
    # Beispiel-Filter: carrier == 'hydro'. Passe an falls anders markiert.
    if 'carrier' in n.generators.columns:
        hydro = n.generators[n.generators['carrier'] == 'hydro'].copy()
    else:
        # falls keine carrier-Spalte, you must adapt
        hydro = pd.DataFrame()
    return hydro

def compute_inflow_timeseries(gen, n: pypsa.Network):
    # zwei Fälle:
    # - wenn es eine time series p_max_pu (gen_t or p_max_pu) existiert: use it
    # - sonst, if there is per-generator profile in n.generators_t.p or similar.
    # Hier ein robustes Beispiel:
    if hasattr(n, "generators_t"):
        # p_max_pu * p_nom gives absolute available power each time step
        if 'p_max_pu' in n.generators_t.columns:
            return n.generators_t['p_max_pu'][gen.name] * gen.p_nom
        # falls p gen dispatch time-series vorhanden ist:
        if 'p' in n.generators_t.columns:
            return n.generators_t['p'][gen.name]  # reale Produktion -> als referenz
    # fallback: konstante sehr kleine inflow
    return pd.Series(0.0, index=n.snapshots)

# === Hauptroutine: Hydro -> PHS retrofit ===
def retrofit_hydro_to_phs(n: pypsa.Network, marker_tag="retrofit_candidate", remove_old=True):
    hydro = identify_hydro_generators(n)
    if hydro.empty:
        print("Keine Hydro-Generatoren gefunden (Filter anpassen).")
        return

    created = []
    for gid, gen_row in hydro.iterrows():
        gen = gen_row  # pandas Series
        bus = gen['bus']
        p_nom = float(gen.get('p_nom', 0.0))
        if p_nom <= 0:
            print(f"Generator {gid} hat p_nom<=0, übersprungen.")
            continue

        # inflow TS (Watt oder MW je nach Einheiten deines Netzes)
        inflow_ts = compute_inflow_timeseries(gen_row, n)  # pd.Series aligned to n.snapshots
        # Energiezufuhr pro Zeitschritt (MWh) - falls snapshots stündlich, multipliziere mit 1h
        # (Achte auf units in deinem network!)
        # --- einfache Heuristik für e_nom: RESERVOIR_HOURS * p_nom
        e_nom = p_nom * RESERVOIR_HOURS

        # Erstelle eindeutige Namen
        base_name = f"PHS_{gid}"
        storage_name = f"{base_name}_store"
        turbine_name = f"{base_name}_turb"
        pump_name = f"{base_name}_pump"

        # 1) StorageUnit (Reservoir)
        n.add("StorageUnit",
              name=storage_name,
              bus=bus,
              p_nom=None,            # doppelseitig gesteuert vom pump/turb
              e_nom=e_nom,
              efficiency_store=1.0,  # wenn du Verluste modellieren willst, setze <1
              standing_loss=0.0,
              capital_cost=CAPEX_RES_PER_MWH * e_nom)  # optional für Invest-Objekt

        # 2) Turbine (Erzeuger) - produziert Strom durch Entladung
        n.add("Generator",
              name=turbine_name,
              bus=bus,
              p_nom=p_nom,
              marginal_cost=0.0,
              efficiency=EFF_TURBINE,
              # falls du Investitionsentscheidung erlauben willst:
              p_nom_extendable=False,
              capital_cost=CAPEX_PHS_PER_MW * p_nom)  # optional

        # 3) Pump (neg. Erzeuger bzw. Verbraucher) - modelliert als "Link" oder "Generator" mit negative output
        # Empfehlung: Link component to model conversion from electricity to store (pump) and back (turbine).
        # We'll add two Links: pump: bus->store, turb: store->bus. If Link not supported, use Generator + Store power balance.
        # Use Link to represent conversion with efficiencies.
        n.add("Link",
              name=f"{base_name}_pump_link",
              bus0=bus,
              bus1=bus,  # if store is on same bus, we keep bus; or create virtual bus for store if needed
              p_nom=p_nom,
              efficiency=EFF_PUMP,
              capital_cost=CAPEX_PHS_PER_MW * p_nom)  # pump capex

        # Mark created items
        created.append({
            "gen_id": gid,
            "bus": bus,
            "p_nom": p_nom,
            "storage": storage_name,
            "turbine": turbine_name,
            "pump_link": f"{base_name}_pump_link"
        })

        # optional: kopieren von Metadaten / tags
        # z.B. markiere, dass der Standort retrofitfähig ist
        # (du kannst später auf diese Markierung filtern)
        if 'tags' in n.generators.columns:
            # das ist pseudo; du musst ggf. ein Feld anlegen oder n.meta verwenden
            pass

        # 4) Optional: entferne Generator (deferred until we processed all)
        if remove_old:
            n.remove("Generator", gid)

    print(f"PHS für {len(created)} Standorte erzeugt.")
    return created

# === Anwendung ===
n = pypsa.Network("./data/networks/elec_s_37.nc")  # zB .nc oder .yaml
created = retrofit_hydro_to_phs(n, remove_old=True)
# n.lopf(...)  # löse das Modell wie gewohnt

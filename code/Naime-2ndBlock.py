# Description of this code:
# - inflow of all hydro-storage is scaled by 1.5.
# - p_nom of hydro and phs is made extendable.
# - norway?
# = scale and extendableness of system is altered

import pypsa
import pandas as pd
import os

# -------------------------------
# Utility: Annuity Function
# -------------------------------
def annuity(n, r):
    if r > 0:
        return r / (1.0 - (1.0 + r) ** -n)
    else:
        return 1.0 / n

# -------------------------------
# Step 1: Load Network
# -------------------------------
def load_and_create_base_network():
    file_path = './data/networks/elec_s_37.nc'
    network = pypsa.Network()
    network.import_from_netcdf(file_path)
    return network

# -------------------------------
# Step 2: Add Battery Storage
# -------------------------------
def add_battery_storage(network):
    max_hours = 8
    for bus in network.buses.index:
        network.add("StorageUnit",
                    name=f"{bus} battery",
                    bus=bus,
                    carrier="batteries",
                    p_nom_extendable=True,
                    max_hours=max_hours,
                    efficiency_store=0.92,
                    efficiency_dispatch=0.92,
                    capital_cost=annuity(16, 0.05) * 81e3 * (1 + 0.021) + max_hours * annuity(16, 0.05) * 236e3,
                    cyclic_state_of_charge=True)
    return network

# -------------------------------
# Step 3: Add CO2 Constraint
# -------------------------------
def add_co2_limit(network, co2_limit=1.0):
    base_co2_emissions = 1012028560.7495946
    network.add("GlobalConstraint",
                "CO2Limit",
                carrier_attribute="co2_emissions",
                sense="<=",
                constant=co2_limit * base_co2_emissions)
    return network

# -------------------------------
# Step 4: Solve Network
# -------------------------------
def solve_network(network):
    solver_options = {
        "NumericFocus": 3,
        "ScaleFlag": 2,
        "Method": 2,
        "Crossover": 0,
        "Presolve": 2,
        "AggFill": 0
    }
    network.optimize(solver_name='gurobi', solver_options=solver_options)
    return network

# -------------------------------
# Step 5: Export Network to NetCDF
# -------------------------------
def export_network(network, co2_limit=1.0):
    folder = f"norway_hydro_scaled_results/network_co2_{int(co2_limit * 100)}"
    os.makedirs(folder, exist_ok=True)
    network.export_to_netcdf(f"{folder}/network.nc")

# -------------------------------
# Step 6: Export Network Tables to CSV
# -------------------------------
def export_network_to_csv(network, co2_limit=1.0):
    folder = f"norway_hydro_scaled_results/network_co2_{int(co2_limit * 100)}"
    os.makedirs(folder, exist_ok=True)
    network.export_to_csv_folder(folder)
    
# -------------------------------
# Step 7: Rescale Loads
# -------------------------------
def rescale_loads(n):
    """
    Rescales the electricity demand (load) of each country in the network 
    based on external energy consumption data (in Gtoe) converted to TWh.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.

    Returns
    -------
    pypsa.Network
        The updated network with rescaled load values.
    """

    # Electricity consumption per country (in Gtoe) - from CFE or external data
    gtoe_electricity = {
    "AT0 0": 11.975,
    "BE0 0": 15.118,
    "BG0 0": 5.364,
    "CZ0 0": 12.211,
    "DE0 0": 89.121,
    "DK0 0": 0.57 * 5.929,
    "DK6 0": 0.43 * 5.929,
    "EE5 0": 1.516,
    "ES0 0": 0.98 * 39.8473, #no data for Spain from CFE, data taken from Portugal and rescaled accordingly, i.e. d_old / d_new
    "ES3 0": 0.02 * 39.8473, 
    "FI6 0": 8.476,
    "FR0 0": 70.668,
    "GB4 0": 0.98 * 51.032,
    "GB2 0": 0.02 * 51.032,
    "GR0 0": 8.002,
    "HR0 0": 3.054,
    "HU0 0": 8.583,
    "IE2 0": 4.972,
    "IT0 0": 0.96 * 49.571,
    "IT1 0": 0.04 * 49.571,
    "LT5 0": 1.512,
    "LU0 0": 1.180,
    "LV5 0": 2.320,
    "NL0 0": 19.456,
    "PL0 0": 35.345,
    "PT0 0": 7.941,
    "RO0 0": 11.551,
    "SE6 0": 14.514,
    "SI0 0": 2.510,
    "SK0 0": 4.362
}


    # Conversion factor from Gtoe to TWh
    gtoe_to_twh = 11.630 # preferred conversion
    target_twh = pd.Series(gtoe_electricity) * gtoe_to_twh  # Convert Gtoe to TWh

    # Identify country codes from the 'bus' column in the loads table
    load_country = n.loads['bus'].apply(lambda x: x.split('_')[-1])

    # Aggregate current total load per country from the model (in MWh), then convert to TWh
    load_sums_by_country = n.loads_t.p_set.sum().groupby(load_country).sum() / 1e6

    # Calculate scaling factor per country: desired TWh / current TWh
    scaling_factors = target_twh / load_sums_by_country

    # Remove any countries for which we don't have data
    scaling_factors = scaling_factors.dropna()

    # Apply the scaling factor to all load time series per country
    for load in n.loads.index:
        country = load_country[load]
        if country in scaling_factors:
            n.loads_t.p_set[load] *= scaling_factors[country]

    return n

# -------------------------------
# Step 7.5: Scale Hydro Inflow 
# -------------------------------
def scale_hydro_inflow_all_countries(network, factor):
    """
    Scale inflow for all hydro storage units across all countries.
    Skips PHS units if needed.
    """
    for su_name, su in network.storage_units.iterrows():
        carrier = str(su.carrier).lower()
        if "phs" in carrier:
            continue  # skip PHS if desired
        # Scale inflow time series if it exists
        if hasattr(network.storage_units_t, "inflow") and su_name in network.storage_units_t.inflow:
            network.storage_units_t.inflow[su_name] *= factor
        # Optionally scale static inflow attribute
        if "inflow" in network.storage_units.columns:
            try:
                network.storage_units.loc[su_name, "inflow"] *= factor
            except Exception:
                pass

# -------------------------------
# Step 7.6: Make Hydro & PHS Extendable
# -------------------------------
def scale_hydro_and_phs(network):
    network.generators.loc[network.generators['carrier'] == 'hydro', 'p_nom_extendable'] = True
    network.storage_units.loc[network.storage_units['carrier'].str.lower() == 'phs', 'p_nom_extendable'] = True
    return network

# -------------------------------
# Step 8: Run Scenarios
# -------------------------------
if __name__ == "__main__":
    co2_limits = [0]  # Set your desired CO2 limit fraction here

    for co2_limit in co2_limits:
        network = load_and_create_base_network()
        network = rescale_loads(network)
        #scale_hydro_inflow_in_norway(network, factor=1.5)  # <-- Increase inflow in Norway
        scale_hydro_inflow_all_countries(network, factor=1.5)
        network = scale_hydro_and_phs(network)
        network = add_battery_storage(network)
        network = add_co2_limit(network, co2_limit)
        network.snapshots = network.snapshots[:168]  # Optional: restrict time
        network = solve_network(network)
        export_network(network, co2_limit)
        export_network_to_csv(network, co2_limit)

# Plot result
hydro_gens = network.generators[network.generators.carrier == "hydro"].index
hydro_generation = network.generators_t.p[hydro_gens].sum(axis=1)
phs_units = network.storage_units[network.storage_units.carrier.str.contains("hydro|PHS", case=False)].index
phs_power = network.storage_units_t.p[phs_units].sum(axis=1)

import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

plt.plot(hydro_generation, label="Hydro Generation (Generators)")
plt.plot(phs_power, label="PHS Power (Storage Units)")

plt.axhline(0, color="black", linewidth=0.8)
plt.title("Hydropower: Generation vs Pumping")
plt.ylabel("Power [MW]")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()

# why is hydro 0?
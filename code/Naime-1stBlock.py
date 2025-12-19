# Desription of this code:
# - all hydro is converted to phs with p_min_pu = -1 and efficiency_store = 0.9
# - Extandable grid to prevent congestion
# = implementation of hydro to phs conversion under perfect

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
    file_path = './data/networks/elec_s_37.nc'  # 2018
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
#def solve_network(network):
#    solver_options = {
#        "NumericFocus": 3,
#        "ScaleFlag": 2,
#        "Method": 2,
#        "Crossover": 1, # Changed from 0 to 1 as Alex suggested
#       "Presolve": 2,
#        "AggFill": 0
#    }
#    network.optimize(solver_name='gurobi', solver_options=solver_options)
#    return network

def solve_network(network):
    solver_options = {
        "OutputFlag": 1,
        "FeasibilityTol": 1e-6,
        "Method": 2,        # Selects the algorithm to solve the linear problem (Use barrier method)
        "Crossover": 1      # Changed from 0 to 1 as Alex suggested
    }
    network.optimize(solver_name='gurobi', solver_options=solver_options)
    return network

# -------------------------------
# Step 5: Export Network to NetCDF
# -------------------------------
def export_network(network, co2_limit=1.0):
    folder = f"all_hydro_to_phs_expand_transmission/netcdf/co2_{int(co2_limit * 100)}"
    os.makedirs(folder, exist_ok=True)
    network.export_to_netcdf(f"{folder}/network.nc")

# -------------------------------
# Step 6: Export Network Tables to CSV
# -------------------------------
def export_network_to_csv(network, co2_limit=1.0):
    folder = f"./results/all_hydro_to_phs_expand_transmission/csv/co2_{int(co2_limit * 100)}"
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
# Step 8: Convert Hydro to Pumped Hydro
# -------------------------------
def convert_hydro_to_phs(network):
    # Apply to all hydro units, not just selected countries
    hydro_mask = network.storage_units.carrier == "hydro"

    # Update operational characteristics for all hydro units
    network.storage_units.loc[hydro_mask, "p_min_pu"] = -1
    network.storage_units.loc[hydro_mask, "efficiency_store"] = 0.9

    return network

# -------------------------------
# Step 9: Expand Transmission Capacity
# -------------------------------
def expand_transmission_capacity(network):
    network.links.loc[:, "p_nom_extendable"] = True  # Set a high nominal power for all transmission links
    
    return network
    
# -------------------------------
# Step 10: Remove Battery Units
# -------------------------------
#def remove_batteries(network):
#    battery_units = network.storage_units[network.storage_units.carrier == "batteries"].index
#    network.storage_units.drop(battery_units, inplace=True)
#    return network

# -------------------------------
# Step 11: Run Scenarios  
# -------------------------------
if __name__ == "__main__":
    co2_limits = [0]  # Example scenario with 0, fully decarbonized

    for co2_limit in co2_limits:
        network = load_and_create_base_network()
        network = rescale_loads(network)
        network = convert_hydro_to_phs(network)
        network = expand_transmission_capacity(network)  
        network = add_battery_storage(network)
        network = add_co2_limit(network, co2_limit)
        network.snapshots = network.snapshots[:168]  # Optional: restrict time
        network = solve_network(network)
        export_network(network, co2_limit)
        export_network_to_csv(network, co2_limit)

# -------------------------------
# Step 12: Show Results
# -------------------------------
print(len(network.storage_units))
print(len(network.snapshots))

#show generator dataframe
network.generators_t.p

# 
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


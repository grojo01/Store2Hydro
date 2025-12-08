import pypsa
import pandas as pd
import os
import rainflow

# -------------------------------
# Utility: Annuity Function
# -------------------------------
def annuity(n, r):
    """
    Calculates the annuity factor.
    """
    if r > 0:
        return r / (1.0 - (1.0 + r) ** -n)
    else:
        return 1.0 / n

# -------------------------------
# Step 1: Load Network
# -------------------------------
def load_and_create_base_network():
    """
    Loads the base PyPSA network from the NetCDF file.
    """
    file_path = './elec_s_37.nc'  # 2018
    network = pypsa.Network()
    network.import_from_netcdf(file_path)
    return network

# -------------------------------
# Step 2: Add Battery Storage
# -------------------------------
def add_battery_storage(network):
    """
    Adds extendable battery storage units to every bus.
    """
    max_hours = 8
    # Calculate capital cost (using the annuity function)
    # The cost calculation remains in the original form to preserve logic.
    capital_cost_per_bus = annuity(16, 0.05) * 81e3 * (1 + 0.021) + max_hours * annuity(16, 0.05) * 236e3

    for bus in network.buses.index:
        network.add("StorageUnit",
                    name=f"{bus} battery",
                    bus=bus,
                    carrier="batteries",
                    p_nom_extendable=True,
                    max_hours=max_hours,
                    efficiency_store=0.92,
                    efficiency_dispatch=0.92,
                    capital_cost=capital_cost_per_bus,
                    cyclic_state_of_charge=True)
    return network

# -------------------------------
# Step 3: Add CO2 Constraint
# -------------------------------
def add_co2_limit(network, co2_limit=1.0):
    """
    Adds a global constraint on CO2 emissions.
    """
    base_co2_emissions = 1012028560.7495946 # Network's base emissions
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
    """
    Solves the network optimization problem using Gurobi.
    """
    solver_options = {
        "OutputFlag": 1,
        "FeasibilityTol": 1e-6,
        "Method": 2,      # Barrier Method
        "Crossover": 1    # Crossover activated (as suggested by Alex)
    }
    network.optimize(solver_name='gurobi', solver_options=solver_options)
    return network

# -------------------------------
# Step 5: Export Network to NetCDF
# -------------------------------
def export_network(network, co2_limit=1.0):
    """
    Exports the optimized network to a NetCDF file in a scenario-specific folder.
    """
    folder = f"all_hydro_to_phs_expand_transmission/netcdf/co2_{int(co2_limit * 100)}"
    os.makedirs(folder, exist_ok=True)
    network.export_to_netcdf(f"{folder}/network.nc")

# -------------------------------
# Step 6: Export Network Tables to CSV
# -------------------------------
def export_network_to_csv(network, co2_limit=1.0):
    """
    Exports all network tables as CSV files in a scenario-specific folder.
    """
    folder = f"all_hydro_to_phs_expand_transmission/csv/co2_{int(co2_limit * 100)}"
    os.makedirs(folder, exist_ok=True)
    network.export_to_csv_folder(folder)

# -------------------------------
# Step 7: Rescale Loads
# -------------------------------
def rescale_loads(n):
    """
    Rescales the electricity demand (load) of each country in the network 
    based on external consumption data (in Gtoe).
    """

    # Electricity consumption per country (in Gtoe)
    gtoe_electricity = {
    "AT0 0": 11.975, "BE0 0": 15.118, "BG0 0": 5.364, "CZ0 0": 12.211, "DE0 0": 89.121,
    "DK0 0": 0.57 * 5.929, "DK6 0": 0.43 * 5.929, "EE5 0": 1.516, 
    "ES0 0": 0.98 * 39.8473, "ES3 0": 0.02 * 39.8473, 
    "FI6 0": 8.476, "FR0 0": 70.668, 
    "GB4 0": 0.98 * 51.032, "GB2 0": 0.02 * 51.032, 
    "GR0 0": 8.002, "HR0 0": 3.054, "HU0 0": 8.583, "IE2 0": 4.972, 
    "IT0 0": 0.96 * 49.571, "IT1 0": 0.04 * 49.571, 
    "LT5 0": 1.512, "LU0 0": 1.180, "LV5 0": 2.320, "NL0 0": 19.456, 
    "PL0 0": 35.345, "PT0 0": 7.941, "RO0 0": 11.551, "SE6 0": 14.514, 
    "SI0 0": 2.510, "SK0 0": 4.362
    }

    # Conversion factor from Gtoe to TWh
    gtoe_to_twh = 11.630
    target_twh = pd.Series(gtoe_electricity) * gtoe_to_twh  # Target consumption in TWh

    # Identify the country code per load series
    load_country = n.loads['bus'].apply(lambda x: x.split('_')[-1])

    # Current aggregated total load per country (in MWh) and conversion to TWh
    load_sums_by_country = n.loads_t.p_set.sum().groupby(load_country).sum() / 1e6

    # Calculate scaling factor: Target TWh / Current TWh
    scaling_factors = target_twh / load_sums_by_country
    scaling_factors = scaling_factors.dropna() # Remove countries without data

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
    """
    Converts all hydro units to pumped hydro storage (PHS).
    """
    hydro_mask = network.storage_units.carrier == "hydro"

    # Update operational characteristics
    network.storage_units.loc[hydro_mask, "p_min_pu"] = -1   # Allows storing (pumping)
    network.storage_units.loc[hydro_mask, "efficiency_store"] = 0.9 # Sets storing efficiency

    return network

# -------------------------------
# Step 9: Expand Transmission Capacity
# -------------------------------
def expand_transmission_capacity(network):
    """
    Sets all transmission lines to be extendable.
    """
    network.links.loc[:, "p_nom_extendable"] = True
    return network
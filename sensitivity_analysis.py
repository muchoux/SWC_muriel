import os
import numpy as np
import pandas as pd
from datetime import datetime
import numpy_financial as npf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ajouter la possibilité de choisir le scenario, en ensuite d'enregistrer 

# Import all functions from step_by_step.py except run_simulation
from step_by_step import (
    EPV, EPVs, EPVg, PVI, PVOM, EPV_discounted,
    PWCO, PWCI, LCOE, NPV,
    build_cashflow_series, compute_irr, compute_dpbt
)

# -----------------------------
# SETTINGS
# -----------------------------
param_name = "SCI"       # <<< parameter to test (example: "Cu", "ps", "d", "SCI")
param_values = np.linspace(0, 100, 101)  # <<< values to test for this parameter
indicator = "DPBT_project"      # <<< Choose: "LCOE", "NPV", "IRR_project", "IRR_client", "DPBT_project", "DPBT_client"

# Load parameter file
df_params = pd.read_csv("parameters/rapport_2025.csv", sep=";", decimal=",")

# Extract scenarios (all columns except 'parameter')
scenarios = [col for col in df_params.columns if "_" in col]

# Output directory
output_dir = "sensitivity_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# SENSITIVITY LOOP (Talavera only)
# -----------------------------
all_results = []

for sc in scenarios:
    base_params = dict(zip(df_params["parameter"], df_params[sc]))

    for val in param_values:
        # overwrite the chosen parameter
        current_params = base_params.copy()
        current_params[param_name] = val

        # --- Load parameters with same logic as step_by_step ---
        Hopt = float(current_params["Hopt"])
        P = float(current_params["P"])
        PR = float(current_params["PR"]) / 100
        rd = float(current_params["rd"]) / 100
        El = float(current_params["El"])
        N = int(current_params["N"])
        SCI = float(current_params["SCI"]) / 100
        Cu = float(current_params["Cu"])
        COM = float(current_params["COM"]) / 100
        d = float(current_params["d"]) / 100
        pg = float(current_params["pg"])
        ps = float(current_params["ps"])
        rpg = float(current_params["rpg"]) / 100
        rps = float(current_params["rps"]) / 100
        rom = float(current_params["rom"]) / 100
        T = float(current_params["T"]) / 100
        Nd = int(current_params["Nd"])
        Xd = float(current_params["Xd"]) / 100
        Xl = float(current_params["Xl"]) / 100
        Xec = float(current_params["Xec"]) / 100
        Xis = float(current_params["Xis"]) / 100
        il = float(current_params["il"]) / 100
        Nis = int(current_params["Nis"])
        Nl = int(current_params["Nl"])
        dec = float(current_params["dec"]) / 100

        # --- Shared factors ---
        q = 1 / (1 + d)
        Kp = (1 + rom) / (1 + d)
        Ks = (1 + rps) * (1 - rd) / (1 + d)
        Kg = (1 + rpg) * (1 - rd) / (1 + d)

        # --- Core project parameters ---
        pvi = PVI(P, Cu)
        epv = EPV(Hopt, P, PR)
        epvs = EPVs(epv, SCI)
        epvg = EPVg(epv, SCI)
        pvom = PVOM(pvi, COM)

        # --- Compute only the chosen indicator (Talavera) ---
        if indicator == "LCOE":
            value_result = LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd, "talavera")
        elif indicator == "NPV":
            value_result = NPV(pvi, pvom, pg, ps, rpg, rps, rd, N, T, Nd, Xd, epvs, epvg, "talavera")
        elif indicator == "IRR_project":
            cashflows_project = build_cashflow_series(
                pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T, pvom, rom, "talavera",
                Xl, Xec, Xis, il, Nis, Nl, dec,
                perspective="project"
            )
            irr_proj = compute_irr(cashflows_project)
            value_result = irr_proj * 100 if irr_proj is not None else None
        elif indicator == "DPBT_project":
            cashflows_project = build_cashflow_series(
                pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T, pvom, rom, "talavera",
                Xl, Xec, Xis, il, Nis, Nl, dec,
                perspective="project"
            )
            dpbt_proj = compute_dpbt(abs(cashflows_project[0]), cashflows_project, d)
            value_result = dpbt_proj if dpbt_proj is not None else "Not recovered"
        elif indicator == "IRR_client":
            cashflows_client = build_cashflow_series(
                pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T, pvom, rom, "talavera",
                Xl, Xec, Xis, il, Nis, Nl, dec,
                perspective="client"
            )
            irr_client = compute_irr(cashflows_client)
            value_result = irr_client * 100 if irr_client is not None else None
        elif indicator == "DPBT_client":
            cashflows_client = build_cashflow_series(
                pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T, pvom, rom, "talavera",
                Xl, Xec, Xis, il, Nis, Nl, dec,
                perspective="client"
            )
            dpbt_client = compute_dpbt(abs(cashflows_client[0]), cashflows_client, d)
            value_result = dpbt_client if dpbt_client is not None else "Not recovered"
        else:
            raise ValueError(f"Unknown indicator: {indicator}")

        # save result
        all_results.append({
            "Scenario": sc,
            param_name: val,
            indicator: value_result
        })

# -----------------------------
# EXPORT TO EXCEL
# -----------------------------
df_out = pd.DataFrame(all_results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# -----------------------------
# PLOT DATA
# -----------------------------

correct_scenario = False
chosen_scenario = "all"

print("Choose a scenario")
scenarios_to_choose_from = np.append(df_out["Scenario"].unique(), "all")

for i,s in enumerate(scenarios_to_choose_from):
    print(f"{i}. {s}")
    
choice = input("Choose the corresponding number : ")

try:
    choice_idx = int(choice)
    if 0 <= choice_idx < len(scenarios_to_choose_from):
        chosen_scenario = scenarios_to_choose_from[choice_idx]
    else:
        raise ValueError
except ValueError:
    print("Choix invalide, 'all' sera utilisé par défaut.")
    chosen_scenario = "all"
    
print(f"✅ Scénario choisi : {chosen_scenario}")

# Variables for case DPBT_project
max_val = 0
min_val = 0
y_ticks = []
not_recovered_val = 0
plt.figure(figsize=(20, 15))

# Set up data for y axis 
if indicator == "DPBT_project":
    all_y_values = [y for y in df_out[indicator] if isinstance(y, (int, float))]
    min_val = int(np.floor(min(all_y_values)))
    max_val = int(np.ceil(max(all_y_values)))
    y_ticks = ["Not Recovered"] + [str(y) for y in range(min_val, max_val + 1)]
    not_recovered_val = max_val + 1  # value to replace Not recovered

scenarios = df_out["Scenario"].unique() if chosen_scenario == "all" else [chosen_scenario]
for scenario in scenarios:
    df_s = df_out[df_out["Scenario"] == scenario]
    x_data = df_s[param_name].tolist()

    if indicator == "DPBT_project":
        y_data = [y if isinstance(y, (int, float)) else not_recovered_val for y in df_s[indicator]]
    else:
        y_data = df_s[indicator].tolist()

    plt.plot(x_data, y_data, label=scenario)

# Labels and title
plt.xlabel(param_name)
plt.ylabel(indicator)
plt.title(f"{indicator} vs {param_name} by Scenario")
plt.legend(title="Scenario", loc="upper left", bbox_to_anchor=(1,1)) 
plt.grid(True)

# Ticks to have a proper y axis
if indicator == "DPBT_project":
    plt.yticks([not_recovered_val] + list(range(min_val, max_val + 1)), y_ticks)

plt.tight_layout()

# Save under a folder thats named after the param_name, scenario, and indicator
# Save both the excel and the graph
save_dir_per_scenario = (
    f"all_scenarios"
    if chosen_scenario == "all"
    else f"scenario_{chosen_scenario}"
)

os.makedirs(os.path.join(output_dir, save_dir_per_scenario), exist_ok=True)

# Save excel
excel_filename = os.path.join(os.path.join(output_dir, save_dir_per_scenario), f"sensitivity_{param_name}_{indicator}_talavera_{timestamp}.xlsx")
data_to_save = df_out if chosen_scenario == "all" else df_out.loc[df_out["Scenario"] == chosen_scenario]
data_to_save.to_excel(excel_filename, index=False)

# Save png
png_filename = os.path.join(os.path.join(output_dir, save_dir_per_scenario), f"sensitivity_{param_name}_{indicator}_talavera_{timestamp}.png")
plt.savefig(png_filename)

# Print ending
print(f"\n✅ Sensitivity analysis (Talavera only) completed for {indicator} varying {param_name}.")
print(f"Results saved : excel to {excel_filename} and png to {png_filename}")
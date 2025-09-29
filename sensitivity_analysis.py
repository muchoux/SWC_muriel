import os
import numpy as np
import pandas as pd
from datetime import datetime
import numpy_financial as npf

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
filename = os.path.join(output_dir, f"sensitivity_{param_name}_{indicator}_talavera_{timestamp}.xlsx")
df_out.to_excel(filename, index=False)

print(f"\n✅ Sensitivity analysis (Talavera only) completed for {indicator} varying {param_name}.")
print(f"Results saved to: {filename}")

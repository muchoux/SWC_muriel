import os
import numpy as np
import pandas as pd
from datetime import datetime
import numpy_financial as npf
import matplotlib.pyplot as plt

# --- Fonctions du modèle (inchangées) ---
from final import (
    EPV, EPVs, EPVg, PVI, PVOM, EPV_discounted,
    PWCO, PWCI, LCOE, NPV_cashflows,
    build_cashflow_series, compute_irr, compute_dpbt
)
from utils import get_metric_by_kwp_interval


# ==============================
# Loader robuste des tarifs
# ==============================
def load_tariffs(path: str) -> pd.DataFrame:
    """
    Charge un CSV de tarifs avec séparateur ';', nettoie les colonnes,
    supprime une éventuelle colonne d'index, force les types numériques
    et renvoie un df trié par 'energy'.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", engine="python")

    # supprimer éventuelles colonnes d'index (Unnamed: 0, etc.)
    drop_me = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_me:
        df = df.drop(columns=drop_me)

    # noms de colonnes normalisés
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "energy" not in df.columns:
        raise ValueError(
            f"Colonne 'energy' absente. Colonnes trouvées: {list(df.columns)}."
        )

    # conversions numériques
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    for c in df.columns:
        if c != "energy":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # tri/clean
    df = df.sort_values("energy").reset_index(drop=True)
    return df


# ==============================
# Helper: ps depuis elec_tariffs
# ==============================
def ps_from_tariffs(city: str, monthly_kwh: float, df: pd.DataFrame) -> float:
    """
    Retourne ps (monnaie/kWh) pour une ville et un kWh/mois donné.
    - df contient 'energy' (seuils: 0, 30, 140, ...) et une colonne par ville.
    - Affectation: [seuil_i, seuil_{i+1}) ; au-delà du dernier -> dernière tranche.
    - Les valeurs du CSV sont en centimes/kWh (ex. 14.12), on renvoie en monnaie/kWh (0.1412).
    """
    city = city.lower().strip()
    if city not in df.columns:
        raise ValueError(f"City '{city}' not found in elec_tariffs.csv columns: {list(df.columns)}")

    edges = df["energy"].to_numpy()
    idx = np.searchsorted(edges, monthly_kwh, side="right") - 1
    idx = max(0, min(idx, len(df) - 1))

    price_ct_per_kwh = float(df.loc[idx, city])   # ex. 14.12
    return price_ct_per_kwh / 100.0               # -> 0.1412


# -----------------------------
# SETTINGS
# -----------------------------
scenario = "2"
param_name = "P"       # <<< paramètre à tester (ex: "Cu", "ps", "d", "SCI")
param_values = np.linspace(0.5, 200, 400)  # <<< valeurs à tester pour ce paramètre
indicator = "LCOE"  # <<< "LCOE", "NPV", "IRR_project", "IRR_client", "DPBT_project", "DPBT_client"

# -----------------------------
# DATA INPUTS
# -----------------------------
# Paramètres par ville_scénario
df_params = pd.read_csv("parameters/rapport_2025.csv", sep=";", decimal=",")

# Prix capex par kWc (ton utilitaire)
df_price = pd.read_excel("parameters/final_price.xlsx", header=[0, 1])

# Tarifs élec par tranches mensuelles (kWh/mois), colonnes = villes
df_tariffs = load_tariffs("parameters/elec_tariffs.csv")

# Colonnes des scénarios dans le CSV de paramètres (toutes sauf 'parameter')
scenarios = [col for col in df_params.columns if "_" in col]

# Dossier de sortie
output_dir = "sensitivity_analysis_results"
os.makedirs(output_dir, exist_ok=True)


# -----------------------------
# SENSITIVITY LOOP
# -----------------------------
all_results = []

for sc in scenarios:
    base_params = dict(zip(df_params["parameter"], df_params[sc]))

    # if param_name == "P":
    #     Hopt0 = float(base_params["Hopt"])
    #     PR0   = float(base_params["PR"]) / 100.0
    #     # seuils théoriques de P où ps change
    #     P_30   = 30.0  * 12.0 / (Hopt0 * PR0)
    #     P_140  = 140.0 * 12.0 / (Hopt0 * PR0)
    #     print(f"[{sc}] ruptures ps aux ~ P={P_30:.3f} kWc et P={P_140:.3f} kWc")


    for val in param_values:
        # overwrite le paramètre à tester
        current_params = base_params.copy()
        current_params[param_name] = val

        # --- Lecture et conversions ---
        Hopt = float(current_params["Hopt"])
        P = float(current_params["P"])
        PR = float(current_params["PR"]) / 100
        rd = float(current_params["rd"]) / 100
        El = float(current_params["El"])
        N = int(current_params["N"])
        SCI = float(current_params["SCI"]) / 100
        COM = float(current_params["COM"]) / 100
        d = float(current_params["d"]) / 100
        pg = float(current_params["pg"])
        # ps sera défini depuis df_tariffs (on ignore current_params["ps"])
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

        # Ville et techno à partir du nom de colonne 'sc' (ex: 'lima_2')
        city_code = sc.split('_')[0].lower()
        techno_code = sc.split('_')[1]

        # Capex unitaire (ton utilitaire d'échelonnement par kWc)
        Cu = round(float(get_metric_by_kwp_interval(df_price, city_code, techno_code, P)["value"]))

        # --- Calculs de base ---
        pvi = PVI(P, Cu)                        # investissement initial
        epv = EPV(Hopt, P, PR)                  # énergie annuelle (kWh/an)
        epvs = EPVs(epv, SCI)                   # autoconsommée (kWh/an)
        epvg = EPVg(epv, SCI)                   # injectée/évité réseau (kWh/an)
        pvom = PVOM(pvi, COM)                   # O&M annuel (monnaie/an)

        # NEW: ps depuis les tranches mensuelles
        monthly_prod = epv / 12.0               # kWh/mois moyen
        ps = ps_from_tariffs(city_code, monthly_prod, df_tariffs)

        # LCOE
        lcoe_result = LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd)

        # Série de cashflows (perspective projet par défaut ici)
        cashflows = build_cashflow_series(
            pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T,
            pvom, rom, Xl, Xec, Xis, il, Nis, Nl, dec, lcoe_result,
            perspective="project"
        )

        # --- Calcul de l'indicateur demandé ---
        if indicator == "LCOE":
            value_result = lcoe_result

        elif indicator == "NPV":
            value_result = NPV_cashflows(cashflows, d)

        elif indicator == "IRR_project":
            cashflows_project = build_cashflow_series(
                pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T,
                pvom, rom, Xl, Xec, Xis, il, Nis, Nl, dec, lcoe_result,
                perspective="project"
            )
            irr_proj = compute_irr(cashflows_project)
            value_result = irr_proj * 100 if irr_proj is not None else None

        elif indicator == "DPBT_project":
            cashflows_project = build_cashflow_series(
                pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T,
                pvom, rom, Xl, Xec, Xis, il, Nis, Nl, dec, lcoe_result,
                perspective="project"
            )
            dpbt_proj = compute_dpbt(cashflows_project, d)
            value_result = dpbt_proj if dpbt_proj is not None else "Not recovered"

        elif indicator == "IRR_client":
            cashflows_client = build_cashflow_series(
                pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T,
                pvom, rom, Xl, Xec, Xis, il, Nis, Nl, dec, lcoe_result,
                perspective="client"
            )
            irr_client = compute_irr(cashflows_client)
            value_result = irr_client * 100 if irr_client is not None else None

        elif indicator == "DPBT_client":
            cashflows_client = build_cashflow_series(
                pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T,
                pvom, rom, Xl, Xec, Xis, il, Nis, Nl, dec, lcoe_result,
                perspective="client"
            )
            dpbt_client = compute_dpbt(cashflows_client, d)
            value_result = dpbt_client if dpbt_client is not None else "Not recovered"

        else:
            raise ValueError(f"Unknown indicator: {indicator}")

        # Enregistre le résultat
        all_results.append({
            "Scenario": sc,
            param_name: val,
            indicator: value_result
        })


# -----------------------------
# EXPORT / PLOT
# -----------------------------
df_out = pd.DataFrame(all_results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("Choose a scenario")
cities = df_out["Scenario"].str.split("_").str[0].unique()
techs = df_out["Scenario"].str.split("_").str[1].unique()
scenarios_to_choose_from = np.append(techs, cities)
scenarios_to_choose_from = np.append("all", scenarios_to_choose_from)

for i, s in enumerate(scenarios_to_choose_from):
    print(f"{i}. {s}")

choice = input("Choose the corresponding number : ")

try:
    choice_idx = int(choice)
    if 0 <= choice_idx < len(scenarios_to_choose_from):
        chosen_scenario = scenarios_to_choose_from[choice_idx]
    else:
        raise ValueError
except ValueError:
    print("Invalid choice, 'all' is default.")
    chosen_scenario = "all"

print(f"✅ Scénario choisi : {chosen_scenario}")

city_chosen = False  # vouloir tracer toutes les technos d'une ville
tech_chosen = False
if chosen_scenario != "all" and int(choice) >= (len(scenarios_to_choose_from) - len(cities)):
    city_chosen = True
if chosen_scenario != "all" and int(choice) < (len(scenarios_to_choose_from) - len(cities)):
    tech_chosen = True

# --- Figure ---
max_val = 0
min_val = 0
y_ticks = []
not_recovered_val = 0
plt.figure(figsize=(12, 6))

# Axe Y spécifique au DPBT
if indicator in ("DPBT_project", "DPBT_client"):
    all_y_values = [y for y in df_out[indicator] if isinstance(y, (int, float))]
    if len(all_y_values) > 0:
        min_val = int(np.floor(min(all_y_values)))
        max_val = int(np.ceil(max(all_y_values)))
    else:
        min_val, max_val = 0, 1
    y_ticks = ["Not Recovered"] + [str(y) for y in range(min_val, max_val + 1)]
    not_recovered_val = max_val + 1  # valeur factice pour "Not recovered"

scenarios_plot = df_out["Scenario"].unique() if chosen_scenario == "all" else [chosen_scenario]
if city_chosen:
    scenarios_plot = df_out[df_out["Scenario"].str.startswith(chosen_scenario)]["Scenario"].unique()
if tech_chosen:
    scenarios_plot = df_out[df_out["Scenario"].str.endswith(chosen_scenario)]["Scenario"].unique()
    print(scenarios_plot)

for scenario_name in scenarios_plot:
    df_s = df_out[df_out["Scenario"] == scenario_name]
    x_data = df_s[param_name].tolist()

    if indicator in ("DPBT_project", "DPBT_client"):
        y_data = [y if isinstance(y, (int, float)) else not_recovered_val for y in df_s[indicator]]
    else:
        y_data = df_s[indicator].tolist()

    plt.plot(x_data, y_data, label=scenario_name)
if param_name == "P":
    xlabel = "Installed power (kWp)"
if param_name == "SCI":
    xlabel = "Self-consumption rate (%)"
if indicator == "LCOE":
    ylabel = "LCOE (¢/kWh)"
if indicator == "DPBT_project":
    ylabel = "DPBT (years)"
plt.xlabel(xlabel, fontsize=16)
plt.ylabel(ylabel, fontsize=16)
plt.title(f"{indicator} vs {param_name}")
plt.legend(title="Scenario", title_fontsize=14, fontsize=8, loc="upper right")
plt.grid(True)

if indicator in ("DPBT_project", "DPBT_client"):
    plt.yticks([not_recovered_val] + list(range(min_val, max_val + 1)), y_ticks)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(f"DPBT vs {param_name}", fontsize=20)

# Sauvegardes
save_dir_per_scenario = "all_scenarios" if chosen_scenario == "all" else f"scenario_{chosen_scenario}"
os.makedirs(os.path.join(output_dir, save_dir_per_scenario), exist_ok=True)

excel_filename = os.path.join(output_dir, save_dir_per_scenario,
                              f"sensitivity_{param_name}_{indicator}_noSource_{timestamp}.xlsx")
data_to_save = df_out if chosen_scenario == "all" else df_out.loc[df_out["Scenario"] == chosen_scenario]
data_to_save.to_excel(excel_filename, index=False)

png_filename = os.path.join(output_dir, save_dir_per_scenario,
                            f"sensitivity_{param_name}_{indicator}_noSource_{timestamp}.png")
plt.savefig(png_filename)

print(f"\n✅ Sensitivity analysis completed for {indicator} varying {param_name}.")
print(f"Results saved : excel to {excel_filename} and png to {png_filename}")

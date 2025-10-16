import numpy as np
import pandas as pd
import os
from datetime import datetime
import numpy_financial as npf
import matplotlib.pyplot as plt


# --- Select scenario and city ---
# This section is the only one meant to be modified by the user.
scenario = "perc"
leasing = "LCOE"  # only relevant if scenario 1; options: "LCOE" or "PS"
city = "arequipa"
save_results = False #whether to save results to Excel or not

# --- Import data ---
# Load the CSV file containing parameters for different cities and scenarios.
df = pd.read_csv("parameters/rapport_2025.csv", sep=";", decimal=",")

# Dynamically build the column name based on selected city and scenario.
column_key = f"{city.lower()}_{scenario}"

# Check that the specified column exists in the CSV; raise error if not.
if column_key not in df.columns:
    available = [col for col in df.columns if "_" in col]
    raise ValueError(
        f"\n Error: The column '{column_key}' does not exist in the parameter file.\n"
        f" Available combinations: {', '.join(available)}\n"
        f"  Please check the 'city' and 'scenario' inputs."
    )

# Create a dictionary mapping parameter names to their values for the selected case.
param_dict = dict(zip(df["parameter"], df[column_key]))

# --- Parameters ---
Hopt = float(param_dict["Hopt"])         # Global irradiation on optimum inclined plane (kWh/m²/year)
P = float(param_dict["P"])               # Installed PV peak power (kWp)
PR = float(param_dict["PR"]) / 100       # Performance ratio (%)
rd = float(param_dict["rd"]) / 100       # Annual degradation rate (%/year)
El = float(param_dict["El"])             # Annual household electricity consumption (kWh/year)
N = int(param_dict["N"])                 # Lifetime of the project (years)
SCI = float(param_dict["SCI"]) / 100     # Self-consumption index (%)
Cu = float(param_dict["Cu"])             # Unit cost of the PV system (USD/kWp)
COM = float(param_dict["COM"]) / 100     # opex cost as % of initial investment (%)
d = float(param_dict["d"]) / 100         # Discount rate (%/year)
pg = float(param_dict["pg"])             # Grid sale price (USD/kWh)
ps = float(param_dict["ps"])             # Self-consumption electricity price (USD/kWh)
rpg = float(param_dict["rpg"]) / 100     # Annual increase in grid sale price (%/year)
rps = float(param_dict["rps"]) / 100     # Annual increase in self-consumption price (%/year)
rom = float(param_dict["rom"]) / 100     # Annual increase in opex cost (%/year)
T = float(param_dict["T"]) / 100         # Income tax rate (%/year)
Nd = int(param_dict["Nd"])               # Depreciation period for tax purposes (years)
Xd = float(param_dict["Xd"]) / 100       # Total depreciation allowance (% of investment)
Xl = float(param_dict["Xl"]) / 100       # Fraction of investment financed by loan (%)
Xec = float(param_dict["Xec"]) / 100     # Fraction of investment financed by equity capital (%)
Xis = float(param_dict["Xis"]) / 100     # Fraction of investment financed by subsidy (%)
il = float(param_dict["il"]) / 100       # Annual interest rate of the loan (%/year)
Nis = int(param_dict["Nis"])             # Duration of subsidy repayment (years)
Nl = int(param_dict["Nl"])               # Loan amortization period (years)
dec = float(param_dict["dec"]) / 100     # Annual return on equity (%/year)

# --- WACC and discount rate ---
def WACC(Xl, Xec, il, dec, T):
    """Weighted Average Cost of Capital (%)"""
    if scenario == "1":
        wacc = ((Xec/(Xec+Xl)) * dec + (Xl/(Xec+Xl)) * il * (1 - T)) * 100
    else:
        wacc = ((Xec/(Xec+Xl)) * dec + (Xl/(Xec+Xl)) * il ) * 100
    return wacc

wacc_result = WACC(Xl, Xec, il, dec, T)

if scenario == "2":
    d = wacc_result/100  # use WACC as discount rate if scenario 2
    
# --- Shared factors ---
q = 1 / (1 + d)
Kp = (1 + rom) / (1 + d)
Ks = (1 + rps) * (1 - rd) / (1 + d)
Ks_2 = (1 + rps) * (1 - rd)
Kg = (1 + rpg) * (1 - rd) / (1 + d)


# --- Functions ---
def EPV(Hopt, P, PR): 
    """Annual PV energy output without degradation (kWh/year)"""
    Eyield = Hopt * PR  # kWh/kWp/year
    return P * Eyield

def EPVs(epv, SCI): 
    """Annual self-consumed energy (kWh/year)"""
    return epv * SCI

def EPVg(epv, SCI): 
    """Annual grid-injected energy (kWh/year)"""
    return epv * (1 - SCI)

def PVI(P, Cu): 
    """Initial investment cost (USD)"""
    PVCOST = P * Cu
    return PVCOST

def PVOM(pvi, COM): 
    """Annual OPEX cost (USD/year)"""
    return COM * pvi

def EPV_discounted(Hopt, P, PR, N, rd, d):
    """Total PV electricity generation over N years with degradation (kWh)"""
    factor = (1 - rd) / (1 + d)
    sum_factors = sum(factor ** i for i in range(1, N + 1))
    result = EPV(Hopt, P, PR) * sum_factors
    return result

def PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl):
    """
    Present Worth of Cash Outflows (USD) -- ESPINOZA LOGIC
    Includes investment financing (loan, equity, subsidies), O&M costs, and depreciation.
    """
    # Espinoza: O&M discounted (no (1-T) factor here)
    PWPVOM = pvom * Kp * (1 - Kp**N) / (1 - Kp)

    # Depreciation present worth
    DEP = 0 if Nd == 0 else pvi * Xd / Nd
    PWDEP = DEP * q * (1 - q ** Nd) / (1 - q)

    # Financing terms (Espinoza)
    PVl = Xl * pvi
    PVec = Xec * pvi
    PVis = Xis * pvi

    loan_term = 0
    equity_term = 0
    dividend_term = 0
    subsidy_term = 0

    if Xl > 0:
        loan_annuity = (PVl * il * (1 - T)) / (1 - (1 + il * (1 - T)) ** (-Nl))
        loan_term = loan_annuity * (q * (1 - q ** Nl)) / (1 - q)

    if Xec > 0:
        equity_term = dec * PVec * (q * (1 - q ** N)) / (1 - q)
        dividend_term = PVec * (q ** N)

    if Xis > 0 and Nis > 0:
        subsidy_term = (PVis / Nis) * T * (q * (1 - q ** Nis)) / (1 - q)

    pvinv = loan_term + equity_term + dividend_term + subsidy_term

    return pvinv + PWPVOM - PWDEP * T

def PWCI(ps, pg, N, T, epvs, epvg, lcoe):
    """
    Present Worth of Cash Inflows (PWCI) -- ESPINOZA LOGIC
    Calculates discounted revenue from self-consumed and grid-injected electricity.
    """
    if scenario == "1":
        if leasing == "LCOE":
            price = lcoe
        else:
            price = ps
    elif scenario == "2":
        price = ps
    return (price * epvs * Ks_2 * (1 - Ks_2**(N-1)) / (1 - Ks_2))


def LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd):
    """Levelized Cost of Electricity (USD/kWh) -- ESPINOZA LOGIC"""
    epv_discounted = EPV_discounted(Hopt, P, PR, N, rd, d)
    pwco = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl)
    lcoe_result = pwco / epv_discounted
    return lcoe_result

# def NPV(pvi, pvom, pg, ps, rpg, rps, rd, N, T, Nd, Xd, epvs, epvg, lcoe):
#     """
#     Net Present Value of the project (USD) -- ESPINOZA LOGIC
#     Difference between present value of cash inflows and outflows
#     """
#     pwci = PWCI(ps, pg, N, T, epvs, epvg, lcoe)
#     dep = pvi * Xd * T
#     return pwci - pvi + dep

def build_cashflow_series(
    pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T, pvom, rom,
    Xl, Xec, Xis, il, Nis, Nl, dec,lcoe,
    perspective="project"
):
    """
    Série de cash-flows (nominaux) années 0..N.

    - perspective="project" (IRR projet / non levier) :
        CF0 = -PVI
        CFt = income - opex
        (pas de dette, pas de dividendes, pas de retrait d'equity en année N)

    - perspective="client" (logique Talavera / NCB exploité + financement) :
        CF0 = -PVI            <-- on inclut bien l'investissement initial
        Pour t=1..N :
           CFt = income - opex - annuité_dette - dividendes
        Et en année N :
           CFN -= PVec        <-- retrait du capital equity (flux négatif final)

    Conventions temporelles (Talavera) :
      * Dégradation appliquée à partir de l'année 2 : (1 - rd)^(t-1)
      * Indexations (prix élec., O&M) à partir de l'année 2 : (1 + r)^(t-1)
      * AUCUNE actualisation ici (le discount rate 'd' sert pour NPV/DPBT, pas pour la série)
    """
    flows = []

    # --- Montants de financement ---
    PVl  = Xl  * pvi      # dette
    PVec = Xec * pvi      # equity
    PVis = Xis * pvi      # subvention (si cash étalée, à ajouter si pertinent)

    # # --- Annuité de dette (taux nominal) ---
    # if PVl > 0 and Nl > 0:
    #     loan_annuity = PVl * il / (1 - (1 + il) ** (-Nl))
    # else:
    #     loan_annuity = 0.0

    # --- Année 0 : investissement initial ---
    if perspective == "project":
        CF0 = -pvi
    # elif perspective == "client":
    #     CF0 = -pvi * Xec
    flows.append(CF0)

    if scenario == "1":
        if leasing == "LCOE":
            price = lcoe
        else:
            price = ps
    else:
        price = ps

    # --- Années 1..N ---
    for n in range(1, N + 1):
        # Prix électricité indexés (base = année 1)
        infl_ps = price * ((1 + rps) ** (n - 1))
        infl_pg = pg * ((1 + rpg) ** (n - 1))

        # Production dégradée (base = année 1)
        energy_self = epvs * ((1 - rd) ** (n - 1))
        energy_grid = epvg * ((1 - rd) ** (n - 1))

        income = infl_ps * energy_self + infl_pg * energy_grid

        # O&M indexé (base = année 1)
        opex = pvom * ((1 + rom) ** (n - 1))

        # Amortissement linéaire (base = année 1)
        if Nd == 0:
            depreciation = 0
        else:
            depreciation = T * (pvi * Xd / Nd) if n <= Nd else 0

        if perspective == "project":
            # IRR projet (non levier) : pas de dette, pas de dividendes
            net = income - opex + depreciation

        # elif perspective == "client":
        #     # Logique Talavera (NCB exploité + financement)
        #     debt_service = loan_annuity if (PVl > 0 and n <= Nl) else 0.0
        #     dividend = dec * PVec if PVec > 0 else 0.0

        #     # Si la subvention est un cash effectivement perçu au fil de l'eau :
        #     subsidy_term = (PVis / Nis) if (Xis > 0 and Nis > 0 and n <= Nis) else 0.0
        #     # NB : dans l'exemple Madrid, typiquement Xis = 0 -> terme nul.

        #     net = income - opex - debt_service - dividend + subsidy_term

        #     # Retrait du capital equity en dernière année (flux négatif)
        #     if n == N and PVec > 0:
        #         net -= PVec

        else:
            raise ValueError("perspective must be 'project' or 'client'")

        flows.append(net)

    return flows

def NPV_cashflows(cashflows, d):
    """
    Net Present Value (NPV) based on full cashflow series (USD).
    """
    return npf.npv(d, cashflows)


def compute_irr(cashflows):
    """
    Internal Rate of Return (IRR) based on full cashflow series (decimal).
    """
    return npf.irr(cashflows)


def compute_dpbt(cashflows, d):
    """
    Discounted Payback Time (years).
    Number of years needed to recover the initial investment.
    """
     # t = 0
    cum = cashflows[0]
    if cum >= 0:
        return 0.0

    # t >= 1 (actualisation exponent (t-1))
    for year, cf in enumerate(cashflows[1:], start=1):
        prev_cum = cum
        disc_cf = cf / ((1 + d) ** (year - 1))
        cum += disc_cf
        if cum >= 0:
            # interpolation linéaire entre (year-1) et year
            frac = abs(prev_cum) / disc_cf if disc_cf != 0 else 0.0
            return (year - 1) + frac

    return None


def run_simulation():
    # --- Core project parameters ---
    pvi  = PVI(P, Cu)
    epv  = EPV(Hopt, P, PR)
    epvs = EPVs(epv, SCI)
    epvg = EPVg(epv, SCI)
    pvom = PVOM(P, COM)

    # --- Calcul du LCOE (unique) ---
    lcoe_result = LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd)
    if scenario == "1":
        lcoe_display = lcoe_result * (1 + 0.18 + 0.28)
    else:
        lcoe_display = lcoe_result

    # --- Fonction interne de calcul pour un type de leasing ---
    def _compute_for_leasing(leasing_type):
        """Retourne un DataFrame des flux + indicateurs financiers."""
        global leasing
        leasing = leasing_type

        cashflows = build_cashflow_series(
            pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T,
            pvom, rom, Xl, Xec, Xis, il, Nis, Nl, dec, lcoe_result,
            perspective="project"
        )

        irr_val  = compute_irr(cashflows)
        dpbt_val = compute_dpbt(cashflows, d)
        npv_val  = NPV_cashflows(cashflows, d)

        # Construire un DataFrame colonne
        df = pd.DataFrame({
            "Year": list(range(0, len(cashflows))),
            "Cashflow (USD)": cashflows,
        })
        df["Leasing type"] = leasing_type
        df["LCOE (USD/kWh)"] = round(lcoe_display, 4)
        df["NPV (USD)"] = round(npv_val, 3) if npv_val is not None else None
        df["IRR (%)"] = round(irr_val * 100, 2) if irr_val is not None else None
        df["DPBT (years)"] = dpbt_val if dpbt_val is not None else "Not recovered"

        # Placer les indicateurs sur la première ligne uniquement pour lisibilité
        df.loc[1:, ["LCOE (USD/kWh)", "NPV (USD)", "IRR (%)", "DPBT (years)"]] = ""

        return df

    # --- Construction du résultat final ---
    if scenario == "1":
        df_lcoe = _compute_for_leasing("LCOE")
        df_ps   = _compute_for_leasing("PS")
        result_df = pd.concat([df_lcoe, df_ps], ignore_index=True)
    else:
        result_df = _compute_for_leasing("PS")

    print("\n--- ESPINOZA MODEL ---")
    print(result_df)

    return result_df

# --- Batch summary to CSV over all city_scenarios in the sheet ---
def _set_globals_from_param_dict(param_dict, scenario_suffix):
    """
    Remplit les variables globales attendues par tes fonctions (LCOE, PWCO, PWCI, etc.)
    à partir d'un param_dict et fixe 'scenario' (ex. 'perc' ou 'hit').
    """
    global scenario, leasing
    global Hopt, P, PR, rd, El, N, SCI, Cu, COM, d, pg, ps, rpg, rps, rom, T, Nd, Xd
    global Xl, Xec, Xis, il, Nis, Nl, dec
    global q, Kp, Ks, Ks_2, Kg

    scenario = scenario_suffix  # ex: 'perc', 'hit' (ta logique PWCI tombera dans le cas 'else' = prix ps)

    Hopt = float(param_dict["Hopt"])
    P    = float(param_dict["P"])
    PR   = float(param_dict["PR"]) / 100
    rd   = float(param_dict["rd"]) / 100
    El   = float(param_dict["El"])
    N    = int(float(param_dict["N"]))
    SCI  = float(param_dict["SCI"]) / 100
    Cu   = float(param_dict["Cu"])
    COM  = float(param_dict["COM"]) / 100
    d    = float(param_dict["d"]) / 100
    pg   = float(param_dict["pg"])
    ps   = float(param_dict["ps"])
    rpg  = float(param_dict["rpg"]) / 100
    rps  = float(param_dict["rps"]) / 100
    rom  = float(param_dict["rom"]) / 100
    T    = float(param_dict["T"]) / 100
    Nd   = int(float(param_dict["Nd"]))
    Xd   = float(param_dict["Xd"]) / 100
    Xl   = float(param_dict["Xl"]) / 100
    Xec  = float(param_dict["Xec"]) / 100
    Xis  = float(param_dict["Xis"]) / 100
    il   = float(param_dict["il"]) / 100
    Nis  = int(float(param_dict["Nis"]))
    Nl   = int(float(param_dict["Nl"]))
    dec  = float(param_dict["dec"]) / 100

    # WACC -> d si tu veux l'utiliser pour un scénario spécifique (dans ton code: scenario == "2")
    wacc_result = WACC(Xl, Xec, il, dec, T)
    if scenario == "2":
        # Rare pour tes suffixes 'perc'/'hit', mais on respecte ta logique
        # d devient WACC (décimal)
        # (wacc_result est en %, ton d est en décimal)
        d = wacc_result / 100.0

    # Partagés
    q    = 1 / (1 + d)
    Kp   = (1 + rom) / (1 + d)
    Ks   = (1 + rps) * (1 - rd) / (1 + d)
    Ks_2 = (1 + rps) * (1 - rd)
    Kg   = (1 + rpg) * (1 - rd) / (1 + d)

    # Choix par défaut, éventuellement écrasé lors du calcul
    leasing = "PS"


def _compute_metrics_for_column(df_params, column_key):
    """
    Calcule (paramètres + indicateurs) pour une colonne du type 'arequipa_perc' et
    renvoie une liste de lignes (dicos) pour le DataFrame final.
    """
    rows = []

    # Paramètres bruts pour cette combinaison
    param_dict_raw = dict(zip(df_params["parameter"], df_params[column_key]))

    # On garde en parallèle une version 'affichable' des paramètres (convertis)
    # pour les poser tels quels dans le CSV final
    # (on réutilise la même conversion que celle de ton script)
    _set_globals_from_param_dict(param_dict_raw, scenario_suffix=column_key.rsplit("_", 1)[-1])

    # --- Calculs de base ---
    pvi  = PVI(P, Cu)
    epv  = EPV(Hopt, P, PR)
    epvs = EPVs(epv, SCI)
    epvg = EPVg(epv, SCI)
    pvom = PVOM(pvi, COM)        # ⟵ attention: PVOM prend bien pvi, pas P

    lcoe_result = LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd)
    # Affichage LCOE (ta règle spéciale pour scenario == "1")
    if scenario == "1":
        lcoe_display = lcoe_result * (1 + 0.18 + 0.28)
    else:
        lcoe_display = lcoe_result

    city = column_key.rsplit("_", 1)[0]
    scen = column_key.rsplit("_", 1)[-1]

    # Leasing à calculer (dans ton code, "1" -> 2 leasing; sinon un seul)
    leasing_types = ["LCOE", "PS"] if scenario == "1" else ["PS"]

    for leasing_type in leasing_types:
        # fixe la variable globale utilisée par tes fonctions
        globals()["leasing"] = leasing_type

        cashflows = build_cashflow_series(
            pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T,
            pvom, rom, Xl, Xec, Xis, il, Nis, Nl, dec, lcoe_result,
            perspective="project"
        )
        irr_val  = compute_irr(cashflows)
        dpbt_val = compute_dpbt(cashflows, d)
        npv_val  = NPV_cashflows(cashflows, d)

        # Ligne résultat + TOUS les paramètres (convertis)
        row = {
            "city": city,
            "scenario": scen,
            "leasing": leasing_type,
            "LCOE_USD_per_kWh": round(lcoe_display, 6),
            "NPV_USD": round(float(npv_val), 3) if npv_val is not None else None,
            "IRR_%": round(float(irr_val) * 100, 3) if irr_val is not None else None,
            "DPBT_years": dpbt_val if dpbt_val is not None else None,

            # Paramètres (version convertie utilisée dans les calculs)
            "Hopt_kWhm2y": Hopt,
            "P_kWp": P,
            "PR_frac": PR,
            "rd_frac": rd,
            "El_kWh_y": El,
            "N_years": N,
            "SCI_frac": SCI,
            "Cu_USD_kWp": Cu,
            "COM_frac": COM,
            "d_disc_frac": d,
            "pg_USD_kWh": pg,
            "ps_USD_kWh": ps,
            "rpg_frac": rpg,
            "rps_frac": rps,
            "rom_frac": rom,
            "T_tax_frac": T,
            "Nd_years": Nd,
            "Xd_frac": Xd,
            "Xl_frac": Xl,
            "Xec_frac": Xec,
            "Xis_frac": Xis,
            "il_frac": il,
            "Nis_years": Nis,
            "Nl_years": Nl,
            "dec_frac": dec,
            "PVI_USD": pvi,
            "EPV_kWh_y": epv,
            "EPVs_kWh_y": epvs,
            "EPVg_kWh_y": epvg,
            "PVOM_USD_y1": pvom,
        }
        rows.append(row)

    return rows


def generate_summary_csv(
    input_csv_path="parameters/rapport_2025.csv",
    output_csv_path="summary_results_all_scenarios_2025.csv",
    sep=";", decimal=","
):
    """
    Parcourt toutes les colonnes {ville}_{scenario} et produit un CSV
    avec tous les paramètres + indicateurs (LCOE, NPV, IRR, DPBT).
    """
    df_params = pd.read_csv(input_csv_path, sep=sep, decimal=decimal)
    scenario_cols = [c for c in df_params.columns if c != "parameter" and "_" in c]

    all_rows = []
    for col in scenario_cols:
        all_rows.extend(_compute_metrics_for_column(df_params, col))

    summary_df = pd.DataFrame(all_rows)

    # Tri optionnel pour lisibilité
    summary_df = summary_df.sort_values(by=["city", "scenario", "leasing"]).reset_index(drop=True)

    summary_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Résumé écrit dans: {output_csv_path}\n")
    return summary_df


# Exemple d'utilisation (décommente pour lancer directement) :
if __name__ == "__main__":
    generate_summary_csv()

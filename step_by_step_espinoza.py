import numpy as np
import pandas as pd
import os
from datetime import datetime
import numpy_financial as npf
import matplotlib.pyplot as plt


# --- Select scenario and city ---
# This section is the only one meant to be modified by the user.
scenario = "2"
city = "tacna"
save_results = False #whether to save results to Excel or not

# --- Import data ---
# Load the CSV file containing parameters for different cities and scenarios.
df = pd.read_csv("parameters/parameters_clean.csv", sep=";", decimal=",")

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
COM = float(param_dict["COM"])           # opex cost as % of initial investment (%)
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
    # wacc = ((Xec/(Xec+Xl)) * dec + (Xl/(Xec+Xl)) * il * (1 - T)) * 100
    wacc = ((Xec/(Xec+Xl)) * dec + (Xl/(Xec+Xl)) * il ) * 100
    return wacc

wacc_result = WACC(Xl, Xec, il, dec, T)

if scenario == "2":
    d = wacc_result/100  # use WACC as discount rate if scenario 2
    
# --- Shared factors ---
q = 1 / (1 + d)
Kp = (1 + rom) / (1 + d)
Ks = (1 + rps) * (1 - rd) / (1 + d)
Ks_2 = (1 - rd) / (1 + d)
Kg = (1 + rpg) * (1 - rd) / (1 + d)


# --- Functions ---
def EPV(Hopt, P, PR): 
    """Annual PV energy output without degradation (kWh/year)"""
    # Eyield = Hopt * PR  # kWh/kWp/year
    if city == "arequipa":
        Eyield=1900                      # Annual Yield AREQUIPA (kWh/kWp/año)
    elif city == "tacna":
        Eyield=1576                      # Annual Yield TACNA (kWh/kWp/año)
    elif city == "lima":
        Eyield=1304                      # Annual Yield LIMA THINFILM (kWh/kWp/año)
    return P * Eyield

def EPVs(epv, SCI): 
    """Annual self-consumed energy (kWh/year)"""
    return epv * SCI

def EPVg(epv, SCI): 
    """Annual grid-injected energy (kWh/year)"""
    return epv * (1 - SCI)

def PVI(P, Cu): 
    """Initial investment cost (USD)"""
    # PVCOST = P * Cu
    if city == "arequipa" or city == "tacna":
        if scenario=="1":
            PVCOST_kw=2180                # CAPEX Arequipa/TAcna (USD$/Kwp instalado) SIN IVA
        elif scenario=="2":
            PVCOST_kw=2180*1.18          # CAPEX Arequipa/TAcna (USD$/Kwp instalado)
           
    elif city == "lima":
        if scenario=="1":
            PVCOST_kw=2030               # CAPEX ThinFilm Lima (USD$/Kwp instalado) SIN IVA
        elif scenario=="2":
            PVCOST_kw=2030*1.18          # CAPEX ThinFilm Lima (USD$/Kwp instalado)
        
    PVCOST=PVCOST_kw*P+102.27+151.52+60.61            # Initial investment of PV project(€)
    return PVCOST

def PVOM(P, COM): 
    """Annual OPEX cost (USD/year)"""
    return COM * P

def PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl):
    """
    Present Worth of Cash Outflows (USD) -- ESPINOZA LOGIC
    Includes investment financing (loan, equity, subsidies), O&M costs, and depreciation.
    """
    # Espinoza: O&M discounted (no (1-T) factor here)
    PWPVOM = pvom * Kp * (1 - Kp**N) / (1 - Kp)

    # Depreciation present worth
    if Nd == 0:
        DEP = 0
    else:
        DEP = pvi * Xd / Nd
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
    if scenario == "2":
        return (ps * (1 - T) * epvs * Ks * (1 - Ks**N) / (1 - Ks))
    else:
        return (ps * epvs * Ks * (1 - Ks**N) / (1 - Ks)) + (pg * epvg * Kg * (1 - Kg**N) / (1 - Kg))


def EPV_discounted(Hopt, P, PR, N, rd, d):
    """Total PV electricity generation over N years with degradation (kWh)"""
    factor = (1 - rd) / (1 + d)
    sum_factors = sum(factor ** i for i in range(1, N + 1))
    result = EPV(Hopt, P, PR) * sum_factors
    return result

def LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd):
    """Levelized Cost of Electricity (USD/kWh) -- ESPINOZA LOGIC"""
    epv_discounted = EPV_discounted(Hopt, P, PR, N, rd, d)
    pwco = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl)
    lcoe_result = pwco / epv_discounted
    if scenario == "1":
        lcoe_result *= (1 + 0.18 + 0.28)
    return lcoe_result

def NPV(pvi, pvom, pg, ps, rpg, rps, rd, N, T, Nd, Xd, epvs, epvg, lcoe):
    """
    Net Present Value of the project (USD) -- ESPINOZA LOGIC
    Difference between present value of cash inflows and outflows
    """
    pwci = PWCI(ps, pg, N, T, epvs, epvg, lcoe)
    pwco = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl)
    return pwci - pwco

def build_cashflow_series(
    pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T, pvom, rom,
    Xl, Xec, Xis, il, Nis, Nl, dec,
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

    # --- Annuité de dette (taux nominal) ---
    if PVl > 0 and Nl > 0:
        loan_annuity = PVl * il / (1 - (1 + il) ** (-Nl))
    else:
        loan_annuity = 0.0

    # --- Année 0 : investissement initial ---
    if perspective == "project":
        CF0 = -pvi
    elif perspective == "client":
        CF0 = -pvi * Xec
    flows.append(CF0)

    # --- Années 1..N ---
    for n in range(1, N + 1):
        # Prix électricité indexés (base = année 1)
        infl_ps = ps * ((1 + rps) ** (n))
        infl_pg = pg * ((1 + rpg) ** (n))

        # Production dégradée (base = année 1)
        energy_self = epvs * ((1 - rd) ** (n - 1))
        energy_grid = epvg * ((1 - rd) ** (n - 1))

        income = infl_ps * energy_self + infl_pg * energy_grid

        # O&M indexé (base = année 1)
        opex = pvom * ((1 + rom) ** (n))

        if perspective == "project":
            # IRR projet (non levier) : pas de dette, pas de dividendes
            net = income - opex

        elif perspective == "client":
            # Logique Talavera (NCB exploité + financement)
            debt_service = loan_annuity if (PVl > 0 and n <= Nl) else 0.0
            dividend = dec * PVec if PVec > 0 else 0.0

            # Si la subvention est un cash effectivement perçu au fil de l'eau :
            subsidy_term = (PVis / Nis) if (Xis > 0 and Nis > 0 and n <= Nis) else 0.0
            # NB : dans l'exemple Madrid, typiquement Xis = 0 -> terme nul.

            net = income - opex - debt_service - dividend + subsidy_term

            # Retrait du capital equity en dernière année (flux négatif)
            if n == N and PVec > 0:
                net -= PVec

        else:
            raise ValueError("perspective must be 'project' or 'client'")

        flows.append(net)

    return flows


def compute_irr(cashflows):
    """
    Internal Rate of Return (IRR) based on full cashflow series (decimal).
    """
    return npf.irr(cashflows)


def compute_dpbt(initial_investment, cashflows, d):
    """
    Discounted Payback Time (years).
    Number of years needed to recover the initial investment.
    """
    discounted_sum = 0
    for year, cf in enumerate(cashflows[1:], start=1):  # exclude year 0
        discounted_sum += cf / ((1 + d) ** year)
        if discounted_sum >= initial_investment:
            return year
    return None  # if never recovered


def run_simulation():
    # --- Core project parameters ---
    pvi = PVI(P, Cu)
    epv = EPV(Hopt, P, PR)
    epvs = EPVs(epv, SCI)
    epvg = EPVg(epv, SCI)
    pvom = PVOM(P, COM)
 
    # --- Economic indicators (legacy) ---
    pvcost = pvi
    epv_discounted = EPV_discounted(Hopt, P, PR, N, rd, d)
    pwco_result = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl)
    lcoe_result = LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd)
    pwci_result = PWCI(ps, pg, N, T, epvs, epvg, lcoe_result)
    npv_result = NPV(pvi, pvom, pg, ps, rpg, rps, rd, N, T, Nd, Xd, epvs, epvg, lcoe_result)

    # --- Cashflows for both perspectives ---
    results_perspectives = {}
    for perspective in ["project", "client"]:
        cashflows = build_cashflow_series(
            pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T, pvom, rom,
            Xl, Xec, Xis, il, Nis, Nl, dec,
            perspective=perspective
        )
        irr_result = compute_irr(cashflows)
        dpbt_result = compute_dpbt(abs(cashflows[0]), cashflows, d)
        results_perspectives[perspective] = {
            "IRR (%)": round(irr_result * 100, 2) if irr_result is not None else None,
            "DPBT (years)": dpbt_result if dpbt_result is not None else "Not recovered",
        }

    return {
        "model": "espinoza",
        #"PVI (USD)": round(pvi, 3),
        #"PVOM (USD/year)": round(pvom, 3),
        #"EPV (kWh)": round(epv, 3),
        #"EPVs (kWh)": round(epvs, 3),
        #"EPVg (kWh)": round(epvg, 3),
        #"EPV discounted (kWh)": round(epv_discounted, 3),
        "WACC (%)": round(wacc_result, 3),
        "PWCO (USD)": round(pwco_result, 3),
        "PWCI (USD)": round(pwci_result, 3),
        "LCOE (USD/kWh)": round(lcoe_result, 4),
        "NPV (USD)": round(npv_result, 3),
        "Project perspective": results_perspectives["project"],
        "Client perspective": results_perspectives["client"],
    }


def save_results_to_excel(city, scenario, parameters_dict, result, save=True):
    if not save:
        return
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    filename = f"results_{city.lower()}_{scenario}.xlsx"
    sheet_name = timestamp
    data = {
        "Parameter": list(parameters_dict.keys()) + list(result.keys()),
        "Value": list(parameters_dict.values()) + list(result.values())
    }
    df_out = pd.DataFrame(data)
    with pd.ExcelWriter(filename, mode='a' if os.path.exists(filename) else 'w', engine='openpyxl') as writer:
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    result = run_simulation()
    df_results = pd.DataFrame([result])
    print("\n--- ESPINOZA MODEL ---")
    print(df_results)

    # save_results_to_excel(
    #     city=city,
    #     scenario=scenario,
    #     parameters_dict=param_dict,
    #     result=result,
    #     save=save_results
    # )

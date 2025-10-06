import numpy as np
import pandas as pd
import os
from datetime import datetime
import numpy_financial as npf
import matplotlib.pyplot as plt


# --- Select scenario and city ---
# This section is the only one meant to be modified by the user.
scenario = "1"
city = "lima"
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

# --- Shared factors ---
q = 1 / (1 + d)
Kp = (1 + rom) / (1 + d)
Ks = (1 + rps) * (1 - rd) / (1 + d)
Kg = (1 + rpg) * (1 - rd) / (1 + d)

# --- Functions ---
def EPV(Hopt, P, PR): 
    """Annual PV energy output without degradation (kWh/year)"""
    return P * Hopt * PR

def EPVs(epv, SCI): 
    """Annual self-consumed energy (kWh/year)"""
    return epv * SCI

def EPVg(epv, SCI): 
    """Annual grid-injected energy (kWh/year)"""
    return epv * (1 - SCI)

def PVI(P, Cu): 
    """Initial investment cost (USD)"""
    return Cu * P

def PVOM(PVI, COM): 
    """Annual OPEX cost (USD/year)"""
    return COM * PVI

def PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, source):
    """
    Present Worth of Cash Outflows (USD)
    Includes investment financing (loan, equity, subsidies), O&M costs, and depreciation.
    Adapts formula based on whether source is 'talavera' (with taxes) or 'espinoza'.
    """
    #print("KOPEX:", Kp)
    PWPVOM = pvom * (1 - T) * Kp * (1 - Kp**N) / (1 - Kp) if source == "talavera" else pvom * Kp * (1 - Kp**N) / (1 - Kp)
    if Nd == 0:
        DEP = 0
    else:
        DEP = pvi * Xd / Nd
    #print("DEP:", DEP)
    PWDEP = DEP * q * (1 - q ** Nd) / (1 - q)
    #print("PVIF:", q * (1 - q ** Nd) / (1 - q))
    #print("PW[OPEX]:", PWPVOM)
    #print("PW[Dep]:", PWDEP * T)

    if source == "espinoza":
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
    else:
        pvinv = pvi

    return pvinv + PWPVOM - PWDEP * T

def PWCI(ps, pg, N, T, epvs, epvg, source):
    """
    Present Worth of Cash Inflows (PWCI).
    Calculates discounted revenue from self-consumed and grid-injected electricity.
    Applies tax logic for 'talavera' model.
    """
    if source == "talavera":
        return (ps * epvs * (1 - T) * Ks * (1 - Ks**N) / (1 - Ks)) + (pg * epvg * (1 - T) * Kg * (1 - Kg**N) / (1 - Kg))
    else:
        return (ps * epvs * Ks * (1 - Ks**N) / (1 - Ks)) + (pg * epvg * Kg * (1 - Kg**N) / (1 - Kg))

def WACC(Xl, Xec, il, dec, T):
    """Weighted Average Cost of Capital (%)"""
    return ((Xec/(Xec+Xl)) * dec + (Xl/(Xec+Xl)) * il * (1 - T)) * 100

def EPV_discounted(Hopt, P, PR, N, rd, d):
    """Total PV electricity generation over N years with degradation (kWh)"""
    factor = (1 - rd) / (1 + d)
    sum_factors = sum(factor ** i for i in range(1, N + 1))
    #print("(1 - rd) / (1 + d):", factor)
    #print("sum:", sum_factors)
    return EPV(Hopt, P, PR) * sum_factors

def LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd, source):
    """Levelized Cost of Electricity (USD/kWh)"""
    epv_discounted = EPV_discounted(Hopt, P, PR, N, rd, d)
    #print("epv_discounted:", epv_discounted)
    pwco = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, source)
    return pwco / epv_discounted

def NPV(pvi, pvom, pg, ps, rpg, rps, rd, N, T, Nd, Xd, epvs, epvg, source):
    """
    Net Present Value of the project (USD)
    Difference between present value of cash inflows and outflows
    """
    pwci = PWCI(ps, pg, N, T, epvs, epvg, source)
    pwco = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, source)
    #print("PW[CI]:", pwci)
    #print("LCC:", pwco)
    return pwci - pwco

# def build_cashflow_series(pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T, pvom, source):  #only for 100% equity
#     """
#     Builds annual cash flow series considering income and costs with escalation (USD/year)
#     """
#     flows = [-pvi]
#     for n in range(1, N+1):
#         inflation_ps = ps * ((1 + rps) ** (n - 1))
#         inflation_pg = pg * ((1 + rpg) ** (n - 1))
#         energy_self = epvs * ((1 - rd) ** n)
#         energy_grid = epvg * ((1 - rd) ** n)
#         income = (inflation_ps * energy_self + inflation_pg * energy_grid)
#         opex = pvom * ((1 + rom) ** (n - 1))

# ########### wrong, to change #################

#         # if source == "talavera":
#         #     income *= (1 - T)
#         #     opex *= (1 - T)

# ###########################################

#         net = income - opex
#         flows.append(net)
#     return flows

def build_cashflow_series(
    pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T, pvom, rom, source,
    Xl, Xec, Xis, il, Nis, Nl, dec,
    perspective="project"
):

    flows = []

    # --- Year 0 (initial investment) ---
    if perspective == "project":
        CF0 = -pvi
    elif perspective == "client":
        CF0 = -(Xec * pvi)
    #     if Xis > 0 and Nis == 0:
    #         CF0 += Xis * pvi  # subsidy received upfront
    #     if Xl > 0:
    #         CF0 += Xl * pvi   # loan disbursement
    # else:
    #     raise ValueError("perspective must be 'project' or 'client'")

    flows.append(CF0)

    # --- Financing parameters ---
    PVl = Xl * pvi       # loan amount
    PVec = Xec * pvi     # equity capital
    PVis = Xis * pvi     # subsidy amount

    if PVl > 0 and Nl > 0:
        loan_annuity = (PVl * il * (1 - T)) / (1 - (1 + il * (1 - T)) ** (-Nl))
    else:
        loan_annuity = 0

    # --- Annual loop ---
    for n in range(1, N + 1):
        # Energy revenues
        infl_ps = ps * ((1 + rps) ** (n - 1))
        infl_pg = pg * ((1 + rpg) ** (n - 1))
        energy_self = epvs * ((1 - rd) ** n)
        energy_grid = epvg * ((1 - rd) ** n)
        income = infl_ps * energy_self + infl_pg * energy_grid

        # O&M costs
        opex = pvom * ((1 + rom) ** (n - 1))

        # Loan payments
        loan_payment = loan_annuity if (n <= Nl and PVl > 0) else 0

        # Dividends on equity
        dividend = dec * PVec if PVec > 0 else 0

        # Subsidy amortization (if spread over Nis years)
        subsidy_term = (PVis / Nis) if (Xis > 0 and Nis > 0 and n <= Nis) else 0

        # Net cash flow
        if perspective == "project":
            net = income - opex
        elif perspective == "client":
            net = income - opex - loan_payment - dividend + subsidy_term

        flows.append(net)

    return flows


def compute_irr(cashflows):
    """
    Internal Rate of Return (IRR) based on full cashflow series (decimal).
    """
    return npf.irr(cashflows)


def compute_dpbt(initial_investment, cashflows, d):
    """
    Discounted Payback Time (years, possibly fractional).
    """
    discounted_sum = 0
    for year, cf in enumerate(cashflows[1:], start=1):
        discounted_cf = cf / ((1 + d) ** year)
        discounted_sum += discounted_cf
        if discounted_sum >= initial_investment:
            prev_sum = discounted_sum - discounted_cf
            remaining = initial_investment - prev_sum
            fraction = remaining / discounted_cf  # part of the year needed
            return year - 1 + fraction
    return None


def run_simulation(source):
    if source not in ["talavera", "espinoza"]:
        raise ValueError("source must be 'talavera' or 'espinoza'")

    # --- Core project parameters ---
    pvi = PVI(P, Cu)
    epv = EPV(Hopt, P, PR)
    epvs = EPVs(epv, SCI)
    epvg = EPVg(epv, SCI)
    pvom = PVOM(pvi, COM)
    epv_discounted = EPV_discounted(Hopt, P, PR, N, rd, d)

    # --- Economic indicators (legacy) ---
    wacc_result = WACC(Xl, Xec, il, dec, T)
    pwco_result = PWCO(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, source)
    pwci_result = PWCI(ps, pg, N, T, epvs, epvg, source)
    lcoe_result = LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd, source)
    npv_result = NPV(pvi, pvom, pg, ps, rpg, rps, rd, N, T, Nd, Xd, epvs, epvg, source)

    # --- Cashflows for both perspectives ---
    results_perspectives = {}
    for perspective in ["project", "client"]:
        cashflows = build_cashflow_series(
            pvi, epvs, epvg, ps, pg, rps, rpg, rd, d, N, T, pvom, rom, source,
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
        "model": source,
        #"PVI (USD)": round(pvi, 3),
        #"EPV (kWh)": round(epv, 3),
        #"EPVs (kWh)": round(epvs, 3),
        #"EPVg (kWh)": round(epvg, 3),
        #"EPV discounted (kWh)": round(epv_discounted, 3),
        "WACC (%)": round(wacc_result, 3),
        #"PWCO (USD)": round(pwco_result, 3),
        #"PWCI (USD)": round(pwci_result, 3),
        "LCOE (USD/kWh)": round(lcoe_result, 4),
        "NPV (USD)": round(npv_result, 3),
        "Project perspective": results_perspectives["project"],
        "Client perspective": results_perspectives["client"],
    }


def save_results_to_excel(city, scenario, parameters_dict, talavera_result, espinoza_result, save=True):
    if not save:
        return
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    filename = f"results_{city.lower()}_{scenario}.xlsx"
    sheet_name = timestamp
    data = {
        "Parameter": list(parameters_dict.keys()) + list(talavera_result.keys()) + list(espinoza_result.keys()),
        "Value": list(parameters_dict.values()) + list(talavera_result.values()) + list(espinoza_result.values())
    }
    df_out = pd.DataFrame(data)
    with pd.ExcelWriter(filename, mode='a' if os.path.exists(filename) else 'w', engine='openpyxl') as writer:
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)

def plot_lcoe_vs_tariff(lcoe_talavera, lcoe_espinoza, base_tariff, rps, start_year=2018, N=25):
    years = np.arange(start_year, start_year + N)
    tariff_projection = base_tariff * (1 + rps) ** (years - start_year)

    plt.figure(figsize=(10, 6))
    plt.plot(years, [lcoe_talavera] * len(years), label="LCOE Talavera (2018)", linestyle="--", color="blue")
    plt.plot(years, [lcoe_espinoza] * len(years), label="LCOE Espinoza (2018)", linestyle="--", color="green")
    plt.plot(years, tariff_projection, label="Projected Electricity Tariff", color="gray")

    plt.title("LCOE vs Electricity Tariff (Scenario 1)")
    plt.xlabel("Year")
    plt.ylabel("USD/kWh")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_lcoe_tariff_scenarios(lcoe_results, ps_by_city, rps_by_city, start_year=2018, N=25):
    """
    Affiche deux graphiques comparant LCOE et tarifs projetés pour chaque ville et scénario.
    """
    years = np.arange(start_year, start_year + N)

    for scenario in sorted(lcoe_results.keys()):
        plt.figure(figsize=(10, 6))
        for city in lcoe_results[scenario]:
            lcoe = lcoe_results[scenario][city]
            ps = ps_by_city[city]
            rps = rps_by_city[city]
            tariff_proj = ps * (1 + rps) ** (years - start_year)
            plt.plot(years, tariff_proj, label=f"{city.title()} Tariff", linestyle="-")
            plt.plot(years, [lcoe] * len(years), label=f"{city.title()} LCOE", linestyle="--")
        
        plt.title(f"LCOE vs Projected Electricity Tariff — Scenario {scenario}")
        plt.xlabel("Year")
        plt.ylabel("USD/kWh")
        plt.ylim(0, 0.5)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    espinoza_result = run_simulation("espinoza")
    talavera_result = run_simulation("talavera")

    df_results = pd.DataFrame([espinoza_result, talavera_result])
    print("\n--- COMPARISON OF MODELS ---")
    print(df_results)

    save_results_to_excel(
        city=city,
        scenario=scenario,
        parameters_dict=param_dict,
        talavera_result=talavera_result,
        espinoza_result=espinoza_result,
        save=save_results
    )

    plot_lcoe_vs_tariff(
        lcoe_talavera=talavera_result["LCOE (USD/kWh)"],
        lcoe_espinoza=espinoza_result["LCOE (USD/kWh)"],
        base_tariff=ps,
        rps=rps,
        start_year=2018,
        N=N
    )

    # # --- Build dynamic LCOE graph using real results ---
    # # Étape 1 : calculer les LCOE pour toutes les villes/scénarios
    # columns = [col for col in df.columns if "_" in col]
    # lcoe_results = {1: {}, 2: {}}
    # ps_by_city = {}
    # rps_by_city = {}

    # for col in columns:
    #     city, sc_str = col.split("_")
    #     scenario = int(sc_str)
    #     city = city.lower()

    #     current_params = dict(zip(df["parameter"], df[col]))
    #     param_dict = {k: float(current_params[k].replace(",", ".")) if isinstance(current_params[k], str) else current_params[k] for k in current_params}
        
    #     # Extraire les paramètres nécessaires
    #     required_keys = ["Hopt", "P", "PR", "rd", "d", "N", "rom", "Cu", "COM", "T",
    #                     "Xd", "Nd", "Xl", "Xec", "Xis", "il", "dec", "Nis", "Nl"]
    #     inputs = {k: float(param_dict[k]) for k in required_keys}
    #     # Convertir les pourcentages
    #     PR = float(param_dict["PR"]) / 100
    #     rd = float(param_dict["rd"]) / 100
    #     d = float(param_dict["d"]) / 100
    #     rom = float(param_dict["rom"]) / 100
    #     T = float(param_dict["T"]) / 100
    #     Xd = float(param_dict["Xd"]) / 100
    #     Xl = float(param_dict["Xl"]) / 100
    #     Xec = float(param_dict["Xec"]) / 100
    #     Xis = float(param_dict["Xis"]) / 100
    #     il = float(param_dict["il"]) / 100
    #     dec = float(param_dict["dec"]) / 100

    #     # Paramètres requis
    #     P = float(param_dict["P"])
    #     Cu = float(param_dict["Cu"])
    #     COM = float(param_dict["COM"]) / 100
    #     Hopt = float(param_dict["Hopt"])
    #     N = int(param_dict["N"])
    #     Nd = int(param_dict["Nd"])
    #     Nis = int(param_dict["Nis"])
    #     Nl = int(param_dict["Nl"])

    #     # Calculs intermédiaires
    #     pvi = PVI(P, Cu)
    #     pvom = PVOM(pvi, COM)

    #     # Calcul du LCOE avec la fonction LCOE existante
    #     lcoe = LCOE(pvi, Hopt, P, PR, rd, d, N, rom, pvom, T, Xd, Nd, source="talavera")


    #     lcoe_results[scenario][city] = lcoe
    #     ps_by_city[city] = float(param_dict["ps"])
    #     rps_by_city[city] = float(param_dict["rps"])

    # # Étape 2 : tracer les deux graphes
    # plot_lcoe_tariff_scenarios(lcoe_results, ps_by_city, rps_by_city)
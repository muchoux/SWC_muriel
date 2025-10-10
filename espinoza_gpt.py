import numpy as np
import pandas as pd
import os
from datetime import datetime
import numpy_financial as npf

# =========================
# User inputs
# =========================
scenario = "1"          # "1" ou "2"
city = "arequipa"          # "lima" | "arequipa" | "tacna"
save_results = False
PARAM_FILE = "parameters/parameters_clean.csv"

# TVA (entreprise) pour convertir un prix TTC -> revenu encaissé HT
VAT = 0.18   # ajuste si besoin

# =========================
# Load parameters
# =========================
df = pd.read_csv(PARAM_FILE, sep=";", decimal=",")

column_key = f"{city.lower()}_{scenario}"
if column_key not in df.columns:
    available = [col for col in df.columns if "_" in col]
    raise ValueError(
        f"\n Error: The column '{column_key}' does not exist in the parameter file.\n"
        f" Available combinations: {', '.join(available)}\n"
        f" Please check the 'city' and 'scenario' inputs."
    )

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

# =========================
# LCOE (base) pour le mode "lcoe"
# =========================
def EPV_discounted(Hopt, P, PR, N, rd, d):
    factor = (1 - rd) / (1 + d)
    return EPV(Hopt, P, PR) * sum(factor ** i for i in range(1, N + 1))

def PWCO_espinoza_like(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, d):
    """
    Présent des coûts (style Espinoza agrégé) pour un LCOE 'base' (indicatif).
    On garde ici la structure simple : O&M actualisé + bouclier via DEP en présent + termes financement agrégés.
    """
    q = 1 / (1 + d)
    Kp = (1 + rom) / (1 + d)

    # O&M présent (à partir de PVOM tel que défini: COM%*P ; on respecte ta convention)
    PWPVOM = pvom * Kp * (1 - Kp**N) / (1 - Kp)

    # Amortissement fiscal (présent)
    DEP = (pvi * Xd / Nd) if Nd > 0 else 0.0
    PWDEP = DEP * q * (1 - q ** Nd) / (1 - q)

    # Financement agrégé (présent)
    PVl = Xl * pvi
    PVec = Xec * pvi
    PVis = Xis * pvi

    loan_term = 0.0
    equity_term = 0.0
    dividend_term = 0.0
    subsidy_term = 0.0

    if Xl > 0 and Nl > 0:
        # annuité "après impôt" en présent agrégé (raccourci Espinoza)
        loan_ann_after_tax = (PVl * il * (1 - T)) / (1 - (1 + il * (1 - T)) ** (-Nl))
        loan_term = loan_ann_after_tax * (q * (1 - q ** Nl)) / (1 - q)

    if Xec > 0:
        equity_term   = dec * PVec * (q * (1 - q ** N)) / (1 - q)
        dividend_term = PVec * (q ** N)

    if Xis > 0 and Nis > 0:
        subsidy_term = (PVis / Nis) * T * (q * (1 - q ** Nis)) / (1 - q)

    pvinv = loan_term + equity_term + dividend_term + subsidy_term
    return pvinv + PWPVOM - PWDEP * T

def LCOE_base(pvi, pvom, Hopt, P, PR, rd, d, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl):
    epv_disc = EPV_discounted(Hopt, P, PR, N, rd, d)
    pwco = PWCO_espinoza_like(pvi, pvom, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl, d)
    return pwco / epv_disc

# =========================
# ESCO S1 – Deux modes de prix (LCOE vs Retail)
# =========================
def effective_energy_price_kwh(price_mode, ps, lcoe_base):
    """
    Prix/kWh HORS TVA encaissé par l'ESCO :
      - 'lcoe'  : LCOE/(1+VAT)
      - 'retail': ps/(1+VAT)
    L'IS est géré plus bas sur le bénéfice (recettes - charges).
    """
    if price_mode == "lcoe":
        return lcoe_base / (1.0 + VAT)
    elif price_mode == "retail":
        return ps / (1.0 + VAT)
    else:
        raise ValueError("price_mode must be 'lcoe' or 'retail'")

def build_cf_esco_after_tax(
    price_mode,
    pvi, epvs, epvg, ps, rps, pg, rpg, rd,
    N, pvom, rom, T, Xd, Nd,
    Xl, Xec, Xis, il, Nis, Nl,
    lcoe_base
):
    """
    Cash-flows ESCO (Scénario 1) APRÈS IMPÔT, avec financement.
      CF0 = -pvi
      Pour n=1..N :
        - Recettes_n = prix_net(n) * EPVs_n + pg_n * EPVg_n   (prix_net HORS TVA)
        - OPEX_n
        - Intérêt_n & Principal_n via annuité nominale
        - DEP_n = pvi*Xd/Nd (si Nd>0)
        - IS_n = T * max(0, Recettes_n - OPEX_n - Intérêt_n - DEP_n)
        => CF_n = Recettes_n - OPEX_n - IS_n - (Intérêt_n + Principal_n)
    """
    flows = [-pvi]

    PVl  = Xl * pvi
    PVec = Xec * pvi

    # Annuité nominale & solde
    if PVl > 0 and Nl > 0:
        annuity = PVl * il / (1 - (1 + il) ** (-Nl))
    else:
        annuity = 0.0
    balance = PVl

    price_base_net = effective_energy_price_kwh(price_mode, ps, lcoe_base)

    for n in range(1, N + 1):
        # Prix année n (indexation à partir d'année 2)
        price_n = price_base_net * ((1 + rps) ** (n - 1))
        pg_n = pg * ((1 + rpg) ** (n - 1))

        # Énergies (dégradation à partir d'année 2)
        epvs_n = epvs * ((1 - rd) ** (n - 1))
        epvg_n = epvg * ((1 - rd) ** (n - 1))

        # Recettes (hors TVA)
        revenue_n = price_n * epvs_n + pg_n * epvg_n

        # OPEX
        opex_n = pvom * ((1 + rom) ** (n - 1))

        # Dette (intérêt + principal)
        if PVl > 0 and n <= Nl:
            interest_n  = balance * il
            principal_n = annuity - interest_n
            balance = max(0.0, balance - principal_n)
            debt_service_n = annuity
        else:
            interest_n = 0.0
            principal_n = 0.0
            debt_service_n = 0.0

        # Amortissement fiscal
        DEP_n = (pvi * Xd / Nd) if (Nd > 0 and n <= Nd) else 0.0

        # Impôt IS (sur bénéfice)
        taxable = revenue_n - opex_n - interest_n - DEP_n
        tax_n = T * taxable if taxable > 0 else 0.0

        # CF net après impôt
        cf_n = revenue_n - opex_n - tax_n - debt_service_n
        flows.append(cf_n)

    return flows

# =========================
# Indicators
# =========================
def compute_irr(cashflows):
    return npf.irr(cashflows)

def compute_dpbt(cashflows, d):
    discounted = 0.0
    inv = -cashflows[0]
    for year, cf in enumerate(cashflows[1:], start=1):
        discounted += cf / ((1 + d) ** year)
        if discounted >= inv:
            return year
    return None

def NPV_from_flows(cashflows, d):
    return sum(cf / ((1 + d) ** t) for t, cf in enumerate(cashflows))

# =========================
# Run (S1 ESCO, deux modes)
# =========================
def run_s1_esco(price_mode):
    # Base techno/éco
    pvi = PVI(P, Cu)
    epv = EPV(Hopt, P, PR)
    epvs_val = EPVs(epv, SCI)
    epvg_val = EPVg(epv, SCI)
    pvom_val = PVOM(P, COM)  # on garde ta convention EXACTE

    # LCOE base (sert pour price_mode='lcoe')
    lcoe = LCOE_base(pvi, pvom_val, Hopt, P, PR, rd, d, N, T, Xd, Nd, Xl, Xec, Xis, il, dec, Nis, Nl)

    # CF après impôt
    flows = build_cf_esco_after_tax(
        price_mode=price_mode,
        pvi=pvi, epvs=epvs_val, epvg=epvg_val, ps=ps, rps=rps, pg=pg, rpg=rpg, rd=rd,
        N=N, pvom=pvom_val, rom=rom, T=T, Xd=Xd, Nd=Nd,
        Xl=Xl, Xec=Xec, Xis=Xis, il=il, Nis=Nis, Nl=Nl,
        lcoe_base=lcoe
    )

    irr = compute_irr(flows)
    npv = NPV_from_flows(flows, d)
    dpbt = compute_dpbt(flows, d)

    return {
        "city": city,
        "scenario": scenario,
        "price_mode": price_mode,
        "WACC_used(%)": round(wacc_result, 3),
        "LCOE_base(USD/kWh)": round(lcoe, 6),
        "NPV(USD)": round(npv, 2),
        "IRR(%)": None if irr is None else round(irr * 100, 2),
        "DPBT(years)": dpbt
    }, flows

# =========================
# Main
# =========================
if __name__ == "__main__":
    res_lcoe, flows_lcoe = run_s1_esco("lcoe")
    res_retail, flows_retail = run_s1_esco("retail")

    df_out = pd.DataFrame([res_lcoe, res_retail])
    print("\n--- ESCO Scenario 1 (Espinoza) – Two pricing modes ---")
    print(df_out.to_string(index=False))

    if save_results:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
        filename = f"results_{city.lower()}_{scenario}_esco_s1.xlsx"
        with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
            df_out.to_excel(writer, sheet_name="summary", index=False)
            pd.DataFrame({"flows_lcoe": flows_lcoe}).to_excel(writer, sheet_name="flows_lcoe", index=False)
            pd.DataFrame({"flows_retail": flows_retail}).to_excel(writer, sheet_name="flows_retail", index=False)
        print(f"\nSaved to {filename}")

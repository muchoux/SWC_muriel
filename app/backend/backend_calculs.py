import numpy as np


# --- Functions ---
def calculate_npv(ps, T, N, PVI, pg, EPV, SCI, rps, rd, rpg, d,P, opex, rOM, Xd, Nd):
    
    EPVs, EPVg, Ks, Kg, PVom, Kp, PWDEP = intermediate_calculations(EPV, SCI, rps, rd, rpg, d, PVI, P, opex, rOM, Xd, Nd)
    # Revenus actualisés
    PWCI_self = ps * EPVs * (1 - T) * Ks * (1 - Ks**N) / (1 - Ks)
    PWCI_grid = pg * EPVg * (1 - T) * Kg * (1 - Kg**N) / (1 - Kg)
    PWCI = PWCI_self + PWCI_grid

    # Coûts actualisés
    PMPVOM = PVom * (1 - T) * Kp * (1 - Kp**N) / (1 - Kp)
    PWCO = PVI + PMPVOM - PWDEP * T

    NPV = PWCI - PWCO
    return NPV, PWCO


def calculate_lcoe(PWCO, N, rd, d, EPV):
    sum_LCOE = 0
    for i in range(1, N):
        sum_LCOE += ((1 - rd) ** i) / ((1 + d) ** i)

    LCOE = PWCO / (EPV * sum_LCOE)
    return LCOE



def intermediate_calculations(EPV, SCI, rps, rd, rpg, d, PVI, P, opex, rOM, Xd, Nd):
    EPVs = EPV * SCI
    EPVg = EPV - EPVs
    Ks = (1 + rps) * (1 - rd) / (1 + d)
    Kg = (1 + rpg) * (1 - rd) / (1 + d)
    Cu = PVI / P
    PVom = opex * PVI
    Kp = (1 + rOM) / (1 + d)
    q = 1 / (1 + d)
    DEP = Xd * PVI / Nd
    PWDEP = DEP * q * (1 - q**Nd) / (1 - q)
    
    return EPVs, EPVg, Ks, Kg, PVom, Kp, PWDEP


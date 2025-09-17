from pydantic import BaseModel

class TechnicalParameters(BaseModel):
    Hopt : float #Hopt global irradiation in optimum inclined plane (kWh/m2*year):
    P : float    #P (peak power) (kWp)
    PR : float   #PR (performance ratio) (%)
    rd : float   #rd (annual degradation rate) (%/year)
    N : int      #N (life cycle) (years)
    SCI : float  #SCI (self consumption index) (%)


class EconomicParameters(BaseModel):
    d : float    #d (discount rate) (%): ")) / 100
    pg : float   #pg (price at which electricity is sold to the grid) (USD/kWh): "))
    ps : float   #ps (price at which electricity is self consumed) (USD/kWh): "))
    rpg : float  #rpg (annual escalation rate of pg) (%): ")) / 100
    rps : float  #rps (annual escalation rate of ps) (%): ")) / 100
    rOM : float  #rom (annual escalation rate of opex) (%): ")) / 100
    T : float    #T (income tax rate) (%): ")) / 100
    Nd : int     #Nd (period of amortization of the investment for tax purposes) (years): "))
    Xd : float   #Xd (total depreciation allowance) (%): ")) / 100
    Cu : float   #Cu Unit cost of the PV system (USD/kWp)
    COM : float  #COM opex cost as % of initial investment (%)
    
class FinancialParameters(BaseModel):
    Xl : float   #Xl (part of PVi financed with loan) (%): ")) / 100
    Xec : float  #Xec (part of PVi financed with equity capital) (%): ")) / 100
    Xis : float  #Xis (part of PVi financed with subsidy or grant) (%): ")) / 100
    il : float   #il (annual loan interest) (%): ")) / 100
    Nis : int    #Nis (amortization of investment subsidy) (years): "))
    Nl : int     #Nl (amortization of loan) (years): "))
    dec : float   #dec (annual dividend of the equity capital or return on equity) (%): ")) /t 100

class Results(BaseModel):
    Result : dict
    msg : str
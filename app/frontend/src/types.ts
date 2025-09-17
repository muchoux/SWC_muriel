export interface Results {
    Result: {
        "model": string;
        "PVI (USD)": number;
        "EPV (kWh)": number;
        "EPVs (kWh)": number;
        "EPVg (kWh)": number;
        "EPV discounted (kWh)": number;
        "WACC (%)": number;
        "PWCO (USD)": number;
        "PWCI (USD)": number;
        "LCOE (USD/kWh)": number;
        "NPV (USD)": number;
        "IRR (%)": number;
        "DPBT (years)": string;
    },
    msg : string;
}


export interface TechnicalParameters {
    Hopt : number;
    P : number;
    PR : number;
    rd : number;
    N : number;
    SCI : number;
}

export interface EconomicParameters {
    d : number;
    pg : number;
    ps : number;
    rpg : number;
    rps : number;
    rOM : number;
    T : number;
    Nd : number;
    Xd : number;
    Cu : number;
    COM : number;
}

export interface FinancialParameters {
    Xl : number;
    Xec : number;
    Xis : number;
    il : number;
    Nis : number;
    Nl : number;
    dec : number;
}

export interface Params {
    financialParams : FinancialParameters;
    economicParams : EconomicParameters;
    technicalParams: TechnicalParameters;
}
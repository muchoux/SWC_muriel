import { EconomicParameters, FinancialParameters, TechnicalParameters } from "./types";

export function fillParams(data: { Hopt: { [x: string]: any; }; P: { [x: string]: any; }; PR: { [x: string]: any; }; rd: { [x: string]: any; }; N: { [x: string]: any; }; SCI: { [x: string]: any; }; d: { [x: string]: any; }; pg: { [x: string]: any; }; ps: { [x: string]: any; }; rpg: { [x: string]: any; }; rps: { [x: string]: any; }; rom: { [x: string]: any; }; T: { [x: string]: any; }; Nd: { [x: string]: any; }; Xd: { [x: string]: any; }; Cu: { [x: string]: any; }; COM: { [x: string]: any; }; Xl: { [x: string]: any; }; Xec: { [x: string]: any; }; Xis: { [x: string]: any; }; il: { [x: string]: any; }; Nis: { [x: string]: any; }; Nl: { [x: string]: any; }; dec: { [x: string]: any; }; },
     cityKey: string) {
  const technicalParams: TechnicalParameters = {
    Hopt: data.Hopt[cityKey],
    P: data.P[cityKey],
    PR: data.PR[cityKey],
    rd: data.rd[cityKey],
    N: data.N[cityKey],
    SCI: data.SCI[cityKey],
  };

  const economicParams: EconomicParameters = {
    d: data.d[cityKey],
    pg: data.pg[cityKey],
    ps: data.ps[cityKey],
    rpg: data.rpg[cityKey],
    rps: data.rps[cityKey],
    rOM: data.rom[cityKey],
    T: data.T[cityKey],
    Nd: data.Nd[cityKey],
    Xd: data.Xd[cityKey],
    Cu: data.Cu[cityKey],
    COM: data.COM[cityKey],
  };

  const financialParams: FinancialParameters = {
    Xl: data.Xl[cityKey],
    Xec: data.Xec[cityKey],
    Xis: data.Xis[cityKey],
    il: data.il[cityKey],
    Nis: data.Nis[cityKey],
    Nl: data.Nl[cityKey],
    dec: data.dec[cityKey],
  };

  return { technicalParams, economicParams, financialParams };
}
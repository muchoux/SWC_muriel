import { EconomicParameters, FinancialParameters, TechnicalParameters } from "./types";

export function fillParams(data: { Hopt: { [x: string]: any; }; P: { [x: string]: any; }; PR: { [x: string]: any; }; rd: { [x: string]: any; }; N: { [x: string]: any; }; SCI: { [x: string]: any; }; d: { [x: string]: any; }; pg: { [x: string]: any; }; ps: { [x: string]: any; }; rpg: { [x: string]: any; }; rps: { [x: string]: any; }; rOM: { [x: string]: any; }; T: { [x: string]: any; }; Nd: { [x: string]: any; }; Xd: { [x: string]: any; }; Cu: { [x: string]: any; }; COM: { [x: string]: any; }; Xl: { [x: string]: any; }; Xec: { [x: string]: any; }; Xis: { [x: string]: any; }; il: { [x: string]: any; }; Nis: { [x: string]: any; }; Nl: { [x: string]: any; }; dec: { [x: string]: any; }; },
     cityKey: string, scenario : string, financingScheme:string, year? : string) {
      console.log(year,scenario, data.Hopt.values[scenario][cityKey])
  const technicalParams: TechnicalParameters = {
    Hopt: year ? data.Hopt.values[scenario][cityKey][financingScheme][year] : data.Hopt.values[scenario][cityKey][financingScheme],
    P: year ? data.P.values[scenario][cityKey][financingScheme][year] : data.P.values[scenario][cityKey][financingScheme],
    PR: year ? data.PR.values[scenario][cityKey][financingScheme][year] : data.PR.values[scenario][cityKey][financingScheme],
    rd: year ? data.rd.values[scenario][cityKey][financingScheme][year] : data.rd.values[scenario][cityKey][financingScheme],
    N: year ? data.N.values[scenario][cityKey][financingScheme][year] : data.N.values[scenario][cityKey][financingScheme],
    SCI: year ? data.SCI.values[scenario][cityKey][financingScheme][year] : data.SCI.values[scenario][cityKey][financingScheme],
  };

  const economicParams: EconomicParameters = {
    d: year ? data.d.values[scenario][cityKey][financingScheme][year] : data.d.values[scenario][cityKey][financingScheme],
    pg: year ? data.pg.values[scenario][cityKey][financingScheme][year] : data.pg.values[scenario][cityKey][financingScheme],
    ps: year ? data.ps.values[scenario][cityKey][financingScheme][year] : data.ps.values[scenario][cityKey][financingScheme],
    rpg: year ? data.rpg.values[scenario][cityKey][financingScheme][year] : data.rpg.values[scenario][cityKey][financingScheme],
    rps: year ? data.rps.values[scenario][cityKey][financingScheme][year] : data.rps.values[scenario][cityKey][financingScheme],
    rOM: year ? data.rOM.values[scenario][cityKey][financingScheme][year] : data.rOM.values[scenario][cityKey][financingScheme],
    T: year ? data.T.values[scenario][cityKey][financingScheme][year] : data.T.values[scenario][cityKey][financingScheme],
    Nd: year ? data.Nd.values[scenario][cityKey][financingScheme][year] : data.Nd.values[scenario][cityKey][financingScheme],
    Xd: year ? data.Xd.values[scenario][cityKey][financingScheme][year] : data.Xd.values[scenario][cityKey][financingScheme],
    Cu: year ? data.Cu.values[scenario][cityKey][financingScheme][year] : data.Cu.values[scenario][cityKey][financingScheme],
    COM: year ? data.COM.values[scenario][cityKey][financingScheme][year] : data.COM.values[scenario][cityKey][financingScheme],
  };

  const financialParams: FinancialParameters = {
    Xl: year ? data.Xl.values[scenario][cityKey][financingScheme][year] : data.Xl.values[scenario][cityKey][financingScheme],
    Xec: year ? data.Xec.values[scenario][cityKey][financingScheme][year] : data.Xec.values[scenario][cityKey][financingScheme],
    Xis: year ? data.Xis.values[scenario][cityKey][financingScheme][year] : data.Xis.values[scenario][cityKey][financingScheme],
    il: year ? data.il.values[scenario][cityKey][financingScheme][year] : data.il.values[scenario][cityKey][financingScheme],
    Nis: year ? data.Nis.values[scenario][cityKey][financingScheme][year] : data.Nis.values[scenario][cityKey][financingScheme],
    Nl: year ? data.Nl.values[scenario][cityKey][financingScheme][year] : data.Nl.values[scenario][cityKey][financingScheme],
    dec: year ? data.dec.values[scenario][cityKey][financingScheme][year] : data.dec.values[scenario][cityKey][financingScheme],
  };

  return { technicalParams, economicParams, financialParams };
}
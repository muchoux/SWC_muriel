import React, { useState } from "react";
import Parameters from "./components/Parameters";
import {
  TechnicalParameters,
  EconomicParameters,
  FinancialParameters,
  Results,
  Params,
} from "../types";
import {
  Section,
  SectionTitle,
  ResultsContainer,
  SubmitButton,
  Result,
  FormContainer,
  PageContainer,
  FormWrapper,
  ControlButton,
  ControlsContainer,
  ControlSelect,
} from "./MainPage.style";
import parameters from "./parameters.json";
import {fillParams} from "./../utils"

const initialTechnicalParams: TechnicalParameters = { Hopt: 0, P: 0, PR: 0, rd: 0, N: 0, SCI: 0};
const initialEconomicParams: EconomicParameters = { d: 0, pg: 0, ps: 0, rpg: 0, rps: 0, rOM: 0, T: 0, Nd: 0, Xd: 0, Cu: 0, COM: 0};
const initialFinancialParams: FinancialParameters = { Xl: 0, Xec: 0, Xis: 0, il: 0, Nis: 0, Nl: 0, dec: 0 };
const scenarioCities = {
  espinoza: {
    arequipa: ["leasing", "owner_financed"],
    tacna: ["leasing", "owner_financed"],
    lima: ["leasing", "owner_financed"]
  },
  mater: {
    arequipa: ["hit", "perc"],
    tacna: ["hit", "perc"],
    lima: ["hit", "perc"],
    chachapapas: ["hit", "perc"],
    juliaca: ["hit", "perc"]
  }
} as const;
const years = ["Year","2020", "2025"];

type Scenario = keyof typeof scenarioCities; // "espinoza" | "mater"
//type City<S extends Scenario> = keyof typeof scenarioCities[S];



const MainPage: React.FC = () => {
  const API_URL = process.env.REACT_APP_API_URL || "";

  const [params, setParams] = useState<Params>({
    economicParams: initialEconomicParams,
    financialParams: initialFinancialParams,
    technicalParams: initialTechnicalParams,
  });
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<Results | undefined>(undefined);
  const [scenario, setScenario] = useState<Scenario | "">("");
  const [city, setCity] = useState<string>(""); 
  const [financingScheme, setFinancingScheme] = useState<string>("");
  const [year, setYear] = useState<string>("");


  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>,
    section: keyof Params
  ) => {
    const { name, value } = e.target;
    setParams((prevParams) => ({
      ...prevParams,
      [section]: {
        ...prevParams[section],
        [name]: isNaN(Number(value)) ? value : Number(value),
      },
    }));
  };

  const setParametersValue = () => {
    const parameters_prefilled = fillParams(parameters,city ?? "",scenario, financingScheme, scenario === "mater" ? year : undefined );
    setParams(parameters_prefilled);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/calculate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          technical_parameters: params.technicalParams,
          economic_parameters: params.economicParams,
          financial_parameters: params.financialParams,
          source : scenario
        }),
      });

      if (!response.ok) throw new Error(`Erreur HTTP: ${response.status}`);
      const data = await response.json();
      setResults(data);
    } catch (err: any) {
      console.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageContainer>
  <FormWrapper>
    <FormContainer onSubmit={handleSubmit}>
      <Section>
        <SectionTitle>Economic Parameters</SectionTitle>
        <Parameters
          sectionKey="economicParams"
          sectionLabel=""
          sectionParams={params.economicParams}
          handleChange={handleChange}
        />
      </Section>

      <Section>
        <SectionTitle>Financial Parameters</SectionTitle>
        <Parameters
          sectionKey="financialParams"
          sectionLabel=""
          sectionParams={params.financialParams}
          handleChange={handleChange}
        />
      </Section>

      <Section>
        <SectionTitle>Technical Parameters</SectionTitle>
        <Parameters
          sectionKey="technicalParams"
          sectionLabel=""
          sectionParams={params.technicalParams}
          handleChange={handleChange}
        />
      </Section>

    <ControlsContainer>
      <ControlSelect
      name="Choose a scenario"
      key='Select-Scenario'
      onChange={(e) => {
        setScenario(e.target.value as Scenario);
        setCity("");
        setFinancingScheme("");
      }}
      value={scenario}
    >
      <option value="" hidden>
        Choose
      </option>
      {Object.keys(scenarioCities).map((c) => (
        <option key={c} value={c}>
          {c}
        </option>
      ))}
    </ControlSelect>

    {scenario && <ControlSelect
      name="Choose a city"
      key='Select-City'
      onChange={(e) => {
        setCity(e.target.value);
        setFinancingScheme("");
      }}
      value={city}
      >
        <option value="" disabled>
        Choose
        </option>
          {Object.keys(scenarioCities[scenario]).map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
      </ControlSelect>
    }

    {scenario && city && <ControlSelect
      name="Choose a financing scheme"
      key='Select-Financing'
      onChange={(e) => setFinancingScheme(e.target.value)}
      value={financingScheme}
    >
      <option value="" hidden>
        Choose
      </option>
      {scenarioCities[scenario][city as keyof typeof scenarioCities[typeof scenario]].map((c) => (
        <option key={c} value={c}>
          {c}
        </option>
      ))}
    </ControlSelect>
    
    }
    
     {scenario === "mater" && city && financingScheme && <ControlSelect
      name="Choose a year"
      key='Select-Year'
      onChange={(e) => setYear(e.target.value)}
      value={year}
    >
      {years.map((c) => (
        <option key={c} value={c}>
          {c}
        </option>
      ))}
    </ControlSelect>
    
    };

    

    <ControlButton type="button" disabled={city === "" || financingScheme === ""} onClick={() => setParametersValue()}>
      Fill data
    </ControlButton>

    <ControlButton
      type="reset"
      onClick={() => {
        setParams({
          economicParams: initialEconomicParams,
          financialParams: initialFinancialParams,
          technicalParams: initialTechnicalParams,
        });
        setResults(undefined);
        setScenario("");
        setCity("");
        setFinancingScheme("");
        setYear("Choose a Year")
      }}
    >
      Reset
    </ControlButton>
  </ControlsContainer>

      <SubmitButton type="submit" disabled={loading}>
        {loading ? "In progress..." : "Submit"}
      </SubmitButton>
    </FormContainer>
  </FormWrapper>


  <ResultsContainer>
    {results &&
      Object.entries(results.Result).map(([key, value]) => (
        <Result key={key}>
          {key}: {value}
        </Result>
      ))}
      <div>{results ? undefined : "No calculations has been made yet"}</div>
  </ResultsContainer>
</PageContainer>

  );
  
};

export default MainPage;

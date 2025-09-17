import React from "react";
import { SectionWrapper, Label, Field, Input, TooltipText, TooltipWrapper } from "./Parameters.style";
import { Params } from "../../types";
import parameters from "./../parameters.json";

type ParametersProps<T extends Record<string, any>> = {
  sectionKey: keyof Params;
  sectionLabel: string;
  sectionParams: T;
  handleChange: (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>,
    section: keyof Params
  ) => void;
};

function Parameters<T extends Record<string, any>>({
  sectionKey,
  sectionParams,
  handleChange,
}: ParametersProps<T>) {
  return (
    <SectionWrapper>
      {Object.keys(sectionParams).map((key) => {
        const value = sectionParams[key as keyof T];
        const meta = (parameters as Record<string, any>)[key];

        return (
          <Field key={key}>
            <TooltipWrapper>
              <Label>
                {key} {meta?.unity ? `(${meta.unity})` : ""}
              </Label>
              {meta?.definition && <TooltipText>{meta.definition}</TooltipText>}
            </TooltipWrapper>

            <Input
              type="number"
              name={key}
              value={value as number}
              onChange={(e) => handleChange(e, sectionKey)}
              placeholder={`Entrez ${key}${meta?.unity ? ` (${meta.unity})` : ""}`}
            />
          </Field>
        );
      })}
    </SectionWrapper>
  );
}

export default Parameters;

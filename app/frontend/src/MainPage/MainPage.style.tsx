import styled from "styled-components";
import React from "react";

export const PageContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr; // gauche 50%, droite 50%
  gap: 40px;
  padding: 15px;
  margin: 40px auto;

  @media (max-width: 900px) {
    grid-template-columns: 1fr; // stack sur mobile
    gap: 20px;
  }
`;

export const FormWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;



export const Section = styled.div`
  padding: 5px 10px;
  border: 1px solid #eaeaea;
  border-radius: 12px;
  background: #fafafa;
  transition: all 0.2s ease;

  &:hover {
    background: #f0f0f0;
    transform: translateY(-2px);
  }
`;



export const FormContainer: React.FC<
  React.FormHTMLAttributes<HTMLFormElement>
> = styled.form`
  max-width: 900px;
  margin: 0px auto;
  padding: 15px;
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.08);
  display: contents;
  flex-direction: column;
  gap: 15px;
  font-family: "Inter", sans-serif;
`;



export const SectionTitle = styled.h2`
  margin-bottom: 10px;
  font-size: 1.3rem;
  font-weight: 600;
  color: #333;
`;

export const Field = styled.div`
  display: flex;
  flex-direction: column;
  margin-bottom: 12px;
`;

export const Label = styled.label`
  font-size: 0.9rem;
  color: #444;
  margin-bottom: 6px;
`;

export const Input = styled.input`
  padding: 10px 14px;
  font-size: 1rem;
  border: 1px solid #ddd;
  border-radius: 10px;
  outline: none;
  transition: all 0.2s ease;

  &:focus {
    border-color: #0077ff;
    box-shadow: 0 0 0 3px rgba(0, 119, 255, 0.2);
  }
`;

export const SubmitButton: React.FC<
  React.ButtonHTMLAttributes<HTMLButtonElement>
> = styled.button`
  padding: 14px 20px;
  background: linear-gradient(135deg, #0077ff, #00c4ff);
  color: #fff;
  font-weight: 600;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.2s ease;
  align-self: center;
  min-width: 200px;

  &:hover {
    background: linear-gradient(135deg, #005fcc, #009bcc);
    transform: translateY(-2px);
  }

  &:disabled {
    background: #bbb;
    cursor: not-allowed;
  }
`;


export const ResultsContainer = styled.div`
  background: #ffffff;
  border-radius: 16px;
  padding: 30px;
  box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.08);
  display: flex;
  flex-direction: column;
  gap: 16px;
  font-family: "Inter", sans-serif;
  max-height: max-content;
  align-self: anchor-center;
  align-items: center;
`;


export const Result: React.FC<{ children?: React.ReactNode }> = styled.p`
  font-weight: bold;
  color: #222;
  margin-bottom: 8px;
`;


export const ControlsContainer = styled.div`
  display: flex;
  direction: column;
  gap: 16px;
  margin-bottom: 24px;
  flex-wrap: wrap; 
  justify-content: center;
`;

export const ControlSelect: React.FC<React.SelectHTMLAttributes<HTMLSelectElement>> = styled.select`
  padding: 10px 14px;
  font-size: 1rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  outline: none;
  background: #fff;
  transition: all 0.2s ease;
  height: fit-content;

  &:focus {
    border-color: #0077ff;
    box-shadow: 0 0 0 3px rgba(0, 119, 255, 0.2);
  }
`;

export const ControlButton : React.FC<
  React.ButtonHTMLAttributes<HTMLButtonElement>>
= styled.button`
  padding: 12px 20px;
  font-size: 1rem;
  font-weight: 600;
  border: none;
  border-radius: 12px;
  background: linear-gradient(135deg, #0077ff, #00c4ff);
  color: #fff;
  cursor: pointer;
  transition: all 0.2s ease;
  min-width: 120px;
  height: fit-content;

  &:hover {
    background: linear-gradient(135deg, #005fcc, #009bcc);
    transform: translateY(-2px);
  }

  &:disabled {
    background: #bbb;
    cursor: not-allowed;
  }
`;

import styled from "styled-components";

export const SectionWrapper = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 5px;
`;


export const Label: React.FC<React.HTMLAttributes<HTMLLabelElement>> = styled.label`
  font-size: 0.9rem;
  color: #333;
  margin-bottom: 8px;
  font-weight: 500;
`;

export const Field: React.FC<React.HTMLAttributes<HTMLDivElement>> = styled.div`
  display: flex;
  flex-direction: column;
  background: #ffffff;
  padding: 12px 16px;
  border-radius: 12px;
  border: 1px solid #eaeaea;
  transition: all 0.2s ease;
  box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.04);

  &:hover {
    transform: translateY(-3px);
    box-shadow: 0px 6px 16px rgba(0, 0, 0, 0.08);
  }
`;

export const Input: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = styled.input`
  padding: 10px 14px;
  font-size: 1rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  outline: none;
  transition: all 0.2s ease;

  &:focus {
    border-color: #0077ff;
    box-shadow: 0 0 0 3px rgba(0, 119, 255, 0.2);
  }
`;

export const TooltipWrapper = styled.div`
  position: relative;
  display: inline-block;

  &:hover span {
    opacity: 1;
    visibility: visible;
    transform: translateY(-5px);
  }
`;

export const TooltipText = styled.span`
  position: absolute;
  bottom: 125%;
  left: 30%;
  transform: translateX(-50%);
  background: #222;
  color: #fff;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 0.85rem;
  white-space: nowrap;
  opacity: 0;
  visibility: hidden;
  transition: all 0.2s ease;
  pointer-events: none;

`;

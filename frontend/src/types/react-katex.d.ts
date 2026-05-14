declare module "react-katex" {
  import type { ReactNode } from "react";
  export interface KaTeXProps {
    math: string;
    block?: boolean;
    errorColor?: string;
    renderError?: (error: Error) => ReactNode;
    settings?: Record<string, unknown>;
  }
  export const InlineMath: (props: KaTeXProps) => JSX.Element;
  export const BlockMath: (props: KaTeXProps) => JSX.Element;
}

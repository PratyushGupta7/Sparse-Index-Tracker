export interface RegimeWindow {
  id: string;
  label: string;
  start: string;
  end: string;
  description: string;
}

export const REGIME_WINDOWS: RegimeWindow[] = [
  {
    id: "covid",
    label: "COVID Crash",
    start: "2020-02-01",
    end: "2020-04-30",
    description: "VIX 82, circuit breakers, fastest -34% drawdown ever.",
  },
  {
    id: "volmageddon",
    label: "Volmageddon",
    start: "2018-01-01",
    end: "2018-04-30",
    description: "VIX 17→50, XIV collapse, single-day -10% on the Nasdaq.",
  },
  {
    id: "rate-hikes-2022",
    label: "2022 Rate Hikes",
    start: "2022-01-01",
    end: "2022-12-31",
    description: "Growth → value rotation; SPY -25% peak-to-trough.",
  },
  {
    id: "ai-bull",
    label: "AI Bull (2023)",
    start: "2023-03-01",
    end: "2024-06-30",
    description: "NVDA +200%, mega-cap tech surge.",
  },
  {
    id: "quiet-2024",
    label: "Quiet 2024",
    start: "2024-08-01",
    end: "2024-11-30",
    description: "Post-election stability, VIX ~14.",
  },
];

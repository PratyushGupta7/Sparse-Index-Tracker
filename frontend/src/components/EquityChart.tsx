"use client";

import { ChartFrame } from "@/components/ChartFrame";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface EquityPoint {
  date: string;
  [series: string]: string | number;
}

const SERIES_COLORS: Record<string, string> = {
  admm: "#22C55E",
  benchmark: "#F59E0B",
  lasso: "#60a5fa",
  omp: "#a78bfa",
  equal_weight_topn: "#94a3b8",
};

export interface EquityChartProps {
  data: EquityPoint[];
  series: string[];
  height?: number;
  showLegend?: boolean;
}

export function EquityChart({ data, series, height = 360, showLegend = true }: EquityChartProps) {
  return (
    <ChartFrame height={height}>
      {({ width }) => (
        <LineChart
          data={data}
          width={width}
          height={height}
          margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
        >
          <CartesianGrid stroke="var(--grid)" strokeDasharray="2 4" />
          <XAxis dataKey="date" stroke="var(--muted-foreground)" fontSize={11} minTickGap={40} />
          <YAxis
            stroke="var(--muted-foreground)"
            fontSize={11}
            tickFormatter={(v) => `$${(Number(v) / 1_000_000).toFixed(2)}M`}
            width={70}
          />
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "var(--foreground)" }}
            formatter={((value: unknown) => `$${(Number(value) / 1_000_000).toFixed(3)}M`) as never}
          />
          {showLegend && <Legend wrapperStyle={{ fontSize: 12 }} />}
          {series.map((name) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={SERIES_COLORS[name] ?? "#64748b"}
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      )}
    </ChartFrame>
  );
}

export interface DrawdownChartProps {
  data: EquityPoint[];
  series: string;
  height?: number;
}

export function DrawdownChart({ data, series, height = 180 }: DrawdownChartProps) {
  return (
    <ChartFrame height={height}>
      {({ width }) => (
        <AreaChart
          data={data}
          width={width}
          height={height}
          margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
        >
          <CartesianGrid stroke="var(--grid)" strokeDasharray="2 4" />
          <XAxis dataKey="date" stroke="var(--muted-foreground)" fontSize={11} minTickGap={40} />
          <YAxis
            stroke="var(--muted-foreground)"
            fontSize={11}
            tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
            width={50}
          />
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={((value: unknown) => `${(Number(value) * 100).toFixed(2)}%`) as never}
          />
          <Area
            type="monotone"
            dataKey={series}
            stroke="#ef4444"
            fill="#ef4444"
            fillOpacity={0.18}
            isAnimationActive={false}
          />
        </AreaChart>
      )}
    </ChartFrame>
  );
}

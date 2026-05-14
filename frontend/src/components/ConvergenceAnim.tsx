"use client";

import { ChartFrame } from "@/components/ChartFrame";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface ConvergencePayload {
  primal: number[];
  dual: number[];
  tol: number;
}

export function ConvergenceAnim({ data }: { data: ConvergencePayload }) {
  const points = data.primal.map((p, i) => ({
    iter: i + 1,
    primal: Math.max(p, 1e-12),
    dual: Math.max(data.dual[i] ?? p, 1e-12),
  }));
  return (
    <ChartFrame height={288}>
      {({ width, height }) => (
        <LineChart
          data={points}
          width={width}
          height={height}
          margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
        >
          <CartesianGrid stroke="var(--grid)" strokeDasharray="2 4" />
          <XAxis
            dataKey="iter"
            label={{
              value: "iteration",
              fill: "var(--muted-foreground)",
              fontSize: 11,
              offset: -4,
              position: "insideBottom",
            }}
            stroke="var(--muted-foreground)"
            fontSize={11}
            minTickGap={20}
          />
          <YAxis
            stroke="var(--muted-foreground)"
            fontSize={11}
            scale="log"
            domain={["auto", "auto"]}
            tickFormatter={(v) => Number(v).toExponential(0)}
            width={70}
          />
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={((value: unknown) => Number(value).toExponential(2)) as never}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine
            y={data.tol}
            stroke="#F59E0B"
            strokeDasharray="4 4"
            label={{ value: "tol", fill: "#F59E0B", fontSize: 10 }}
          />
          <Line
            type="monotone"
            dataKey="primal"
            stroke="#22C55E"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="dual"
            stroke="#60a5fa"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      )}
    </ChartFrame>
  );
}

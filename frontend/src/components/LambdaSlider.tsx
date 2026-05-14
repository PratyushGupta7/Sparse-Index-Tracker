"use client";

import { ChartFrame } from "@/components/ChartFrame";
import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, ComposedChart, Line, Tooltip, XAxis, YAxis } from "recharts";

export interface LambdaPoint {
  lam_frac: number;
  nnz: number;
  oos_te: number;
  in_sample_r2: number;
}

export function LambdaSlider({ points }: { points: LambdaPoint[] }) {
  const sorted = useMemo(() => [...points].sort((a, b) => a.lam_frac - b.lam_frac), [points]);
  const [idx, setIdx] = useState(Math.floor(sorted.length / 2));
  const current = sorted[idx];

  useEffect(() => {
    if (idx >= sorted.length) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setIdx(sorted.length - 1);
    }
  }, [sorted.length, idx]);

  if (sorted.length === 0) return null;

  return (
    <div className="space-y-4 rounded-lg border border-[var(--border)] bg-[var(--card)] p-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <p className="text-sm text-[var(--muted-foreground)]">λ / λ_max</p>
          <p className="font-mono text-2xl">{current.lam_frac.toFixed(3)}</p>
        </div>
        <div className="flex gap-4 text-sm font-mono">
          <span className="text-[var(--primary)]">{current.nnz} stocks</span>
          <span className="text-[var(--accent)]">OOS TE {(current.oos_te * 100).toFixed(2)}%</span>
          <span className="text-[var(--muted-foreground)]">
            R² {current.in_sample_r2.toFixed(3)}
          </span>
        </div>
      </div>

      <input
        type="range"
        min={0}
        max={sorted.length - 1}
        step={1}
        value={idx}
        onChange={(e) => setIdx(Number(e.target.value))}
        className="w-full accent-[var(--primary)]"
        aria-label="Regularisation strength"
      />

      <ChartFrame height={224}>
        {({ width, height }) => (
          <ComposedChart
            data={sorted}
            width={width}
            height={height}
            margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
          >
            <CartesianGrid stroke="var(--grid)" strokeDasharray="2 4" />
            <XAxis
              dataKey="lam_frac"
              stroke="var(--muted-foreground)"
              fontSize={11}
              tickFormatter={(v) => Number(v).toFixed(2)}
            />
            <YAxis yAxisId="left" stroke="#22C55E" fontSize={11} />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#F59E0B"
              fontSize={11}
              tickFormatter={(v) => `${(Number(v) * 100).toFixed(1)}%`}
            />
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
              }}
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="nnz"
              stroke="#22C55E"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              name="# stocks"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="oos_te"
              stroke="#F59E0B"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              name="OOS TE"
            />
          </ComposedChart>
        )}
      </ChartFrame>
    </div>
  );
}

"use client";

import { ChartFrame } from "@/components/ChartFrame";
import { Tooltip, Treemap } from "recharts";

export interface AllocationLeaf {
  ticker: string;
  weight: number;
  shares: number;
  cost: number;
}

export function AllocationTreemap({ data }: { data: AllocationLeaf[] }) {
  const treeData = data.map((d) => ({
    name: d.ticker,
    size: Math.max(d.weight, 0.0001),
    weight: d.weight,
    shares: d.shares,
    cost: d.cost,
  }));
  return (
    <ChartFrame height={288}>
      {({ width, height }) => (
        <Treemap
          data={treeData}
          width={width}
          height={height}
          dataKey="size"
          stroke="var(--background)"
          fill="#22C55E"
          isAnimationActive={false}
        >
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={
              ((_value: unknown, _name: unknown, item: unknown) => {
                const p = (item as { payload?: AllocationLeaf })?.payload;
                if (!p) return null;
                return [
                  `${(p.weight * 100).toFixed(2)}% · ${p.shares} sh · $${p.cost.toFixed(0)}`,
                  p.ticker,
                ] as [string, string];
              }) as never
            }
          />
        </Treemap>
      )}
    </ChartFrame>
  );
}

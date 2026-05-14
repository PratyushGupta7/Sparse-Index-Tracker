"use client";

import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

interface ChartFrameProps {
  children: (size: { width: number; height: number }) => ReactNode;
  height: number;
}

export function ChartFrame({ children, height }: ChartFrameProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let raf = 0;
    const update = () => {
      const next = Math.floor(el.getBoundingClientRect().width);
      setWidth((prev) => (next > 0 && next !== prev ? next : prev));
    };
    const schedule = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(update);
    };

    schedule();
    const observer = new ResizeObserver(schedule);
    observer.observe(el);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, []);

  return (
    <div ref={ref} className="w-full min-w-0" style={{ height }}>
      {width > 0 ? (
        children({ width, height })
      ) : (
        <div
          aria-hidden="true"
          className="h-full w-full animate-pulse rounded-2xl bg-[var(--card)]"
        />
      )}
    </div>
  );
}

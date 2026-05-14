"use client";

import dynamic from "next/dynamic";

export const HeroSphereClient = dynamic(() => import("./HeroSphere").then((m) => m.HeroSphere), {
  ssr: false,
  loading: () => (
    <div className="aspect-square w-full max-w-md animate-pulse rounded-2xl bg-[var(--card)]" />
  ),
});

"use client";

import { motion, useReducedMotion } from "framer-motion";

/**
 * Animated SVG showing the classic L1-vs-L2 intuition: the L1 ball (rotated
 * square) hits the constraint hyperplane at corners (sparse solutions),
 * whereas the L2 ball (circle) tends to clip in the interior of an axis.
 */
export function L1L2Geometry() {
  const reduce = useReducedMotion();
  const transition = reduce ? { duration: 0 } : { duration: 1.2, ease: "easeOut" as const };
  return (
    <svg
      viewBox="-50 -50 100 100"
      role="img"
      aria-label="Geometric intuition for L1 vs L2 regularisation"
      className="w-full max-w-md text-[var(--foreground)]"
    >
      <defs>
        <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
          <path d="M 10 0 L 0 0 0 10" fill="none" stroke="var(--grid)" strokeWidth="0.4" />
        </pattern>
      </defs>
      <rect x="-50" y="-50" width="100" height="100" fill="url(#grid)" />
      <line x1="-50" y1="0" x2="50" y2="0" stroke="currentColor" strokeWidth="0.5" />
      <line x1="0" y1="-50" x2="0" y2="50" stroke="currentColor" strokeWidth="0.5" />

      <motion.circle
        cx="0"
        cy="0"
        r="25"
        fill="rgba(96,165,250,0.15)"
        stroke="#60a5fa"
        strokeWidth="1.4"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={transition}
      />

      <motion.polygon
        points="-25,0 0,-25 25,0 0,25"
        fill="rgba(34,197,94,0.18)"
        stroke="#22C55E"
        strokeWidth="1.4"
        initial={{ scale: 0, rotate: -45 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={transition}
      />

      <motion.line
        x1="-46"
        y1="-15"
        x2="46"
        y2="42"
        stroke="#F59E0B"
        strokeWidth="1.6"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ ...transition, delay: 0.4 }}
      />

      <motion.circle
        cx="0"
        cy="-25"
        r="2.4"
        fill="#22C55E"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ ...transition, delay: 1.0 }}
      />

      <text x="-46" y="-32" fontSize="4.5" fill="currentColor">
        L1 ball (sparse)
      </text>
      <text x="14" y="32" fontSize="4.5" fill="currentColor">
        L2 ball
      </text>
      <text x="14" y="-30" fontSize="4.5" fill="#F59E0B">
        y = Xw constraint
      </text>
    </svg>
  );
}

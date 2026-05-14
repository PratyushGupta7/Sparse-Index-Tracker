"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);
  const current = mounted ? (theme === "system" ? resolvedTheme : theme) : "dark";
  const next = current === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      className="rounded-md border border-[var(--border)] bg-[var(--card)] p-2 hover:bg-[var(--card)]/80 focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
    >
      {current === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}

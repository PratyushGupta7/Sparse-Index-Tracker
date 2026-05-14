import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/react";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { Toaster } from "sonner";
import { ThemeProvider } from "@/components/ThemeProvider";
import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Sparse Index Tracker — Replicate the S&P 500 with 50 stocks",
    template: "%s · Sparse Index Tracker",
  },
  description:
    "Custom ADMM solver replicates the S&P 500 with ~50 stocks (R²=0.97 across 8 regimes). Walk-forward 2018–2025, 4 markets, live API.",
  keywords: [
    "sparse portfolio",
    "index tracking",
    "ADMM",
    "compressed sensing",
    "quantitative finance",
    "lasso",
  ],
  authors: [{ name: "Pratyush Gupta" }],
  openGraph: {
    type: "website",
    title: "Sparse Index Tracker",
    description: "Replicate the S&P 500 with 50 stocks. Mathematically. R²=0.97 across 8 regimes.",
  },
  twitter: { card: "summary_large_image" },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0B1220" },
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
  ],
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      data-scroll-behavior="smooth"
      className={`${inter.variable} ${jetbrains.variable}`}
    >
      <body>
        <ThemeProvider>
          <div className="flex min-h-screen flex-col">
            <Nav />
            <main className="flex-1">{children}</main>
            <Footer />
          </div>
          <Toaster richColors position="bottom-right" />
        </ThemeProvider>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}

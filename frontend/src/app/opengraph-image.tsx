import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "Sparse Index Tracker — replicate the S&P 500 with 50 stocks";

export default function OG() {
  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "#0B1220",
        color: "#f1f5f9",
        padding: "80px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <div
          style={{
            width: 56,
            height: 56,
            borderRadius: 12,
            background: "#22C55E",
            color: "#052e16",
            fontSize: 36,
            fontWeight: 800,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          Σ
        </div>
        <span style={{ fontSize: 28, fontWeight: 600 }}>Sparse Index Tracker</span>
      </div>
      <div>
        <p style={{ fontSize: 64, lineHeight: 1.05, fontWeight: 800, margin: 0 }}>
          Replicate the S&P 500 with <span style={{ color: "#22C55E" }}>50 stocks</span>.
        </p>
        <p style={{ fontSize: 32, color: "#94a3b8", marginTop: 32 }}>
          Custom ADMM solver · R²=0.97 across 8 regimes · 2018–2025 walk-forward · 4 markets
        </p>
      </div>
      <p style={{ fontSize: 24, color: "#F59E0B" }}>
        github.com/PratyushGupta7/Sparse-Index-Tracker
      </p>
    </div>,
    { ...size }
  );
}

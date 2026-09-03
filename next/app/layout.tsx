import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OceanEmbed — Subsurface Temperature Reconstruction",
  description: "Reconstructing 0–1000 m ocean temperature from satellite surface fields, validated against Argo.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

import type { Field2D } from "./api";

export type Region = "full" | "as" | "bob";
export const REGIONS: { key: Region; label: string; range: [number, number] }[] = [
  { key: "full", label: "Full basin", range: [0, 200] },
  { key: "as", label: "Arabian Sea", range: [55, 78] },
  { key: "bob", label: "Bay of Bengal", range: [78, 100] },
];

// Crop a Field2D to a longitude range (the region selector). lat is untouched.
export function cropLon<T extends Field2D>(field: T, region: Region): T {
  if (region === "full") return field;
  const [lo, hi] = REGIONS.find((r) => r.key === region)!.range;
  const idx = field.lon.map((v, i) => [v, i] as const).filter(([v]) => v >= lo && v <= hi).map(([, i]) => i);
  if (idx.length === 0) return field;
  const j0 = idx[0], j1 = idx[idx.length - 1];
  return {
    ...field,
    lon: field.lon.slice(j0, j1 + 1),
    values: field.values.map((row) => row.slice(j0, j1 + 1)),
  };
}

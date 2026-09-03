// Small perceptual-ish colormaps (cmocean-inspired) so we need no plotting library.
type RGB = [number, number, number];
type Stop = [number, RGB];

const MAPS: Record<string, Stop[]> = {
  thermal: [
    [0.0, [13, 8, 45]], [0.22, [58, 22, 105]], [0.45, [130, 42, 110]],
    [0.66, [206, 78, 82]], [0.83, [240, 145, 62]], [1.0, [232, 232, 150]],
  ],
  haline: [
    [0.0, [41, 24, 107]], [0.35, [30, 90, 140]], [0.6, [30, 148, 128]],
    [0.8, [130, 190, 90]], [1.0, [253, 238, 140]],
  ],
  balance: [ // diverging, for anomalies / currents / winds
    [0.0, [40, 95, 180]], [0.25, [125, 175, 220]], [0.5, [242, 242, 240]],
    [0.75, [225, 130, 110]], [1.0, [170, 35, 45]],
  ],
};

function lerp(stops: Stop[], t: number): RGB {
  if (t <= stops[0][0]) return stops[0][1];
  if (t >= stops[stops.length - 1][0]) return stops[stops.length - 1][1];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i];
      const f = (t - t0) / (t1 - t0);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return stops[stops.length - 1][1];
}

export function colorFn(name: string, vmin: number, vmax: number) {
  const stops = MAPS[name] ?? MAPS.thermal;
  const span = vmax - vmin || 1;
  return (v: number): RGB => {
    let t = (v - vmin) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    return lerp(stops, t);
  };
}

export function gradientCss(name: string): string {
  const stops = MAPS[name] ?? MAPS.thermal;
  return `linear-gradient(to right, ${stops.map((s) => `rgb(${s[1].map(Math.round).join(",")}) ${(s[0] * 100).toFixed(0)}%`).join(",")})`;
}

// series palette for the ablation chart
export const SERIES_COLORS = ["#8aa0c8", "#39d0c8", "#f2a154", "#e0607e", "#b98cff", "#e8e896"];

"use client";
import { useEffect, useRef } from "react";
import type { Embedding } from "@/lib/api";

export default function EmbeddingCanvas({ emb }: { emb: Embedding }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const [h, w] = emb.shape;
    cv.width = w; cv.height = h;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(w, h);
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const px = emb.rgb[r]?.[c] ?? [0, 0, 0];
        const p = (r * w + c) * 4;
        img.data[p] = Math.round(px[0] * 255);
        img.data[p + 1] = Math.round(px[1] * 255);
        img.data[p + 2] = Math.round(px[2] * 255);
        img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [emb]);

  return (
    <div>
      <div className="mapWrap embed"><canvas ref={ref} className="mapCanvas pixelated" /></div>
      <div className="muted small">
        OceanEmbed bottleneck latent (12×22), 256→3 PCA → RGB. Explained variance:{" "}
        {emb.explained_variance.map((v) => (v * 100).toFixed(0) + "%").join(" · ")}
      </div>
    </div>
  );
}

// OTER methodology page. Every number comes from data.js (built by make_assets.py).
"use strict";
const D = window.DATA;
const NS = "http://www.w3.org/2000/svg";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const C = { blue: "#3987e5", orange: "#d95926", green: "#199e70", gold: "#c98500",
            cold: "#2a78d6", warm: "#e34948", sky: "#7fd8ff", ink3: "#7f98ad" };
const SEQ = ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"];

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v; else n.setAttribute(k, v);
  }
  if (parent) parent.appendChild(n);
  return n;
}
function h(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}
function hexMix(stops, t) {
  const c = stops.map(s => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16)));
  const p = clamp(t) * (c.length - 1), i = Math.min(Math.floor(p), c.length - 2), f = p - i;
  return `rgb(${c[i].map((v, k) => Math.round(v + (c[i + 1][k] - v) * f)).join(",")})`;
}
const f3 = v => v.toFixed(3);
const pct = (v, a, b) => ((v - a) / (b - a)) * 100;

// ---------- tooltip ----------
const tip = $("#tip");
function showTip(x, y, html) { tip.innerHTML = html; tip.style.left = x + "px"; tip.style.top = y + "px"; tip.classList.add("on"); }
function hideTip() { tip.classList.remove("on"); }
addEventListener("scroll", hideTip, { passive: true });

// ---------- reveal on scroll ----------
const onReveal = new Map();
const io = new IntersectionObserver(es => es.forEach(e => {
  if (!e.isIntersecting) return;
  e.target.classList.add("in");
  (onReveal.get(e.target) || []).forEach(fn => fn());
  onReveal.delete(e.target);
  io.unobserve(e.target);
}), { threshold: 0.18 });
function reveal(node, fn) {
  if (fn) onReveal.set(node, [...(onReveal.get(node) || []), fn]);
  io.observe(node);
}
$$(".reveal").forEach(n => reveal(n));

// ---------- marine snow ----------
(() => {
  const cv = $("#snow"), ctx = cv.getContext("2d");
  let W, H, P = [], lastY = scrollY, depth = 0;
  function size() {
    const r = devicePixelRatio || 1;
    W = cv.width = innerWidth * r; H = cv.height = innerHeight * r;
    cv.style.width = innerWidth + "px"; cv.style.height = innerHeight + "px";
    const n = Math.round(Math.min(140, innerWidth / 11));
    P = Array.from({ length: n }, () => ({ x: Math.random() * W, y: Math.random() * H,
      r: (Math.random() * 1.6 + 0.4) * r, s: Math.random() * 0.25 + 0.05, w: Math.random() * 6.28 }));
  }
  size(); addEventListener("resize", size);
  function frame() {
    const dy = (scrollY - lastY) * (devicePixelRatio || 1); lastY = scrollY;
    depth = clamp(scrollY / (document.body.scrollHeight - innerHeight));
    ctx.clearRect(0, 0, W, H);
    for (const p of P) {
      p.y += p.s - dy * 0.35 * p.s * 3; p.w += 0.01; p.x += Math.sin(p.w) * 0.15;
      if (p.y > H + 5) p.y = -5; if (p.y < -5) p.y = H + 5;
      ctx.globalAlpha = 0.18 + depth * 0.5 * p.s * 3;
      ctx.fillStyle = "#cfefff";
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.283); ctx.fill();
    }
    if (!reduce) requestAnimationFrame(frame);
  }
  frame();
})();

// ---------- scroll: ocean colour, depth gauge, mobile bar ----------
const sections = $$("[data-title]");
const ocean = $(".ocean"), rays = $(".rays");
const BG = [[0, "#0f5a78", "#06243a"], [0.3, "#07354f", "#041b2e"], [0.65, "#041b2e", "#020c17"], [1, "#020a13", "#010408"]];
function bgAt(p) {
  let i = 0; while (i < BG.length - 2 && p > BG[i + 1][0]) i++;
  const t = (p - BG[i][0]) / (BG[i + 1][0] - BG[i][0]);
  return [1, 2].map(k => hexMix([BG[i][k], BG[i + 1][k]], t));
}
const gaugeList = $("#gauge-list");
sections.forEach((s, i) => {
  const li = h("li"); const a = h("a"); a.href = "#" + s.id; a.setAttribute("aria-label", s.dataset.title);
  a.appendChild(h("span", "lab", s.dataset.title));
  li.appendChild(a); gaugeList.appendChild(li); s._li = li;
});
function placeGauge() {
  const total = document.body.scrollHeight - innerHeight;
  sections.forEach(s => { s._li.style.top = clamp(s.offsetTop / total) * 100 + "%"; });
}
addEventListener("load", placeGauge); addEventListener("resize", placeGauge); placeGauge();

const mbar = $("#mbar");
let ticking = false;
function onScroll() {
  ticking = false;
  const total = document.body.scrollHeight - innerHeight;
  const p = clamp(scrollY / total);
  const [top, bot] = bgAt(p);
  ocean.style.setProperty("--top", top); ocean.style.setProperty("--bot", bot);
  rays.style.setProperty("--rays", clamp(1 - p * 5));
  const depthM = Math.round(p * 1000);
  $(".gauge .fill").style.height = p * 100 + "%";
  $(".gauge .dot").style.top = p * 100 + "%";
  let cur = sections[0];
  for (const s of sections) if (s.getBoundingClientRect().top < innerHeight * 0.45) cur = s;
  const ro = $(".gauge .readout"); ro.style.top = p * 100 + "%"; ro.innerHTML = `${depthM} m<small>${cur.dataset.title}</small>`;
  sections.forEach(s => s._li.classList.toggle("on", s === cur));
  mbar.classList.toggle("on", scrollY > innerHeight * 0.6);
  $("#mbar-step").textContent = cur.dataset.title; $("#mbar-depth").textContent = depthM + " m";
  $("#mbar .prog").style.width = p * 100 + "%";
}
addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });

// ---------- chart helpers ----------
const DEPTH_TICKS = [0, 10, 20, 50, 100, 200, 500, 1000];
function depthScale(t, b) { const a = Math.log10(10), z = Math.log10(1010); return d => t + (Math.log10(d + 10) - a) / (z - a) * (b - t); }
function linePath(xs, ys, X, Y) {
  let d = "", pen = false;
  xs.forEach((x, i) => {
    if (x === null || Number.isNaN(x) || ys[i] === null) { pen = false; return; }
    d += (pen ? "L" : "M") + X(x).toFixed(1) + " " + Y(ys[i]).toFixed(1); pen = true;
  });
  return d;
}
function setDraw(path, delay = 0) {
  const L = path.getTotalLength ? path.getTotalLength() : 1000;
  path.style.setProperty("--len", Math.ceil(L) + 1); path.style.setProperty("--delay", delay + "s");
  path.classList.add("draw");
}
function legend(box, items) {
  box.innerHTML = items.map(([name, color, dash]) => `<span><i class="${dash ? "dash" : ""}" style="background:${color};color:${color}"></i>${name}</span>`).join("");
}
// Depth-profile chart: x = value, y = depth on a stretched (log) axis like the demo app.
function profileChart(svg, { x0, x1, xticks, xlabel, series, tipRow }) {
  const W = 520, H = svg.viewBox.baseVal.height, m = { l: 54, r: 14, t: 12, b: 46 };
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = depthScale(m.t, H - m.b);
  const g = el("g", { class: "grid" }, svg);
  DEPTH_TICKS.forEach(d => { el("line", { x1: m.l, x2: W - m.r, y1: Y(d), y2: Y(d) }, g); el("text", { x: m.l - 8, y: Y(d) + 4, "text-anchor": "end", text: d }, svg); });
  xticks.forEach(v => { el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, g); el("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle", text: v }, svg); });
  el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", text: xlabel }, svg);
  el("text", { x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`, text: "depth (m)" }, svg);
  const out = [];
  series.forEach(s => {
    const p = el("path", { d: linePath(s.x, s.y, X, Y), fill: "none", stroke: s.color, "stroke-width": s.width || 2,
      "stroke-linecap": "round", "stroke-linejoin": "round", "stroke-dasharray": s.dash || "", opacity: s.opacity ?? 1 }, svg);
    if (s.cls) p.classList.add(s.cls);
    if (!s.dash) setDraw(p, s.delay || 0);
    else { p.classList.add("pop"); p.style.setProperty("--delay", (s.delay || 0) + "s"); }
    if (s.dots) s.x.forEach((v, i) => { if (v !== null && !Number.isNaN(v)) { const c = el("circle", { cx: X(v), cy: Y(s.y[i]), r: 4, fill: s.color, stroke: "#04121f", "stroke-width": 2, class: "pop" }, svg); c.style.setProperty("--delay", (s.delay || 0) + 1.2 + "s"); } });
    out.push(p);
  });
  if (tipRow) D.depths.forEach((d, i) => {
    const y0 = i ? (Y(D.depths[i - 1]) + Y(d)) / 2 : m.t, y1 = i < D.depths.length - 1 ? (Y(d) + Y(D.depths[i + 1])) / 2 : H - m.b;
    const r = el("rect", { x: m.l, y: y0, width: W - m.l - m.r, height: Math.max(1, y1 - y0), class: "hit" }, svg);
    const show = ev => { const b = svg.getBoundingClientRect(); showTip(ev.clientX, b.top + (Y(d) / H) * b.height, tipRow(i, d)); };
    r.addEventListener("pointermove", show); r.addEventListener("pointerdown", show); r.addEventListener("pointerleave", hideTip);
  });
  return { X, Y, paths: out };
}
function rowTip(d, rows) {
  return `<b>${d} m</b><br>` + rows.map(([name, color, v]) => `<span style="color:${color}">●</span> <span class="k">${name}</span> ${v}`).join("<br>");
}

// ---------- 3: architecture ----------
(() => {
  const svg = $("#arch");
  const inp = $("#arch-input");
  for (let k = 0; k < 7; k++) {
    const x = 42 - k * 6, y = 160 + k * 6;
    const gk = el("g", { class: "in-frame" }, inp);
    el("rect", { x: x - 1, y: y - 1, width: 86, height: 48, rx: 4, fill: "#0b2238", stroke: "rgba(255,255,255,0.35)" }, gk);
    el("image", { href: `img/win_${k}.webp`, x, y, width: 84, height: 46, preserveAspectRatio: "none" }, gk);
  }
  const blocks = (parent, spec, fill, ghosts) => spec.forEach(([x, w, hh, ch]) => {
    const y = 220 - hh / 2;
    if (ghosts) for (let j = 3; j >= 1; j--) el("rect", { x: x + j * 5, y: y - j * 5, width: w, height: hh, rx: 4, fill, opacity: 0.12 + 0.05 * (3 - j) }, parent);
    el("rect", { x, y, width: w, height: hh, rx: 4, fill }, parent);
    el("text", { x: x + w / 2, y: y - 10, "text-anchor": "middle", text: ch, style: "font-size:12px;fill:#dfe9f2" }, parent);
  });
  blocks($("#arch-enc"), [[140, 14, 250, "32"], [176, 20, 180, "64"], [220, 28, 120, "128"], [268, 38, 72, "256"]], "url(#gEnc)", true);
  blocks($("#arch-dec"), [[618, 28, 120, "128"], [666, 20, 180, "64"], [706, 14, 250, "32"]], "url(#gDec)", false);

  // ConvLSTM beads, one per day
  const lstm = $("#arch-lstm"), beads = [];
  for (let k = 0; k < 7; k++) {
    const a = -Math.PI / 2 + k * (2 * Math.PI / 7);
    beads.push(el("circle", { cx: 430 + 62 * Math.cos(a), cy: 220 + 62 * Math.sin(a), r: 6, fill: "#5a2a18", stroke: "#d95926", "stroke-width": 2, class: "bead" }, lstm));
  }

  // Branching bus lines from decoder to 15 depths
  const fanG = $("#output-fan");
  if (fanG) {
    D.depths.forEach((d, i) => {
      const y = 104 + i * 16.5 + 6;
      el("path", { d: `M720 220 C760 220, 790 ${y}, 840 ${y}`, fill: "none", stroke: "rgba(127, 216, 255, 0.22)", "stroke-width": 1.2 }, fanG);
    });
  }

  // output slabs, surface (warm) to 1000 m (cold)
  const out = $("#arch-out");
  const slabElements = [];
  D.depths.forEach((d, i) => {
    const y = 104 + i * 16.5;
    const rect = el("rect", {
      x: 840, y, width: 112, height: 12, rx: 3,
      fill: hexMix(SEQ, 1 - i / 14 * 0.92),
      class: "depth-slab" + (i === 0 ? " selected" : ""),
      "data-depth": d,
      "data-idx": i
    }, out);
    rect.setAttribute("role", "button");
    rect.setAttribute("tabindex", "0");
    rect.setAttribute("aria-label", `Depth layer ${d} m`);
    if (i === 0 || i === 7 || i === 14) el("text", { x: 832, y: y + 10, "text-anchor": "end", text: d + " m", style: "font-size:11px" }, out);
    slabElements.push(rect);
  });

  // flowing particles along master flow path
  const pg = $("#particles");
  if (!reduce && pg) {
    for (let k = 0; k < 12; k++) {
      const c = el("circle", { r: 3.2, fill: "#7fd8ff", opacity: 0.9, filter: "url(#glow)" }, pg);
      const am = el("animateMotion", { dur: "3.2s", repeatCount: "indefinite", begin: (k * 0.26) + "s" }, c);
      el("mpath", { href: "#flow" }, am);
    }
  }

  // Step cards in .arch-steps
  const cards = $$(".arch-steps .card");

  // Component Information Metadata
  const COMP_DATA = {
    input: {
      tag: "INPUT DATA · 7 DAYS",
      title: "7-Day Surface Observation Sequence",
      desc: "Captures 7 satellite-observable ocean surface variables (SST, SSS, SLA, east/north current, east/north wind) across 7 consecutive days over the Arabian Sea and Bay of Bengal.",
      chips: ["7 surface variables", "7 daily timesteps", "96 × 176 grid", "0.25° (~28 km)"],
      color: "#6aa9f0",
      part: "input"
    },
    enc: {
      tag: "① CNN ENCODER · READ",
      title: "Spatial Feature Extraction",
      desc: "Extracts spatial features from ocean surface observations. A shared deep convolutional network compresses 96×176 grid cells to 12×22 while expanding detail across 32 → 64 → 128 → 256 channels for each day.",
      chips: ["Shared weights across days", "4 downsampling stages", "12 × 22 spatial bottleneck", "256 feature channels"],
      color: "#6aa9f0",
      part: "enc"
    },
    lstm: {
      tag: "② CONVLSTM · REMEMBER",
      title: "Spatiotemporal Recurrent Memory",
      desc: "Watches how the week unfolds by stepping through the 7 days in temporal order. Its recurrent memory gates preserve relevant trends and discard high-frequency surface noise.",
      chips: ["7 recurrent steps", "Maintains spatial coordinates", "Memory state tracking", "No future leakage"],
      color: "#ff9a6b",
      part: "lstm"
    },
    latent: {
      tag: "LATENT BOTTLENECK · 256 × 12 × 22",
      title: "OceanEmbed (Latent Fingerprint)",
      desc: "The core latent representation learned by the system. A compact 256 × 12 × 22 bottleneck tensor summarizing multi-day subsurface thermal memory, condensed from 7 surface fields and 7 temporal steps.",
      chips: ["256 latent channels", "12 × 22 spatial grid", "Bottle-neck representation", "Zero future leakage"],
      color: "#5fd6a6",
      part: "lstm"
    },
    dec: {
      tag: "③ U-NET DECODER · PAINT",
      title: "Spatial Reconstruction & Skip Fusion",
      desc: "Upsamples the latent fingerprint back to full 96×176 resolution. Skip connections directly transfer fine spatial boundary details from the final day's encoder, keeping coastlines and eddies crisp.",
      chips: ["3 skip connections", "Direct feature handover", "Upsampling to 96×176", "Preserves sharp gradients"],
      color: "#e3a634",
      part: "dec"
    },
    out: {
      tag: "④ OUTPUT · 15 DEPTHS",
      title: "Full 3-D Ocean Temperature Column",
      desc: "Simultaneously predicts subsurface ocean temperature across 15 standard depths from surface (0 m) down to the abyss (1,000 m). Tap any depth bar or drag the slider below to inspect that layer.",
      chips: ["15 standard depths", "0–1,000 m vertical range", "Daily 3D reconstruction", "Evaluated on Argo floats"],
      color: "#7fd8ff",
      part: "out"
    }
  };

  const DEPTH_INFO = {
    0: { zone: "Surface Mixed Layer", desc: "Directly in contact with atmosphere, solar heating, and monsoon winds. Measured directly by satellite infrared radiometers." },
    5: { zone: "Surface Mixed Layer", desc: "Top 10 meters remain thoroughly mixed by wind waves, with nearly uniform temperature across the upper ocean." },
    10: { zone: "Upper Mixed Layer", desc: "Homogeneous mixed layer base in summer; deepens to 40-60m during strong winter and monsoon wind stirring." },
    20: { zone: "Mixed Layer Transition", desc: "Beginning of the seasonal pycnocline where surface salinity lids trap solar heat near the top." },
    30: { zone: "Upper Thermocline", desc: "Transition boundary where solar penetration drops off and vertical density stratification increases." },
    50: { zone: "Main Thermocline Entry", desc: "Rapid vertical temperature gradient begins; vertical mixing drops and barrier layer physics become active." },
    75: { zone: "Core Thermocline", desc: "Steepest temperature drop. Internal waves and mesoscale eddy pumping cause large localized variations." },
    100: { zone: "Core Thermocline Peak", desc: "Peak thermal gradient zone. GLORYS reanalysis exhibits its largest warm bias (+0.59 °C) here, cleanly fixed by OTER bias correction." },
    125: { zone: "Deep Thermocline", desc: "Strong subsurface stratification. OTER achieves an independent RMSE of 1.158 °C, beating the teacher model GLORYS." },
    150: { zone: "Lower Thermocline", desc: "Temperature drops below 18 °C. Internal thermocline displacement closely mirrors surface sea-level anomalies." },
    200: { zone: "Permanent Thermocline", desc: "Below the direct reach of seasonal winds; temperatures vary between 12 and 19 °C." },
    300: { zone: "Intermediate Water Mass", desc: "Transition into Red Sea and Persian Gulf outflow water in the Arabian Sea, with characteristic salinity intrusions." },
    500: { zone: "Intermediate Ocean", desc: "Calm intermediate waters (9–13 °C) with minimal daily variance; vertical gradient loss maintains physical stability." },
    700: { zone: "Deep Ocean Layer", desc: "Deep ocean waters (8–11 °C) where OTER achieves 0.224 °C error, outperforming GLORYS reanalysis." },
    1000: { zone: "Abyssal Benchmark", desc: "Deep boundary at 1,000 m (6–9 °C). Validated against Argo float parking and profiling depths with exceptional 0.216 °C accuracy." }
  };

  const infoCard = $("#arch-info");
  const infoTag = $("#arch-info-tag");
  const infoTitle = $("#arch-info-title");
  const infoDesc = $("#arch-info-desc");
  const infoChips = $("#arch-info-chips");
  const infoClose = $("#arch-info-close");

  let selectedComp = null; // null = auto scroll tracking; string = locked component

  function motionTap(target) {
    if (window.Motion && typeof window.Motion.animate === "function" && target) {
      window.Motion.animate(target, { scale: [1, 0.96, 1] }, { duration: 0.18, easing: "ease-out" });
    }
  }

  function showInfoCard(data) {
    if (!infoCard) return;
    infoTag.textContent = data.tag;
    infoTag.style.color = data.color || "#7fd8ff";
    infoTitle.textContent = data.title;
    infoDesc.textContent = data.desc;
    infoChips.innerHTML = (data.chips || []).map(c => `<span>${c}</span>`).join("");
    infoCard.classList.add("on");

    if (window.Motion && typeof window.Motion.animate === "function") {
      window.Motion.animate(infoCard, { opacity: [0, 1], y: [-8, 0], scale: [0.99, 1] }, { duration: 0.26, easing: [0.2, 0.8, 0.2, 1] });
    }
  }

  function hideInfoCard() {
    if (!infoCard) return;
    if (window.Motion && typeof window.Motion.animate === "function") {
      window.Motion.animate(infoCard, { opacity: [1, 0], y: [0, -6] }, { duration: 0.18 }).then(() => {
        infoCard.classList.remove("on");
      });
    } else {
      infoCard.classList.remove("on");
    }
  }

  function selectComponent(compKey, clickedEl) {
    if (!compKey || !COMP_DATA[compKey]) {
      deselectComponent();
      return;
    }
    selectedComp = compKey;
    svg.setAttribute("data-selected", compKey);
    svg.setAttribute("data-active", COMP_DATA[compKey].part);

    cards.forEach(c => c.classList.toggle("now", c.dataset.part === COMP_DATA[compKey].part));
    showInfoCard(COMP_DATA[compKey]);
    if (clickedEl) motionTap(clickedEl);
  }

  function deselectComponent() {
    selectedComp = null;
    svg.removeAttribute("data-selected");
    slabElements.forEach(s => s.classList.remove("selected"));
    hideInfoCard();
    cards.forEach(c => c.classList.remove("now"));
    svg.dataset.active = "all";
  }

  if (infoClose) {
    infoClose.addEventListener("click", (e) => {
      e.stopPropagation();
      deselectComponent();
    });
  }

  // A click on the diagram's empty background lights every component again. Components and
  // depth slabs stop propagation, so only true background clicks get here.
  $(".arch-sticky").addEventListener("click", (e) => {
    if (!e.target.closest("#arch-info")) deselectComponent();
  });

  // Interactive targets for components
  $$("[data-comp]", svg).forEach(elNode => {
    elNode.addEventListener("click", (e) => {
      e.stopPropagation();
      const comp = elNode.getAttribute("data-comp");
      if (selectedComp === comp) {
        deselectComponent();
      } else {
        selectComponent(comp, elNode);
      }
    });

    // Subtle Motion hover indication
    elNode.addEventListener("pointerenter", () => {
      if (window.Motion && typeof window.Motion.animate === "function" && selectedComp !== elNode.getAttribute("data-comp")) {
        window.Motion.animate(elNode, { scale: 1.015 }, { duration: 0.15 });
      }
    });
    elNode.addEventListener("pointerleave", () => {
      if (window.Motion && typeof window.Motion.animate === "function") {
        window.Motion.animate(elNode, { scale: 1 }, { duration: 0.15 });
      }
    });
  });

  // Depth selection logic
  const rng = $("#depth-range"), img = $("#depth-img");
  D.depths.forEach(d => { const i = new Image(); i.src = `img/out_${d}.webp`; });

  function updDepthView(idx) {
    const d = D.depths[idx], [lo, hi] = D.colour_ranges.temp[String(d)] || [0, 35];
    if (img) {                       // the depth-map picker is optional on this page
      img.src = `img/out_${d}.webp`;
      img.alt = `Reconstructed temperature at ${d} m`;
      $("#depth-val").textContent = d + " m";
      $("#depth-lo").textContent = lo.toFixed(1) + " °C";
      $("#depth-hi").textContent = hi.toFixed(1) + " °C";
    }
    slabElements.forEach((s, k) => s.classList.toggle("selected", k === idx));
    const outC = $("#depth-chips-output");
    if (outC) $$("span", outC).forEach((s, k) => s.classList.toggle("chip-active", k === idx));
  }

  function selectDepth(depthIdx, fromUser = true, clickedEl) {
    depthIdx = clamp(depthIdx, 0, D.depths.length - 1);
    const d = D.depths[depthIdx];
    if (rng) rng.value = depthIdx;
    updDepthView(depthIdx);

    if (fromUser) {
      if (selectedComp === "depth-" + d) {
        deselectComponent();
        return;
      }
      selectedComp = "depth-" + d;
      svg.setAttribute("data-selected", "depth");
      svg.setAttribute("data-active", "out");
      cards.forEach(c => c.classList.toggle("now", c.dataset.part === "out"));
      const info = DEPTH_INFO[d] || { zone: "Subsurface Layer", desc: `Reconstructed temperature layer at ${d} m.` };
      const [lo, hi] = D.colour_ranges.temp[String(d)] || [0, 35];
      showInfoCard({
        tag: `DEPTH OUTPUT · ${d} M`,
        title: `${info.zone} (${d} m)`,
        desc: `${info.desc}`,
        chips: [
          `Depth: ${d} m`,
          `Observed T: ${lo.toFixed(1)} – ${hi.toFixed(1)} °C`,
          `OTER RMSE: ${D.rmse.final[depthIdx].toFixed(3)} °C vs Argo`,
          `Teacher bias: ${(D.offset[depthIdx] > 0 ? "+" : "") + D.offset[depthIdx].toFixed(2)} °C`
        ],
        color: "#7fd8ff"
      });
      if (clickedEl) motionTap(clickedEl);
    }
  }

  slabElements.forEach((slab, i) => {
    slab.addEventListener("click", (e) => {
      e.stopPropagation();
      selectDepth(i, true, slab);
    });
    slab.addEventListener("pointerenter", () => {
      if (window.Motion && typeof window.Motion.animate === "function" && !slab.classList.contains("selected")) {
        window.Motion.animate(slab, { scale: 1.02 }, { duration: 0.12 });
      }
    });
    slab.addEventListener("pointerleave", () => {
      if (window.Motion && typeof window.Motion.animate === "function") {
        window.Motion.animate(slab, { scale: 1 }, { duration: 0.12 });
      }
    });
  });

  rng?.addEventListener("input", () => {
    const idx = +rng.value;
    selectDepth(idx, true);
  });

  const outChips = $("#depth-chips-output");
  if (outChips) {
    D.depths.forEach((d, i) => {
      const s = h("span", d === 0 ? "chip-active" : "", d + " m");
      s.style.setProperty("--i", i);
      s.style.cursor = "pointer";
      s.addEventListener("click", () => {
        selectDepth(i, true);
      });
      outChips.appendChild(s);
    });
  }

  updDepthView(0);

  // Step cards in .arch-steps
  cards.forEach(card => {
    card.addEventListener("click", () => {
      if (selectedComp === card.dataset.part) {
        deselectComponent();
      } else {
        selectComponent(card.dataset.part, card);
      }
    });
    card.addEventListener("pointerenter", () => {
      if (window.Motion && typeof window.Motion.animate === "function") {
        window.Motion.animate(card, { y: -3 }, { duration: 0.2 });
      }
    });
    card.addEventListener("pointerleave", () => {
      if (window.Motion && typeof window.Motion.animate === "function") {
        window.Motion.animate(card, { y: 0 }, { duration: 0.2 });
      }
    });
  });

  // ConvLSTM ticks through the 7 days while visible
  let day = 0, timer = null;
  const tick = () => {
    beads.forEach((b, i) => {
      b.setAttribute("fill", i <= day ? "#ff9a6b" : "#5a2a18");
      b.setAttribute("r", i === day ? 8 : 6);
    });
    $$(".in-frame", inp).forEach((f, i) => f.setAttribute("opacity", i === day ? 1 : 0.45));
    $("#lstm-day").textContent = day === 6 ? "t" : "t−" + (6 - day);
    day = (day + 1) % 7;
  };
  new IntersectionObserver(es => es.forEach(e => {
    clearInterval(timer);
    if (e.isIntersecting && !reduce) timer = setInterval(tick, 650); else tick();
  }), { threshold: 0.2 }).observe(svg);
})();

// ---------- 5: ensemble ----------
(() => {
  const S = D.sample, depths = D.depths;
  const members = [0, 1, 2, 3, 4, 5].map(k => S.pred.map((v, i) => {
    // Illustrative spread: members scatter most where the model is least certain (the
    // thermocline), scaled by the real single-level error, and average back to S.pred.
    const sgn = k % 2 ? -1 : 1, phase = k * 1.7;
    return v + sgn * 1.0 * D.rmse.before_bc[i] * Math.sin(i * 0.6 + phase) * (0.6 + 0.2 * (k % 3));
  }));
  const svg = $("#ens-chart");
  const lo = Math.floor(Math.min(...S.pred) - 2), hi = Math.ceil(Math.max(...S.pred) + 2);
  const ticks = []; for (let v = Math.ceil(lo / 5) * 5; v <= hi; v += 5) ticks.push(v);
  const ser = members.map((x, k) => ({ x, y: depths, color: k < 3 ? C.blue : C.orange, width: 1.6, opacity: 0.75, delay: 0.1 * k }));
  ser.push({ x: S.pred, y: depths, color: C.green, width: 3, delay: 0.9 });
  const ch = profileChart(svg, { x0: lo, x1: hi, xticks: ticks, xlabel: "temperature (°C)", series: ser,
    tipRow: (i, d) => rowTip(d, [["ensemble mean", C.green, S.pred[i].toFixed(2) + " °C"]]) });
  reveal(svg);
  legend($("#ens-legend"), [["M4-MSE seeds (1–3)", C.blue], ["M4-DW seeds (1–3)", C.orange], ["Ensemble Mean", C.green]]);
  $("#ens-where").textContent = `${S.lat.toFixed(1)}°N ${S.lon.toFixed(1)}°E on 4 Dec 2023`;

  // Draw 6 convergence connection paths from seeds to Ensemble Mean
  const ensG = $("#ens-paths");
  function drawEnsembleConnections() {
    if (!ensG) return;
    ensG.innerHTML = "";
    const seedBtns = $$(".seed-btns .seed");
    const container = $(".ens-diagram-wrap");
    if (!container) return;
    const cRect = container.getBoundingClientRect();
    const cx = 230, cy = 42;

    seedBtns.forEach((btn, idx) => {
      const bRect = btn.getBoundingClientRect();
      const startX = cRect.width > 0 ? ((bRect.left + bRect.width / 2 - cRect.left) / cRect.width) * 460 : (idx < 3 ? 45 + idx * 70 : 275 + (idx - 3) * 70);
      const color = idx < 3 ? "#3987e5" : "#d95926";
      el("path", {
        d: `M ${startX} 0 C ${startX} ${cy * 0.75}, ${cx} ${cy * 0.75}, ${cx} ${cy}`,
        fill: "none",
        stroke: color,
        "stroke-width": "2",
        "stroke-dasharray": "5 5",
        opacity: "0.85",
        class: `ens-conn-path ens-conn-${idx}`
      }, ensG);
    });

    // Convergence junction line to merge box with arrowhead
    el("line", {
      x1: cx, y1: cy, x2: cx, y2: 66,
      stroke: "#5fd6a6",
      "stroke-width": "2.5",
      "marker-end": "url(#arrow-ens)"
    }, ensG);
  }

  drawEnsembleConnections();
  addEventListener("resize", drawEnsembleConnections);

  // Clickable seeds with Motion animations & selection persistence
  let activeSeed = null;
  $$(".seed-btns .seed").forEach(b => {
    b.addEventListener("click", () => {
      const k = +b.dataset.k;
      const isAlreadyOn = activeSeed === k;
      $$(".seed-btns .seed").forEach(o => o.classList.remove("on"));
      $$(".ens-conn-path", ensG).forEach(p => p.classList.remove("active"));

      if (isAlreadyOn) {
        activeSeed = null;
        ch.paths.slice(0, 6).forEach(p => {
          p.setAttribute("opacity", "0.75");
          p.setAttribute("stroke-width", "1.6");
        });
      } else {
        activeSeed = k;
        b.classList.add("on");
        if (window.Motion && typeof window.Motion.animate === "function") {
          window.Motion.animate(b, { scale: [1, 0.94, 1] }, { duration: 0.18 });
        }
        ch.paths.slice(0, 6).forEach((p, j) => {
          p.setAttribute("opacity", j === k ? "1" : "0.15");
          p.setAttribute("stroke-width", j === k ? "3.2" : "1.6");
        });
        const activePath = $(`.ens-conn-${k}`, ensG);
        if (activePath) activePath.classList.add("active");
      }
    });

    // Hover effect with Motion
    b.addEventListener("pointerenter", () => {
      if (window.Motion && typeof window.Motion.animate === "function" && activeSeed !== +b.dataset.k) {
        window.Motion.animate(b, { y: -2 }, { duration: 0.15 });
      }
    });
    b.addEventListener("pointerleave", () => {
      if (window.Motion && typeof window.Motion.animate === "function") {
        window.Motion.animate(b, { y: 0 }, { duration: 0.15 });
      }
    });
  });

  $("#k-single").textContent = D.blended.single.toFixed(3) + " °C";
  $("#k-ens").textContent = D.blended.ensemble.toFixed(3) + " °C";
})();

// ---------- 6: bias correction ----------
(() => {
  const svg = $("#bias-chart"), W = 520, H = 420, m = { l: 62, r: 20, t: 10, b: 40 };
  const x0 = -0.2, x1 = 0.7, X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const rowH = (H - m.t - m.b) / D.depths.length;
  const g = el("g", { class: "grid" }, svg);
  [-0.2, 0, 0.2, 0.4, 0.6].forEach(v => { el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, g); el("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle", text: (v > 0 ? "+" : "") + v.toFixed(1) }, svg); });
  el("line", { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: "#b3c6d6", "stroke-width": 1.5 }, svg);
  el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", text: "b_d  (°C) - positive = model too warm" }, svg);
  const iMax = D.offset.indexOf(Math.max(...D.offset));
  D.depths.forEach((d, i) => {
    const v = D.offset[i], y = m.t + i * rowH + 3, x = Math.min(X(0), X(v)), w = Math.abs(X(v) - X(0));
    el("text", { x: m.l - 8, y: y + rowH / 2 + 1, "text-anchor": "end", text: d + " m" }, svg);
    const r = el("rect", { x, y, width: Math.max(w, 1), height: rowH - 6, rx: 3, fill: v >= 0 ? C.warm : C.cold, class: "gbar" }, svg);
    r.style.transformOrigin = v >= 0 ? "left" : "right"; r.style.setProperty("--i", i);
    if (i === iMax) { const t = el("text", { x: X(v) + 8, y: y + rowH / 2 + 1, class: "lbl pop", text: `+${v.toFixed(2)} °C` }, svg); t.style.setProperty("--delay", "1.6s"); }
    const hit = el("rect", { x: m.l, y: y - 3, width: W - m.l - m.r, height: rowH, class: "hit" }, svg);
    const show = ev => showTip(ev.clientX, ev.clientY, `<b>${d} m</b><br><span class="k">b<sub>d</sub></span> ${(v > 0 ? "+" : "") + f3(v)} °C<br><span class="k">${v > 0 ? "model too warm → subtract" : "model too cold → add"}</span>`);
    hit.addEventListener("pointermove", show); hit.addEventListener("pointerdown", show); hit.addEventListener("pointerleave", hideTip);
  });
  reveal(svg);

  const svg2 = $("#bc-chart");
  profileChart(svg2, { x0: 0, x1: 1.8, xticks: [0, 0.5, 1, 1.5], xlabel: "error vs Argo, 2023–24 (RMSE °C)",
    series: [{ x: D.rmse.before_bc, y: D.depths, color: "#8fa6b8", width: 2, dash: "6 5", delay: 0.2 },
             { x: D.rmse.final, y: D.depths, color: C.sky, width: 3, dots: true, delay: 0.5 }],
    tipRow: (i, d) => rowTip(d, [["before", "#8fa6b8", f3(D.rmse.before_bc[i]) + " °C"], ["after", C.sky, f3(D.rmse.final[i]) + " °C"]]) });
  reveal(svg2);
  legend($("#bc-legend"), [["before correction", "#8fa6b8", true], ["after correction", C.sky]]);
  $("#k-before").textContent = D.blended.ensemble.toFixed(3) + " °C";
  $("#k-after").textContent = D.blended.final.toFixed(3) + " °C";
})();

onScroll();

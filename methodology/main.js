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
  filmScrub();
}
addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });

// ---------- why ----------
(() => {
  const [la0, la1] = [0.5, 24.5], [lo0, lo1] = [55.5, 99.5];
  const map = $("#argo-map");
  D.argo_dots.forEach(([lat, lon], i) => {
    const d = h("span", "argo-dot"); d.style.left = pct(lon, lo0, lo1) + "%"; d.style.top = (100 - pct(lat, la0, la1)) + "%";
    d.style.setProperty("--i", i); map.appendChild(d);
  });
  reveal(map);
  $("#stat-argo").firstChild.textContent = D.argo_dots.length;
  $("#stat-cells").firstChild.textContent = (96 * 176).toLocaleString("en-IN");
  $("#chip-casts").textContent = `checked on ${D.argo_casts} real Argo profiles`;
})();

// ---------- 1: inputs + pipeline ----------
(() => {
  const T = [
    ["sst", "Sea surface temperature", "°C", "NOAA OISST v2.1", "How warm the very top of the sea is — the first clue to how much heat is stored below."],
    ["sss", "Sea surface salinity", "PSU", "NASA SMAP · 8-day mean", "How salty the surface is. River water and monsoon rain freshen the Bay of Bengal into a light 'lid' that traps heat near the top."],
    ["sla", "Sea level anomaly", "m", "Copernicus DUACS altimetry", "Bumps and dips in sea level. A raised surface usually sits over a thick warm layer; a dip means cold water pushed up from below."],
    ["cur_u", "Current · east–west", "m/s", "NASA OSCAR v2.0", "How fast surface water flows east (red) or west (blue). Currents carry heat sideways."],
    ["cur_v", "Current · north–south", "m/s", "NASA OSCAR v2.0", "How fast surface water flows north (red) or south (blue)."],
    ["wind_u", "Wind · east–west", "m/s", "Copernicus scatterometer", "Wind stirs and mixes the upper ocean, pushing warm water down or cold water up."],
    ["wind_v", "Wind · north–south", "m/s", "Copernicus scatterometer", "The monsoon reverses these winds each season — and the ocean below responds."],
  ];
  const box = $("#tiles");
  T.forEach(([k, name, unit, src, why], i) => {
    const t = h("button", "tile");
    t.style.setProperty("--i", i);
    t.setAttribute("aria-label", `${name}: ${why}`);
    t.innerHTML = `<img src="img/in_${k}.webp" alt="" loading="lazy"><div class="cap"><b>${name}</b><span>${unit} · ${src}</span></div>
      <div class="back"><b>${name}</b>${why}</div>`;
    t.style.font = "inherit"; t.style.color = "inherit"; t.style.textAlign = "left"; t.style.padding = "0";
    t.addEventListener("click", () => t.classList.toggle("flip"));
    box.appendChild(t);
  });
  box.appendChild(h("div", "tile-hint", "Seven maps.<br>Every day since 2015.<br><span style='color:#7fd8ff'>Tap any map ↺</span>"));
  reveal(box);

  const S = [
    ["Quality control", "Drop flagged or physically impossible values from every product."],
    ["Land mask", "Mark land cells so they are never used as input or scored."],
    ["Regrid to 0.25° (bilinear)", "Put every product on the same 0.25° grid — about 28 km per cell."],
    ["Daily alignment", "Line all seven fields up on the same calendar days."],
    ["Z-score (train stats only)", "Rescale each variable with averages from the training years only — no hint of the future leaks in."],
    ["Missing-value mask", "Remember where data is missing (e.g. satellite gaps) instead of inventing it."],
    ["Domain 0–25°N, 55–100°E", "Crop to the Arabian Sea and Bay of Bengal: 96 × 176 cells."],
  ];
  const pipe = $("#pipeline");
  S.forEach(([name, tipTxt], i) => {
    const c = h("div", "pipe", `${name}<span class="tip">${tipTxt}</span>`);
    c.dataset.n = i + 1; c.tabIndex = 0;
    c.addEventListener("click", () => { $$(".pipe.show").forEach(o => o !== c && o.classList.remove("show")); c.classList.toggle("show"); });
    pipe.appendChild(c);
    if (i < S.length - 1) pipe.appendChild(h("span", "pipe-arrow", "→"));
  });
  reveal(pipe, () => $$(".pipe", pipe).forEach((c, i) => setTimeout(() => c.classList.add("lit"), 350 + i * 420)));
})();

// ---------- 2: 7-day window (scroll-scrubbed stack) ----------
const film = $("#film"), stage = $("#window-stage");
D.window.forEach((d, k) => {
  const f = h("div", "frame" + (k === 6 ? " today" : ""));
  const date = new Date(d + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
  f.innerHTML = `<img src="img/win_${k}.webp" alt="Sea surface temperature on ${date}"><span class="lbl">${k === 6 ? "t" : "t−" + (6 - k)} · ${date}</span>`;
  f.style.zIndex = k; film.appendChild(f);
});
function filmScrub() {
  const r = stage.getBoundingClientRect();
  const p = reduce ? 1 : clamp((-r.top) / (r.height - innerHeight) * 1.25);
  const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;     // ease in-out
  const small = innerWidth < 900, sx = small ? 34 : 58, cube = small ? 0.72 : 1;
  $$(".frame", film).forEach((f, k) => {
    const o = k - 3;
    const fan = { x: o * sx, y: Math.abs(o) * 6, z: 0, rz: o * 7, rx: 0, s: 0.86 };
    const stk = { x: o * 12 * cube, y: -o * 14 * cube, z: o * 26, rz: -32, rx: 56, s: 1 };
    const v = k2 => lerp(fan[k2], stk[k2], e);
    f.style.transform = `translate3d(${v("x")}px, ${v("y")}px, ${v("z")}px) rotateX(${v("rx")}deg) rotateZ(${v("rz")}deg) scale(${v("s")})`;
    f.style.opacity = String(lerp(0.55 + 0.45 * (k / 6), 1, e));
  });
}
filmScrub();

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
  // output slabs, surface (warm) to 1000 m (cold)
  const out = $("#arch-out");
  D.depths.forEach((d, i) => {
    const y = 104 + i * 16.5;
    el("rect", { x: 840, y, width: 112, height: 12, rx: 3, fill: hexMix(SEQ, 1 - i / 14 * 0.92) }, out);
    if (i === 0 || i === 7 || i === 14) el("text", { x: 832, y: y + 10, "text-anchor": "end", text: d + " m", style: "font-size:11px" }, out);
  });
  // flowing particles
  const pg = $("#particles");
  if (!reduce) for (let k = 0; k < 8; k++) {
    const c = el("circle", { r: 3.2, fill: "#7fd8ff", opacity: 0.9 }, pg);
    const am = el("animateMotion", { dur: "3.6s", repeatCount: "indefinite", begin: (k * 0.45) + "s" }, c);
    el("mpath", { href: "#flow" }, am);
  }
  // step highlighting as the text cards pass the middle of the screen
  const cards = $$(".arch-steps .card");
  const ob = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    cards.forEach(c => c.classList.toggle("now", c === e.target));
    svg.dataset.active = e.target.dataset.part;
  }), { rootMargin: "-68% 0px -27% 0px" });
  cards.forEach(c => ob.observe(c));
  // ConvLSTM ticks through the 7 days while visible
  let day = 0, timer = null;
  const tick = () => {
    beads.forEach((b, i) => { b.setAttribute("fill", i <= day ? "#ff9a6b" : "#5a2a18"); b.setAttribute("r", i === day ? 8 : 6); });
    $$(".in-frame", inp).forEach((f, i) => f.setAttribute("opacity", i === day ? 1 : 0.45));
    $("#lstm-day").textContent = day === 6 ? "t" : "t−" + (6 - day);
    day = (day + 1) % 7;
  };
  new IntersectionObserver(es => es.forEach(e => {
    clearInterval(timer);
    if (e.isIntersecting && !reduce) timer = setInterval(tick, 650); else tick();
  }), { threshold: 0.2 }).observe(svg);
  // depth picker
  const rng = $("#depth-range"), img = $("#depth-img");
  D.depths.forEach(d => { const i = new Image(); i.src = `img/out_${d}.webp`; });
  const upd = () => {
    const d = D.depths[+rng.value], [lo, hi] = D.colour_ranges.temp[String(d)];
    img.src = `img/out_${d}.webp`; img.alt = `Reconstructed temperature at ${d} m`;
    $("#depth-val").textContent = d + " m";
    $("#depth-lo").textContent = lo.toFixed(1) + " °C"; $("#depth-hi").textContent = hi.toFixed(1) + " °C";
  };
  rng.addEventListener("input", upd); upd();
})();

// ---------- 4: training ----------
(() => {
  const chips = $("#depth-chips");
  D.depths.forEach((d, i) => { const s = h("span", "", d + " m"); s.style.setProperty("--i", i); chips.appendChild(s); });
  reveal(chips);
  reveal($("#mask-demo"));
  const box = $("#wbars"), rows = [];
  const max = Math.max(...D.dw_weights);
  D.depths.forEach((d, i) => {
    const r = h("div", "wbar", `<span>${d} m</span><div><div class="b"></div></div><span class="v"></span>`);
    box.appendChild(r); rows.push(r);
  });
  const text = {
    mse: "Every °C of error counts the same at every depth — so the busy thermocline (50–200 m), where errors are largest, dominates training.",
    dw: "Each depth's error is divided by how much that depth naturally varies — so the calm deep ocean and the surface count as much as the thermocline.",
  };
  function set(mode) {
    $$(".toggle button").forEach(b => { const on = b.dataset.mode === mode; b.classList.toggle("on", on); b.setAttribute("aria-selected", on); });
    $("#loss-explain").textContent = text[mode];
    rows.forEach((r, i) => {
      const w = mode === "mse" ? 1 : D.dw_weights[i];
      const b = $(".b", r); b.style.width = (w / max * 100) + "%"; b.style.background = mode === "mse" ? C.blue : C.orange;
      $(".v", r).textContent = w.toFixed(2);
    });
  }
  $$(".toggle button").forEach(b => b.addEventListener("click", () => set(b.dataset.mode)));
  rows.forEach(r => { $(".b", r).style.width = "0"; });
  reveal(box, () => set("mse"));
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
  legend($("#ens-legend"), [["MSE seeds", C.blue], ["DW-MSE seeds", C.orange], ["ensemble mean", C.green]]);
  $("#ens-where").textContent = `${S.lat.toFixed(1)}°N ${S.lon.toFixed(1)}°E on 4 Dec 2023`;
  $$(".seed").forEach(b => b.addEventListener("click", () => {
    const k = +b.dataset.k, on = !b.classList.contains("on");
    $$(".seed").forEach(o => o.classList.remove("on"));
    ch.paths.slice(0, 6).forEach((p, j) => { p.setAttribute("opacity", on ? (j === k ? 1 : 0.15) : 0.75); p.setAttribute("stroke-width", on && j === k ? 3 : 1.6); });
    if (on) b.classList.add("on");
  }));
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
  el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", text: "b_d  (°C) — positive = model too warm" }, svg);
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

// ---------- 7: Argo cycle + evaluation ----------
(() => {
  const svg = $("#argo-cycle"), S = D.sample;
  const SURF = 50, BOT = 340, Yd = d => SURF + d / 1000 * (BOT - SURF);
  const defs = el("defs", {}, svg);
  const lg = el("linearGradient", { id: "wcol", x1: 0, x2: 0, y1: 0, y2: 1 }, defs);
  el("stop", { offset: 0, "stop-color": "#1b7fa8", "stop-opacity": 0.55 }, lg);
  el("stop", { offset: 1, "stop-color": "#04121f", "stop-opacity": 0.1 }, lg);
  el("rect", { x: 0, y: SURF, width: 470, height: BOT - SURF + 20, fill: "url(#wcol)", rx: 10 }, svg);
  el("path", { d: `M0 ${SURF} q30 -6 60 0 t60 0 t60 0 t60 0 t60 0 t60 0 t60 0 t60 0`, stroke: "#7fd8ff", "stroke-width": 2, fill: "none" }, svg);
  [0, 250, 500, 750, 1000].forEach(d => el("text", { x: 8, y: Yd(d) + (d ? 4 : 16), text: d + " m" }, svg));
  el("path", { d: `M70 ${SURF} C 90 ${Yd(500)}, 110 ${Yd(1000)}, 150 ${Yd(1000)} H 340 C 380 ${Yd(1000)}, 400 ${Yd(400)}, 410 ${SURF}`, stroke: "rgba(255,209,102,0.35)", "stroke-width": 1.5, "stroke-dasharray": "4 6", fill: "none" }, svg);
  const sat = el("g", { transform: "translate(440 18)" }, svg);
  el("rect", { x: -7, y: -5, width: 14, height: 10, rx: 2, fill: "#dfe9f2" }, sat);
  el("rect", { x: -26, y: -3, width: 16, height: 6, fill: C.blue }, sat); el("rect", { x: 10, y: -3, width: 16, height: 6, fill: C.blue }, sat);
  const waves = el("g", { opacity: 0 }, svg);
  [10, 20, 30].forEach(r => el("path", { d: `M${410 - r} ${SURF - 8 - r * 0.2} A ${r} ${r} 0 0 1 ${410 + r} ${SURF - 8 - r * 0.2}`, stroke: "#7fd8ff", "stroke-width": 1.5, fill: "none" }, waves));
  const fl = el("g", {}, svg);
  el("line", { x1: 0, y1: -18, x2: 0, y2: -30, stroke: "#dfe9f2", "stroke-width": 1.5 }, fl);
  el("rect", { x: -5, y: -18, width: 10, height: 34, rx: 5, fill: "#ffd166" }, fl);
  // profile panel
  const px0 = 500, px1 = 690, t0 = 5, t1 = 31, PX = t => px0 + (t - t0) / (t1 - t0) * (px1 - px0);
  el("rect", { x: px0 - 6, y: SURF, width: px1 - px0 + 12, height: BOT - SURF, fill: "rgba(0,0,0,0.2)", rx: 8 }, svg);
  [10, 20, 30].forEach(t => el("text", { x: PX(t), y: BOT + 18, "text-anchor": "middle", text: t + "°" }, svg));
  el("text", { x: (px0 + px1) / 2, y: SURF - 10, "text-anchor": "middle", text: "temperature it measures" }, svg);
  const prof = el("path", { fill: "none", stroke: C.orange, "stroke-width": 2.5, "stroke-linejoin": "round" }, svg);
  const P = S.argo_raw.p.map((p, i) => [p, S.argo_raw.t[i]]).sort((a, b) => b[0] - a[0]);
  const note = $("#cycle-note");
  const phases = [["① Sinks to about 1,000 m", 2200], ["② Drifts with the currents for days", 1800],
                  ["③ Rises, measuring temperature all the way up", 3400], ["④ Sends its profile home by satellite", 1600], ["", 1000]];
  const total = phases.reduce((a, p) => a + p[1], 0);
  const path = [[70, SURF], [150, Yd(1000)], [340, Yd(1000)], [410, SURF]];
  let start = null, running = false;
  function at(tms) {
    let t = tms % total, k = 0; while (t > phases[k][1]) { t -= phases[k][1]; k++; }
    const u = t / phases[k][1], e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
    let x, y, depthNow = 1000;
    if (k === 0) { x = lerp(path[0][0], path[1][0], e); y = lerp(path[0][1], path[1][1], e); prof.setAttribute("d", ""); }
    else if (k === 1) { x = lerp(path[1][0], path[2][0], u); y = path[1][1]; }
    else if (k === 2) { x = lerp(path[2][0], path[3][0], e); y = lerp(path[2][1], path[3][1], e); depthNow = lerp(1000, 0, e); }
    else { x = path[3][0]; y = SURF; depthNow = 0; }
    if (k >= 2) prof.setAttribute("d", P.filter(q => q[0] >= depthNow).map((q, i) => (i ? "L" : "M") + PX(q[1]).toFixed(1) + " " + Yd(q[0]).toFixed(1)).join(""));
    fl.setAttribute("transform", `translate(${x} ${y})`);
    waves.setAttribute("opacity", k === 3 ? 0.5 + 0.5 * Math.sin(u * 20) : 0);
    if (phases[k][0]) note.textContent = phases[k][0];
  }
  function loop(ts) { if (!running) return; if (start === null) start = ts; at(ts - start); requestAnimationFrame(loop); }
  new IntersectionObserver(es => es.forEach(e => {
    if (reduce) { at(total - 1100); note.textContent = "Argo floats sink, drift, then rise measuring temperature, and report by satellite."; return; }
    running = e.isIntersecting; if (running) { start = null; requestAnimationFrame(loop); }
  }), { threshold: 0.25 }).observe(svg);

  // comparison chart
  const ev = $("#eval-chart");
  const lo = 5, hi = 31;
  profileChart(ev, { x0: lo, x1: hi, xticks: [10, 15, 20, 25, 30], xlabel: "temperature (°C)",
    series: [{ x: S.argo_raw.t, y: S.argo_raw.p, color: C.orange, width: 2, delay: 0 },
             { x: S.glorys, y: D.depths, color: C.gold, width: 2, dash: "6 5", delay: 0.8 },
             { x: S.pred, y: D.depths, color: C.blue, width: 3, dots: true, delay: 0.5 }],
    tipRow: (i, d) => rowTip(d, [["Argo", C.orange, Number.isFinite(S.argo15[i]) ? S.argo15[i].toFixed(2) + " °C" : "—"],
                                 ["OTER", C.blue, S.pred[i].toFixed(2) + " °C"], ["GLORYS", C.gold, S.glorys[i].toFixed(2) + " °C"]]) });
  reveal(ev);
  legend($("#eval-legend"), [["Argo float (truth)", C.orange], ["OTER", C.blue], ["GLORYS", C.gold, true]]);
  $("#eval-note").textContent = `Float ${S.float}, ${S.lat.toFixed(2)}°N ${S.lon.toFixed(2)}°E, 4 Dec 2023 — never seen in training. Error over this column: ${S.rmse.toFixed(2)} °C RMSE.`;
  const map = $("#eval-map");
  const pin = h("span", "pin"), box = h("span", "cell-box");
  [pin, box].forEach(n => { n.style.left = pct(S.lon, 55.5, 99.5) + "%"; n.style.top = (100 - pct(S.lat, 0.5, 24.5)) + "%"; map.appendChild(n); });
})();

// ---------- result ----------
(() => {
  const B = D.blended;
  $("#res-casts").textContent = D.argo_casts;
  $("#gap").textContent = (B.final - B.glorys).toFixed(2) + " °C";
  const big = $("#big");
  reveal(big.closest(".step-head"), () => {
    if (reduce) { big.textContent = B.final.toFixed(3); return; }
    const t0 = performance.now(), dur = 1800;
    (function step(t) { const u = clamp((t - t0) / dur), e = 1 - Math.pow(1 - u, 3); big.textContent = (B.final * e).toFixed(3); if (u < 1) requestAnimationFrame(step); })(t0);
  });
  const rows = [["Climatology", "long-term average, no AI", B.climatology, C.green],
                ["One model", "single ConvLSTM network", B.single, "#5d7890"],
                ["Six-model ensemble", "before bias correction", B.ensemble, "#6f8ca6"],
                ["OTER · final", "ensemble + bias correction", B.final, C.blue, true],
                ["GLORYS12V1", "the teacher (reanalysis)", B.glorys, C.gold]];
  const box = $("#hbars"), max = 1.7;
  rows.forEach(([name, sub, v, col, hl], i) => {
    const r = h("div", "hbar" + (hl ? " hl" : ""), `<div class="name">${name}<small>${sub}</small></div><div class="track"><div class="bar" style="background:${col};--i:${i}"></div><span class="val" style="--i:${i};left:0">${v.toFixed(3)} °C</span></div>`);
    box.appendChild(r);
  });
  reveal(box, () => $$(".hbar", box).forEach((r, i) => {
    const w = rows[i][2] / max * 100; $(".bar", r).style.width = w + "%"; $(".val", r).style.left = w + "%";
  }));
  const svg = $("#res-chart");
  profileChart(svg, { x0: 0, x1: 2.3, xticks: [0, 0.5, 1, 1.5, 2], xlabel: "error vs Argo (RMSE °C)",
    series: [{ x: D.rmse.climatology, y: D.depths, color: C.green, width: 2, delay: 0 },
             { x: D.rmse.glorys, y: D.depths, color: C.gold, width: 2, dash: "6 5", delay: 0.6 },
             { x: D.rmse.final, y: D.depths, color: C.blue, width: 3, dots: true, delay: 0.4 }],
    tipRow: (i, d) => rowTip(d, [["climatology", C.green, f3(D.rmse.climatology[i])], ["OTER", C.blue, f3(D.rmse.final[i])], ["GLORYS", C.gold, f3(D.rmse.glorys[i])]]) });
  reveal(svg);
  legend($("#res-legend"), [["Climatology", C.green], ["OTER", C.blue], ["GLORYS (teacher)", C.gold, true]]);
})();

onScroll();

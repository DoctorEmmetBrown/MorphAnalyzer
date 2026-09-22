/* Petit moteur de graphiques SVG, sans dependance.
 *
 * Volontairement minimal : courbes et barres, une seule echelle d'ordonnees
 * (jamais deux axes y — c'est l'erreur de lecture la plus courante), une
 * legende des qu'il y a plus d'une serie, un survol qui donne les valeurs
 * exactes, et une grille en retrait. Les couleurs viennent des variables CSS
 * --s1..--s8, donc elles suivent le theme clair ou sombre. */
(function (global) {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs) => {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  };
  const css = (name, fallback) => {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  };
  const seriesColors = () =>
    ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"].map((n, i) =>
      css(n, ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4", "#008300", "#e34948"][i])
    );

  function niceTicks(lo, hi, count) {
    if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
    const raw = (hi - lo) / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
    return out;
  }

  const fmt = (v) => {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e9) return (v / 1e9).toFixed(a >= 1e10 ? 0 : 1) + " G";
    if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + " M";
    if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + " k";
    if (a < 1e-4) return v.toExponential(1);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 100) return v.toFixed(1);
    if (a >= 10) return v.toFixed(2);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(3);
  };

  function render(container, opts) {
    container.innerHTML = "";
    const data = opts.series.filter((s) => s.values.some((v) => v !== null && isFinite(v)));
    if (!data.length || !opts.x.length) {
      const p = document.createElement("p");
      p.className = "empty";
      p.textContent = opts.emptyText || "Rien à tracer.";
      container.appendChild(p);
      return;
    }

    const W = container.clientWidth || 340;
    const H = container.clientHeight || 220;
    const legendH = data.length > 1 ? 18 : 0;
    const M = { t: 8, r: 10, b: 30 + legendH, l: 46 };
    const iw = Math.max(10, W - M.l - M.r);
    const ih = Math.max(10, H - M.t - M.b);
    const ink = css("--text", "#111");
    const dim = css("--text-dim", "#555");
    const mute = css("--text-mute", "#888");
    const grid = css("--border", "#ddd");
    const surface = css("--surface-1", "#fff");
    const colors = seriesColors();

    const logX = !!opts.logX && opts.x.every((v) => v > 0);
    const tx = (v) => (logX ? Math.log10(v) : v);
    let xs = opts.x.map(tx);
    let xlo = Math.min(...xs), xhi = Math.max(...xs);
    if (xlo === xhi) { xlo -= 0.5; xhi += 0.5; }
    const rev = !!opts.reverseX;
    const X = (v) => {
      const t = (tx(v) - xlo) / (xhi - xlo);
      return M.l + (rev ? 1 - t : t) * iw;
    };

    let ylo = Infinity, yhi = -Infinity;
    for (const s of data) for (const v of s.values) {
      if (v === null || !isFinite(v)) continue;
      if (v < ylo) ylo = v;
      if (v > yhi) yhi = v;
    }
    if (!isFinite(ylo)) { ylo = 0; yhi = 1; }
    if (opts.kind === "bar") ylo = Math.min(0, ylo);
    if (ylo === yhi) { ylo -= 0.5; yhi += 0.5; }
    const pad = (yhi - ylo) * 0.06;
    ylo -= pad; yhi += pad;
    const Y = (v) => M.t + ih - ((v - ylo) / (yhi - ylo)) * ih;

    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img" });
    svg.style.overflow = "visible";

    /* grille et axes */
    for (const t of niceTicks(ylo, yhi, 4)) {
      const y = Y(t);
      svg.appendChild(el("line", { x1: M.l, x2: M.l + iw, y1: y, y2: y, stroke: grid, "stroke-width": 1 }));
      const lab = el("text", { x: M.l - 6, y: y + 3.5, "text-anchor": "end", fill: mute, "font-size": 10 });
      lab.textContent = fmt(t);
      svg.appendChild(lab);
    }
    for (const t of niceTicks(xlo, xhi, 4)) {
      const v = logX ? Math.pow(10, t) : t;
      const x = X(v);
      if (x < M.l - 1 || x > M.l + iw + 1) continue;
      svg.appendChild(el("line", { x1: x, x2: x, y1: M.t + ih, y2: M.t + ih + 4, stroke: grid }));
      const lab = el("text", { x, y: M.t + ih + 15, "text-anchor": "middle", fill: mute, "font-size": 10 });
      lab.textContent = fmt(v);
      svg.appendChild(lab);
    }
    svg.appendChild(el("line", { x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih, stroke: grid }));

    if (opts.xLabel) {
      const t = el("text", { x: M.l + iw, y: M.t + ih + 26, "text-anchor": "end", fill: dim, "font-size": 10.5 });
      t.textContent = opts.xLabel;
      svg.appendChild(t);
    }

    /* marques */
    if (opts.kind === "bar") {
      const n = opts.x.length;
      const slot = iw / Math.max(n, 1);
      const bw = Math.max(1, (slot * 0.78) / data.length - 1);
      data.forEach((s, si) => {
        s.values.forEach((v, i) => {
          if (v === null || !isFinite(v)) return;
          const cx = X(opts.x[i]) - (data.length * (bw + 1)) / 2 + si * (bw + 1);
          const y = Y(Math.max(v, 0)), y0 = Y(0);
          svg.appendChild(el("rect", {
            x: cx, y: Math.min(y, y0), width: bw, height: Math.max(1, Math.abs(y0 - y)),
            fill: s.color || colors[si % 8], rx: Math.min(3, bw / 2),
          }));
        });
      });
    } else {
      data.forEach((s, si) => {
        const color = s.color || colors[si % 8];
        let d = "", pen = false;
        s.values.forEach((v, i) => {
          if (v === null || !isFinite(v)) { pen = false; return; }
          const x = X(opts.x[i]), y = Y(v);
          d += (pen ? " L" : " M") + x.toFixed(2) + " " + y.toFixed(2);
          pen = true;
        });
        svg.appendChild(el("path", {
          d, fill: "none", stroke: color, "stroke-width": 2,
          "stroke-linejoin": "round", "stroke-linecap": "round",
        }));
        if (opts.x.length <= 40) {
          s.values.forEach((v, i) => {
            if (v === null || !isFinite(v)) return;
            svg.appendChild(el("circle", {
              cx: X(opts.x[i]), cy: Y(v), r: 2.8, fill: color, stroke: surface, "stroke-width": 1.5,
            }));
          });
        }
      });
    }

    /* legende */
    if (data.length > 1) {
      let lx = M.l;
      const ly = H - 4;
      data.forEach((s, si) => {
        const color = s.color || colors[si % 8];
        svg.appendChild(el("rect", { x: lx, y: ly - 8, width: 9, height: 9, rx: 2, fill: color }));
        const t = el("text", { x: lx + 13, y: ly, fill: dim, "font-size": 10.5 });
        t.textContent = s.name;
        svg.appendChild(t);
        lx += 13 + s.name.length * 5.8 + 14;
      });
    }

    /* survol */
    const hairX = el("line", { y1: M.t, y2: M.t + ih, stroke: mute, "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0 });
    svg.appendChild(hairX);
    const dots = data.map((s, si) =>
      svg.appendChild(el("circle", { r: 4, fill: s.color || colors[si % 8], stroke: surface, "stroke-width": 2, opacity: 0 }))
    );
    const tip = document.createElement("div");
    tip.className = "readout";
    tip.hidden = true;
    container.style.position = "relative";
    container.appendChild(tip);

    const hit = el("rect", { x: M.l, y: M.t, width: iw, height: ih, fill: "transparent" });
    svg.appendChild(hit);
    hit.addEventListener("mousemove", (ev) => {
      const box = svg.getBoundingClientRect();
      const px = ev.clientX - box.left;
      let best = 0, bd = Infinity;
      opts.x.forEach((v, i) => { const d = Math.abs(X(v) - px); if (d < bd) { bd = d; best = i; } });
      const gx = X(opts.x[best]);
      hairX.setAttribute("x1", gx); hairX.setAttribute("x2", gx); hairX.setAttribute("opacity", 1);
      let html = `<b>${opts.xLabel || "x"} = ${fmt(opts.x[best])}</b>`;
      data.forEach((s, si) => {
        const v = s.values[best];
        dots[si].setAttribute("cx", gx);
        if (v === null || !isFinite(v)) { dots[si].setAttribute("opacity", 0); return; }
        dots[si].setAttribute("cy", Y(v)); dots[si].setAttribute("opacity", 1);
        html += `<br><span class="k">${s.name}</span> ${fmt(v)}`;
      });
      tip.innerHTML = html;
      tip.hidden = false;
      const tw = tip.offsetWidth;
      tip.style.left = Math.min(Math.max(gx - tw / 2, 2), W - tw - 2) + "px";
      tip.style.top = "2px";
    });
    hit.addEventListener("mouseleave", () => {
      tip.hidden = true;
      hairX.setAttribute("opacity", 0);
      dots.forEach((d) => d.setAttribute("opacity", 0));
    });

    container.appendChild(svg);
  }

  global.Chart = { render, fmt, seriesColors };
})(window);

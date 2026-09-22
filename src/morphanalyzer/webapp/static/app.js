/* morphanalyzer — interface web.
 *
 * Le navigateur ne voit jamais un volume : il demande une coupe composee au
 * serveur, qui la rend en PNG la ou les donnees sont deja memmappees. Tout le
 * reste est de l'API JSON. */
"use strict";

const S = {
  project: null,
  layers: [],          // ordre de dessin : le premier est au fond
  views: {},           // nom -> reglages d'affichage
  axis: 0,
  index: null,
  zoom: 1,
  pan: { x: 0, y: 0 },
  steps: [],
  queue: [],
  job: null,
  poll: null,
  table: null,
  chartKind: "line",
  logX: false,
  reverseX: false,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const AXES = ["Z", "Y", "X"];
const round4 = (v) => (v === null || v === undefined ? "" : +(+v).toPrecision(4));

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const d = (await res.json()).detail;
      if (typeof d === "string") msg = d;
      else if (Array.isArray(d)) msg = d.map((x) => `${(x.loc || []).join(".")} : ${x.msg}`).join(" ; ");
      else if (d) msg = JSON.stringify(d);
    } catch (e) { /* corps non JSON */ }
    throw new Error(msg);
  }
  return res.json();
}

function toast(message, isError) {
  const t = $("#toast");
  t.textContent = message;
  t.className = "toast" + (isError ? " err" : "");
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), isError ? 6000 : 2600);
}

const debounce = (fn, ms) => {
  let id;
  return (...a) => { clearTimeout(id); id = setTimeout(() => fn(...a), ms); };
};

/* ───────────────────────────── projet ─────────────────────────────── */

async function loadProject() {
  const p = await api("/api/project");
  S.project = p;
  $("#project-name").textContent = p.name;
  const vs = p.voxel_size.map((v) => (+v).toPrecision(3)).join(" × ");
  $("#project-meta").textContent = p.shape
    ? `${p.shape.join(" × ")} voxels — ${vs} ${p.unit}`
    : "projet vide";

  const known = new Set(p.layers.map((l) => l.name));
  S.layers = [...S.layers.filter((n) => known.has(n)),
              ...p.layers.map((l) => l.name).filter((n) => !S.layers.includes(n))];
  for (const layer of p.layers) {
    if (!S.views[layer.name]) {
      const first = S.layers.indexOf(layer.name) === 0;
      S.views[layer.name] = {
        visible: first,                       // un seul calque au depart : on empile a la demande
        ramp: layer.kind === "grey" ? "grey" : "blue",
        color: "#2a78d6",
        alpha: first ? 1 : 0.7,
        vmin: layer.vmin, vmax: layer.vmax,
        kind: layer.kind,
      };
    }
  }
  renderLayers();
  fillSelect($("#input-layer"), S.layers, "volume");
  const continuous = p.layers.filter((l) => l.kind === "scalar" || l.kind === "grey").map((l) => l.name);
  fillSelect($("#hist-layer"), S.layers, continuous[0]);
  renderHistory(p.history);
  renderInfo(p);
  renderTables(p.tables);

  const n = p.shape ? p.shape[S.axis] : 0;
  $("#slice").max = Math.max(0, n - 1);
  if (S.index === null || S.index >= n) S.index = Math.floor(n / 2);
  $("#slice").value = S.index;
  $("#viewport-empty").hidden = !!p.layers.length;
  updateSliceLabel();
  refreshSlice();
}

function fillSelect(sel, names, preferred) {
  const before = sel.value;
  sel.innerHTML = "";
  for (const n of names) {
    const o = document.createElement("option");
    o.value = o.textContent = n;
    sel.appendChild(o);
  }
  if (names.includes(before)) sel.value = before;
  else if (preferred && names.includes(preferred)) sel.value = preferred;
}

/* ───────────────────────────── calques ────────────────────────────── */

const RAMP_PREVIEW = {
  blue: "linear-gradient(90deg,#f2f7fe,#6da7ec,#0d366b)",
  orange: "linear-gradient(90deg,#fdf4ef,#f29566,#6b280f)",
  teal: "linear-gradient(90deg,#eefaf5,#5ecda3,#08462f)",
  violet: "linear-gradient(90deg,#f3f2fb,#948ad6,#241c54)",
  grey: "linear-gradient(90deg,#ffffff,#83837e,#111111)",
};
const BINARY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"];

function renderLayers() {
  const ul = $("#layers");
  ul.innerHTML = "";
  const byName = Object.fromEntries((S.project?.layers || []).map((l) => [l.name, l]));
  for (const name of S.layers) {
    const info = byName[name];
    if (!info) continue;
    const view = S.views[name];
    const li = document.createElement("li");
    li.className = "layer" + (view.visible ? " active" : "");
    li.dataset.name = name;

    const head = document.createElement("div");
    head.className = "layer-head";
    head.innerHTML =
      `<button class="eye ${view.visible ? "on" : ""}" title="Afficher / masquer">${view.visible ? "●" : "○"}</button>` +
      `<span class="name" title="${info.description || name}">${name}</span>` +
      `<span class="badge">${info.kind}</span>`;
    head.querySelector(".eye").addEventListener("click", (e) => {
      e.stopPropagation();
      view.visible = !view.visible;
      renderLayers();
      refreshSlice();
    });
    head.addEventListener("click", () => {
      li.classList.toggle("open");
    });
    li.appendChild(head);

    const body = document.createElement("div");
    body.className = "layer-body";

    if (info.kind === "binary") {
      const sw = document.createElement("div");
      sw.className = "swatches";
      for (const c of BINARY_COLORS) {
        const b = document.createElement("button");
        b.className = "sw" + (view.color === c ? " on" : "");
        b.style.background = c;
        b.title = c;
        b.addEventListener("click", () => { view.color = c; renderLayers(); refreshSlice(); });
        sw.appendChild(b);
      }
      body.appendChild(sw);
    } else if (info.kind !== "labels") {
      const sw = document.createElement("div");
      sw.className = "swatches";
      for (const r of (S.project.ramps || Object.keys(RAMP_PREVIEW))) {
        const b = document.createElement("button");
        b.className = "sw" + (view.ramp === r ? " on" : "");
        b.style.background = RAMP_PREVIEW[r] || "#888";
        b.title = r;
        b.addEventListener("click", () => { view.ramp = r; renderLayers(); refreshSlice(); });
        sw.appendChild(b);
      }
      body.appendChild(sw);

      const grid = document.createElement("div");
      grid.className = "grid2";
      grid.innerHTML =
        `<div><label>min</label><input type="number" step="any" value="${round4(view.vmin ?? info.vmin)}" data-k="vmin"></div>` +
        `<div><label>max</label><input type="number" step="any" value="${round4(view.vmax ?? info.vmax)}" data-k="vmax"></div>`;
      grid.querySelectorAll("input").forEach((inp) =>
        inp.addEventListener("change", () => {
          view[inp.dataset.k] = inp.value === "" ? null : +inp.value;
          refreshSlice();
        })
      );
      body.appendChild(grid);
    }

    const alpha = document.createElement("div");
    alpha.innerHTML = `<label>opacité <span class="mono">${view.alpha.toFixed(2)}</span></label>`;
    const range = document.createElement("input");
    range.type = "range"; range.min = 0; range.max = 1; range.step = 0.05; range.value = view.alpha;
    range.addEventListener("input", () => {
      view.alpha = +range.value;
      alpha.querySelector("span").textContent = view.alpha.toFixed(2);
      refreshSlice();
    });
    alpha.appendChild(range);
    body.appendChild(alpha);

    const meta = document.createElement("div");
    meta.className = "layer-meta";
    meta.textContent = `${info.dtype} · ${info.shape.join("×")}` +
      (info.n_labels ? ` · ${info.n_labels} étiquettes` : "") +
      (info.step ? ` · ${info.step}` : "");
    body.appendChild(meta);

    if (name !== "volume" && !S.project.read_only) {
      const del = document.createElement("button");
      del.className = "layer-del";
      del.textContent = "supprimer ce calque";
      del.addEventListener("click", async () => {
        if (!confirm(`Supprimer définitivement le calque « ${name} » ?`)) return;
        try {
          await fetch(`/api/layer/${encodeURIComponent(name)}`, { method: "DELETE" });
          S.layers = S.layers.filter((n) => n !== name);
          delete S.views[name];
          await loadProject();
          toast(`calque « ${name} » supprimé`);
        } catch (e) { toast(e.message, true); }
      });
      body.appendChild(del);
    }

    li.appendChild(body);
    ul.appendChild(li);
  }
  $("#layers-empty").hidden = S.layers.length > 0;
}

/* ───────────────────────────── slicer ─────────────────────────────── */

function buildSpec() {
  return {
    axis: S.axis,
    index: S.index,
    layers: S.layers
      .filter((n) => S.views[n]?.visible)
      .map((n) => ({ name: n, ...S.views[n] })),
  };
}

const refreshSlice = debounce(() => {
  const spec = buildSpec();
  if (!spec.layers.length) {
    $("#slice-img").removeAttribute("src");
    return;
  }
  const url = "/api/slice.png?spec=" + encodeURIComponent(JSON.stringify(spec));
  const img = $("#slice-img");
  img.src = url;
  $("#snapshot").href = url;
  $("#snapshot").download = `${S.project.name}_${AXES[S.axis]}${S.index}.png`;
}, 40);

function updateSliceLabel() {
  const n = S.project?.shape ? S.project.shape[S.axis] : 0;
  $("#slice-label").textContent = `${S.index} / ${Math.max(0, n - 1)}`;
}

function applyTransform() {
  const img = $("#slice-img");
  const w = img.naturalWidth || 1, h = img.naturalHeight || 1;
  $("#stage").style.transform =
    `translate(${S.pan.x}px, ${S.pan.y}px) scale(${S.zoom}) translate(${-w / 2}px, ${-h / 2}px)`;
  $("#scale-label").textContent = `${Math.round(S.zoom * 100)} %`;
}

function fitView() {
  const img = $("#slice-img");
  const vp = $("#viewport");
  if (!img.naturalWidth) return;
  const k = Math.min(vp.clientWidth / img.naturalWidth, vp.clientHeight / img.naturalHeight) * 0.92;
  S.zoom = Math.max(0.05, k);
  S.pan = { x: 0, y: 0 };
  applyTransform();
}

function setAxis(axis) {
  S.axis = axis;
  $$(".segmented button").forEach((b) => b.classList.toggle("on", +b.dataset.axis === axis));
  const n = S.project?.shape ? S.project.shape[axis] : 0;
  $("#slice").max = Math.max(0, n - 1);
  S.index = Math.floor(n / 2);
  $("#slice").value = S.index;
  updateSliceLabel();
  refreshSlice();
  setTimeout(fitView, 120);
}

/* coordonnees voxel sous le curseur */
function voxelAt(ev) {
  const img = $("#slice-img");
  if (!img.naturalWidth || !S.project?.shape) return null;
  const r = img.getBoundingClientRect();
  const u = (ev.clientX - r.left) / r.width;
  const v = (ev.clientY - r.top) / r.height;
  if (u < 0 || u > 1 || v < 0 || v > 1) return null;
  const shape = S.project.shape;
  const other = [0, 1, 2].filter((a) => a !== S.axis);
  const c = [0, 0, 0];
  c[S.axis] = S.index;
  c[other[0]] = Math.min(shape[other[0]] - 1, Math.floor(v * shape[other[0]]));
  c[other[1]] = Math.min(shape[other[1]] - 1, Math.floor(u * shape[other[1]]));
  return c;
}

const probe = debounce(async (c, ev) => {
  try {
    const r = await api(`/api/value?z=${c[0]}&y=${c[1]}&x=${c[2]}`);
    const rows = Object.entries(r.values)
      .map(([k, v]) => `<span class="k">${k}</span> ${v.value === null ? "—" : v.value}`)
      .join("<br>");
    const box = $("#readout");
    box.innerHTML = `<b>z ${c[0]} · y ${c[1]} · x ${c[2]}</b><br>${rows}`;
    box.hidden = false;
    const vp = $("#viewport").getBoundingClientRect();
    const bw = box.offsetWidth, bh = box.offsetHeight;
    let left = ev.clientX - vp.left + 14, top = ev.clientY - vp.top + 14;
    if (left + bw > vp.width - 4) left = ev.clientX - vp.left - bw - 14;
    if (top + bh > vp.height - 4) top = ev.clientY - vp.top - bh - 14;
    box.style.left = left + "px";
    box.style.top = top + "px";
  } catch (e) { /* le projet a pu changer */ }
}, 60);

/* ───────────────────────────── pipeline ───────────────────────────── */

async function loadSteps() {
  S.steps = await api("/api/steps");
  renderStepList("");
}

const MODULE_ORDER = ["filters", "metrics", "distance", "granulometry", "segmentation",
                      "skeleton", "shape", "tortuosity", "network", "cortical", "mesh"];

function moduleOf(st) {
  const head = st.module.split(".")[0];
  return MODULE_ORDER.includes(head) ? head : st.module.split(".")[0];
}

function renderStepList(filter) {
  const sel = $("#step-picker");
  const f = filter.trim().toLowerCase();
  sel.innerHTML = "";
  const groups = new Map();
  for (const st of S.steps) {
    if (f && !st.name.includes(f) && !st.summary.toLowerCase().includes(f) && !st.module.includes(f)) continue;
    const g = moduleOf(st);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(st);
  }
  const rank = (m) => (MODULE_ORDER.indexOf(m) < 0 ? 99 : MODULE_ORDER.indexOf(m));
  const ordered = [...groups.keys()].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  for (const g of ordered) {
    const grp = document.createElement("optgroup");
    grp.label = g;
    for (const st of groups.get(g)) {
      const o = document.createElement("option");
      o.value = st.name;
      o.textContent = st.name;
      o.title = st.summary;
      grp.appendChild(o);
    }
    sel.appendChild(grp);
  }
  if (sel.options.length) {
    sel.selectedIndex = 0;
    renderParams(sel.value);
  } else {
    $("#step-summary").textContent = "";
    $("#step-params").innerHTML = "";
  }
}

function renderParams(name) {
  const st = S.steps.find((s) => s.name === name);
  const box = $("#step-params");
  box.innerHTML = "";
  if (!st) return;
  $("#step-summary").textContent = st.summary;

  const outRow = document.createElement("div");
  outRow.className = "param";
  outRow.innerHTML = `<label title="nom du calque ou de la table produite">out</label>` +
                     `<input type="text" data-p="out" placeholder="${name}">`;
  box.appendChild(outRow);

  for (const p of st.parameters) {
    const row = document.createElement("div");
    row.className = "param";
    const hint = (p.annotation || "") + (p.required ? " (requis)" : "");
    row.innerHTML = `<label title="${hint}">${p.name}</label>`;
    const wrap = document.createElement("div");
    wrap.className = "with-layer";
    const inp = document.createElement("input");
    inp.type = "text";
    inp.dataset.p = p.name;
    inp.placeholder = p.default === null || p.default === undefined ? (p.required ? "requis" : "défaut") : String(p.default);
    wrap.appendChild(inp);
    const pick = document.createElement("select");
    pick.title = "insérer un calque du projet";
    pick.innerHTML = `<option value="">@</option>` + S.layers.map((n) => `<option value="@${n}">${n}</option>`).join("");
    pick.addEventListener("change", () => { if (pick.value) { inp.value = pick.value; pick.value = ""; } });
    wrap.appendChild(pick);
    row.appendChild(wrap);
    box.appendChild(row);
  }
}

function collectParams() {
  const out = {};
  for (const inp of $$("#step-params input")) {
    const v = inp.value.trim();
    if (!v) continue;
    out[inp.dataset.p] = parseValue(v);
  }
  return out;
}

function parseValue(v) {
  if (v.startsWith("@")) return v;
  if (v === "true") return true;
  if (v === "false") return false;
  if (v === "null" || v === "none") return null;
  try { return JSON.parse(v); } catch (e) { return v; }
}

function renderQueue() {
  const ol = $("#queue");
  ol.innerHTML = "";
  S.queue.forEach((entry, i) => {
    const li = document.createElement("li");
    const params = Object.entries(entry.params)
      .filter(([k]) => k !== "out")
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ");
    li.innerHTML = `<span class="qname">${entry.step}</span>` +
                   `<span class="qparams">${params || ""}</span>` +
                   `<span class="badge">${entry.params.out || entry.step}</span>`;
    const del = document.createElement("button");
    del.textContent = "✕";
    del.title = "retirer";
    del.addEventListener("click", () => { S.queue.splice(i, 1); renderQueue(); });
    li.appendChild(del);
    ol.appendChild(li);
  });
  $("#run").disabled = S.queue.length === 0 || (S.job && S.job.status === "running");
}

async function runQueue() {
  const body = { steps: S.queue, input_layer: $("#input-layer").value };
  try {
    S.job = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("#job").hidden = false;
    $("#run").disabled = true;
    pollJob();
  } catch (e) { toast(e.message, true); }
}

function pollJob() {
  clearInterval(S.poll);
  S.poll = setInterval(async () => {
    try {
      const j = await api(`/api/jobs/${S.job.id}`);
      S.job = j;
      renderJob(j);
      if (j.status === "done" || j.status === "error") {
        clearInterval(S.poll);
        $("#run").disabled = S.queue.length === 0;
        await loadProject();
        if (j.status === "done") {
          toast(`terminé en ${j.elapsed.toFixed(1)} s`);
          S.queue = [];
          renderQueue();
        } else {
          toast(j.error, true);
        }
      }
    } catch (e) { clearInterval(S.poll); }
  }, 600);
}

function renderJob(j) {
  const frac = j.n_steps ? Math.min(1, (j.current + (j.status === "running" ? 0 : 1)) / j.n_steps) : 0;
  $("#job-bar").style.width = (frac * 100).toFixed(0) + "%";
  $("#job-status").textContent =
    j.status === "running" ? `étape ${Math.min(j.current + 1, j.n_steps)} / ${j.n_steps} — ${j.steps[j.current] || ""}`
    : j.status === "done" ? `terminé en ${j.elapsed.toFixed(1)} s`
    : j.status === "error" ? "erreur" : "en attente";
  const ol = $("#job-log");
  ol.innerHTML = "";
  for (const entry of j.log) {
    const li = document.createElement("li");
    if (entry.traceback) {
      li.className = "err";
      li.textContent = entry.traceback.trim().split("\n").slice(-3).join("\n");
    } else {
      li.textContent = `${entry.step} · ${entry.seconds}s → ${entry.produced}`;
    }
    ol.appendChild(li);
  }
  ol.scrollTop = ol.scrollHeight;
}

/* Chaines types : ce que l'on enchaine presque toujours. Elles partent du
 * calque d'entree choisi, qui doit etre le masque de la phase fluide. */
const PRESETS = {
  granulo: [
    { step: "distance_transform", params: { out: "distance" } },
    { step: "aperture_map", params: { out: "ouverture", n_radii: 24 } },
    { step: "pore_size_distribution", params: { out: "granulometrie", bins: 30 } },
  ],
  cellules: [
    { step: "distance_transform", params: { out: "distance" } },
    { step: "cell_markers", params: { out: "marqueurs", distance: "@distance", fill_ratio: 0.55 } },
    { step: "watershed_cells", params: { out: "cellules", markers: "@marqueurs", mask: "@volume" } },
    { step: "cell_morphometry", params: { out: "morphometrie" } },
    { step: "throats", params: { out: "cols" } },
  ],
  drainage: [
    { step: "drainage", params: { out: "drainage", method: "hilpert", step: 0.5, surface_tension: 0.0728 } },
  ],
};

/* ───────────────────────────── tables ─────────────────────────────── */

function renderTables(tables) {
  const names = tables.map((t) => t.name);
  fillSelect($("#table-picker"), names);
  $("#curves-empty").hidden = names.length > 0;
  $("#chart").hidden = names.length === 0;
  if (names.length && !names.includes(S.table)) {
    S.table = names[0];
    $("#table-picker").value = S.table;
  }
  if (S.table) loadTable(S.table);
}

async function loadTable(name) {
  try {
    const t = await api(`/api/table/${encodeURIComponent(name)}`);
    S.tableData = t;
    const numeric = t.columns.filter((c) => t.rows.some((r) => typeof r[c] === "number"));
    fillSelect($("#col-x"), numeric);
    const ysel = $("#col-y");
    ysel.innerHTML = "";
    for (const c of numeric) {
      const o = document.createElement("option");
      o.value = o.textContent = c;
      ysel.appendChild(o);
    }
    if (numeric.length) {
      $("#col-x").value = numeric[0];
      const guess = numeric.find((c) => /satur|poros|fraction|count|cumul/.test(c)) || numeric[1] || numeric[0];
      Array.from(ysel.options).forEach((o) => (o.selected = o.value === guess));
    }
    $("#chart-csv").href = `/api/table/${encodeURIComponent(name)}.csv`;
    $("#chart-csv").download = `${name}.csv`;
    drawChart();
  } catch (e) { toast(e.message, true); }
}

function drawChart() {
  const t = S.tableData;
  if (!t) return;
  const xcol = $("#col-x").value;
  const ycols = Array.from($("#col-y").selectedOptions).map((o) => o.value);
  if (!xcol || !ycols.length) return;
  const x = t.rows.map((r) => r[xcol]);
  const series = ycols.map((c) => ({ name: c, values: t.rows.map((r) => r[c]) }));
  Chart.render($("#chart"), {
    x, series, kind: S.chartKind, xLabel: xcol,
    logX: S.logX, reverseX: S.reverseX,
  });
  const box = $("#chart-values");
  if (!box.hidden) {
    const cols = [xcol, ...ycols];
    box.innerHTML =
      "<table><thead><tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>" +
      t.rows.map((r) => "<tr>" + cols.map((c) => `<td>${Chart.fmt(r[c])}</td>`).join("") + "</tr>").join("") +
      "</tbody></table>";
  }
}

/* ───────────────────────── historique et info ─────────────────────── */

function renderHistory(history) {
  const ol = $("#history");
  ol.innerHTML = "";
  for (const h of [...history].reverse()) {
    const li = document.createElement("li");
    const params = Object.entries(h.params || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ");
    li.innerHTML = `<div class="h-step">${h.step}</div>` +
      `<div class="h-meta">${h.at} · ${h.duration ? h.duration.toFixed(2) + " s" : ""}` +
      `${h.outputs?.length ? " → " + h.outputs.join(", ") : ""}${h.note ? " · " + h.note : ""}</div>` +
      (params ? `<div class="h-meta">${params}</div>` : "");
    ol.appendChild(li);
  }
  $("#history-empty").hidden = history.length > 0;
}

function renderInfo(p) {
  const dl = $("#volume-info");
  const phys = p.shape ? p.shape.map((n, i) => (n * p.voxel_size[i]).toPrecision(4)).join(" × ") : "—";
  dl.innerHTML =
    `<dt>projet</dt><dd>${p.name}</dd>` +
    `<dt>dossier</dt><dd>${p.path}</dd>` +
    `<dt>dimensions</dt><dd>${p.shape ? p.shape.join(" × ") : "—"}</dd>` +
    `<dt>voxel</dt><dd>${p.voxel_size.map((v) => (+v).toPrecision(3)).join(" × ")} ${p.unit}</dd>` +
    `<dt>taille physique</dt><dd>${phys} ${p.unit}</dd>` +
    `<dt>calques</dt><dd>${p.layers.length}</dd>` +
    `<dt>tables</dt><dd>${p.tables.length}</dd>`;
  if ($("#hist-layer").value) drawHistogram($("#hist-layer").value);
}

async function drawHistogram(layer) {
  try {
    const h = await api(`/api/histogram?layer=${encodeURIComponent(layer)}&bins=48`);
    const centres = h.edges.slice(0, -1).map((e, i) => (e + h.edges[i + 1]) / 2);
    Chart.render($("#histogram"), {
      x: centres,
      series: [{ name: layer, values: h.counts }],
      kind: "bar", xLabel: "valeur",
    });
  } catch (e) { /* calque supprime */ }
}

/* ───────────────────────────── evenements ─────────────────────────── */

function wire() {
  /* theme */
  const stored = localStorage.getItem("ma-theme");
  if (stored) document.documentElement.dataset.theme = stored;
  $("#theme-toggle").addEventListener("click", () => {
    const now = document.documentElement.dataset.theme;
    const next = now === "dark" ? "light" : now === "light" ? "auto" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("ma-theme", next);
    if (S.tableData) drawChart();
    if ($("#hist-layer").value) drawHistogram($("#hist-layer").value);
  });

  /* axe et coupe */
  $$(".segmented button").forEach((b) =>
    b.addEventListener("click", () => setAxis(+b.dataset.axis))
  );
  $("#slice").addEventListener("input", (e) => {
    S.index = +e.target.value;
    updateSliceLabel();
    refreshSlice();
  });
  $("#zoom-fit").addEventListener("click", fitView);

  /* zoom, deplacement, sonde */
  const vp = $("#viewport");
  vp.addEventListener("wheel", (e) => {
    e.preventDefault();
    const k = Math.exp(-e.deltaY * 0.0015);
    const r = vp.getBoundingClientRect();
    const mx = e.clientX - r.left - r.width / 2 - S.pan.x;
    const my = e.clientY - r.top - r.height / 2 - S.pan.y;
    S.pan.x -= mx * (k - 1);
    S.pan.y -= my * (k - 1);
    S.zoom = Math.min(40, Math.max(0.05, S.zoom * k));
    applyTransform();
  }, { passive: false });

  let drag = null;
  vp.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    drag = { x: e.clientX - S.pan.x, y: e.clientY - S.pan.y };
    vp.style.cursor = "grabbing";
  });
  window.addEventListener("mouseup", () => { drag = null; vp.style.cursor = ""; });
  vp.addEventListener("mousemove", (e) => {
    if (drag) {
      S.pan.x = e.clientX - drag.x;
      S.pan.y = e.clientY - drag.y;
      applyTransform();
      return;
    }
    const c = voxelAt(e);
    const hair = $("#crosshair");
    if (!c) { $("#readout").hidden = true; hair.hidden = true; return; }
    const r = vp.getBoundingClientRect();
    hair.hidden = false;
    hair.children[0].style.top = (e.clientY - r.top) + "px";
    hair.children[1].style.left = (e.clientX - r.left) + "px";
    $("#cursor").textContent = `z ${c[0]} · y ${c[1]} · x ${c[2]}`;
    probe(c, e);
  });
  vp.addEventListener("mouseleave", () => {
    $("#readout").hidden = true;
    $("#crosshair").hidden = true;
    $("#cursor").textContent = "—";
  });
  vp.addEventListener("dblclick", fitView);
  vp.addEventListener("keydown", (e) => {
    const n = S.project?.shape ? S.project.shape[S.axis] : 1;
    if (e.key === "ArrowUp" || e.key === "ArrowRight") S.index = Math.min(n - 1, S.index + 1);
    else if (e.key === "ArrowDown" || e.key === "ArrowLeft") S.index = Math.max(0, S.index - 1);
    else if (e.key === "1") return setAxis(0);
    else if (e.key === "2") return setAxis(1);
    else if (e.key === "3") return setAxis(2);
    else return;
    e.preventDefault();
    $("#slice").value = S.index;
    updateSliceLabel();
    refreshSlice();
  });
  $("#slice-img").addEventListener("load", () => {
    if (!$("#slice-img").dataset.fitted) {
      $("#slice-img").dataset.fitted = "1";
      fitView();
    } else applyTransform();
  });

  /* pipeline */
  $("#step-search").addEventListener("input", (e) => renderStepList(e.target.value));
  $("#step-picker").addEventListener("change", (e) => renderParams(e.target.value));
  $("#step-add").addEventListener("click", () => {
    const name = $("#step-picker").value;
    if (!name) return;
    S.queue.push({ step: name, params: collectParams() });
    renderQueue();
  });
  $("#queue-clear").addEventListener("click", () => { S.queue = []; renderQueue(); });
  $$("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => {
      S.queue = PRESETS[b.dataset.preset].map((s) => ({ step: s.step, params: { ...s.params } }));
      renderQueue();
      toast(`${S.queue.length} étapes ajoutées à la file`);
    })
  );
  $("#run").addEventListener("click", runQueue);
  $("#pipeline-replay").addEventListener("click", () => window.open("/api/pipeline.yaml", "_blank"));

  /* onglets */
  $$(".tabs button").forEach((b) =>
    b.addEventListener("click", () => {
      $$(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
      for (const pane of ["curves", "history", "info"]) {
        $("#tab-" + pane).hidden = pane !== b.dataset.tab;
      }
      if (b.dataset.tab === "curves" && S.tableData) drawChart();
      if (b.dataset.tab === "info" && $("#hist-layer").value) drawHistogram($("#hist-layer").value);
    })
  );

  /* courbes */
  $("#table-picker").addEventListener("change", (e) => { S.table = e.target.value; loadTable(S.table); });
  $("#col-x").addEventListener("change", drawChart);
  $("#col-y").addEventListener("change", drawChart);
  $("#chart-kind-line").addEventListener("click", () => {
    S.chartKind = "line";
    $("#chart-kind-line").classList.add("on");
    $("#chart-kind-bar").classList.remove("on");
    drawChart();
  });
  $("#chart-kind-bar").addEventListener("click", () => {
    S.chartKind = "bar";
    $("#chart-kind-bar").classList.add("on");
    $("#chart-kind-line").classList.remove("on");
    drawChart();
  });
  $("#chart-logx").addEventListener("click", (e) => {
    S.logX = !S.logX;
    e.target.classList.toggle("on", S.logX);
    drawChart();
  });
  $("#chart-reverse").addEventListener("click", (e) => {
    S.reverseX = !S.reverseX;
    e.target.classList.toggle("on", S.reverseX);
    drawChart();
  });
  $("#chart-table").addEventListener("click", (e) => {
    const box = $("#chart-values");
    box.hidden = !box.hidden;
    e.target.classList.toggle("on", !box.hidden);
    drawChart();
  });
  $("#hist-layer").addEventListener("change", (e) => drawHistogram(e.target.value));

  window.addEventListener("resize", debounce(() => {
    if (S.tableData && !$("#tab-curves").hidden) drawChart();
    if (!$("#tab-info").hidden && $("#hist-layer").value) drawHistogram($("#hist-layer").value);
  }, 150));
}

/* ───────────────────────────── demarrage ──────────────────────────── */

(async function start() {
  wire();
  try {
    await loadProject();
    await loadSteps();
    renderQueue();
    $("#viewport").focus();
  } catch (e) {
    toast("impossible de charger le projet : " + e.message, true);
  }
})();

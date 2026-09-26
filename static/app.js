const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const papers = new Map();   // key -> paper, for every row on screen
let libEntries = [];
let libFilter = "saved";
let libFolder = null;       // library filtered to one folder, or null for any
let allFolders = [];        // [{name, count}], for the folder picker and library filters
let modelName = "the model";

const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

// ---------- your library: on the server locally, in this browser on the public demo ----------
// Both versions answer the same calls. Writes return the paper's {saved, folders} and every
// folder as {all: [{name, count}]}, like the server's /api/save and /api/folder.

const serverLib = {
  where: "server",
  entries: async () => (await api("/api/library")).entries,
  folders: async () => (await api("/api/folders")).folders,
  save: (p, saved) => api("/api/save", { paper: p, saved }),
  folder: (p, name, add) => api("/api/folder", { paper: p, folder: name, add }),
  rate: (p, rating) => api("/api/rate", { paper: p, rating }),
  remove: key => api("/api/library/remove", { key }),
  noteSummary: () => {},        // the server keeps summaries itself
  overlay: p => p,              // server rows already carry your rating, saved, folders
  searchContext: () => ({}),    // the server reads your ratings itself
  mapContext: () => ({}),
};

// A visitor's library: everything they save, rate, file or summarize, kept in localStorage.
// No account and nothing on the server; clearing the site's data erases it.
const browserLib = (() => {
  const KEY = "research-copilot:library:v1";
  let data = {};
  let persisted = true;         // false when the browser won't store (private window, blocked site data)
  const read = () => {
    try { data = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch { persisted = false; }
  };
  const write = () => {
    try { localStorage.setItem(KEY, JSON.stringify(data)); persisted = true; } catch { persisted = false; }
  };
  read();
  addEventListener("storage", e => { if (e.key === KEY) read(); });  // another tab changed it

  const PAPER_ONLY = ["rating", "saved", "folders", "summary", "full_text"];
  function upsert(p, changes) {
    const now = Date.now() / 1000;
    const en = data[p.key] || { key: p.key, rating: 0, saved: false, folders: [], summary: "", full_text: false, added: now };
    const paper = Object.fromEntries(Object.entries(p).filter(([k]) => !PAPER_ONLY.includes(k)));
    Object.assign(en, { paper, bibtex: p.bibtex || en.bibtex || "" }, changes, { updated: now });
    data[p.key] = en;
    write();
    return en;
  }
  const folders = () => {
    const counts = {};
    Object.values(data).forEach(en => (en.folders || []).forEach(f => { counts[f] = (counts[f] || 0) + 1; }));
    return Object.entries(counts).map(([name, count]) => ({ name, count }))
      .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
  };
  const state = en => ({ saved: en.saved, folders: en.folders, all: folders() });
  const recent = sign => Object.values(data).filter(en => Math.sign(en.rating) === sign)
    .sort((a, b) => a.updated - b.updated).slice(-15).map(en => en.paper.title);

  return {
    where: "browser",
    get persisted() { return persisted; },
    entries: async () => Object.values(data).sort((a, b) => b.updated - a.updated),
    folders: async () => folders(),
    save: async (p, saved) => state(upsert(p, saved ? { saved: true } : { saved: false, folders: [] })),
    folder: async (p, name, add) => {
      name = name.split(/\s+/).filter(Boolean).join(" ").slice(0, 60);
      if (!name) throw new Error("folder name is empty");
      const current = (data[p.key]?.folders || []).filter(f => f !== name);
      const next = add ? [...current, name] : current;
      next.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
      return state(upsert(p, add ? { folders: next, saved: true } : { folders: next }));
    },
    rate: async (p, rating) => { upsert(p, { rating }); },
    remove: async key => { delete data[key]; write(); },
    noteSummary: (p, d) => { upsert(p, { summary: d.summary, full_text: d.full_text }); },
    overlay: p => {
      const en = data[p.key];
      return { ...p, rating: en?.rating || 0, saved: !!en?.saved, folders: en?.folders || [] };
    },
    searchContext: () => ({ liked: recent(1), disliked: recent(-1) }),
    mapContext: () => {
      const mine = Object.values(data).filter(en => en.saved || en.rating);
      return { papers: mine.map(en => ({ ...en.paper, key: en.key })),
               ratings: Object.fromEntries(mine.filter(en => en.rating).map(en => [en.key, en.rating])) };
    },
  };
})();

let lib = serverLib;  // init() switches to browserLib on the public demo

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${r.status})`);
  return data;
}

// Calls fn(seconds) every second until the returned stop() is called.
function ticker(fn) {
  const t0 = Date.now();
  fn(0);
  const id = setInterval(() => fn(Math.round((Date.now() - t0) / 1000)), 1000);
  return () => clearInterval(id);
}

function setStatus(text, busy = false) {
  const el = $("#status");
  el.textContent = text;
  el.classList.toggle("busy", busy);
}

// ---------- markdown + math ----------

function renderMd(md) {
  const math = [];
  const stash = (tex, display) => {
    math.push(katex.renderToString(tex, { displayMode: display, throwOnError: false }));
    return `@@MATH${math.length - 1}@@`;
  };
  md = md
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => stash(t, true))
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => stash(t, true))
    .replace(/\\\((.+?)\\\)/g, (_, t) => stash(t, false))
    .replace(/(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?!\d)/g, (_, pre, t) => pre + stash(t, false));
  const html = DOMPurify.sanitize(marked.parse(md));
  return html.replace(/@@MATH(\d+)@@/g, (_, i) => math[+i]);
}

// ---------- paper rows ----------

function authorLine(authors = []) {
  if (!authors.length) return "";
  return authors.length > 3 ? `${authors.slice(0, 3).join(", ")} et al.` : authors.join(", ");
}

function row(p, entry = null, { trending = false, note = "" } = {}) {
  if (!entry) p = lib.overlay(p);  // library rows are built from the entry itself
  papers.set(p.key, p);
  // Ranked papers show their relevance score. Only the trending list shows upvotes there: search
  // results from Hugging Face carry upvotes too, and ▲ in a search or the library reads as "trending".
  const votes = trending && p.score == null && p.upvotes;
  const score = p.score != null ? (+p.score).toFixed(1).replace(/\.0$/, "") : votes ? `▲${p.upvotes}` : "–";
  const meta = [
    authorLine(p.authors), p.year,
    p.venue && p.venue !== p.source ? p.venue : "",
    p.citations ? `${p.citations} cites` : "",
    p.source && p.source.toLowerCase(),
  ].filter(Boolean).map(esc).join(" · ");
  const hasSummary = entry && entry.summary;

  // Each dataset links to its homepage (Papers with Code catalogue), else a Hugging Face search.
  // Papers saved before links existed have none, so they get the search.
  const datasetLink = (p) => (name) => {
    const url = (p.dataset_links && p.dataset_links[name]) || `https://huggingface.co/datasets?search=${encodeURIComponent(name)}`;
    return `<a href="${esc(url)}" target="_blank" rel="noopener" title="${esc(url)}">${esc(name)}</a>`;
  };
  // Signals behind the ranking. Lit = meets your priority, dim = doesn't, absent = unknown.
  const sig = (ok, text) => `<span class="${ok ? "ok" : "no"}">${text}</span>`;
  const codeBits = [p.code_official ? "official" : "", p.code_framework, p.stars ? `★${p.stars}` : ""].filter(Boolean).join(" · ");
  const signals = [
    p.code_url
      ? sig(true, `<a href="${esc(p.code_url)}" target="_blank" rel="noopener">code ↗</a>${codeBits ? ` ${esc(codeBits)}` : ""}`)
      : sig(false, "no code"),
    p.datasets && p.datasets.length
      ? sig(p.datasets.length >= 2, `${p.datasets.length} dataset${p.datasets.length > 1 ? "s" : ""}: ${p.datasets.slice(0, 4).map(datasetLink(p)).join(", ")}${p.datasets.length > 4 ? "…" : ""}`)
      : (p.recruiter != null ? sig(false, "no named datasets") : ""),
    p.needs_gpu == null ? "" : sig(!p.needs_gpu, `${p.needs_gpu ? "needs gpu" : "cpu ok"}${p.compute_note ? ` · ${esc(p.compute_note)}` : ""}`),
    p.recruiter == null ? "" : sig(p.recruiter >= 7, `recruiter ${(+p.recruiter).toFixed(0)}/10`),
    p.similarity == null ? "" : sig(p.similarity >= 0.8, `similarity ${(+p.similarity).toFixed(2)}`),
    p.preference == null ? "" : sig(p.preference >= 0.5, `you'd like ${Math.round(100 * p.preference)}%`),
  ].filter(Boolean).join("");

  return `
  <li class="paper" data-key="${esc(p.key)}"${p.rank != null ? ` data-rank="${+p.rank}"` : ""}>
    <div class="score ${p.score != null && p.score < 5 ? "low" : ""} ${votes ? "votes" : ""}"
         ${votes ? `title="${p.upvotes} upvotes on Hugging Face"` : ""}>${score}</div>
    <div>
      <h2><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></h2>
      <p class="meta">${meta}</p>
      ${signals ? `<p class="signals">${signals}</p>` : ""}
      ${note ? `<p class="reason">${esc(note)}</p>` : p.reason ? `<p class="reason">${esc(p.reason)}</p>` : p.tldr ? `<p class="reason">${esc(p.tldr)}</p>` : ""}
      ${p.recruiter_reason ? `<p class="reason recruiter"><span>recruiter</span> ${esc(p.recruiter_reason)}</p>` : ""}
      ${p.abstract ? `<details class="abstract"><summary>abstract</summary><p>${esc(p.abstract)}</p></details>` : ""}
      <div class="actions">
        <button data-act="summarize">${hasSummary ? "summary" : "summarize"}</button>
        ${p.pdf_url ? `<a href="${esc(p.pdf_url)}" target="_blank" rel="noopener">pdf</a>` : ""}
        <button data-act="similar">similar</button>
        <button data-act="bib">bibtex</button>
        <button data-act="save" class="${p.saved ? "on" : ""}" aria-expanded="false">${saveLabel(p)}</button>
        <span class="rate">
          <button data-act="up" class="${p.rating > 0 ? "on" : ""}" aria-label="More like this" title="More like this">+</button>
          <button data-act="down" class="${p.rating < 0 ? "on" : ""}" aria-label="Less like this" title="Less like this">−</button>
        </span>
        ${entry ? `<button data-act="remove">remove</button>` : ""}
      </div>
      <div class="folder-picker" hidden></div>
      <div class="similar-box" hidden></div>
      <div class="summary" hidden ${hasSummary ? `data-summary='${esc(JSON.stringify({ summary: entry.summary, full_text: entry.full_text }))}'` : ""}></div>
    </div>
  </li>`;
}

// ---------- folders ----------

// One button for saving and filing: "save" saves and opens the folder picker; once saved it
// toggles the picker, which also holds "unsave".
const saveLabel = p => !p.saved ? "save"
  : !p.folders?.length ? "saved"
  : p.folders.length === 1 ? `saved · ${esc(p.folders[0])}` : `saved · ${p.folders.length} folders`;

function showSaved(li, p) {
  const btn = $('[data-act="save"]', li);
  btn.innerHTML = saveLabel(p);
  btn.classList.toggle("on", !!p.saved);
}

function togglePicker(li, p, open) {
  const box = $(".folder-picker", li);
  box.hidden = !open;
  $('[data-act="save"]', li).setAttribute("aria-expanded", String(open));
  if (open) {
    renderPicker(li, p);
    $(".new-folder", box).focus();
  }
}

// Keep the library's copy of this paper in step without re-rendering it (that would close the picker).
function syncLibrary(p, all) {
  allFolders = all;
  const en = libEntries.find(x => x.key === p.key);
  if (en) {
    Object.assign(en, { folders: p.folders, saved: p.saved });
    showLibCount();
  } else refreshLibCount();
  if (!$("#view-library").hidden) renderFolderFilters();
}

function renderPicker(li, p) {
  const box = $(".folder-picker", li);
  const mine = new Set(p.folders || []);
  const names = [...new Set([...allFolders.map(f => f.name), ...mine])];
  box.innerHTML = `
    ${names.map(n => `<button data-act="folder-toggle" data-folder="${esc(n)}" class="${mine.has(n) ? "on" : ""}">${esc(n)}</button>`).join("")}
    <input class="new-folder" placeholder="${names.length ? "new folder…" : "add to a folder…"}" maxlength="60" aria-label="New folder name">
    <button data-act="unsave" class="unsave">unsave</button>`;
}

async function setFolder(li, p, folder, add) {
  const d = await lib.folder(p, folder, add);
  p.folders = d.folders;
  p.saved = d.saved;
  showSaved(li, p);
  renderPicker(li, p);
  syncLibrary(p, d.all);
}

document.addEventListener("keydown", async e => {
  if (e.key !== "Enter" || !e.target.matches(".new-folder")) return;
  e.preventDefault();
  const name = e.target.value.trim();
  const li = e.target.closest(".paper");
  if (!name || !li) return;
  try {
    await setFolder(li, papers.get(li.dataset.key), name, true);
    $(".new-folder", li)?.focus();
  } catch (err) {
    setStatus(err.message);
  }
});

function renderFolderFilters() {
  const box = $("#lib-folders");
  if (libFolder && !allFolders.some(f => f.name === libFolder)) libFolder = null;  // emptied folder
  box.hidden = !allFolders.length;
  box.innerHTML = `<span class="label">folders</span>`
    + `<button data-folder="" class="${libFolder ? "" : "on"}">any</button>`
    + allFolders.map(f => `<button data-folder="${esc(f.name)}" class="${libFolder === f.name ? "on" : ""}">${esc(f.name)} <span>${f.count}</span></button>`).join("");
}

$("#lib-folders").addEventListener("click", e => {
  const b = e.target.closest("[data-folder]");
  if (!b) return;
  libFolder = b.dataset.folder || null;
  renderFolderFilters();
  renderLibrary();
});

// ---------- similar papers (from the embedding graph) ----------

function similarList(items) {
  if (!items.length) return `<p class="note">nothing close enough yet. search more and the graph grows.</p>`;
  const item = s => `<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a>
    <span class="sim">${[s.similarity != null ? (+s.similarity).toFixed(2) : "", s.year ? esc(s.year) : ""].filter(Boolean).join(" · ")}</span>
    ${s.via ? `<span class="via">via “${esc(s.via)}”</span>` : ""}</li>`;
  const close = items.filter(s => s.direct), reached = items.filter(s => !s.direct);
  return `<p class="note">closest</p><ol>${close.map(item).join("")}</ol>`
    + (reached.length ? `<p class="note">found through the graph</p><ol>${reached.map(item).join("")}</ol>` : "");
}

async function similarRow(li, p) {
  const btn = $('[data-act="similar"]', li);
  const box = $(".similar-box", li);
  if (box.dataset.loaded) {
    box.hidden = !box.hidden;
    btn.textContent = box.hidden ? "similar" : "hide similar";
    return;
  }
  btn.disabled = true;
  btn.textContent = "finding…";
  try {
    const d = await api("/api/similar", { paper: p });
    box.innerHTML = similarList(d.similar);
    box.dataset.loaded = "1";
    box.hidden = false;
    btn.textContent = "hide similar";
  } catch (err) {
    box.hidden = false;
    box.innerHTML = `<p class="note">${esc(err.message)}</p>`;
    btn.textContent = "similar";
  } finally {
    btn.disabled = false;
  }
}

// ---------- similarity graph drawing (map tab and the graph beside search results) ----------

// Draws nodes + links with d3-force into `box`. opts: height, fill(d), big(d), onSelect(d).
// Returns { highlight(key | null), refresh() } so rows can light up their node, and nodes can be
// redrawn after their data changes (a paper added to the results turns bright).
function drawGraph(box, data, opts) {
  box._graph?.stop();  // an earlier layout in this box stops ticking
  const css = getComputedStyle(document.documentElement), col = v => css.getPropertyValue(v).trim();
  const w = box.clientWidth, h = opts.height;
  box.innerHTML = "";
  const svg = d3.select(box).append("svg").attr("viewBox", [0, 0, w, h]).attr("height", h);
  const g = svg.append("g");
  // Plain scrolling keeps scrolling the page; zoom with ⌘/ctrl + scroll or a pinch.
  const zoom = d3.zoom().scaleExtent([0.3, 6]).filter(e => e.type !== "wheel" || e.ctrlKey || e.metaKey)
    .on("zoom", e => g.attr("transform", e.transform));
  svg.call(zoom);

  const nodes = data.nodes.map(d => ({ ...d })), links = data.links.map(d => ({ ...d }));
  const link = g.append("g").attr("stroke", col("--faint")).selectAll("line").data(links).join("line")
    .attr("stroke-opacity", d => 0.15 + (d.sim - 0.6) * 1.5).attr("stroke-width", 0.8);
  const node = g.append("g").selectAll("circle").data(nodes).join("circle")
    .attr("r", d => opts.big(d) ? 5.5 : 3.5)
    .attr("fill", d => opts.fill(d, col))
    .attr("stroke", d => d.rating < 0 ? col("--muted") : "none").attr("stroke-width", 1.2)
    .style("cursor", "pointer")
    .on("click", (e, d) => { highlight(d.key); opts.onSelect(d); });
  node.append("title").text(d => `${d.title}${d.year ? ` (${d.year})` : ""}`);

  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.key).distance(d => 20 + 140 * (1 - d.sim)).strength(0.4))
    .force("charge", d3.forceManyBody().strength(-38))
    .force("center", d3.forceCenter(w / 2, h / 2))
    .force("collide", d3.forceCollide(7))
    .force("x", d3.forceX(w / 2).strength(0.05))  // keeps small disconnected groups from drifting off
    .force("y", d3.forceY(h / 2).strength(0.08))
    .on("tick", () => {
      link.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      node.attr("cx", d => d.x).attr("cy", d => d.y);
    })
    .on("end", () => {  // once settled, zoom to fit everything
      const xs = nodes.map(d => d.x), ys = nodes.map(d => d.y), pad = 24;
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      const k = Math.min(3, 0.95 * Math.min(w / (x1 - x0 + 2 * pad), h / (y1 - y0 + 2 * pad)));
      svg.transition().duration(500).call(zoom.transform,
        d3.zoomIdentity.translate(w / 2, h / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2));
    });
  box._graph = sim;
  node.call(d3.drag()
    .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.2).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = d.fy = null; }));

  function highlight(key) {
    const touches = l => l.source.key === key || l.target.key === key;
    const linked = new Set([key]);
    links.forEach(l => { if (touches(l)) { linked.add(l.source.key); linked.add(l.target.key); } });
    node.attr("opacity", n => key == null || linked.has(n.key) ? 1 : 0.2)
      .attr("r", n => n.key === key ? 8 : opts.big(n) ? 5.5 : 3.5);
    link.attr("stroke", l => key != null && touches(l) ? col("--soft") : col("--faint"));
  }
  function refresh() {
    node.attr("fill", d => opts.fill(d, col)).attr("r", d => opts.big(d) ? 5.5 : 3.5);
  }
  return { highlight, refresh };
}

async function showCard(card, d) {
  const head = `<h2><a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.title)}</a></h2>`;
  card.hidden = false;
  card.innerHTML = `${head}<p class="note">${d.year ? esc(d.year) + " · " : ""}finding similar papers…</p>`;
  try {
    const s = await api("/api/similar", { paper: { title: d.title }, key: d.key });
    card.innerHTML = head + similarList(s.similar);
  } catch (err) {
    $(".note", card).textContent = err.message;
  }
}

// The map tab: everything around what's on screen and your library.
async function drawMap() {
  const note = $("#map-note"), box = $("#map");
  if (!window.d3) return (note.textContent = "couldn't load the graph library (d3). check your connection.");
  note.textContent = "building the map…";
  const keys = $$("#results .paper, #sample-list .paper").map(li => li.dataset.key);
  const data = await api("/api/map", { keys, ...lib.mapContext() });
  if (!data.nodes.length) {
    box.innerHTML = "";
    return (note.textContent = "nothing to map yet. run a search or rate a few papers.");
  }
  note.textContent = `${data.nodes.length} of ${data.total_papers} papers you've seen · lines join similar papers · `
    + `bright = liked, ring = disliked, light = on screen · drag, ⌘/ctrl + scroll to zoom, click a dot`;
  drawGraph(box, data, {
    height: Math.max(420, Math.min(680, innerHeight - 240)),
    big: d => d.rating || d.on_screen,
    fill: (d, col) => d.rating > 0 ? col("--fg") : d.rating < 0 ? "none" : d.on_screen ? col("--soft") : col("--faint"),
    onSelect: d => showCard($("#map-card"), d),
  });
}

// The graph beside search results: the results and their nearest neighbours from earlier searches.
let resultsGraph = null;

async function drawResultsGraph(keys) {
  const wrap = $("#results-graph"), note = $("#rg-note");
  resultsGraph = null;
  $("#rg-card").hidden = true;
  if (!window.d3 || !keys.length) return (wrap.hidden = true);
  const data = await api("/api/map", { keys, include_library: false, limit: keys.length + 35, ...lib.mapContext() });
  if (data.nodes.length < 2) return (wrap.hidden = true);
  wrap.hidden = false;
  const related = data.nodes.length - data.nodes.filter(n => n.on_screen).length;
  note.textContent = `your results (bright) and ${related} related papers from earlier searches · `
    + `hover a result to find it · click a dot`;
  // 960 px and up: a column beside the results that stays in view (see #view-search in the CSS).
  const beside = matchMedia("(min-width: 960px)").matches;
  resultsGraph = drawGraph($("#rg"), data, {
    height: beside ? mapHeight() : 300,
    big: d => d.on_screen,
    fill: (d, col) => d.on_screen ? col("--fg") : d.rating < 0 ? "none" : col("--faint"),
    onSelect: d => {
      const li = $(`#results .paper[data-key="${CSS.escape(d.key)}"]`);
      if (li) {
        $("#rg-card").hidden = true;
        return flash(li);
      }
      if (!searchId) return showCard($("#rg-card"), d);
      addFromMap(d);
    },
  });
}

// A paper clicked on the map joins the results as a full row: the server rebuilds it, reads it
// for this search (relevance, datasets, recruiter score, GPU needs) and returns its rank value,
// the same one the results carry, so it lands where the ranking would have put it.
const adding = new Set();
let addedNodes = [];  // map nodes whose papers were added, to undo with "back to the original results"

function backToOriginal() {
  $$("#results > .paper[data-added]").forEach(li => li.remove());
  addedNodes.forEach(d => { d.on_screen = false; });
  addedNodes = [];
  resultsGraph?.refresh();
  $("#rg-card").hidden = true;
  $("#back-original").hidden = true;
}
$("#back-original").addEventListener("click", backToOriginal);

async function addFromMap(d) {
  if (adding.has(d.key)) return;
  adding.add(d.key);
  const card = $("#rg-card");
  card.hidden = false;
  const stop = ticker(s => {
    card.innerHTML = `<p class="note">adding “${esc(d.title)}” to your results · ${esc(modelName)} is reading it · ${s}s`
      + (s >= 8 ? " · on a free tier, right after a search this waits up to a minute for the model's quota" : "") + "</p>";
  });
  try {
    const { paper, errors } = await api("/api/place", {
      search_id: searchId, key: d.key,
      paper: { title: d.title, abstract: d.abstract, year: d.year, url: d.url, code_url: d.code_url, source: d.source },
      ...lib.searchContext(),
    });
    if ($(`#results .paper[data-key="${CSS.escape(paper.key)}"]`)) return;  // added twice by quick clicks
    const tmp = document.createElement("ol");
    tmp.innerHTML = row(paper);
    const li = tmp.firstElementChild;
    li.dataset.added = "1";
    const after = $$("#results > .paper").find(el => el.dataset.rank != null && +el.dataset.rank < paper.rank);
    $("#results").insertBefore(li, after || null);
    d.on_screen = true;
    addedNodes.push(d);
    resultsGraph?.refresh();
    $("#back-original").hidden = false;
    card.hidden = !errors.length;
    card.innerHTML = errors.length
      ? `<p class="note">added, but ${esc(errors.map(e => e.message).join(" · "))}</p>` : "";
    flash(li);
  } catch (err) {
    card.innerHTML = `<p class="note">${esc(err.message)}</p>`;
  } finally {
    stop();
    adding.delete(d.key);
  }
}

// The map beside a list: at most ~38% of the screen, so a clicked paper's card below it has room.
const mapHeight = () => Math.round(Math.max(240, Math.min(innerHeight * 0.38, 400)));

function flash(li) {
  li.scrollIntoView({ behavior: "smooth", block: "center" });
  li.classList.remove("flash");
  void li.offsetWidth;  // restart the animation
  li.classList.add("flash");
}

// The map above the saved / liked lists: those papers, how they relate, and similar papers you
// haven't rated yet, listed under the map with the usual buttons so you can like or save them.
let libGraph = null, libGraphRun = 0;

async function drawLibraryGraph(list) {
  const wrap = $("#lib-graph"), run = ++libGraphRun;
  libGraph = null;
  $("#lg-card").hidden = true;
  if (!window.d3 || !["saved", "liked"].includes(libFilter) || !list.length) {
    $("#lg-related-box").hidden = true;
    return (wrap.hidden = true);
  }
  $("#lg-note").textContent = "mapping your papers…";
  wrap.hidden = false;
  let data;
  try {
    data = await api("/api/map", { keys: list.map(en => en.key), include_library: false,
                                   limit: list.length + 40, related: 12, ...lib.mapContext() });
  } catch (err) {
    return ($("#lg-note").textContent = err.message);
  }
  if (run !== libGraphRun) return;  // the filter or folder changed while this was loading
  const related = data.related || [];
  $("#lg-note").textContent = `your ${libFilter} papers (bright) and the papers around them · `
    + `lines join similar papers · hover a paper to find it · click a dot`;
  // 960 px and up: a column beside the list that stays in view (see #view-library in the CSS).
  const beside = matchMedia("(min-width: 960px)").matches;
  if (data.nodes.length >= 2) {
    libGraph = drawGraph($("#lg"), data, {
      height: beside ? mapHeight() : 320,
      big: d => d.on_screen,
      fill: (d, col) => d.on_screen ? col("--fg") : d.rating < 0 ? "none" : col("--faint"),
      onSelect: d => {
        const sel = `.paper[data-key="${CSS.escape(d.key)}"]`;
        const li = $(`#library ${sel}`) || $(`#lg-related ${sel}`);
        if (!li) return showCard($("#lg-card"), d);
        $("#lg-card").hidden = true;
        flash(li);
      },
    });
  } else {
    $("#lg").innerHTML = "";
  }
  $("#lg-related").innerHTML = related.map(p => row(p, null, { note: `similar to “${p.via}”` })).join("");
  $("#lg-related-count").textContent = related.length ? `· ${related.length}` : "";
  $("#lg-related-box").hidden = !related.length;
}

for (const id of ["#library", "#lg-related"]) {
  $(id).addEventListener("mouseover", e => {
    const li = e.target.closest(".paper");
    if (li && libGraph) libGraph.highlight(li.dataset.key);
  });
  $(id).addEventListener("mouseleave", () => libGraph?.highlight(null));
}

$("#results").addEventListener("mouseover", e => {
  const li = e.target.closest(".paper");
  if (li && resultsGraph) resultsGraph.highlight(li.dataset.key);
});
$("#results").addEventListener("mouseleave", () => resultsGraph?.highlight(null));

function showSummary(li, d) {
  const box = $(".summary", li);
  const source = d.full_text ? "from the full text" : "from the abstract only (no open-access pdf)";
  box.innerHTML = `<p class="note">${source} · <button data-act="regen">regenerate</button></p>${renderMd(d.summary)}`;
  box.dataset.loaded = "1";
  box.hidden = false;
  $('[data-act="summarize"]', li).textContent = "hide summary";
}

async function summarizeRow(li, p, refresh = false) {
  const btn = $('[data-act="summarize"]', li);
  const box = $(".summary", li);
  if (!refresh && box.dataset.loaded) {
    box.hidden = !box.hidden;
    btn.textContent = box.hidden ? "summary" : "hide summary";
    return;
  }
  if (!refresh && box.dataset.summary) {
    showSummary(li, JSON.parse(box.dataset.summary));
    return;
  }
  btn.disabled = true;
  const stop = ticker(s => { btn.textContent = `reading · ${s}s`; });
  try {
    const d = await api("/api/summarize", { paper: p, refresh });
    lib.noteSummary(p, d);
    showSummary(li, d);
    refreshLibCount();
  } catch (err) {
    box.hidden = false;
    box.innerHTML = `<p class="note">${esc(err.message)}</p>`;
    btn.textContent = "summarize";
  } finally {
    stop();
    btn.disabled = false;
  }
}

document.addEventListener("click", async e => {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const li = btn.closest(".paper");
  const p = papers.get(li.dataset.key);
  try {
    switch (btn.dataset.act) {
      case "summarize": return summarizeRow(li, p);
      case "similar": return similarRow(li, p);
      case "folder-toggle":
        return setFolder(li, p, btn.dataset.folder, !(p.folders || []).includes(btn.dataset.folder));
      case "regen": return summarizeRow(li, p, true);
      case "bib":
        await navigator.clipboard.writeText(p.bibtex);
        btn.textContent = "copied";
        setTimeout(() => { btn.textContent = "bibtex"; }, 1400);
        return;
      case "save": {
        if (p.saved) return togglePicker(li, p, $(".folder-picker", li).hidden);
        const d = await lib.save(p, true);
        Object.assign(p, { saved: d.saved, folders: d.folders });
        showSaved(li, p);
        syncLibrary(p, d.all);
        return togglePicker(li, p, true);  // saved; now optionally file it
      }
      case "unsave": {
        const d = await lib.save(p, false);
        Object.assign(p, { saved: d.saved, folders: d.folders });
        showSaved(li, p);
        togglePicker(li, p, false);
        return syncLibrary(p, d.all);
      }
      case "up":
      case "down": {
        const value = btn.dataset.act === "up" ? 1 : -1;
        p.rating = p.rating === value ? 0 : value;
        await lib.rate(p, p.rating);
        $('[data-act="up"]', li).classList.toggle("on", p.rating > 0);
        $('[data-act="down"]', li).classList.toggle("on", p.rating < 0);
        return refreshLibCount();
      }
      case "remove":
        await lib.remove(p.key);
        libEntries = libEntries.filter(en => en.key !== p.key);
        li.remove();
        return refreshLibCount();
    }
  } catch (err) {
    setStatus(err.message);
  }
});

// ---------- search ----------

const ERROR_LABELS = { rewrite_failed: "used your words as typed", timeout: "timed out", rate_limit: "rate limited", network: "offline", http_error: "server error", bad_response: "bad response" };

const errorRow = e => `<li class="err-${esc(e.error_type)}"><b>${esc(e.source)}${e.field ? ` · ${esc(e.field)}` : ""}</b> `
  + `${esc(ERROR_LABELS[e.error_type] || e.error_type.replace("_", " "))} — ${esc(e.message)}</li>`;

const selectedFields = () => $$("#fields input:checked").map(i => i.value);
let searching = false;
let searchId = null;  // the search on screen, to add papers from its map into (addFromMap)

$("#search-form").addEventListener("submit", async e => {
  e.preventDefault();
  const query = $("#q").value.trim();
  const fields = selectedFields();
  if (!query || searching) return;
  if (!fields.length) return setStatus("pick at least one field");

  searching = true;
  $("#trending").hidden = true;
  $("#results").innerHTML = "";
  $("#errors").innerHTML = "";
  $("#sample").hidden = true;
  $("#results-graph").hidden = true;
  $("#back-original").hidden = true;
  const stop = ticker(s => setStatus(`searching, then ${modelName} reads the shortlist (a few minutes on a local model) · ${s}s`, true));
  try {
    const data = await api("/api/search", { query, fields, code_only: $("#code-only").checked,
                                            ...lib.searchContext() });
    stop();
    const index = { building: " · code index still building, using hugging face links only", missing: " · code index not built" }[data.code_index] || "";
    const searched = data.rewrite && data.rewrite.keywords.toLowerCase() !== query.toLowerCase()
      ? `searched for “${data.rewrite.keywords}” · ` : "";
    setStatus(searched + (data.papers.length
      ? `${data.candidates} candidates · ${data.with_code} with code · showing the top ${data.papers.length}${index}`
      : "nothing matched. try broader words, another field, or turn off code only"));
    $("#errors").innerHTML = data.errors.map(errorRow).join("");
    searchId = data.search_id;
    addedNodes = [];
    $("#back-original").hidden = true;
    $("#results").innerHTML = data.papers.map(p => row(p)).join("");
    drawResultsGraph(data.papers.map(p => p.key)).catch(() => { $("#results-graph").hidden = true; });
    $("#sample-list").innerHTML = (data.unranked_sample || []).map(p => row(p)).join("");
    $("#sample").hidden = !(data.unranked_sample || []).length;
  } catch (err) {
    stop();
    setStatus(err.message);
  } finally {
    searching = false;
  }
});

// ---------- library ----------

function renderLibrary() {
  // Every rated paper is kept (ratings are the eval's and preference model's labels), but the
  // library only lists what you chose to keep: saved, liked, or summarized.
  const keep = {
    saved: en => en.saved,
    liked: en => en.rating > 0,
    summarized: en => en.summary,
  }[libFilter];
  const list = libEntries.filter(en => keep(en) && (!libFolder || (en.folders || []).includes(libFolder)));
  $("#library").innerHTML = list.length
    ? list.map(en => row({ ...en.paper, key: en.key, rating: en.rating, saved: en.saved, folders: en.folders || [], bibtex: en.bibtex }, en)).join("")
    : `<li class="empty">${{
        saved: "nothing saved yet. save a paper and it lands here.",
        liked: "nothing liked yet. rate a paper + and it lands here.",
        summarized: "no summaries yet. summarize a paper and it lands here.",
      }[libFilter]}</li>`;
  drawLibraryGraph(list);
  return list;
}

async function loadLibrary() {
  [libEntries, allFolders] = await Promise.all([
    lib.entries(), lib.folders(),
  ]);
  renderFolderFilters();
  renderLibrary();
  showLibCount();
  const where = $("#lib-where");
  where.hidden = lib.where !== "browser";
  where.textContent = lib.persisted === false
    ? "this browser isn't letting the page store anything, so your library will be gone when you close the tab."
    : "your library lives in this browser only: no account, nothing stored on the server. clearing this site's data erases it.";
}

const showLibCount = () => { $("#lib-count").textContent = libEntries.filter(en => en.saved).length || ""; };

async function refreshLibCount() {
  try {
    libEntries = await lib.entries();
    showLibCount();
  } catch {}
}

$("#lib-filter").addEventListener("click", e => {
  const b = e.target.closest("[data-filter]");
  if (!b) return;
  libFilter = b.dataset.filter;
  $$("#lib-filter button").forEach(x => x.classList.toggle("on", x === b));
  renderLibrary();
});

$("#export-bib").addEventListener("click", () => {
  // exports what's listed: the current filter and folder
  const keys = new Set($$("#library .paper").map(li => li.dataset.key));
  const bib = libEntries.filter(en => keys.has(en.key)).map(en => en.bibtex).join("\n\n");
  if (!bib) return;
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([bib], { type: "application/x-bibtex" })),
    download: `${(libFolder || "library").replace(/[^\w-]+/g, "-")}.bib`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
});

// ---------- views ----------

function showView(name) {
  $$("nav button").forEach(b => b.classList.toggle("on", b.dataset.view === name));
  $("#view-search").hidden = name !== "search";
  $("#view-library").hidden = name !== "library";
  $("#view-map").hidden = name !== "map";
  if (name === "library") loadLibrary().catch(err => setStatus(err.message));
  else if (name === "map") drawMap().catch(err => { $("#map-note").textContent = err.message; });
  else $("#q").focus();
}
$$("nav button").forEach(b => b.addEventListener("click", () => showView(b.dataset.view)));

// ---------- boot ----------

(async function init() {
  try {
    const prefs = await api("/api/prefs");
    if (prefs.demo) lib = browserLib;  // before anything reads the library
    const remembered = store.get("fields");
    $("#fields").innerHTML = Object.entries(prefs.fields).map(([key, label]) => `
      <label class="toggle"><input type="checkbox" value="${esc(key)}" ${!remembered || remembered.includes(key) ? "checked" : ""}><span>${esc(label)}</span></label>
    `).join("");
    $("#fields").addEventListener("change", () => store.set("fields", selectedFields()));
    for (const id of ["code-only"]) {
      $("#" + id).checked = !!store.get(id);
      $("#" + id).addEventListener("change", e => store.set(id, e.target.checked));
    }
    modelName = prefs.model;
    allFolders = await lib.folders();
    if (prefs.demo) {
      // Public demo: rate-limited; each visitor's library lives in their own browser (browserLib).
      document.body.classList.add("demo");
      $("#foot").innerHTML = `public demo · your library stays in this browser · ${prefs.demo.searches_per_hour} searches an hour · `
        + `${esc(prefs.model)} via ${esc(prefs.provider)} · `
        + `<a href="https://github.com/tryzhaa/Research-Copilot" target="_blank" rel="noopener">run it yourself ↗</a>`;
    } else {
      $("#foot").textContent = `${prefs.model} via ${prefs.provider} · ${prefs.effort} effort · settings live in preferences.yaml`;
    }
    refreshLibCount();
    loadTrending();
  } catch (err) {
    setStatus(`can't reach the server: ${err.message}`);
  }
})();

// Before the first search, the home screen shows what's trending (cached server-side, no LLM).
async function loadTrending() {
  try {
    const { papers, note } = await api("/api/trending");
    if (!papers.length || $("#results").children.length) return;  // a search already started
    $("#trending-list").innerHTML = papers.map(p => row(p, null, { trending: true })).join("");
    $("#trending-note").textContent = note ? `· ${note}` : "";
    $("#trending").hidden = searching;
  } catch {
    // Optional: the search box works without it.
  }
}

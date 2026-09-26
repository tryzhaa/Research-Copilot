const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const papers = new Map();   // key -> paper, for every row on screen
let libEntries = [];
let libFilter = "all";
let libFolder = null;       // library filtered to one folder, or null for any
let allFolders = [];        // [{name, count}], for the folder picker and library filters
let modelName = "the model";

const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

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

function row(p, entry = null) {
  papers.set(p.key, p);
  // Ranked papers show their relevance score; trending ones (never ranked) their upvotes.
  const score = p.score != null ? (+p.score).toFixed(1).replace(/\.0$/, "")
    : p.upvotes ? `▲${p.upvotes}` : "–";
  const meta = [
    authorLine(p.authors), p.year,
    p.venue && p.venue !== p.source ? p.venue : "",
    p.citations ? `${p.citations} cites` : "",
    p.source && p.source.toLowerCase(),
  ].filter(Boolean).map(esc).join(" · ");
  const hasSummary = entry && entry.summary;

  // Signals behind the ranking. Lit = meets your priority, dim = doesn't, absent = unknown.
  const sig = (ok, text) => `<span class="${ok ? "ok" : "no"}">${text}</span>`;
  const codeBits = [p.code_official ? "official" : "", p.code_framework, p.stars ? `★${p.stars}` : ""].filter(Boolean).join(" · ");
  const signals = [
    p.code_url
      ? sig(true, `<a href="${esc(p.code_url)}" target="_blank" rel="noopener">code ↗</a>${codeBits ? ` ${esc(codeBits)}` : ""}`)
      : sig(false, "no code"),
    p.datasets && p.datasets.length
      ? sig(p.datasets.length >= 2, `${p.datasets.length} dataset${p.datasets.length > 1 ? "s" : ""}: ${esc(p.datasets.slice(0, 4).join(", "))}${p.datasets.length > 4 ? "…" : ""}`)
      : (p.recruiter != null ? sig(false, "no named datasets") : ""),
    p.needs_gpu == null ? "" : sig(!p.needs_gpu, `${p.needs_gpu ? "needs gpu" : "cpu ok"}${p.compute_note ? ` · ${esc(p.compute_note)}` : ""}`),
    p.recruiter == null ? "" : sig(p.recruiter >= 7, `recruiter ${(+p.recruiter).toFixed(0)}/10`),
    p.similarity == null ? "" : sig(p.similarity >= 0.8, `similarity ${(+p.similarity).toFixed(2)}`),
    p.preference == null ? "" : sig(p.preference >= 0.5, `you'd like ${Math.round(100 * p.preference)}%`),
  ].filter(Boolean).join("");

  return `
  <li class="paper" data-key="${esc(p.key)}">
    <div class="score ${p.score != null && p.score < 5 ? "low" : ""} ${p.score == null && p.upvotes ? "votes" : ""}"
         ${p.score == null && p.upvotes ? `title="${p.upvotes} upvotes on Hugging Face"` : ""}>${score}</div>
    <div>
      <h2><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></h2>
      <p class="meta">${meta}</p>
      ${signals ? `<p class="signals">${signals}</p>` : ""}
      ${p.reason ? `<p class="reason">${esc(p.reason)}</p>` : p.tldr ? `<p class="reason">${esc(p.tldr)}</p>` : ""}
      ${p.recruiter_reason ? `<p class="reason recruiter"><span>recruiter</span> ${esc(p.recruiter_reason)}</p>` : ""}
      ${p.abstract ? `<details class="abstract"><summary>abstract</summary><p>${esc(p.abstract)}</p></details>` : ""}
      <div class="actions">
        <button data-act="summarize">${hasSummary ? "summary" : "summarize"}</button>
        ${p.pdf_url ? `<a href="${esc(p.pdf_url)}" target="_blank" rel="noopener">pdf</a>` : ""}
        <button data-act="similar">similar</button>
        <button data-act="bib">bibtex</button>
        <button data-act="save" class="${p.saved ? "on" : ""}">${p.saved ? "saved" : "save"}</button>
        <button data-act="folder" class="${p.folders?.length ? "on" : ""}">${folderLabel(p)}</button>
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

const folderLabel = p => !p.folders?.length ? "folder"
  : p.folders.length === 1 ? `in ${esc(p.folders[0])}` : `in ${p.folders.length} folders`;

function renderPicker(li, p) {
  const box = $(".folder-picker", li);
  const mine = new Set(p.folders || []);
  const names = [...new Set([...allFolders.map(f => f.name), ...mine])];
  box.innerHTML = `
    ${names.map(n => `<button data-act="folder-toggle" data-folder="${esc(n)}" class="${mine.has(n) ? "on" : ""}">${esc(n)}</button>`).join("")}
    <input class="new-folder" placeholder="${names.length ? "new folder…" : "name a folder…"}" maxlength="60" aria-label="New folder name">`;
}

async function setFolder(li, p, folder, add) {
  const d = await api("/api/folder", { paper: p, folder, add });
  p.folders = d.folders;
  p.saved = d.saved;
  allFolders = d.all;
  const btn = $('[data-act="folder"]', li);
  btn.innerHTML = folderLabel(p);
  btn.classList.toggle("on", p.folders.length > 0);
  const save = $('[data-act="save"]', li);
  save.classList.toggle("on", p.saved);
  save.textContent = p.saved ? "saved" : "save";
  renderPicker(li, p);
  // Update the library in place: re-rendering it would close the picker mid-use.
  const en = libEntries.find(x => x.key === p.key);
  if (en) Object.assign(en, { folders: d.folders, saved: d.saved });
  else refreshLibCount();
  if (!$("#view-library").hidden) renderFolderFilters();
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
    <span class="sim">${(+s.similarity).toFixed(2)}${s.year ? ` · ${esc(s.year)}` : ""}</span>
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
// Returns { highlight(key | null) } so rows can light up their node.
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
  return { highlight };
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
  const data = await api("/api/map", { keys });
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
  const data = await api("/api/map", { keys, include_library: false, limit: keys.length + 35 });
  if (data.nodes.length < 2) return (wrap.hidden = true);
  wrap.hidden = false;
  const related = data.nodes.length - data.nodes.filter(n => n.on_screen).length;
  note.textContent = `your results (bright) and ${related} related papers from earlier searches · `
    + `hover a result to find it · click a dot`;
  // 960 px and up: a column beside the results that stays in view (see #view-search in the CSS).
  const beside = matchMedia("(min-width: 960px)").matches;
  resultsGraph = drawGraph($("#rg"), data, {
    height: beside ? Math.max(300, Math.min(innerHeight - 200, 520)) : 300,
    big: d => d.on_screen,
    fill: (d, col) => d.on_screen ? col("--fg") : d.rating < 0 ? "none" : col("--faint"),
    onSelect: d => {
      const li = $(`#results .paper[data-key="${CSS.escape(d.key)}"]`);
      if (!li) return showCard($("#rg-card"), d);
      $("#rg-card").hidden = true;
      li.scrollIntoView({ behavior: "smooth", block: "center" });
      li.classList.remove("flash");
      void li.offsetWidth;  // restart the animation
      li.classList.add("flash");
    },
  });
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
    showSummary(li, await api("/api/summarize", { paper: p, refresh }));
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
      case "folder": {
        const box = $(".folder-picker", li);
        box.hidden = !box.hidden;
        if (!box.hidden) {
          renderPicker(li, p);
          $(".new-folder", box).focus();
        }
        return;
      }
      case "folder-toggle":
        return setFolder(li, p, btn.dataset.folder, !(p.folders || []).includes(btn.dataset.folder));
      case "regen": return summarizeRow(li, p, true);
      case "bib":
        await navigator.clipboard.writeText(p.bibtex);
        btn.textContent = "copied";
        setTimeout(() => { btn.textContent = "bibtex"; }, 1400);
        return;
      case "save":
        p.saved = !p.saved;
        await api("/api/save", { paper: p, saved: p.saved });
        btn.classList.toggle("on", p.saved);
        btn.textContent = p.saved ? "saved" : "save";
        return refreshLibCount();
      case "up":
      case "down": {
        const value = btn.dataset.act === "up" ? 1 : -1;
        p.rating = p.rating === value ? 0 : value;
        await api("/api/rate", { paper: p, rating: p.rating });
        $('[data-act="up"]', li).classList.toggle("on", p.rating > 0);
        $('[data-act="down"]', li).classList.toggle("on", p.rating < 0);
        return refreshLibCount();
      }
      case "remove":
        await api("/api/library/remove", { key: p.key });
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
  const stop = ticker(s => setStatus(`searching, then ${modelName} reads the shortlist (a few minutes on a local model) · ${s}s`, true));
  try {
    const data = await api("/api/search", { query, fields, use_s2: $("#use-s2").checked, code_only: $("#code-only").checked });
    stop();
    const index = { building: " · code index still building, using hugging face links only", missing: " · code index not built" }[data.code_index] || "";
    const searched = data.rewrite && data.rewrite.keywords.toLowerCase() !== query.toLowerCase()
      ? `searched for “${data.rewrite.keywords}” · ` : "";
    setStatus(searched + (data.papers.length
      ? `${data.candidates} candidates · ${data.with_code} with code · showing the top ${data.papers.length}${index}`
      : "nothing matched. try broader words, another field, or turn off code only"));
    $("#errors").innerHTML = data.errors.map(errorRow).join("");
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
  const keep = {
    all: () => true,
    saved: en => en.saved,
    liked: en => en.rating > 0,
    summarized: en => en.summary,
  }[libFilter];
  const list = libEntries.filter(en => keep(en) && (!libFolder || (en.folders || []).includes(libFolder)));
  $("#library").innerHTML = list.length
    ? list.map(en => row({ ...en.paper, key: en.key, rating: en.rating, saved: en.saved, folders: en.folders || [], bibtex: en.bibtex }, en)).join("")
    : `<li class="empty">nothing here yet. save, rate or summarize a paper and it lands here.</li>`;
  return list;
}

async function loadLibrary() {
  [libEntries, allFolders] = await Promise.all([
    api("/api/library").then(d => d.entries), api("/api/folders").then(d => d.folders),
  ]);
  renderFolderFilters();
  renderLibrary();
  $("#lib-count").textContent = libEntries.length || "";
}

async function refreshLibCount() {
  try {
    const { entries } = await api("/api/library");
    libEntries = entries;
    $("#lib-count").textContent = entries.length || "";
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
    const remembered = store.get("fields");
    $("#fields").innerHTML = Object.entries(prefs.fields).map(([key, label]) => `
      <label class="toggle"><input type="checkbox" value="${esc(key)}" ${!remembered || remembered.includes(key) ? "checked" : ""}><span>${esc(label)}</span></label>
    `).join("");
    $("#fields").addEventListener("change", () => store.set("fields", selectedFields()));
    for (const id of ["use-s2", "code-only"]) {
      $("#" + id).checked = !!store.get(id);
      $("#" + id).addEventListener("change", e => store.set(id, e.target.checked));
    }
    modelName = prefs.model;
    allFolders = (await api("/api/folders")).folders;
    if (prefs.demo) {
      // Public demo: read-only, rate-limited. CSS hides everything that writes to the shared library.
      document.body.classList.add("demo");
      $("#foot").innerHTML = `public demo · read-only · ${prefs.demo.searches_per_hour} searches an hour · `
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
    $("#trending-list").innerHTML = papers.map(p => row(p)).join("");
    $("#trending-note").textContent = note ? `· ${note}` : "";
    $("#trending").hidden = searching;
  } catch {
    // Optional: the search box works without it.
  }
}

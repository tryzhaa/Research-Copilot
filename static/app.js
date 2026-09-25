const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const papers = new Map();   // key -> paper, for every row on screen
let libEntries = [];
let libFilter = "all";
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
  const score = p.score == null ? "–" : (+p.score).toFixed(1).replace(/\.0$/, "");
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
    <div class="score ${p.score != null && p.score < 5 ? "low" : ""}">${score}</div>
    <div>
      <h2><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></h2>
      <p class="meta">${meta}</p>
      ${signals ? `<p class="signals">${signals}</p>` : ""}
      ${p.reason ? `<p class="reason">${esc(p.reason)}</p>` : ""}
      ${p.recruiter_reason ? `<p class="reason recruiter"><span>recruiter</span> ${esc(p.recruiter_reason)}</p>` : ""}
      ${p.abstract ? `<details class="abstract"><summary>abstract</summary><p>${esc(p.abstract)}</p></details>` : ""}
      <div class="actions">
        <button data-act="summarize">${hasSummary ? "summary" : "summarize"}</button>
        ${p.pdf_url ? `<a href="${esc(p.pdf_url)}" target="_blank" rel="noopener">pdf</a>` : ""}
        <button data-act="bib">bibtex</button>
        <button data-act="save" class="${p.saved ? "on" : ""}">${p.saved ? "saved" : "save"}</button>
        <span class="rate">
          <button data-act="up" class="${p.rating > 0 ? "on" : ""}" aria-label="More like this" title="More like this">+</button>
          <button data-act="down" class="${p.rating < 0 ? "on" : ""}" aria-label="Less like this" title="Less like this">−</button>
        </span>
        ${entry ? `<button data-act="remove">remove</button>` : ""}
      </div>
      <div class="summary" hidden ${hasSummary ? `data-summary='${esc(JSON.stringify({ summary: entry.summary, full_text: entry.full_text }))}'` : ""}></div>
    </div>
  </li>`;
}

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

const ERROR_LABELS = { timeout: "timed out", rate_limit: "rate limited", network: "offline", http_error: "server error", bad_response: "bad response" };

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
  $("#results").innerHTML = "";
  $("#errors").innerHTML = "";
  $("#sample").hidden = true;
  const stop = ticker(s => setStatus(`searching, then ${modelName} reads the shortlist (a few minutes on a local model) · ${s}s`, true));
  try {
    const data = await api("/api/search", { query, fields, use_s2: $("#use-s2").checked, code_only: $("#code-only").checked });
    stop();
    const index = { building: " · code index still building, using hugging face links only", missing: " · code index not built" }[data.code_index] || "";
    setStatus(data.papers.length
      ? `${data.candidates} candidates · ${data.with_code} with code · showing the top ${data.papers.length}${index}`
      : "nothing matched. try broader words, another field, or turn off code only");
    $("#errors").innerHTML = data.errors.map(errorRow).join("");
    $("#results").innerHTML = data.papers.map(p => row(p)).join("");
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
  const list = libEntries.filter(keep);
  $("#library").innerHTML = list.length
    ? list.map(en => row({ ...en.paper, key: en.key, rating: en.rating, saved: en.saved, bibtex: en.bibtex }, en)).join("")
    : `<li class="empty">nothing here yet. save, rate or summarize a paper and it lands here.</li>`;
}

async function loadLibrary() {
  libEntries = (await api("/api/library")).entries;
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
  const bib = libEntries.map(en => en.bibtex).join("\n\n");
  if (!bib) return;
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([bib], { type: "application/x-bibtex" })),
    download: "library.bib",
  });
  a.click();
  URL.revokeObjectURL(a.href);
});

// ---------- views ----------

function showView(name) {
  $$("nav button").forEach(b => b.classList.toggle("on", b.dataset.view === name));
  $("#view-search").hidden = name !== "search";
  $("#view-library").hidden = name !== "library";
  if (name === "library") loadLibrary().catch(err => setStatus(err.message));
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
    $("#foot").textContent = `${prefs.model} via ${prefs.provider} · ${prefs.effort} effort · settings live in preferences.yaml`;
    refreshLibCount();
  } catch (err) {
    setStatus(`can't reach the server: ${err.message}`);
  }
})();

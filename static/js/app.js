// ScribeScan frontend: document tabs, auto-save, multi-device sync and AI scanning.
import { api } from "./api.js";
import { createInsights } from "./insights.js";

const AUTOSAVE_DEBOUNCE_MS = 1200; // save shortly after the user pauses typing…
const AUTOSAVE_INTERVAL_MS = 5000; // …and at least every few seconds while typing
const SYNC_INTERVAL_MS = 15000; // pick up edits made on other devices
const LAST_DOC_KEY = "scribescan:lastDoc";

const $ = (id) => document.getElementById(id);
const els = {
  app: $("app"), docList: $("docList"), docCount: $("docCount"), newDocBtn: $("newDocBtn"),
  menuBtn: $("menuBtn"), scrim: $("scrim"), titleInput: $("titleInput"),
  saveStatus: $("saveStatus"), saveStatusText: document.querySelector("#saveStatus .save-status-text"),
  conflictBanner: $("conflictBanner"), conflictLoadBtn: $("conflictLoadBtn"), conflictKeepBtn: $("conflictKeepBtn"),
  wordCount: $("wordCount"), charCount: $("charCount"),
  modelBadge: $("modelBadge"), modelBadgeText: document.querySelector("#modelBadge .model-badge-text"),
  gauge: $("gauge"), gaugeValue: $("gaugeValue"), gaugeNumber: $("gaugeNumber"), gaugeCaption: $("gaugeCaption"),
  verdict: $("verdict"), verdictLabel: $("verdictLabel"), verdictSub: $("verdictSub"),
  minWordsWrap: $("minWordsWrap"), minWordsText: $("minWordsText"), minWordsBar: $("minWordsBar"),
  scanAlert: $("scanAlert"), scanBtn: $("scanBtn"), scanBtnText: document.querySelector("#scanBtn .btn-scan-text"),
  chunksWrap: $("chunksWrap"), chunkStrip: $("chunkStrip"), chunksCount: $("chunksCount"),
  scanMeta: $("scanMeta"), stalePill: $("stalePill"),
  confirmDialog: $("confirmDialog"), confirmTitle: $("confirmTitle"), confirmBody: $("confirmBody"),
  toast: $("toast"), howMin: $("howMin"), howChunk: $("howChunk"), howThreshold: $("howThreshold"),
};

const state = {
  config: { min_words: 300, human_threshold: 0.2, chunk_words: 200 },
  model: { state: "loading" },
  docs: [], // summaries from the server
  current: null, // { id, title, version, updated_at }
  dirty: false,
  editSeq: 0, // bumps on every edit, so a save can tell if newer edits arrived mid-flight
  saving: null, // Promise while a save request is in flight
  saveFailed: false,
  conflict: null, // server copy when a 409 happened
  lastResult: null,
  scannedText: null, // normalised text the shown result belongs to (null = unknown/stale)
  scanning: false,
};

// ------------------------------------------------------------------ editor
const quill = new Quill("#editor", {
  theme: "snow",
  placeholder: "Start writing, or paste text you want to check…",
  modules: {
    toolbar: [
      [{ header: [1, 2, 3, false] }],
      ["bold", "italic", "underline", "strike"],
      [{ list: "ordered" }, { list: "bullet" }],
      ["blockquote", "code-block", "link"],
      ["clean"],
    ],
    history: { delay: 800, maxStack: 500, userOnly: true },
  },
});
quill.disable();
const insights = createInsights(quill);

// ------------------------------------------------------------------ helpers
const normalize = (t) => t.split(/\s+/).filter(Boolean).join(" ");
const countWords = (t) => (t.match(/\S+/g) || []).length;
const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

function fmtRelative(iso) {
  const d = new Date(iso);
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  if (s < 7 * 86400) return `${Math.round(s / 86400)}d ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
const fmtClock = (d) => d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });

let toastTimer;
function toast(message, kind = "info") {
  els.toast.textContent = message;
  els.toast.dataset.kind = kind;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (els.toast.hidden = true), 3200);
}

function confirmDialog(title, body) {
  if (typeof els.confirmDialog.showModal !== "function") return Promise.resolve(window.confirm(`${title}\n\n${body}`));
  els.confirmTitle.textContent = title;
  els.confirmBody.textContent = body;
  els.confirmDialog.returnValue = "";
  els.confirmDialog.showModal();
  return new Promise((resolve) => {
    els.confirmDialog.addEventListener("close", () => resolve(els.confirmDialog.returnValue === "confirm"), { once: true });
  });
}

function setSaveStatus(kind, text) {
  els.saveStatus.dataset.state = kind;
  els.saveStatusText.textContent = text;
  els.saveStatus.title = text;
}

// ------------------------------------------------------------------ document list
const DOC_ICON = '<svg class="icon doc-icon" viewBox="0 0 24 24"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>';
const TRASH_ICON = '<svg class="icon" viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg>';

function renderDocList() {
  els.docList.replaceChildren();
  els.docCount.textContent = state.docs.length;
  if (!state.docs.length) {
    const li = document.createElement("li");
    li.className = "doc-empty";
    li.textContent = "No documents yet.";
    els.docList.append(li);
    return;
  }
  for (const doc of state.docs) {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.tabIndex = 0;
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", String(state.current?.id === doc.id));
    li.dataset.id = doc.id;
    li.innerHTML = `${DOC_ICON}<div class="doc-item-body"><div class="doc-item-title"></div><div class="doc-item-meta"></div></div>
      <button class="icon-btn delete-btn" type="button">${TRASH_ICON}</button>`;
    li.querySelector(".doc-item-title").textContent = doc.title || "Untitled document"; // textContent: no HTML injection
    li.querySelector(".doc-item-meta").textContent = `${plural(doc.word_count, "word")} · ${fmtRelative(doc.updated_at)}`;
    li.querySelector(".delete-btn").setAttribute("aria-label", `Delete ${doc.title}`);
    els.docList.append(li);
  }
}

function upsertSummary(summary) {
  const i = state.docs.findIndex((d) => d.id === summary.id);
  if (i >= 0) state.docs[i] = { ...state.docs[i], ...summary };
  else state.docs.push(summary);
  state.docs.sort((a, b) => (b.updated_at > a.updated_at ? 1 : b.updated_at < a.updated_at ? -1 : b.id - a.id));
  renderDocList();
}

els.docList.addEventListener("click", (e) => {
  const item = e.target.closest(".doc-item");
  if (!item) return;
  const doc = state.docs.find((d) => d.id === Number(item.dataset.id));
  if (!doc) return;
  if (e.target.closest(".delete-btn")) deleteDocument(doc);
  else if (state.current?.id !== doc.id) openDocument(doc.id);
  else closeSidebar();
});
els.docList.addEventListener("keydown", (e) => {
  const item = e.target.closest(".doc-item");
  if (!item || e.target !== item) return;
  const doc = state.docs.find((d) => d.id === Number(item.dataset.id));
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openDocument(doc.id); }
  if (e.key === "Delete") deleteDocument(doc);
});

// ------------------------------------------------------------------ open / create / delete
function applyDocument(doc, { preserveSelection = false } = {}) {
  const selection = preserveSelection ? quill.getSelection() : null;
  state.current = { id: doc.id, title: doc.title, version: doc.version, updated_at: doc.updated_at };
  state.dirty = false;
  state.saveFailed = false;
  hideConflict();

  els.titleInput.value = doc.title;
  document.title = `${doc.title} — ScribeScan`;
  let delta = null;
  try { delta = doc.content ? JSON.parse(doc.content) : null; } catch { /* fall back to plain text */ }
  if (delta) quill.setContents(delta, "silent");
  else quill.setText(doc.plain_text || "", "silent");
  quill.history.clear();
  quill.enable();
  if (selection) quill.setSelection(Math.min(selection.index, quill.getLength() - 1), 0, "silent");

  setSaveStatus("saved", `Saved ${fmtRelative(doc.updated_at)}`);
  localStorage.setItem(LAST_DOC_KEY, String(doc.id));

  if (doc.latest_scan) {
    renderResult(doc.latest_scan.result, new Date(doc.latest_scan.scanned_at), !doc.latest_scan.is_stale);
    state.scannedText = doc.latest_scan.is_stale ? null : normalize(quill.getText());
  } else if (!preserveSelection) {
    resetResult();
  }
  hideScanAlert();
  updateCounts();
  const { id, title, word_count, version, created_at, updated_at } = doc;
  upsertSummary({ id, title, word_count, version, created_at, updated_at }); // also re-renders the list
}

async function openDocument(id, { silent = false, preserveSelection = false } = {}) {
  if (state.current && state.current.id !== id) await flush();
  try {
    const doc = await api.getDocument(id);
    applyDocument(doc, { preserveSelection });
    if (!silent) { closeSidebar(); quill.focus(); }
  } catch (err) {
    toast(err.message, "error");
    if (err.status === 404) { await refreshList(); }
  }
}

async function createDocument() {
  await flush();
  try {
    const doc = await api.createDocument();
    upsertSummary(doc);
    applyDocument({ ...doc, latest_scan: null });
    closeSidebar();
    quill.focus();
  } catch (err) {
    toast(err.message, "error");
  }
}

async function deleteDocument(doc) {
  const ok = await confirmDialog(`Delete “${doc.title}”?`, "This permanently removes the document and its scan history.");
  if (!ok) return;
  const wasCurrent = state.current?.id === doc.id;
  if (wasCurrent) { clearTimeout(debounceTimer); state.current = null; state.dirty = false; }
  try {
    await api.deleteDocument(doc.id);
  } catch (err) {
    if (err.status !== 404) {
      toast(err.message, "error");
      if (wasCurrent) await openDocument(doc.id, { silent: true });
      return;
    }
  }
  state.docs = state.docs.filter((d) => d.id !== doc.id);
  renderDocList();
  toast("Document deleted");
  if (wasCurrent) {
    if (state.docs.length) await openDocument(state.docs[0].id, { silent: true });
    else await createDocument();
  }
}

async function refreshList() {
  try {
    state.docs = await api.listDocuments();
  } catch { return; }
  renderDocList();
  if (state.current && !state.docs.some((d) => d.id === state.current.id) && !state.dirty) {
    toast("This document was deleted on another device.");
    state.current = null;
    if (state.docs.length) await openDocument(state.docs[0].id, { silent: true });
    else await createDocument();
  }
}

// ------------------------------------------------------------------ auto-save
let debounceTimer;

function markDirty() {
  if (!state.current) return;
  state.dirty = true;
  state.editSeq += 1;
  if (!state.conflict) setSaveStatus("dirty", "Editing…");
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => save(), AUTOSAVE_DEBOUNCE_MS);
}

async function save({ keepalive = false } = {}) {
  while (state.saving) await state.saving; // never run two saves at once
  const doc = state.current;
  if (!doc || !state.dirty || state.conflict) return;

  const seq = state.editSeq;
  const payload = {
    title: els.titleInput.value.trim() || "Untitled document",
    content: JSON.stringify(quill.getContents()),
    plain_text: quill.getText(),
    version: doc.version,
  };
  setSaveStatus("saving", "Saving…");

  const run = (async () => {
    try {
      const saved = await api.updateDocument(doc.id, payload, { keepalive });
      Object.assign(doc, { version: saved.version, updated_at: saved.updated_at, title: saved.title });
      if (state.editSeq === seq) state.dirty = false;
      state.saveFailed = false;
      upsertSummary(saved);
      if (state.current === doc) {
        if (state.dirty) setSaveStatus("dirty", "Editing…");
        else setSaveStatus("saved", `Saved ${fmtClock(new Date())}`);
      }
    } catch (err) {
      if (err.status === 409) return showConflict(err.body?.current);
      if (err.status === 404) return recoverDeletedDocument(payload);
      state.saveFailed = true;
      setSaveStatus("error", err.status === 0 ? "Offline — will retry" : "Save failed — will retry");
    }
  })();
  state.saving = run;
  try { await run; } finally { if (state.saving === run) state.saving = null; }
}

async function flush() {
  clearTimeout(debounceTimer);
  if (state.dirty) await save();
  else while (state.saving) await state.saving;
}

// The doc was deleted on another device while we had unsaved edits: keep the user's work.
async function recoverDeletedDocument(payload) {
  try {
    const created = await api.createDocument(`${payload.title} (recovered)`.slice(0, 200));
    const saved = await api.updateDocument(created.id, { ...payload, title: created.title, version: created.version });
    toast("This document was deleted elsewhere — your text was saved as a new copy.");
    upsertSummary(saved);
    state.docs = state.docs.filter((d) => d.id !== state.current?.id);
    state.current = { id: saved.id, title: saved.title, version: saved.version, updated_at: saved.updated_at };
    state.dirty = false;
    els.titleInput.value = saved.title;
    setSaveStatus("saved", `Saved ${fmtClock(new Date())}`);
    renderDocList();
  } catch (err) {
    setSaveStatus("error", "Save failed — will retry");
  }
}

// ------------------------------------------------------------------ conflicts
function showConflict(current) {
  state.conflict = current || null;
  els.conflictBanner.hidden = false;
  setSaveStatus("error", "Edited elsewhere");
}
function hideConflict() {
  state.conflict = null;
  els.conflictBanner.hidden = true;
}
els.conflictLoadBtn.addEventListener("click", async () => {
  const id = state.current?.id;
  hideConflict();
  state.dirty = false;
  if (id) await openDocument(id, { silent: true });
});
els.conflictKeepBtn.addEventListener("click", async () => {
  if (!state.current) return;
  if (state.conflict?.version) state.current.version = state.conflict.version;
  hideConflict();
  state.dirty = true;
  await save();
});

// ------------------------------------------------------------------ multi-device sync
async function syncFromServer() {
  if (document.hidden || state.saving || state.conflict) return;
  let docs;
  try { docs = await api.listDocuments(); } catch { return; }
  const local = state.current;
  // keep the live (possibly unsaved) title of the open document
  state.docs = docs.map((d) => (local && d.id === local.id && state.dirty ? { ...d, title: els.titleInput.value } : d));
  renderDocList();
  if (!local) return;
  const remote = docs.find((d) => d.id === local.id);
  if (!remote) return refreshList();
  if (remote.version > local.version && !state.dirty && !state.saving) {
    await openDocument(local.id, { silent: true, preserveSelection: true });
  }
}

// ------------------------------------------------------------------ counts & guardrail UI
let countsFrame = 0;
function scheduleCounts() {
  if (!countsFrame) countsFrame = requestAnimationFrame(() => { countsFrame = 0; updateCounts(); });
}

function updateCounts() {
  const text = quill.getText();
  const words = countWords(text);
  const chars = text.replace(/\n$/, "").length;
  els.wordCount.textContent = plural(words, "word");
  els.charCount.textContent = plural(chars, "character");

  const min = state.config.min_words;
  const ok = words >= min;
  els.minWordsBar.style.width = `${Math.min(100, (words / min) * 100)}%`;
  els.minWordsText.textContent = ok ? `${words.toLocaleString()} words ✓` : `${words.toLocaleString()} / ${min} words`;
  els.minWordsWrap.dataset.ok = String(ok);
  if (ok && els.scanAlert.dataset.code === "too_short") hideScanAlert();

  els.stalePill.hidden = !state.lastResult || state.scanning || state.scannedText === normalize(text);
}

// ------------------------------------------------------------------ scan results
const TIER_TEXT = {
  human: "AI likelihood is under 20% — the exact score is withheld.",
  mixed: "Some sections read as machine-generated.",
  likely_ai: "Most of the text reads as machine-generated.",
  ai: "The text strongly matches AI-generated patterns.",
};
const TIER_COLOR = { human: "var(--human)", mixed: "var(--mixed)", likely_ai: "var(--likely)", ai: "var(--ai)" };
const CHECK_SVG = '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';

function tierFor(p) {
  if (p < state.config.human_threshold) return "human";
  if (p >= 0.8) return "ai";
  if (p >= 0.5) return "likely_ai";
  return "mixed";
}

function setGauge(pct) {
  els.gaugeValue.style.strokeDashoffset = String(100 - Math.max(0, Math.min(100, pct)));
}

let countUpFrame = 0;
function countUp(target) {
  cancelAnimationFrame(countUpFrame);
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / 900);
    const eased = 1 - Math.pow(1 - t, 3);
    els.gaugeNumber.textContent = `${Math.round(target * eased)}%`;
    if (t < 1) countUpFrame = requestAnimationFrame(step);
  };
  countUpFrame = requestAnimationFrame(step);
}

function resetResult() {
  state.lastResult = null;
  state.scannedText = null;
  cancelAnimationFrame(countUpFrame);
  els.gauge.dataset.tier = "empty";
  els.verdict.dataset.tier = "empty";
  setGauge(0);
  els.gaugeNumber.textContent = "—";
  els.gaugeCaption.textContent = "AI likelihood";
  els.verdictLabel.textContent = "Not scanned yet";
  els.verdictSub.textContent = `Write at least ${state.config.min_words} words, then run a scan.`;
  els.chunksWrap.hidden = true;
  els.scanMeta.hidden = true;
  els.stalePill.hidden = true;
  insights.clear();
}

// A whole-document score can hide that the text is mixed, so say so when sections disagree.
function verdictNote(r, tier) {
  const sections = r.sections || [];
  const total = sections.reduce((n, s) => n + s.word_count, 0);
  const human = sections.filter((s) => s.tier === "human").reduce((n, s) => n + s.word_count, 0);
  const humanShare = total ? Math.round((human / total) * 100) : 0;
  if (humanShare >= 15) return `Mixed: about ${humanShare}% of the text reads as human-written. See the sections below.`;
  if (r.partial) return "Part of this text reads as AI-generated. See the highlighted sections.";
  return TIER_TEXT[tier] || "";
}

// offsetsValid: the editor text is exactly what was scanned, so section highlights line up.
function renderResult(r, scannedAt = new Date(), offsetsValid = true) {
  state.lastResult = { r, scannedAt };
  const tier = r.verdict;
  els.gauge.dataset.tier = tier;
  els.verdict.dataset.tier = tier;
  els.verdictLabel.textContent = r.label;
  els.verdictSub.textContent = verdictNote(r, tier);
  insights.render(r, offsetsValid);

  if (tier === "human" || r.ai_probability == null) {
    // Server withholds the number below the threshold; show only the "human zone".
    cancelAnimationFrame(countUpFrame);
    setGauge((r.threshold ?? state.config.human_threshold) * 100);
    els.gaugeNumber.innerHTML = CHECK_SVG;
    els.gaugeCaption.textContent = `Below ${Math.round((r.threshold ?? 0.2) * 100)}%`;
    els.chunksWrap.hidden = true;
  } else {
    const pct = Math.round(r.ai_probability * 100);
    setGauge(pct);
    countUp(pct);
    els.gaugeCaption.textContent = "AI likelihood";
  }

  // Per-model agreement (only present when the score is >= the human threshold).
  const modelRows = Object.entries(r.model_scores || {}).map(([name, p]) => [name, `${Math.round(p * 100)}%`]);
  const rows = [
    ...modelRows,
    ["Words", r.word_count?.toLocaleString()],
    ["Sections", r.sections ? r.sections.length : "—"],
    ["Hardware", r.device ? (r.device.startsWith("cuda") ? "GPU (CUDA)" : "CPU") : "—"],
    ["Inference time", r.elapsed_ms != null ? `${r.elapsed_ms.toLocaleString()} ms` : "—"],
    ["Scanned", `${fmtClock(scannedAt)}`],
  ];
  els.scanMeta.replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v ?? "—";
    return [dt, dd];
  }));
  els.scanMeta.hidden = false;
}

function showScanAlert(message, kind = "warn", code = "") {
  els.scanAlert.textContent = message;
  els.scanAlert.className = `alert alert-${kind}`;
  els.scanAlert.dataset.code = code;
  els.scanAlert.hidden = false;
}
function hideScanAlert() {
  els.scanAlert.hidden = true;
  els.scanAlert.dataset.code = "";
}

function setScanning(on) {
  state.scanning = on;
  els.scanBtn.disabled = on;
  els.scanBtn.classList.toggle("is-busy", on);
  els.scanBtnText.textContent = on ? "Scanning…" : "Scan for AI";
  if (on) {
    cancelAnimationFrame(countUpFrame);
    els.gauge.dataset.tier = "scanning";
    els.verdict.dataset.tier = "scanning";
    els.gaugeNumber.textContent = "…";
    els.verdictLabel.textContent = "Analyzing…";
    els.verdictSub.textContent = "Scoring overlapping sections with the detector models.";
    els.stalePill.hidden = true;
    insights.clear();
  }
}

async function runScan() {
  if (state.scanning || !state.current) return;
  const text = quill.getText();
  const words = countWords(text);
  const min = state.config.min_words;

  // Guardrail 1 (client side; the server enforces it again).
  if (words < min) {
    showScanAlert(`Add ${min - words} more word${min - words === 1 ? "" : "s"} to scan — at least ${min} words are needed for a reliable result.`, "warn", "too_short");
    return;
  }
  if (state.model.state !== "ready") {
    showScanAlert(state.model.state === "error"
      ? `The detection model failed to load: ${state.model.error}`
      : "The detection model is still loading — try again in a few seconds.", state.model.state === "error" ? "error" : "warn");
    if (state.model.state !== "error") pollHealth();
    return;
  }

  hideScanAlert();
  setScanning(true);
  const docId = state.current.id;
  const previous = state.lastResult;
  try {
    const result = await api.scan(text, docId);
    setScanning(false);
    if (state.current?.id !== docId) return; // user switched documents meanwhile
    renderResult(result);
    state.scannedText = normalize(text);
  } catch (err) {
    setScanning(false);
    if (previous) renderResult(previous.r, previous.scannedAt, state.scannedText === normalize(text)); else resetResult();
    const code = err.body?.code || "";
    showScanAlert(err.message, code === "too_short" || code === "model_not_ready" ? "warn" : "error", code);
  } finally {
    updateCounts();
  }
}

// ------------------------------------------------------------------ model status
async function pollHealth() {
  try {
    const h = await api.health();
    state.config = h;
    state.model = h.model;
    renderModelBadge();
    els.howMin.textContent = h.min_words;
    els.howChunk.textContent = h.chunk_words;
    els.howThreshold.textContent = `${Math.round(h.human_threshold * 100)}%`;
    const n = (h.model.models || []).length;
    $("howModels").textContent = n > 1 ? `${n} detector models (${h.model.name})` : `the ${h.model.name} detector`;
    $("howEnsemble").textContent = n < 2 ? "" : h.model.ensemble === "min"
      ? "a window only counts as AI if every model agrees"
      : "their scores are averaged";
    if (!state.lastResult) els.verdictSub.textContent = `Write at least ${h.min_words} words, then run a scan.`;
    updateCounts();
    if (["idle", "loading"].includes(h.model.state)) setTimeout(pollHealth, 2000);
  } catch {
    setTimeout(pollHealth, 4000);
  }
}

function renderModelBadge() {
  const m = state.model;
  const b = els.modelBadge;
  if (m.state === "ready") {
    const gpu = (m.device || "").startsWith("cuda");
    b.dataset.state = gpu ? "ready" : "cpu";
    els.modelBadgeText.textContent = gpu ? "Detector ready · GPU" : "Detector ready · CPU";
  } else if (m.state === "error") {
    b.dataset.state = "error";
    els.modelBadgeText.textContent = "Model failed to load";
  } else {
    b.dataset.state = "loading";
    els.modelBadgeText.textContent = "Loading detector model…";
  }
  b.title = [m.name, m.device, m.error].filter(Boolean).join(" · ");
}

// ------------------------------------------------------------------ sidebar (mobile)
function closeSidebar() { els.app.classList.remove("sidebar-open"); els.scrim.hidden = true; }
els.menuBtn.addEventListener("click", () => { els.app.classList.add("sidebar-open"); els.scrim.hidden = false; });
els.scrim.addEventListener("click", closeSidebar);

// ------------------------------------------------------------------ wiring
quill.on("text-change", (_delta, _old, source) => {
  if (source === "user") markDirty();
  scheduleCounts();
});

els.titleInput.addEventListener("input", () => {
  markDirty();
  const title = els.titleInput.value.trim() || "Untitled document";
  document.title = `${title} — ScribeScan`;
  const entry = state.docs.find((d) => d.id === state.current?.id);
  if (entry) { entry.title = title; renderDocList(); }
});
els.titleInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); quill.focus(); }
});

els.newDocBtn.addEventListener("click", createDocument);
els.scanBtn.addEventListener("click", runScan);
// Phones: the scan panel sits below the editor, so jump to it and scan.
$("mobileScanBtn").addEventListener("click", () => {
  document.querySelector(".scan-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  runScan();
});

document.addEventListener("keydown", (e) => {
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key.toLowerCase() === "s") {
    e.preventDefault();
    if (state.dirty) save(); else toast("All changes saved");
  } else if (mod && e.key === "Enter") {
    e.preventDefault();
    runScan();
  }
});

// Save whatever is pending when the tab is hidden or closed (keepalive lets it outlive the page).
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { if (state.dirty) save({ keepalive: true }); }
  else syncFromServer();
});
window.addEventListener("pagehide", () => { if (state.dirty) save({ keepalive: true }); });
window.addEventListener("beforeunload", (e) => {
  if (state.dirty && state.saveFailed) { e.preventDefault(); e.returnValue = ""; }
});

setInterval(() => { if (state.dirty && !state.saving) save(); }, AUTOSAVE_INTERVAL_MS);
setInterval(syncFromServer, SYNC_INTERVAL_MS);
setInterval(() => { if (state.docs.length) renderDocList(); }, 60000); // refresh "5m ago" labels

// ------------------------------------------------------------------ boot
async function init() {
  pollHealth();
  try {
    state.docs = await api.listDocuments();
  } catch (err) {
    setSaveStatus("error", "Server unreachable");
    toast(err.message, "error");
    return;
  }
  renderDocList();
  const lastId = Number(localStorage.getItem(LAST_DOC_KEY));
  const target = state.docs.find((d) => d.id === lastId) || state.docs[0];
  if (target) await openDocument(target.id, { silent: true });
  else await createDocument();
}

init();

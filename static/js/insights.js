// Section-level results: in-editor highlights, per-section reasons + revision tips, text analytics.
//
// Highlights use the CSS Custom Highlight API, which paints ranges without touching the
// document's DOM or the saved content. Browsers without it still get the section list.

const TIERS = {
  human: { label: "Human-like", color: "var(--human)", highlight: null },
  mixed: { label: "Possibly AI", color: "var(--mixed)", highlight: "ss-mixed" },
  likely_ai: { label: "Likely AI", color: "var(--likely)", highlight: "ss-likely" },
  ai: { label: "AI-like", color: "var(--ai)", highlight: "ss-ai" },
};
const HIGHLIGHT_NAMES = ["ss-mixed", "ss-likely", "ss-ai", "ss-active"];
const supportsHighlights = typeof CSS !== "undefined" && !!CSS.highlights && typeof Highlight !== "undefined";

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
};
const pct = (p) => `${Math.round(p * 100)}%`;

export function createInsights(quill) {
  const ui = {
    sectionsCard: $("sectionsCard"), list: $("sectionList"), share: $("tierShare"), toggle: $("hlToggle"),
    toggleWrap: $("hlToggleWrap"), staleNote: $("sectionsStale"),
    stripWrap: $("chunksWrap"), strip: $("chunkStrip"), stripCount: $("chunksCount"),
    analyticsCard: $("analyticsCard"), stats: $("statsGrid"),
  };
  let sections = [];
  let ranges = new Map(); // section index -> live DOM Range
  let active = null;
  let valid = false; // do the character offsets still match the editor text?

  // ---------------------------------------------------------------- highlights
  function rangeFor(start, end) {
    try {
      const [a, aOff] = quill.getLeaf(start);
      const [b, bOff] = quill.getLeaf(end);
      if (!a || !b) return null;
      const range = document.createRange();
      range.setStart(a.domNode, Math.min(aOff, a.domNode.length ?? 0));
      range.setEnd(b.domNode, Math.min(bOff, b.domNode.length ?? 0));
      return range;
    } catch {
      return null;
    }
  }

  function clearPaint() {
    if (supportsHighlights) HIGHLIGHT_NAMES.forEach((n) => CSS.highlights.delete(n));
  }

  function paint() {
    clearPaint();
    if (!supportsHighlights || !valid || !ui.toggle.checked) return;
    const groups = {};
    for (const s of sections) {
      const name = TIERS[s.tier]?.highlight;
      const range = ranges.get(s.index);
      if (name && range) (groups[name] ||= []).push(range);
    }
    for (const [name, list] of Object.entries(groups)) CSS.highlights.set(name, new Highlight(...list));
    const activeRange = active != null && ranges.get(active);
    if (activeRange) CSS.highlights.set("ss-active", new Highlight(activeRange));
  }

  // ---------------------------------------------------------------- selection
  function select(index, { scrollEditor = false, scrollList = false } = {}) {
    active = active === index && !scrollEditor && !scrollList ? null : index;
    for (const li of ui.list.children) {
      const on = Number(li.dataset.index) === active;
      li.classList.toggle("is-open", on);
      li.querySelector(".section-head").setAttribute("aria-expanded", String(on));
      li.querySelector(".section-body").hidden = !on;
      if (on && scrollList) li.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    for (const seg of ui.strip.children) seg.classList.toggle("is-active", Number(seg.dataset.index) === active);
    paint();
    const range = active != null && ranges.get(active);
    if (scrollEditor && range) {
      const node = range.startContainer.nodeType === 1 ? range.startContainer : range.startContainer.parentElement;
      node?.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }

  // Clicking highlighted text in the editor opens that section's notes.
  quill.on("selection-change", (range, _old, source) => {
    if (source !== "user" || !range || range.length !== 0 || !valid || !ui.toggle.checked) return;
    const sel = window.getSelection();
    if (!sel || !sel.anchorNode) return;
    for (const s of sections) {
      const r = ranges.get(s.index);
      if (s.tier !== "human" && r && r.isPointInRange(sel.anchorNode, sel.anchorOffset)) {
        if (active !== s.index) select(s.index, { scrollList: true });
        return;
      }
    }
  });
  ui.toggle.addEventListener("change", paint);

  // ---------------------------------------------------------------- rendering
  function sectionBody(s) {
    const body = el("div", "section-body");
    body.hidden = true;
    if (s.tier === "human") {
      body.append(el("p", "section-ok", "This part reads as human-written. Nothing to change."));
      return body;
    }
    if (s.signals?.length) {
      for (const sig of s.signals) {
        const item = el("div", "signal");
        item.append(el("div", "signal-title", sig.title), el("div", "signal-detail", sig.detail));
        const tip = el("div", "signal-tip");
        tip.append(el("span", "tip-label", "Try"), document.createTextNode(sig.tip));
        item.append(tip);
        body.append(item);
      }
    } else {
      const [lead, ...tips] = s.tips || [];
      if (lead) body.append(el("p", "signal-detail", lead));
      for (const t of tips) {
        const tip = el("div", "signal-tip");
        tip.append(el("span", "tip-label", "Try"), document.createTextNode(t));
        body.append(tip);
      }
    }
    return body;
  }

  function renderList() {
    ui.list.replaceChildren();
    sections.forEach((s, n) => {
      const tier = TIERS[s.tier] || TIERS.mixed;
      const li = el("li", "section-item");
      li.dataset.index = s.index;
      li.dataset.tier = s.tier;
      const head = el("button", "section-head");
      head.type = "button";
      head.setAttribute("aria-expanded", "false");
      const badge = el("span", "section-pct", pct(s.ai_probability));
      const text = el("span", "section-text");
      const snippet = valid ? quill.getText(s.start_char, Math.min(64, s.end_char - s.start_char)).replace(/\s+/g, " ").trim() : "";
      text.append(
        el("span", "section-snippet", snippet ? `${snippet}…` : `Section ${n + 1}`),
        el("span", "section-meta", `${tier.label} · ${s.word_count} words${s.signals?.length ? ` · ${s.signals.length} note${s.signals.length === 1 ? "" : "s"}` : ""}`),
      );
      head.append(badge, text);
      head.addEventListener("click", () => select(s.index, { scrollEditor: active !== s.index }));
      li.append(head, sectionBody(s));
      ui.list.append(li);
    });
  }

  function renderStrip() {
    ui.strip.replaceChildren();
    for (const s of sections) {
      const seg = el("div", "chunk-seg");
      seg.dataset.index = s.index;
      seg.style.flexGrow = String(s.word_count);
      seg.style.background = (TIERS[s.tier] || TIERS.mixed).color;
      seg.style.opacity = String(0.4 + 0.6 * s.ai_probability);
      seg.title = `${pct(s.ai_probability)} AI · ${s.word_count} words`;
      seg.addEventListener("click", () => select(s.index, { scrollEditor: true, scrollList: true }));
      ui.strip.append(seg);
    }
    ui.stripCount.textContent = `${sections.length} section${sections.length === 1 ? "" : "s"}`;
    ui.stripWrap.hidden = sections.length === 0;
  }

  function renderShare() {
    const total = sections.reduce((n, s) => n + s.word_count, 0) || 1;
    const words = {};
    for (const s of sections) words[s.tier] = (words[s.tier] || 0) + s.word_count;
    ui.share.replaceChildren();
    for (const key of ["ai", "likely_ai", "mixed", "human"]) {
      if (!words[key]) continue;
      const item = el("span", "share-item");
      const dot = el("span", "share-dot");
      dot.style.background = TIERS[key].color;
      item.append(dot, document.createTextNode(`${Math.round((words[key] / total) * 100)}% ${TIERS[key].label.toLowerCase()}`));
      ui.share.append(item);
    }
  }

  function renderAnalytics(a) {
    if (!a) { ui.analyticsCard.hidden = true; return; }
    const variation = a.sentence_variation < 0.4 ? "Very even rhythm" : a.sentence_variation < 0.6 ? "Moderate variety" : "Varied, natural rhythm";
    const vocab = a.vocabulary_variety < 0.72 ? "Repetitive wording" : a.vocabulary_variety < 0.8 ? "Typical range" : "Rich vocabulary";
    const ease = a.flesch_reading_ease >= 80 ? "Very easy to read" : a.flesch_reading_ease >= 60 ? "Plain English"
      : a.flesch_reading_ease >= 40 ? "Fairly difficult" : "Difficult";
    const rows = [
      ["Sentences", a.sentences, `${a.paragraphs} paragraph${a.paragraphs === 1 ? "" : "s"}`],
      ["Reading time", `${a.reading_time_min} min`, `${a.words.toLocaleString()} words`],
      ["Avg sentence", `${a.avg_sentence_words} words`, `Shortest ${a.shortest_sentence_words}, longest ${a.longest_sentence_words}`],
      ["Sentence variation", a.sentence_variation.toFixed(2), variation],
      ["Vocabulary variety", `${Math.round(a.vocabulary_variety * 100)}%`, vocab],
      ["Reading ease", a.flesch_reading_ease, ease],
      ["Contractions", `${a.contractions_per_100} / 100 words`, a.contractions_per_100 < 0.3 ? "Formal tone" : "Conversational tone"],
      ["Connector openers", a.transition_openers, "Sentences starting However, Moreover…"],
      ["Stock phrases", a.stock_phrases, a.stock_phrase_examples?.length ? a.stock_phrase_examples.slice(0, 3).join(", ") : "None found"],
    ];
    ui.stats.replaceChildren(...rows.map(([label, value, hint]) => {
      const cell = el("div", "stat");
      cell.append(el("dt", null, label), el("dd", null, String(value)), el("div", "stat-hint", hint));
      return cell;
    }));
    ui.analyticsCard.hidden = false;
  }

  // ---------------------------------------------------------------- public API
  function clear() {
    sections = [];
    ranges = new Map();
    active = null;
    valid = false;
    clearPaint();
    ui.sectionsCard.hidden = true;
    ui.analyticsCard.hidden = true;
    ui.stripWrap.hidden = true;
  }

  /** @param result scan result  @param offsetsValid whether the editor text is exactly what was scanned */
  function render(result, offsetsValid) {
    clear();
    renderAnalytics(result.analytics);
    sections = result.sections || [];
    if (!sections.length) return;
    valid = offsetsValid;
    if (valid) for (const s of sections) {
      const r = rangeFor(s.start_char, s.end_char);
      if (r) ranges.set(s.index, r);
    }
    ui.toggleWrap.hidden = !valid || !supportsHighlights;
    ui.staleNote.hidden = valid;
    renderStrip();
    renderShare();
    renderList();
    ui.sectionsCard.hidden = false;
    paint();
    const first = sections.filter((s) => s.tier !== "human").sort((a, b) => b.ai_probability - a.ai_probability)[0];
    if (first) select(first.index);
  }

  return { render, clear };
}

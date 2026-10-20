// Shared by fragment.html (review cards) and the PDF report.
const KATEX_MACROS = {
  "\\R": "\\mathbb{R}", "\\N": "\\mathbb{N}", "\\Z": "\\mathbb{Z}",
  "\\Q": "\\mathbb{Q}", "\\C": "\\mathbb{C}"
};

function renderMathIn(root) {
  root.querySelectorAll(".m").forEach(function (el) {
    const tex = el.textContent;
    try {
      katex.render(tex, el, {
        displayMode: el.dataset.d === "1",
        throwOnError: false,   // bad LaTeX is shown (in red) instead of crashing
        strict: "ignore",
        trust: false,
        macros: Object.assign({}, KATEX_MACROS)
      });
    } catch (e) {
      el.textContent = tex;
      el.classList.add("bad");
    }
  });
  fitWide(root);
}

// Shrink anything wider than its container (big matrices, long equations).
function fitWide(root) {
  const avail = root.clientWidth;
  if (!avail) return;
  root.querySelectorAll(".m .katex-html").forEach(function (h) {
    const w = h.getBoundingClientRect().width;
    if (w > avail) {
      const k = h.closest(".katex");
      k.style.zoom = String((avail / w) * 0.98);
    }
  });
}

async function renderFragment(html, fontPx) {
  document.body.style.fontSize = fontPx + "px";
  const root = document.getElementById("root");
  root.innerHTML = html;           // html is produced (and escaped) by math_render.to_html
  renderMathIn(root);
  await document.fonts.ready;
  fitWide(root);                   // font metrics can change once the fonts have loaded
}

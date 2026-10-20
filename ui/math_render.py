"""Typeset LaTeX with KaTeX inside one headless Chromium (Playwright).

* render_async()  - PNG of a text fragment (used by the review cards)
* render_pdf()    - HTML file -> PDF (used by the student reports)

Everything here is Kivy-free. KaTeX itself is downloaded once into ui/web/katex/
the first time it is needed (or run `python -m ui.math_render` to do it up front).

One-time setup:
    pip install playwright
    playwright install chromium
"""
import hashlib
import html
import importlib.util
import io
import queue
import re
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"
KATEX_DIR = WEB_DIR / "katex"
KATEX_VERSION = "0.16.11"
KATEX_URL = f"https://registry.npmjs.org/katex/-/katex-{KATEX_VERSION}.tgz"
CACHE_DIR = Path(tempfile.gettempdir()) / "evaluator_math_cache"
SCALE = 2  # PNGs are rendered at 2x so equations stay crisp on high-DPI screens

SETUP_HINT = "pip install playwright  then  playwright install chromium"

# ---------------------------------------------------------------- KaTeX files
_katex_lock = threading.Lock()


def ensure_katex():
    """Download katex.min.js / katex.min.css / fonts once. Raises on network failure."""
    with _katex_lock:
        if (KATEX_DIR / "katex.min.js").is_file() and (KATEX_DIR / "katex.min.css").is_file():
            return
        KATEX_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(KATEX_URL, timeout=60) as resp:
            data = resp.read()
        wanted = ("package/dist/katex.min.js", "package/dist/katex.min.css")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                is_font = name.startswith("package/dist/fonts/") and name.endswith(".woff2")
                if name in wanted or is_font:
                    dest = KATEX_DIR / name[len("package/dist/"):]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tar.extractfile(member).read())


# --------------------------------------------------- text -> safe HTML + math
# Math spans we recognise (the model is free to use any of them):
#   $$...$$   \[...\]   \(...\)   $...$   and BARE \begin{env}...\end{env}
# The $...$ rule is conservative so "costs $5 and $10" is not treated as math.
_ENVS = r"[A-Za-z]+\*?"
_MATH_RE = re.compile(
    "|".join([
        r"\$\$(?P<dd>.+?)\$\$",
        r"\\\[(?P<br>.+?)\\\]",
        r"\\\((?P<pa>.+?)\\\)",
        r"(?<!\\)\$(?!\s)(?P<d>[^$\n]+?)(?<!\s)\$(?!\d)",
        r"(?P<bare>\\begin\{(?P<env>" + _ENVS + r")\}.*?\\end\{(?P=env)\})",
    ]),
    re.S,
)


def _tex_of(match):
    """(latex source, is_display) for one regex match."""
    g = match.groupdict()
    if g["dd"] is not None:
        return g["dd"].strip(), True
    if g["br"] is not None:
        return g["br"].strip(), True
    if g["pa"] is not None:
        return g["pa"].strip(), False
    if g["d"] is not None:
        return g["d"].strip(), False
    return g["bare"].strip(), True  # bare environment = display math


def has_math(text):
    return bool(text) and _MATH_RE.search(text) is not None


def to_html(text):
    """Escape `text` and turn every math span into <span class="m" data-d="0|1">latex</span>.

    Everything (including the latex) is HTML-escaped, so model output can never inject markup.
    """
    text = text or ""
    out, pos = [], 0
    for m in _MATH_RE.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        tex, display = _tex_of(m)
        out.append(f'<span class="m" data-d="{1 if display else 0}">{html.escape(tex)}</span>')
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def plain_text(text):
    """Fallback / loading text: the same string with the math delimiters removed."""
    return _MATH_RE.sub(lambda m: _tex_of(m)[0], text or "")


# ------------------------------------------------------------------ renderer
class MathRenderer:
    """One background thread owns the browser (Playwright's sync API is thread-bound)."""

    def __init__(self):
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._thread = None
        self.error = None  # set once startup fails (no playwright / chromium / network)

    def available(self):
        return self.error is None and importlib.util.find_spec("playwright") is not None

    # ---- worker ----
    def _start(self):
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def _run(self):
        browser = page = None
        try:
            ensure_katex()
            from playwright.sync_api import sync_playwright

            pw = sync_playwright().start()
            browser = pw.chromium.launch()
            page = browser.new_page(device_scale_factor=SCALE)
            page.goto((WEB_DIR / "fragment.html").as_uri())
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if "Executable doesn't exist" in msg or "playwright install" in msg:
                msg = f"Chromium is not installed ({SETUP_HINT})"
            self.error = msg
        while True:
            fn, callback = self._q.get()
            if self.error:
                callback(None, RuntimeError(self.error))
                continue
            try:
                result, err = fn(browser, page), None
            except Exception as e:  # one bad fragment must not kill the worker
                result, err = None, e
            callback(result, err)

    def submit(self, fn, callback):
        self._start()
        self._q.put((fn, callback))

    # ---- PNG fragments ----
    def render_async(self, text, width_px, font_px, callback):
        """Render `text` to a PNG `width_px` wide. callback(path, error) runs on the worker
        thread (or immediately, if the PNG is already cached)."""
        width_px = int(max(60, width_px))
        key = hashlib.sha1(f"{text}|{width_px}|{font_px}|v1".encode("utf-8")).hexdigest()
        out = CACHE_DIR / f"{key}.png"
        if out.is_file():
            callback(str(out), None)
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        body = to_html(text)

        def job(browser, page):
            page.set_viewport_size({"width": width_px, "height": 100})
            page.evaluate("([h, px]) => renderFragment(h, px)", [body, int(font_px)])
            tmp = out.with_suffix(".tmp.png")
            page.locator("#root").screenshot(path=str(tmp), omit_background=True)
            tmp.replace(out)
            return str(out)

        self.submit(job, callback)

    # ---- PDF ----
    def render_pdf(self, html_file, pdf_file, timeout=180):
        """Blocking: print an HTML file (which loads katex/ + render.js) to a PDF."""

        def job(browser, _page):
            p = browser.new_page()
            try:
                p.goto(Path(html_file).as_uri())
                p.wait_for_load_state("load")
                p.evaluate("document.fonts.ready.then(() => true)")
                p.pdf(
                    path=str(pdf_file), format="A4", print_background=True,
                    margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
                )
            finally:
                p.close()

        done, box = threading.Event(), {}

        def callback(_result, err):
            box["err"] = err
            done.set()

        self.submit(job, callback)
        if not done.wait(timeout):
            raise TimeoutError("PDF rendering timed out")
        if box["err"]:
            raise box["err"]


renderer = MathRenderer()


if __name__ == "__main__":  # python -m ui.math_render  -> pre-download KaTeX
    ensure_katex()
    print("KaTeX ready in", KATEX_DIR)

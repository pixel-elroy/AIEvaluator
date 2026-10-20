"""OCR engine: page processing, the model client, and transcription backends.

No Kivy imports, so everything here can run from a terminal or a worker thread.

Sections
  1. Page loading + pre-processing (deskew, enhance, binarize)
  2. LLMClient: talks to any OpenAI-compatible server (Gemini via AI Studio, Ollama, ...)
     with rate limiting, retries and cancellation
  3. Transcription backends (PaddleOCR, or the vision model) and student-ID reading
"""
import base64
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

try:
    import numpy as np
except ImportError:  # imported lazily by the functions that need it
    np = None


class PageError(Exception):
    """A page image or PDF could not be read."""


def safe_name(text):
    """File-system safe version of a label (keeps letters, digits, _ and -)."""
    cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(text))
    return cleaned.strip("_") or "item"


# =====================================================================
# 1. Page loading + pre-processing
#    (needs opencv-python, numpy, PyMuPDF; imported lazily)
# =====================================================================
MAX_SIDE = 3200  # cap huge scans so the viewer stays responsive
PDF_DPI = 220    # sharp PDF rendering: thin pen strokes vanish at low DPI


def _read_image(path):
    import cv2

    data = np.fromfile(path, dtype=np.uint8)  # works with unicode paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise PageError(f"Could not read image: {os.path.basename(path)}")
    return img


def _pixmap_to_bgr(pix):
    import cv2

    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _text_pages(path, lines_per_page=55):
    import fitz

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines() or [""]
    for i in range(0, len(lines), lines_per_page):
        doc = fitz.open()
        page = doc.new_page()  # A4
        rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
        page.insert_textbox(rect, "\n".join(lines[i:i + lines_per_page]), fontsize=11)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), colorspace=fitz.csRGB, alpha=False)
        yield _pixmap_to_bgr(pix)
        doc.close()


def load_pages(sources, dpi=PDF_DPI):
    for src in sources:
        low = src.lower()
        if low.endswith(".pdf"):
            import fitz

            zoom = dpi / 72
            with fitz.open(src) as doc:
                for page in doc:
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False
                    )
                    yield _pixmap_to_bgr(pix)
        elif low.endswith(".txt"):
            yield from _text_pages(src)
        else:
            yield _read_image(src)


def _limit_size(img):
    import cv2

    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def enhance(gray):
    """Flatten uneven lighting and shadows WITHOUT thresholding, so every stroke
    (including faint pencil and thin pen lines) is kept."""
    import cv2

    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    small = cv2.dilate(small, np.ones((5, 5), np.uint8))  # erase ink, keep paper tone
    small = cv2.medianBlur(small, 21)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = np.maximum(bg, 1)
    return cv2.divide(gray, bg, scale=255)  # paper -> white, ink stays as dark as it was


def deskew(gray):
    """Find the rotation (-5..5 deg) that makes text rows sharpest, then apply it."""
    import cv2

    h, w = gray.shape
    scale = 600 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    sh, sw = bw.shape
    center = (sw / 2, sh / 2)

    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-5.0, 5.01, 0.5):
        m = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rot = cv2.warpAffine(bw, m, (sw, sh), flags=cv2.INTER_NEAREST)
        proj = rot.sum(axis=1, dtype=np.float64)
        score = float(np.sum(np.diff(proj) ** 2))
        if score > best_score:
            best_angle, best_score = float(angle), score

    if abs(best_angle) < 0.1:
        return gray
    m = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def binarize(flat_gray):
    """Otsu threshold on an already-flattened page, then drop isolated specks.

    No blur (it erases thin strokes) and no artificially low threshold (it thins pen
    strokes). Tiny 1-3 pixel specks from paper noise are removed afterwards instead."""
    import cv2

    otsu, _ = cv2.threshold(flat_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    t = min(otsu, 240)
    binary = cv2.threshold(flat_gray, t, 255, cv2.THRESH_BINARY)[1]

    ink = (binary == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    tiny = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] <= 3)
    tiny = tiny[tiny != 0]  # label 0 is the background
    if len(tiny):
        binary[np.isin(labels, tiny)] = 255
    return binary


def preprocess(img_bgr):
    """Returns (enhanced_gray, binarized)."""
    import cv2

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = deskew(enhance(gray))
    return enhanced, binarize(enhanced)


def process_document(sources, out_dir, prefix):
    """Returns [(original_jpg, enhanced_png, binarized_png), ...] for every page.
    Raises PageError if a file cannot be read."""
    try:
        import cv2
    except ImportError as e:
        raise PageError("OpenCV is not installed (pip install opencv-python-headless).") from e
    if np is None:
        raise PageError("numpy is not installed.")
    safe = safe_name(prefix)
    pages = []
    try:
        for i, img in enumerate(load_pages(sources), start=1):
            img = _limit_size(img)
            orig_path = os.path.join(out_dir, f"{safe}_p{i}_orig.jpg")
            enh_path = os.path.join(out_dir, f"{safe}_p{i}_enh.png")
            bin_path = os.path.join(out_dir, f"{safe}_p{i}_bin.png")
            cv2.imwrite(orig_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            enhanced, binary = preprocess(img)
            cv2.imwrite(enh_path, enhanced)
            cv2.imwrite(bin_path, binary)
            pages.append((orig_path, enh_path, bin_path))
    except PageError:
        raise
    except ImportError as e:
        raise PageError(f"A required library is missing ({e}). Install PyMuPDF for PDFs.") from e
    except Exception as e:  # corrupt PDF, truncated image, ...
        raise PageError(f"Could not read the pages ({type(e).__name__}: {e}).") from e
    if not pages:
        raise PageError("No pages were found in the file.")
    return pages


# =====================================================================
# 2. Model client (OpenAI-compatible chat/completions)
# =====================================================================
class LLMError(Exception):
    """Any failure talking to the model server."""


class AuthError(LLMError):
    """The server rejected the API key."""


class QuotaExceeded(LLMError):
    """Rate or daily quota exhausted even after retrying."""


class CallBudgetExceeded(LLMError):
    """This script has already used all the model calls it is allowed (see LLMClient.begin_budget)."""


class Cancelled(Exception):
    """The user cancelled the run."""


IMAGE_TOKENS = 2048  # rough image-token estimate for compatible local servers

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
# Cloud endpoints the app may talk to. Anything else is refused, so a typo or a pasted URL can
# never send student data to an unexpected server.
CLOUD_HOSTS = ("generativelanguage.googleapis.com",)


def endpoint_kind(base_url):
    """'local', 'cloud' (an allowed cloud host over https) or None (not permitted)."""
    try:
        parsed = urlparse((base_url or "").strip())
        host = (parsed.hostname or "").lower()
    except ValueError:
        return None
    if parsed.scheme in ("http", "https") and host in LOCAL_HOSTS:
        return "local"
    if parsed.scheme == "https" and host in CLOUD_HOSTS:
        return "cloud"
    return None

# Use an identifiable user agent when talking to the local OpenAI-compatible server.
USER_AGENT = "AnswerScriptEvaluator/1.0"


def explain_forbidden(code, body):
    """('auth', msg) for a genuine key problem, ('blocked', msg) when a firewall refused the request."""
    snippet = (body or "").strip()[:200]
    if code == 401 or (code == 403 and snippet.startswith("{")):  # the API itself answered
        return "auth", f"The server rejected the API key (HTTP {code}). {snippet}".strip()
    return "blocked", (
        f"The request was blocked before it reached the API (HTTP {code}"
        + (", error 1010" if "1010" in snippet else "") + "). This is usually a firewall, proxy or "
        "bot-protection rule, not the API key." + (f" Server said: {snippet}" if snippet else ""))


def _human_wait(seconds):
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} hours"
    if seconds >= 120:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds:.0f} seconds"


def explain_rate_limit(body):
    """Classify a 429 reply from its FULL body. Returns (kind, detail).

    kind: "daily" (a per-day quota: waiting a few seconds will not help), "tokens" (tokens per minute),
    "minute" (requests per minute) or "unknown". detail: short plain-English description, or "".
    Gemini lists the quota that was hit under error.details[].violations[]."""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, list) and data:
        data = data[0]
    err = data.get("error") if isinstance(data, dict) else None
    err = err if isinstance(err, dict) else {}
    violations = []
    for d in err.get("details") or []:
        if isinstance(d, dict) and isinstance(d.get("violations"), list):
            violations += [v for v in d["violations"] if isinstance(v, dict)]
    ids = " ".join(f"{v.get('quotaId', '')} {v.get('quotaMetric', '')}" for v in violations).lower()
    message = str(err.get("message") or "").lower()

    model = limit = ""
    for v in violations:
        dims = v.get("quotaDimensions")
        if isinstance(dims, dict) and dims.get("model"):
            model = str(dims["model"])
        if v.get("quotaValue"):
            limit = str(v["quotaValue"])
    if "perday" in ids or "per_day" in ids or "per day" in message or "daily" in message:
        kind = "daily"
    elif "token" in ids:
        kind = "tokens"
    elif "perminute" in ids or "per_minute" in ids or "per minute" in message:
        kind = "minute"
    else:
        kind = "unknown"
    label = {"daily": "daily request limit", "tokens": "tokens-per-minute limit",
             "minute": "requests-per-minute limit", "unknown": ""}[kind]
    detail = ""
    if label:
        detail = label + (f" for {model}" if model else "") + (f" (limit {limit})" if limit else "")
    return kind, detail


def strip_reasoning(text):
    """Drop <think>...</think> reasoning that some models include in their reply."""
    if "</think>" not in text:
        return text
    if "<think>" not in text:
        return text.rsplit("</think>", 1)[1]  # only the closing tag survived
    kept, rest = [], text
    while "<think>" in rest and "</think>" in rest.split("<think>", 1)[1]:
        before, after = rest.split("<think>", 1)
        kept.append(before)
        rest = after.split("</think>", 1)[1]
    kept.append(rest)
    return "".join(kept)


def parse_json_reply(text):
    """Pull the first JSON object/array out of a model reply (tolerates ``` fences)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        s = s.rstrip()
        if s.endswith("```"):
            s = s[:-3]
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(s[i:])
                return obj
            except ValueError:
                continue
    raise ValueError("no JSON found in the reply")


def _find_key(obj, key):
    """First value stored under `key` anywhere inside nested dicts/lists."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def _image_bytes(path, max_side, quality):
    """JPEG bytes of an image, shrunk so its longest side is <= max_side."""
    try:
        import cv2

        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("unreadable")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            return buf.tobytes(), "image/jpeg"
    except Exception:
        pass
    with open(path, "rb") as f:  # fall back to the raw file
        data = f.read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return data, mime


def image_part(path, max_side=2200, quality=88):
    """One `image_url` message part holding the picture as a base64 data URL."""
    data, mime = _image_bytes(path, max_side, quality)
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


class LLMClient:
    """Minimal client for OpenAI-compatible servers (local, or the Gemini endpoint).
    Thread-safe for one pipeline run."""

    def __init__(self, base_url, model, api_key="", requests_per_minute=10,
                 timeout=180, max_retries=6, cancel=None, tokens_per_minute=0, reasoning=""):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.model = (model or "").strip()
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.cancel = cancel or threading.Event()
        self.backoff_scale = 1.0  # tests shrink this
        self.long_wait = 180.0    # a longer Retry-After means a daily quota, not a per-minute one
        self.window_seconds = 60.0
        self.tokens_per_minute = int(tokens_per_minute or 0)  # 0 = no limit
        self.reasoning = (reasoning or "").strip()            # e.g. "none"; empty = do not send
        self.notice = None        # optional callable(str): tells the user why we are waiting
        self._dropped = set()     # optional request fields this model rejected (never sent again)
        self.calls = 0            # requests actually sent (includes automatic retries after HTTP 429/5xx)
        self.call_budget = None   # max model calls (ask / ask_json attempts) allowed right now; None = no limit
        self.budget_used = 0      # model calls made since begin_budget()
        self.last_reply = ""      # raw text of the last reply (shown in error messages)
        self.last_finish = ""     # finish_reason of the last reply ("length" = cut off)
        self.tokens_used = 0      # as reported by the server
        self._window = []         # [time, tokens] of recent requests
        self._interval = 60.0 / max(1, int(requests_per_minute or 10))
        self._next_slot = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings, cancel=None):
        return cls(
            settings.get("base_url"), settings.get("model"), settings.get("api_key", ""),
            settings.get("requests_per_minute") or 10, cancel=cancel,
            tokens_per_minute=settings.get("tokens_per_minute") or 0,
        )

    def _tell(self, message):
        if self.notice:
            try:
                self.notice(message)
            except Exception:
                pass  # a broken progress callback must never break a request

    # ---- pacing ----
    def _sleep(self, seconds):
        end = time.monotonic() + seconds
        while True:
            if self.cancel.is_set():
                raise Cancelled()
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(0.2, left))

    def _wait_turn(self):
        """Reserve the next request slot so we never exceed requests_per_minute."""
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + self._interval
        self._sleep(start - now)

    def _retry_after(self, error, body):
        """Seconds the server asks us to wait (header or Gemini-style body), else None."""
        header = error.headers.get("Retry-After") if error.headers else None
        try:
            if header:
                return float(header)
        except ValueError:
            pass
        try:
            hint = _find_key(json.loads(body), "retryDelay")
            if hint:
                return float(str(hint).rstrip("s"))
        except (ValueError, TypeError):
            pass
        return None

    # ---- token pacing ----
    @staticmethod
    def _estimate_tokens(messages):
        total = 700  # expected length of the reply
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                total += len(content) // 3 + 1
            else:
                for part in content:
                    if part.get("type") == "text":
                        total += len(part.get("text", "")) // 3 + 1
                    else:
                        total += IMAGE_TOKENS
        return total

    def _wait_tokens(self, estimate):
        """Keep the last minute's token use under tokens_per_minute. Returns the window entry."""
        if not self.tokens_per_minute:
            return None
        estimate = min(estimate, self.tokens_per_minute)  # a bigger request can only go alone
        while True:
            now = time.monotonic()
            with self._lock:
                self._window = [w for w in self._window if now - w[0] < self.window_seconds]
                used = sum(w[1] for w in self._window)
                if not self._window or used + estimate <= self.tokens_per_minute:
                    entry = [now, estimate]
                    self._window.append(entry)
                    return entry
                wait = self.window_seconds - (now - self._window[0][0]) + 0.05
            self._sleep(min(max(wait, 0.05), 5.0))

    # ---- requests ----
    OPTIONAL_FIELDS = (("reasoning_effort", "reasoning"), ("response_format", "response_format"))

    def _degrade(self, payload, error_text, skip=()):
        """Drop ONE optional request field the server rejected. Returns (field, guessed) or (None, False).

        Gemini often says only "Request contains an invalid argument." without naming the field. Then the
        optional fields are tried one at a time (guessed=True) and the caller undoes a wrong guess."""
        low = (error_text or "").lower()
        for field, word in self.OPTIONAL_FIELDS:
            if field in payload and word in low:
                del payload[field]
                return field, False
        if "invalid argument" in low or "invalid_argument" in low or not low.strip():
            for field, _word in self.OPTIONAL_FIELDS:
                if field in payload and field not in skip:
                    del payload[field]
                    return field, True
        return None, False

    def chat(self, messages, temperature=0.0, max_tokens=4096, json_mode=False):
        if not self.base_url or not self.model:
            raise LLMError("Model address / model name are not set. Open Settings first.")
        if endpoint_kind(self.base_url) is None:
            raise LLMError(
                "This model address is not allowed. Use a model server on this computer "
                "(e.g. http://localhost:11434/v1) or the Gemini endpoint "
                "(https://generativelanguage.googleapis.com/v1beta/openai).")
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}  # the server guarantees valid JSON
        if self.reasoning:
            payload["reasoning_effort"] = self.reasoning
        for field in self._dropped:
            payload.pop(field, None)  # this model already rejected it
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.base_url + "/chat/completions"
        entry = self._wait_tokens(self._estimate_tokens(messages))

        last = ""
        rate_limited = False
        trial, wrong = None, set()  # an unconfirmed guess about which optional field the server rejected
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            self.calls += 1
            body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                usage = data.get("usage") if isinstance(data, dict) else None
                if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), (int, float)):
                    self.tokens_used += int(usage["total_tokens"])
                    if entry:
                        entry[1] = int(usage["total_tokens"])  # replace the estimate with the truth
                return self._extract_text(data)
            except urllib.error.HTTPError as e:
                full = e.read().decode("utf-8", "replace") if e.fp else ""
                text = full[:400]  # for messages; parsing below uses the FULL body (the quota details are at the end)
                if e.code in (401, 403):
                    kind, message = explain_forbidden(e.code, text)
                    raise (AuthError if kind == "auth" else LLMError)(message)
                if e.code == 413:
                    raise LLMError("The request is too large for the model's limits (HTTP 413). "
                                   "A very long script may need a plan with a higher tokens-per-minute limit. " + text)
                if e.code == 429 or e.code >= 500:
                    rate_limited = rate_limited or e.code == 429
                    last = f"HTTP {e.code}. {text}"
                    kind, detail = explain_rate_limit(full) if e.code == 429 else ("unknown", "")
                    if kind == "daily":
                        # Retrying cannot help (and burns more requests): stop now, progress is saved.
                        raise QuotaExceeded(
                            f"Gemini's {detail or 'daily request limit'} is used up. Daily limits reset at midnight "
                            "Pacific time. Progress is saved: press Resume after the reset, or use an API key / "
                            f"model with a higher limit. ({self.calls} request(s) were sent in this run.)")
                    wait = self._retry_after(e, full)
                    if e.code == 429 and wait is not None and wait > self.long_wait:
                        raise QuotaExceeded(
                            f"The quota is used up; the server asks you to wait about {_human_wait(wait)}. "
                            "Progress is saved, so resume then.")
                    if attempt < self.max_retries:
                        delay = min(wait if wait is not None else min(5.0 * (2 ** attempt), 60.0), 120.0)
                        why = (f"rate limited ({detail})" if detail else
                               "rate limited" if e.code == 429 else f"server error HTTP {e.code}")
                        hint = " Set GEMINI_TPM in .env to your plan's limit." if kind == "tokens" else ""
                        self._tell(f"Gemini {why}: waiting {delay:.0f}s, then retry {attempt + 1} of "
                                   f"{self.max_retries}...{hint}")
                        self._sleep(delay * self.backoff_scale)
                        continue
                    break
                if e.code == 400:
                    if trial is not None:  # the last guess did not help, so it was not the culprit: put it back
                        payload[trial[0]] = trial[1]
                        self._dropped.discard(trial[0])
                        wrong.add(trial[0])
                        trial = None
                    value_before = dict(payload)
                    dropped, guessed = self._degrade(payload, text, skip=wrong)
                    if dropped:
                        self._dropped.add(dropped)
                        trial = (dropped, value_before[dropped]) if guessed else None
                        self._tell(f"The model rejected the optional setting '{dropped}'; continuing without it.")
                        continue  # retry without the optional field this server does not support
                raise LLMError(f"The server refused the request (HTTP {e.code}). {text}")
            except ValueError:
                raise LLMError("The server sent a reply that is not valid JSON.")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = f"connection problem: {getattr(e, 'reason', e)}"
                if attempt < self.max_retries:
                    self._sleep(min(2.0 * (2 ** attempt), 30.0) * self.backoff_scale)
                    continue
                break
        if rate_limited:
            raise QuotaExceeded(f"Still rate limited after {self.max_retries} retries ({last}). "
                                "Progress is saved; press Resume in a few minutes.")
        raise LLMError(f"Could not get an answer from the model server ({last}).")

    def _extract_text(self, data):
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("The server's reply had no message in it.")
        try:
            self.last_finish = str(data["choices"][0].get("finish_reason") or "")
        except (KeyError, IndexError, TypeError, AttributeError):
            self.last_finish = ""
        if isinstance(content, list):  # some servers return a list of parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = strip_reasoning(str(content or "")).strip()
        if not content:
            raise LLMError("The model returned an empty reply (it may have been blocked by a safety filter).")
        return content

    def begin_budget(self, limit):
        """From now on at most `limit` calls may be made, until end_budget(). Going over raises
        CallBudgetExceeded (an LLMError) BEFORE anything is sent, so callers fall back instead."""
        self.call_budget, self.budget_used = int(limit), 0

    def end_budget(self):
        """Stop limiting; returns how many calls were made since begin_budget()."""
        used, self.call_budget, self.budget_used = self.budget_used, None, 0
        return used

    def ask(self, prompt, images=(), system=None, temperature=0.0, max_tokens=4096, json_mode=False):
        if self.call_budget is not None:
            if self.budget_used >= self.call_budget:
                raise CallBudgetExceeded(f"The limit of {self.call_budget} model calls for this script was reached.")
            self.budget_used += 1
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if images:
            content = [{"type": "text", "text": prompt}, *images]
        else:
            content = prompt
        messages.append({"role": "user", "content": content})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)

    def ask_json(self, prompt, images=(), system=None, attempts=3, max_tokens=4096):
        """Ask for JSON; if the reply is not parseable, ask again (up to `attempts`)."""
        current = prompt
        for _ in range(attempts):
            reply = self.ask(current, images=images, system=system, max_tokens=max_tokens, json_mode=True)
            self.last_reply = reply
            try:
                return parse_json_reply(reply)
            except ValueError:
                current = prompt + "\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON."
                if self.last_finish == "length":  # cut off mid-JSON: allow a longer reply next time
                    max_tokens = min(max_tokens * 2, 16000)
        snippet = (self.last_reply or "").strip().replace("\n", " ")[:300]
        raise LLMError("The model did not return valid JSON." + (f" It replied: {snippet!r}" if snippet else ""))


# =====================================================================
# 3. Transcription backends + student-ID reading
# =====================================================================
class OCRUnavailable(Exception):
    """A requested OCR engine is not installed."""


METHOD_ENGINES = {          # settings value -> engines to run
    "paddleocr": ("paddleocr",),
    "vlm": ("vlm",),
    "compare": ("vlm", "paddleocr"),
}
PRIMARY_ENGINE = {          # which transcription later stages use
    "paddleocr": "paddleocr",
    "vlm": "vlm",
    "compare": "vlm",
}

TRANSCRIBE_PROMPT = (
    "TASK: transcribe_page\n"
    "This is one scanned page of a student's handwritten exam answer script. "
    "Transcribe EXACTLY what is written, in reading order.\n"
    "- Be verbatim: do NOT correct spelling, grammar or mathematics, and do NOT solve or complete anything.\n"
    "- Write mathematics in LaTeX between $ and $.\n"
    "- Write [unclear] for any word or symbol you cannot read.\n"
    "- Write crossed-out text as [struck: ...].\n"
    "- If the student drew a graph, plot, diagram or geometric figure, insert a line [Graph Description: ...] that "
    "describes it (axes and labels, endpoints, slopes, intercepts, shapes, trends and any labelled values).\n"
    "- Keep question numbers and labels exactly as written (for example Q1, 2(a), Ans 3).\n"
    "- If the page is blank, reply with nothing.\n"
    "Output ONLY the transcription, with no commentary."
)

ID_PROMPT = (
    "TASK: read_student_id\n"
    "Below is the transcription of the first page of a student's exam answer script. Find the student's name "
    "and roll number / registration number, usually written at the top.\n"
    "Return ONLY JSON: {\"name\": \"<name or empty string>\", \"roll_no\": \"<roll number or empty string>\", "
    "\"confidence\": <number between 0 and 1>}\n"
    "Copy the text exactly as written. If a field is not present or unreadable, use an empty string.\n\n"
    "TRANSCRIPTION:\n<<<\n"
)


def _strip_fences(text):
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


_paddle = None


import os

def get_paddle(lang="en"):
    """One shared PaddleOCR instance (loading the model is slow)."""
    global _paddle
    if _paddle is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise OCRUnavailable(
                "PaddleOCR is not installed (pip install paddleocr paddlepaddle).") from e

        # 1. Force environment flags to disable the oneDNN crash completely
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

        # 2. Add safe limits to prevent exponential memory ballooning
        kwargs = {
            "lang": lang,
            "use_angle_cls": True,
            "enable_mkldnn": False,
            "det_limit_side_len": 960,  # Caps image matrix dimension scaling
            "det_limit_type": "max"     # Keeps processing matrix size uniform
        }

        try:
            _paddle = PaddleOCR(**kwargs)
        except Exception:
            # Fallback for older versions, while still ensuring memory safety
            try:
                _paddle = PaddleOCR(lang=lang, det_limit_side_len=960)
            except Exception:
                _paddle = PaddleOCR(lang=lang)

    return _paddle



def _paddle_lines(ocr, image_path):
    """Text lines from either the 3.x (predict) or 2.x (ocr) PaddleOCR API."""
    lines = []
    if hasattr(ocr, "predict"):
        for res in ocr.predict(image_path) or []:
            texts = None
            try:
                texts = res["rec_texts"]
            except (KeyError, TypeError):
                try:
                    texts = res.json["res"]["rec_texts"]
                except (KeyError, TypeError, AttributeError):
                    pass
            lines.extend(str(t) for t in (texts or []))
    else:
        for page in ocr.ocr(image_path, cls=True) or []:
            for item in page or []:
                lines.append(str(item[1][0]))
    return lines


def transcribe_page(engine, image_path, client=None):
    """Text of one page image from the named engine ("paddleocr" or "vlm")."""
    if engine == "paddleocr":
        return "\n".join(_paddle_lines(get_paddle(), image_path)).strip()
    if engine == "vlm":
        if client is None:
            raise LLMError("The vision model is not configured. Open Settings first.")
        return _strip_fences(client.ask(TRANSCRIBE_PROMPT, images=[image_part(image_path)]))
    raise ValueError(f"unknown OCR engine: {engine}")


BATCH_TRANSCRIBE_PROMPT = (
    "TASK: transcribe_pages\n"
    "These are {n} consecutive scanned pages of ONE student's handwritten exam answer script, attached in order. "
    "Transcribe EXACTLY what is written on each page, in reading order.\n"
    "- Start each page with a line containing only === PAGE k === (k = 1 for the first attached image, 2 for the "
    "second, and so on). Output exactly {n} such lines, in order, even for a blank page (write nothing after it).\n"
    "- Be verbatim: do NOT correct spelling, grammar or mathematics, and do NOT solve or complete anything.\n"
    "- Write mathematics in LaTeX between $ and $.\n"
    "- Write [unclear] for any word or symbol you cannot read.\n"
    "- Write crossed-out text as [struck: ...].\n"
    "- If the student drew a graph, plot, diagram or geometric figure, insert a line [Graph Description: ...] that "
    "describes it (axes and labels, endpoints, slopes, intercepts, shapes, trends and any labelled values).\n"
    "- Keep question numbers and labels exactly as written (for example Q1, 2(a), Ans 3).\n"
    "Output ONLY the page markers and the transcription, with no commentary."
)
_PAGE_MARK = re.compile(r"^[ \t]*={2,}[ \t]*PAGE[ \t]+(\d+)[ \t]*={2,}[ \t]*$", re.I | re.M)


def _split_pages(reply, n):
    """Per-page texts from a batched reply, or None unless it has exactly the markers 1..n in order."""
    marks = list(_PAGE_MARK.finditer(reply))
    if [int(m.group(1)) for m in marks] != list(range(1, n + 1)):
        return None
    texts = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(reply)
        texts.append(reply[m.end():end].strip())
    return texts


def transcribe_pages(engine, image_paths, client=None):
    """Texts of several page images, one entry per image.

    The vision model reads all the pages in ONE request (fewer calls, so fewer rate-limit waits).
    If its reply cannot be split back into pages, or was cut off, the pages are read one by one."""
    image_paths = list(image_paths)
    if len(image_paths) <= 1 or engine != "vlm":
        return [transcribe_page(engine, p, client) for p in image_paths]
    if client is None:
        raise LLMError("The vision model is not configured. Open Settings first.")
    n = len(image_paths)
    texts = None
    try:
        reply = client.ask(BATCH_TRANSCRIBE_PROMPT.format(n=n),
                           images=[image_part(p) for p in image_paths],
                           max_tokens=min(16000, 3000 * n))
        if client.last_finish != "length":
            texts = _split_pages(_strip_fences(reply), n)
    except (AuthError, QuotaExceeded, Cancelled):
        raise
    except LLMError:
        texts = None  # fall back to one page per request
    if texts is None:
        return [transcribe_page(engine, p, client) for p in image_paths]
    return texts


MAX_OUTPUT_TOKENS = 16000     # largest reply we ask for
MAX_PAGES_PER_CALL = 12       # a script with more pages than this needs more than one reading call


def _split_pages_lenient(reply, n):
    """([text or None per page], last page marker seen). Unlike _split_pages this never rejects a reply:
    markers outside 1..n are ignored, and a page without a marker stays None (= not read)."""
    marks = [m for m in _PAGE_MARK.finditer(reply) if 1 <= int(m.group(1)) <= n]
    texts = [None] * n
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(reply)
        k = int(m.group(1)) - 1
        chunk = reply[m.end():end].strip()
        texts[k] = chunk if texts[k] is None else texts[k] + "\n" + chunk
    return texts, (int(marks[-1].group(1)) if marks else 0)


def read_pages(client, image_paths, note=None):
    """Transcribe EVERY page of a script with ONE model call (one more only if the reply was cut off).

    Returns a list with one entry per page: the text, or None for a page that could not be read
    (the caller must not treat None as a blank page). A reply that does not follow the page-marker
    format is salvaged rather than re-requested."""
    image_paths = list(image_paths)
    out = []
    for start in range(0, len(image_paths), MAX_PAGES_PER_CALL):
        out += _read_group(client, image_paths[start:start + MAX_PAGES_PER_CALL], note, retry=True)
    return out


def _read_group(client, paths, note, retry):
    n = len(paths)
    side = 2200 if n <= 6 else 1800   # keeps a many-page request a sensible size
    images = [image_part(p, max_side=side) for p in paths]
    if n == 1:
        return [_strip_fences(client.ask(TRANSCRIBE_PROMPT, images=images, max_tokens=4096))]
    reply = _strip_fences(client.ask(BATCH_TRANSCRIBE_PROMPT.format(n=n), images=images,
                                     max_tokens=min(MAX_OUTPUT_TOKENS, 2500 * n)))
    cut = client.last_finish == "length"
    texts, last = _split_pages_lenient(reply, n)
    if not last:  # no markers at all: keep the text rather than lose it, attributed to page 1
        if note:
            note("The model did not mark the page breaks; the text is filed under page 1.")
        return [reply] + [""] * (n - 1)
    if cut and retry:  # cut off mid-way: re-read from the last page it reached (that page may be partial)
        first = max(1, last)
        if note:
            note(f"The reply was cut off after page {last}; reading pages {first}-{n} again.")
        try:
            rest = _read_group(client, paths[first - 1:], note, retry=False)
        except (AuthError, QuotaExceeded, Cancelled):
            raise
        except LLMError:
            rest = [None] * (n - first + 1)
        texts[first - 1:] = rest
    elif cut:
        texts[max(0, last - 1)] = None  # the last page is incomplete: do not trust it
    else:
        texts = ["" if t is None else t for t in texts]  # a complete reply: an unmarked page was blank
    return texts


def extract_student_id(client, first_page_text):
    """Find name and roll number in the first page's transcription (a cheap text-only call).
    Returns {"name", "roll_no", "confidence"}."""
    text = (first_page_text or "").strip()
    if not text:
        return {"name": "", "roll_no": "", "confidence": 0.0}
    data = client.ask_json(ID_PROMPT + text[:1500] + "\n>>>")
    if not isinstance(data, dict):
        data = {}

    def clean(v):
        return str(v).strip() if isinstance(v, (str, int, float)) else ""

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"name": clean(data.get("name")), "roll_no": clean(data.get("roll_no")),
            "confidence": confidence}

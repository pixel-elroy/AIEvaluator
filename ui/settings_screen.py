"""Settings screen: the Gemini connection used to read answer scripts and to structure, match and grade.

Question papers are always read with PaddleOCR (on this computer); answer scripts are always read with
the Gemini vision model. Settings are stored as JSON (see settings_path()).
"""
import json
import os
import threading
import urllib.error
import urllib.request
from functools import partial

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDButton, MDButtonIcon, MDButtonText
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.textfield import MDTextField, MDTextFieldHintText

from core import ocr_engine
from ui.widgets import make_app_bar, make_button, make_card, make_row, muted_label, notify

# Fixed by the project: the extraction method is not user-selectable any more.
EXTRACTION_METHODS = (("vlm", "Gemini vision model"),)
METHOD_HELP = {
    "vlm": "Question papers are read on this computer with PaddleOCR. Answer scripts (handwriting and "
           "math) are read by the Gemini vision model configured below.",
}

LOCATIONS = (("cloud", "Cloud (Gemini)"),)
LOCATION_HELP = {
    "cloud": "Answer-script page images and their text, and the question text, are sent to Google's Gemini "
             "API. On the free tier Google may use that data to improve its products, so use sample or "
             "consented scripts. Get a key from Google AI Studio.",
}

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


# ---------------------------------------------------------------- .env file
# Put these in a file called .env next to main.py (see .env.example):
#   GEMINI_API_KEY=...        (GOOGLE_API_KEY also works)
#   GEMINI_MODEL=gemini-3.8-flash
#   GEMINI_RPM=5              (optional: max requests per minute)
#   GEMINI_TPM=0              (optional: max tokens per minute, 0 = no limit)
# Values found here win over the Settings screen, so the key survives restarts.
def _env_files():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = [os.path.join(root, ".env"), os.path.join(os.getcwd(), ".env")]
    return list(dict.fromkeys(paths))


def env_values():
    """Parsed .env values (a later file overrides an earlier one). Never raises."""
    values = {}
    for path in _env_files():
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:].lstrip()
            name, _, value = line.partition("=")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            values[name.strip()] = value
    return values


def _env_overrides():
    """Settings keys supplied by .env (or real environment variables) -> value."""
    env = env_values()

    def get(*names):
        for n in names:
            value = (env.get(n) or os.environ.get(n) or "").strip()
            if value:
                return value
        return ""

    found = {}
    if get("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        found["api_key"] = get("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if get("GEMINI_MODEL"):
        found["model"] = get("GEMINI_MODEL")
    for name, key in (("GEMINI_RPM", "requests_per_minute"), ("GEMINI_TPM", "tokens_per_minute")):
        raw = get(name)
        if raw.isdigit():
            found[key] = int(raw)
    return found

# provider -> (base URL, suggested model)
CLOUD_PROVIDERS = {"Gemini (Google AI Studio)": (GEMINI_URL, "gemini-2.5-flash")}
PROVIDERS = {"cloud": CLOUD_PROVIDERS}

DEFAULTS = {
    "extraction": "vlm",        # fixed: answer scripts are read by the vision model
    "location": "cloud",
    "provider": "Gemini (Google AI Studio)",
    "base_url": GEMINI_URL,
    "model": "gemini-2.5-flash",
    "api_key": "",
    "requests_per_minute": 5,    # stays under the Gemini free-tier limit
    "tokens_per_minute": 0,      # 0 = no limit
    "reasoning_read": "none",   # reasoning effort when reading pages / matching answers ("" = do not send)
    "reasoning_grade": "low",   # reasoning effort when grading
}


# ---------------------------------------------------------------- storage
def settings_path():
    app = MDApp.get_running_app()
    folder = app.user_data_dir if app else os.path.expanduser("~/.answer_evaluator")
    return os.path.join(folder, "settings.json")


def _is_gemini_url(base_url):
    """True only for the Gemini endpoint (https on an allowed cloud host)."""
    return ocr_engine.endpoint_kind(base_url) == "cloud"


def load_settings():
    """Returns a dict with every key in DEFAULTS (missing/corrupt file -> defaults)."""
    data = dict(DEFAULTS)
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data.update({k: v for k, v in loaded.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    # Older versions allowed a local server only. Anything that is not the Gemini endpoint is
    # replaced (and its key dropped: it belonged to a different server).
    if not _is_gemini_url(data.get("base_url")):
        data.update({"provider": DEFAULTS["provider"], "base_url": DEFAULTS["base_url"],
                     "model": DEFAULTS["model"], "api_key": ""})
    data.update(_env_overrides())  # .env wins over the saved file (and must come after the reset above)
    data["location"], data["extraction"] = "cloud", "vlm"
    if data.get("provider") not in CLOUD_PROVIDERS:
        data["provider"] = DEFAULTS["provider"]
    if not str(data.get("model") or "").strip():
        data["model"] = DEFAULTS["model"]
    for name, low, high in (("requests_per_minute", 1, 600), ("tokens_per_minute", 0, 10_000_000)):
        try:
            data[name] = max(low, min(high, int(data[name])))
        except (TypeError, ValueError):
            data[name] = DEFAULTS[name]
    for name in ("reasoning_read", "reasoning_grade"):
        data[name] = str(data[name] or "").strip()
    return data


def save_settings(data):
    base_url = str(data.get("base_url", "") or "").strip()
    if not _is_gemini_url(base_url):
        raise ValueError("The model address must be the Gemini endpoint: " + GEMINI_URL)
    data = dict(data)
    data["location"], data["extraction"] = "cloud", "vlm"
    for name, value in _env_overrides().items():  # .env is the source of truth for these: don't duplicate them
        if name in ("api_key", "model") and data.get(name) == value:
            data[name] = DEFAULTS[name]
    path = settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({k: data.get(k, DEFAULTS[k]) for k in DEFAULTS}, f, indent=2)
    try:
        os.chmod(tmp, 0o600)  # the file may contain an API key
    except OSError:
        pass
    os.replace(tmp, path)


# ----------------------------------------------------------------- screen
def _field(hint, **kwargs):
    field = MDTextField(mode="outlined", **kwargs)
    hint_widget = MDTextFieldHintText(text=hint)
    field.add_widget(hint_widget)
    return field, hint_widget


class SettingsScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = DEFAULTS["extraction"]
        self.location = DEFAULTS["location"]
        self.provider = DEFAULTS["provider"]
        self._drafts = {}  # per-location edits, so switching back and forth loses nothing
        self.menu = None

        root = MDBoxLayout(orientation="vertical")
        root.add_widget(make_app_bar("Settings", self.go_back))

        scroll = MDScrollView(do_scroll_x=False)
        body = MDBoxLayout(
            orientation="vertical",
            padding=[dp(24), dp(24), dp(24), dp(32)],
            spacing=dp(20),
            adaptive_height=True,
        )

        # --- transcription method ---
        method_card = make_card(
            "How documents are read", "Fixed for this project."
        )
        self.method_row = MDBoxLayout(orientation="horizontal", spacing=dp(8), adaptive_height=True)
        self.method_help = muted_label("")
        method_card.add_widget(self.method_row)
        method_card.add_widget(self.method_help)
        body.add_widget(method_card)

        # --- vision-model location + connection ---
        api_card = make_card(
            "Gemini model",
            "Used to read answer scripts and to structure the paper, match answers and grade.",
        )
        self.location_row = MDBoxLayout(orientation="horizontal", spacing=dp(8), adaptive_height=True)
        self.location_help = muted_label("")
        api_card.add_widget(self.location_row)
        api_card.add_widget(self.location_help)

        self.provider_text = MDButtonText(text=self.provider)
        self.provider_btn = MDButton(
            MDButtonIcon(icon="server-network-outline"), self.provider_text, style="outlined"
        )
        self.provider_btn.bind(on_release=self.open_provider_menu)
        api_card.add_widget(make_row(self.provider_btn))

        self.base_input, _ = _field("Base URL")
        self.model_input, _ = _field("Gemini model name (default gemini-2.5-flash)")
        self.key_input, self.key_hint = _field("Gemini API key", password=True)
        self.rpm_input, _ = _field("Max requests per minute (stay under your plan's limit)", input_filter="int")
        self.tpm_input, _ = _field("Max tokens per minute (0 = no limit)", input_filter="int")
        self.read_input, _ = _field("Reasoning effort for reading and matching (e.g. none; blank = don't send)")
        self.grade_input, _ = _field("Reasoning effort for grading (e.g. low, medium; blank = don't send)")
        for w in (self.base_input, self.model_input, self.key_input, self.rpm_input,
                  self.tpm_input,
                  self.read_input, self.grade_input):
            api_card.add_widget(w)

        self.test_label = muted_label("")
        self.test_btn = make_button("Test connection", style="outlined", on_release=self.test_connection)
        api_card.add_widget(make_row(self.test_btn))
        api_card.add_widget(self.test_label)
        body.add_widget(api_card)

        body.add_widget(
            make_button("Save settings", on_release=self.save, pos_hint={"center_x": 0.5})
        )

        scroll.add_widget(body)
        root.add_widget(scroll)
        self.add_widget(root)
        self._render_methods()
        self._render_locations()

    # ------------------------------------------------------- lifecycle
    def on_pre_enter(self, *args):
        data = load_settings()
        self.method = "vlm"
        self.location = "cloud"
        self._drafts = {}
        provider = data["provider"] if data["provider"] in PROVIDERS[self.location] else DEFAULTS["provider"]
        self._apply_fields({
            "provider": provider, "base_url": data["base_url"],
            "model": data["model"], "api_key": data["api_key"],
        })
        self.rpm_input.text = str(data["requests_per_minute"])
        self.tpm_input.text = str(data["tokens_per_minute"])
        self.read_input.text = data["reasoning_read"]
        self.grade_input.text = data["reasoning_grade"]
        from_env = sorted(_env_overrides())
        self.test_label.text = ("Loaded from .env: " + ", ".join(from_env)) if from_env else ""
        self._render_methods()
        self._render_locations()

    def on_leave(self, *args):
        try:
            self._write()  # never lose edits when leaving the screen
        except (OSError, ValueError):
            pass

    # ------------------------------------------------ transcription method
    def _render_methods(self):
        self.method_row.clear_widgets()
        self.method_help.text = METHOD_HELP["vlm"]

    # ----------------------------------------------------- model location
    def _render_locations(self):
        self.location_row.clear_widgets()
        for key, label in LOCATIONS:
            style = "filled" if key == self.location else "outlined"
            self.location_row.add_widget(
                make_button(label, style=style, on_release=partial(self._set_location, key))
            )
        self.location_help.text = LOCATION_HELP[self.location]
        self.key_hint.text = "Gemini API key (stored only on this computer)"

    def _set_location(self, key, *_):
        if key == self.location:
            return
        self._drafts[self.location] = self._model_fields()  # remember what was typed
        self.location = key
        draft = self._drafts.get(key)
        if draft is None:  # first visit: start from the first preset of that kind
            provider = next(iter(PROVIDERS[key]))
            base, model = PROVIDERS[key][provider]
            draft = {"provider": provider, "base_url": base, "model": model, "api_key": ""}
        self._apply_fields(draft)
        self.test_label.text = ""
        self._render_locations()

    # --------------------------------------------------------- provider
    def open_provider_menu(self, *_):
        items = [
            {"text": name, "on_release": partial(self._pick_provider, name)}
            for name in PROVIDERS[self.location]
        ]
        self.menu = MDDropdownMenu(caller=self.provider_btn, items=items)
        self.menu.open()

    def _pick_provider(self, name, *_):
        if self.menu:
            self.menu.dismiss()
        self.provider = name
        self.provider_text.text = name
        base, model = PROVIDERS[self.location][name]
        self.base_input.text = base
        self.model_input.text = model

    # ----------------------------------------------------- field helpers
    def _model_fields(self):
        return {
            "provider": self.provider,
            "base_url": self.base_input.text.strip(),
            "model": self.model_input.text.strip(),
            "api_key": self.key_input.text.strip(),
        }

    def _apply_fields(self, d):
        self.provider = d["provider"]
        self.provider_text.text = d["provider"]
        self.base_input.text = d["base_url"]
        self.model_input.text = d["model"]
        self.key_input.text = d["api_key"]

    def _int_value(self, field, name, low, high):
        try:
            return max(low, min(high, int(field.text)))
        except ValueError:
            return DEFAULTS[name]

    # ------------------------------------------------------------- save
    def _write(self):
        data = {
            "extraction": self.method, "location": self.location,
            "requests_per_minute": self._int_value(self.rpm_input, "requests_per_minute", 1, 600),
            "tokens_per_minute": self._int_value(self.tpm_input, "tokens_per_minute", 0, 10_000_000),
            "reasoning_read": self.read_input.text.strip(),
            "reasoning_grade": self.grade_input.text.strip(),
        }
        data.update(self._model_fields())
        save_settings(data)

    def save(self, *_):
        try:
            self._write()
        except (OSError, ValueError) as e:
            notify(f"Could not save settings: {e}")
            return
        notify("Settings saved.")

    # ------------------------------------------------------ test button
    def test_connection(self, *_):
        base = self.base_input.text.strip().rstrip("/")
        key = self.key_input.text.strip()
        if not base:
            self.test_label.text = "Enter a base URL first."
            return
        if "{" in base:
            self.test_label.text = "Replace the {placeholder} in the URL first (e.g. the machine's address or your account id)."
            return
        if not _is_gemini_url(base):
            self.test_label.text = "The address must be the Gemini endpoint: " + GEMINI_URL
            return
        if not key:
            self.test_label.text = "Enter your Gemini API key first."
            return
        self.test_label.text = "Testing..."
        self.test_btn.disabled = True
        threading.Thread(target=self._probe, args=(base, key), daemon=True).start()

    def _probe(self, base, key):
        # a real User-Agent matters: providers behind Cloudflare answer Python's default one with 403
        headers = {"User-Agent": ocr_engine.USER_AGENT, "Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(base + "/models", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                msg = f"Connected (HTTP {response.status})."
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                body = e.read().decode("utf-8", "replace") if e.fp else ""
                kind, detail = ocr_engine.explain_forbidden(e.code, body)
                msg = (f"Server reached, but it rejected the key (HTTP {e.code}). Check the API key."
                       if kind == "auth" else detail)
            elif e.code == 404:
                msg = "Server reached, but it has no /models endpoint (some servers don't). The URL may still be right."
            else:
                msg = f"Server reached, but answered HTTP {e.code}."
        except Exception as e:
            reason = getattr(e, "reason", e)
            msg = f"Could not connect: {reason}. Is the server running and the address correct?"
        Clock.schedule_once(partial(self._show_test_result, msg))

    def _show_test_result(self, msg, *_):
        self.test_label.text = msg
        self.test_btn.disabled = False

    def go_back(self, *_):
        self.manager.transition.direction = "right"
        self.manager.current = "start"

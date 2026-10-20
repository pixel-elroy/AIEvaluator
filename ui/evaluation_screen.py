"""Evaluation screen: runs the pipeline in a background thread and shows live progress."""
import os
import threading
from functools import partial
from importlib.util import find_spec

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.progressindicator import MDLinearProgressIndicator
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView

from core.evaluator import EvaluationPipeline, Store
from core.ocr_engine import (AuthError, Cancelled, LLMError, OCRUnavailable, PageError, QuotaExceeded,
                             endpoint_kind)
from core.parser import ParseError
from ui.settings_screen import load_settings
from ui.widgets import make_app_bar, make_button, make_card, make_row, muted_label, notify

LOG_LINES = 250


def data_dir():
    """Folder where the database and page images are kept."""
    return MDApp.get_running_app().user_data_dir


def get_store():
    return Store(os.path.join(data_dir(), "evaluation.db"))


class EvaluationScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._token = 0          # invalidates updates from an older run
        self._cancel = None
        self._running = False
        self._log = []

        root = MDBoxLayout(orientation="vertical")
        root.add_widget(make_app_bar("Evaluation", self.go_back))
        scroll = MDScrollView(do_scroll_x=False)
        body = MDBoxLayout(
            orientation="vertical", padding=[dp(24), dp(24), dp(24), dp(32)],
            spacing=dp(20), adaptive_height=True,
        )

        summary_card = make_card("This run")
        self.summary = muted_label("")
        summary_card.add_widget(self.summary)
        body.add_widget(summary_card)

        progress_card = make_card("Progress")
        self.status = MDLabel(text="Ready.", font_style="Body", role="large", adaptive_height=True)
        self.bar = MDLinearProgressIndicator(
            type="determinate", value=0, max=100, size_hint_y=None, height=dp(6))
        self.log_label = muted_label("")
        log_scroll = MDScrollView(size_hint_y=None, height=dp(200))
        log_scroll.add_widget(self.log_label)
        self.log_scroll = log_scroll
        for w in (self.status, self.bar, log_scroll):
            progress_card.add_widget(w)
        body.add_widget(progress_card)

        self.start_btn = make_button("Start evaluation", on_release=self.start)
        self.cancel_btn = make_button("Cancel", style="outlined", on_release=self.cancel)
        self.cancel_btn.disabled = True
        self.review_btn = make_button("Review results", style="tonal", on_release=self.go_review)
        self.review_btn.disabled = True
        self.settings_btn = make_button("Settings", style="text", on_release=self.go_settings)
        row = make_row(self.start_btn, self.cancel_btn, self.review_btn, self.settings_btn)
        row.adaptive_size = True
        row.pos_hint = {"center_x": 0.5}
        body.add_widget(row)

        scroll.add_widget(body)
        root.add_widget(scroll)
        self.add_widget(root)

    # ------------------------------------------------------- lifecycle
    def on_pre_enter(self, *args):
        self._refresh_summary()
        try:
            self.review_btn.disabled = not get_store().list_students()
        except Exception:
            self.review_btn.disabled = True

    def _refresh_summary(self):
        upload = self.manager.get_screen("upload")
        settings = load_settings()
        rubric = upload.rubric_input.text
        subject = (getattr(upload, "subject_input", None) and upload.subject_input.text or "").strip()
        lines = [
            f"Subject: {subject or '(not set — required for the scores table)'}",
            f"Question paper: {os.path.basename(upload.qp_input.text) or '(none)'}  ·  read on this computer with PaddleOCR",
            f"Marking scheme: {os.path.basename(rubric) if rubric else 'none supplied: the AI will draft one for you to review'}",
            f"Answer scripts: {len(upload.answer_scripts)}  ·  read by the Gemini vision model",
            f"Model: {settings['model'] or '(not set)'}  ·  max {settings['requests_per_minute']} requests/min"
            + (f", {settings['tokens_per_minute']} tokens/min" if settings["tokens_per_minute"] else ""),
            "API calls per script: 3 (never more than 4): read all pages, match answers + read name, grade all questions",
            "Privacy: answer-script pages and text, and the question text, are sent to Google's Gemini API.",
        ]
        problem = self._settings_problem(settings)
        if problem:
            lines.append("")
            lines.append("⚠ " + problem)
        self.summary.text = "\n".join(lines)

    @staticmethod
    def _settings_problem(settings):
        if endpoint_kind(settings["base_url"]) != "cloud" or not settings["model"]:
            return "Open Settings and set the Gemini model name (the address must be the Gemini endpoint)."
        if not settings["api_key"]:
            return "Set your Gemini API key: GEMINI_API_KEY in the .env file next to main.py, or in Settings."
        # Question-paper PDFs always use PaddleOCR, independent of the answer-sheet method.
        if find_spec("paddleocr") is None:
            return "PaddleOCR is required for question-paper PDFs (pip install paddleocr paddlepaddle)."
        for module, package in (("cv2", "opencv-python-headless"), ("fitz", "PyMuPDF")):
            if find_spec(module) is None:
                return f"{package} is not installed (pip install {package})."
        return None

    # --------------------------------------------------------- running
    def start(self, *_):
        if self._running:
            return
        upload = self.manager.get_screen("upload")
        if not upload.qp_input.text or not upload.answer_scripts:
            notify("Add a question paper and at least one answer script first.")
            return
        subject = (getattr(upload, "subject_input", None) and upload.subject_input.text or "").strip()
        if not subject:
            notify("Enter the subject name on the Upload screen (it names the scores table).")
            return
        settings = load_settings()
        problem = self._settings_problem(settings)
        if problem:
            self.status.text = problem
            notify(problem)
            return
        scripts = [(s.student_id, list(s.sources)) for s in upload.answer_scripts]
        self._cancel = threading.Event()
        self._token += 1
        self._running = True
        self.start_btn.disabled, self.cancel_btn.disabled = True, False
        self.status.text = "Starting..."
        self.bar.value = 0
        self._append("--- started ---")
        threading.Thread(
            target=self._worker,
            args=(self._token, settings, upload.qp_input.text, upload.rubric_input.text or None,
                  scripts, subject, self._cancel),
            daemon=True,
        ).start()

    def cancel(self, *_):
        if self._cancel and self._running:
            self._cancel.set()
            self.cancel_btn.disabled = True
            self.status.text = "Cancelling... progress so far is saved."

    def _worker(self, token, settings, qp, rubric, scripts, subject, cancel):
        def progress(message, fraction):
            Clock.schedule_once(partial(self._on_progress, token, message, fraction))

        outcome = ("done", "All scripts evaluated.")
        try:
            store = get_store()
            store.set_subject(subject)
            pipe = EvaluationPipeline(settings, store, data_dir(), cancel, progress)
            paper, rubric_obj = pipe.prepare(qp, rubric)
            summary = pipe.run_all(scripts, paper, rubric_obj)
            store.sync_all_scores()
            if summary["failed"]:
                first = "; ".join(f"{k}: {m}" for k, m in summary["failed"][:3])
                outcome = ("done", f"Finished, but {len(summary['failed'])} script(s) failed: {first}")
        except Cancelled:
            outcome = ("cancelled", "Cancelled. Progress is saved; press Start to continue.")
        except QuotaExceeded as e:
            outcome = ("paused", f"Paused: {e}")
        except AuthError as e:
            outcome = ("error", str(e))
        except (LLMError, ParseError, OCRUnavailable, PageError, OSError, ValueError) as e:
            outcome = ("error", f"{e}")
        except Exception as e:  # never let the thread die silently
            outcome = ("error", f"Unexpected error: {type(e).__name__}: {e}")
        Clock.schedule_once(partial(self._on_finished, token, *outcome))

    def _on_progress(self, token, message, fraction, *_):
        if token != self._token:
            return
        self.status.text = message
        if fraction is not None:
            self.bar.value = max(0.0, min(100.0, fraction * 100))
        self._append(message)

    def _on_finished(self, token, status, message, *_):
        if token != self._token:
            return
        self._running = False
        self.start_btn.disabled, self.cancel_btn.disabled = False, True
        self.status.text = message
        self._append(f"--- {status}: {message} ---")
        if status == "done":
            self.bar.value = 100
        self.start_btn.children[0].text = "Resume" if status in ("paused", "cancelled") else "Start evaluation"
        try:
            self.review_btn.disabled = not get_store().list_students()
        except Exception:
            pass
        notify(message[:120])

    def _append(self, line):
        if self._log and self._log[-1] == line:
            return
        self._log.append(line)
        self._log = self._log[-LOG_LINES:]
        self.log_label.text = "\n".join(self._log)
        Clock.schedule_once(lambda dt: setattr(self.log_scroll, "scroll_y", 0), 0)

    # ------------------------------------------------------ navigation
    def go_back(self, *_):
        if self._running:
            notify("An evaluation is running. Cancel it first.")
            return
        self.manager.transition.direction = "right"
        self.manager.current = "inspection"

    def go_review(self, *_):
        self.manager.get_screen("review").back_to = "evaluation"
        self.manager.transition.direction = "left"
        self.manager.current = "review"

    def go_settings(self, *_):
        if self._running:
            notify("An evaluation is running. Cancel it first.")
            return
        self.manager.transition.direction = "left"
        self.manager.current = "settings"

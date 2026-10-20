import os
import shutil
import tempfile
import threading
from functools import partial

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDButton, MDButtonIcon, MDButtonText, MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.screen import MDScreen

from core.ocr_engine import process_document
from ui.widgets import make_app_bar, make_button, make_row, muted_label, notify


QP_LABEL = "Question Paper"
VIEW_MODES = ("Enhanced", "Binarized", "Original")  # index into a page tuple: 1, 2, 0
VIEW_INDEX = {"Enhanced": 1, "Binarized": 2, "Original": 0}


class ZoomableImage(ScrollView):
    """Image viewer. Zoom with the buttons or Ctrl + mouse wheel; drag to pan.

    Zoom 1.0 means "fit to the window"; the image is never cropped at that level."""

    MIN_ZOOM = 0.5
    MAX_ZOOM = 8.0
    STEP = 1.25

    def __init__(self, on_zoom=None, **kwargs):
        super().__init__(
            do_scroll_x=True,
            do_scroll_y=True,
            bar_width=dp(8),
            scroll_type=["bars", "content"],
            **kwargs,
        )
        self.zoom = 1.0
        self._on_zoom = on_zoom

        self.image = Image(
            size_hint=(None, None), pos_hint={"center_x": 0.5, "center_y": 0.5}, fit_mode="fill"
        )
        self._box = FloatLayout(size_hint=(None, None))  # at least viewport-sized, centres the image
        self._box.add_widget(self.image)
        self.add_widget(self._box)

        self.bind(size=self._relayout)
        self.image.bind(texture=self._relayout)

    # ---- public API ----
    def show(self, path, reset=True):
        """Display an image. reset=True returns to fit-to-window."""
        if reset:
            self.zoom = 1.0
            self._notify()
        if self.image.source == path:
            self.image.reload()
        else:
            self.image.source = path
        self._relayout()
        if reset:
            Clock.schedule_once(lambda dt: self._scroll_to(0.5, 1.0), 0)

    def clear(self):
        self.image.source = ""
        self.zoom = 1.0
        self._notify()

    def zoom_by(self, factor):
        self.set_zoom(self.zoom * factor)

    def fit(self):
        self.zoom = 1.0
        self._relayout()
        self._notify()
        Clock.schedule_once(lambda dt: self._scroll_to(0.5, 1.0), 0)

    def set_zoom(self, value):
        value = max(self.MIN_ZOOM, min(self.MAX_ZOOM, value))
        if abs(value - self.zoom) < 1e-3 or self.image.texture is None:
            return
        # remember which point of the image is at the centre of the view...
        cw, ch = self._box.size
        vw, vh = self.size
        rx = (self.scroll_x * (cw - vw) + vw / 2) / cw if cw > vw else 0.5
        ry = (self.scroll_y * (ch - vh) + vh / 2) / ch if ch > vh else 0.5
        self.zoom = value
        self._relayout()
        # ...and keep it centred after resizing
        cw, ch = self._box.size
        fx = (rx * cw - vw / 2) / (cw - vw) if cw > vw else 0.5
        fy = (ry * ch - vh / 2) / (ch - vh) if ch > vh else 0.5
        self._scroll_to(fx, fy)
        Clock.schedule_once(lambda dt: self._scroll_to(fx, fy), 0)  # again once layout settles
        self._notify()

    # ---- internals ----
    def _scroll_to(self, fx, fy):
        self.scroll_x = max(0.0, min(1.0, fx))
        self.scroll_y = max(0.0, min(1.0, fy))

    def _notify(self):
        if self._on_zoom:
            self._on_zoom(self.zoom)

    def _relayout(self, *_):
        tex = self.image.texture
        if tex is None or not self.width or not self.height:
            return
        tw, th = tex.size
        fit = min(self.width / tw, self.height / th)  # scale that fits the whole page
        iw, ih = tw * fit * self.zoom, th * fit * self.zoom
        self.image.size = (iw, ih)
        self._box.size = (max(iw, self.width), max(ih, self.height))

    def on_touch_down(self, touch):
        # Ctrl + wheel = zoom (plain wheel keeps scrolling the page)
        if (
            self.collide_point(*touch.pos)
            and getattr(touch, "is_mouse_scrolling", False)
            and "ctrl" in getattr(Window, "modifiers", [])
        ):
            self.zoom_by(self.STEP if touch.button == "scrollup" else 1 / self.STEP)
            return True
        return super().on_touch_down(touch)


class InspectionScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.docs = {}        # label -> [(orig, enhanced, binarized), ...]
        self.errors = {}      # label -> error message
        self.order = []       # labels in dropdown order
        self.current = None
        self.page = 0
        self.view_mode = 0  # index into VIEW_MODES
        self.signature = None
        self.out_dir = None
        self.menu = None
        self._token = 0       # invalidates stale worker threads

        root = MDBoxLayout(orientation="vertical")
        root.add_widget(make_app_bar("Document Inspection & Preprocessing", self.go_back))

        content = MDBoxLayout(
            orientation="vertical", padding=dp(16), spacing=dp(12)
        )

        # Row 1: document dropdown + original/processed toggle
        self.doc_text = MDButtonText(text="Select document")
        self.doc_btn = MDButton(
            MDButtonIcon(icon="file-document-outline"), self.doc_text, style="outlined"
        )
        self.doc_btn.bind(on_release=self.open_menu)

        self.view_text = MDButtonText(text=f"View: {VIEW_MODES[0]}")
        self.view_btn = MDButton(self.view_text, style="tonal")
        self.view_btn.bind(on_release=self.toggle_view)

        toolbar = MDBoxLayout(orientation="horizontal", spacing=dp(12), adaptive_height=True)
        toolbar.add_widget(self.doc_btn)
        toolbar.add_widget(Widget())
        toolbar.add_widget(self.view_btn)
        content.add_widget(toolbar)

        # Status line
        self.status = muted_label("Waiting for documents...")
        content.add_widget(self.status)

        # Zoomable image viewer (fills remaining space, scales with the window)
        self.viewer = ZoomableImage(on_zoom=self._on_zoom, size_hint_y=1)
        content.add_widget(self.viewer)

        # Bottom bar: page navigation (left) and zoom controls (right)
        self.prev_btn = MDIconButton(icon="chevron-left")
        self.prev_btn.bind(on_release=lambda *_: self.change_page(-1))
        self.next_btn = MDIconButton(icon="chevron-right")
        self.next_btn.bind(on_release=lambda *_: self.change_page(1))
        self.page_label = MDLabel(
            text="", halign="center", size_hint_x=None, width=dp(110), adaptive_height=True,
            pos_hint={"center_y": 0.5},
        )

        zoom_out = MDIconButton(icon="magnify-minus-outline")
        zoom_out.bind(on_release=lambda *_: self.viewer.zoom_by(1 / ZoomableImage.STEP))
        zoom_in = MDIconButton(icon="magnify-plus-outline")
        zoom_in.bind(on_release=lambda *_: self.viewer.zoom_by(ZoomableImage.STEP))
        fit_btn = MDIconButton(icon="fit-to-screen-outline")
        fit_btn.bind(on_release=lambda *_: self.viewer.fit())
        self.zoom_label = MDLabel(
            text="100%", halign="center", size_hint_x=None, width=dp(64), adaptive_height=True,
            pos_hint={"center_y": 0.5},
        )

        controls = MDBoxLayout(orientation="horizontal", adaptive_height=True, spacing=dp(4))
        for w in (self.prev_btn, self.page_label, self.next_btn, Widget(),
                  zoom_out, self.zoom_label, zoom_in, fit_btn):
            controls.add_widget(w)
        content.add_widget(controls)

        nav = make_row(
            make_button("Back to Upload", style="outlined", on_release=self.go_back),
            make_button("Continue to Evaluation", on_release=self.go_evaluate),
        )
        nav.adaptive_size = True
        nav.pos_hint = {"center_x": 0.5}
        content.add_widget(nav)

        root.add_widget(content)
        self.add_widget(root)

    # ------------------------------------------------------------ lifecycle
    def on_pre_enter(self, *args):
        upload = self.manager.get_screen("upload")
        jobs = [(QP_LABEL, [upload.qp_input.text])]
        for s in upload.answer_scripts:
            jobs.append((s.student_id, list(s.sources)))

        signature = tuple((label, tuple(src)) for label, src in jobs)
        if signature != self.signature:  # only re-process when the inputs changed
            self.signature = signature
            self._start_processing(jobs)

    def cleanup(self):
        self._token += 1
        if self.out_dir:
            shutil.rmtree(self.out_dir, ignore_errors=True)

    # ----------------------------------------------------------- processing
    def _start_processing(self, jobs):
        self.cleanup()
        self.out_dir = tempfile.mkdtemp(prefix="evaluator_")
        token = self._token

        self.docs, self.errors = {}, {}
        self.order = [label for label, _ in jobs]
        self.current, self.page = None, 0
        self.doc_text.text = "Select document"
        self.viewer.clear()
        self.page_label.text = ""
        self.status.text = f"Pre-processing 0/{len(jobs)} ..."

        threading.Thread(
            target=self._work, args=(jobs, token, self.out_dir), daemon=True
        ).start()

    def _work(self, jobs, token, out_dir):
        total = len(jobs)
        for i, (label, sources) in enumerate(jobs, start=1):
            if token != self._token:
                return
            pages, err = None, None
            try:
                pages = process_document(sources, out_dir, f"{i}_{label}")
            except Exception as e:  # report per document, keep going
                err = f"{type(e).__name__}: {e}"
            Clock.schedule_once(partial(self._on_doc_done, token, label, pages, err, i, total))

    def _on_doc_done(self, token, label, pages, err, done, total, *_):
        if token != self._token:
            return
        if err:
            self.errors[label] = err
        else:
            self.docs[label] = pages

        if done < total:
            self.status.text = f"Pre-processing {done}/{total} ..."
        elif self.errors:
            first = next(iter(self.errors.items()))
            self.status.text = (
                f"Done. {len(self.errors)} failed (e.g. {first[0]}: {first[1]})"
            )
        else:
            self.status.text = f"All {total} document(s) pre-processed."

        if self.current is None and not err:
            self.select_doc(label, announce=False)

    # ------------------------------------------------------------ viewing
    def open_menu(self, *_):
        if not self.order:
            return
        items = []
        for label in self.order:
            if label in self.docs:
                text = label
            elif label in self.errors:
                text = f"{label} (failed)"
            else:
                text = f"{label} (processing...)"
            items.append({"text": text, "on_release": partial(self.select_doc, label)})
        self.menu = MDDropdownMenu(caller=self.doc_btn, items=items)
        self.menu.open()

    def select_doc(self, label, announce=True, *_):
        if self.menu:
            self.menu.dismiss()
        if label in self.docs:
            self.current, self.page = label, 0
            self.doc_text.text = label
            self._show_page()
        elif announce and label in self.errors:
            notify(f"{label}: {self.errors[label]}")
        elif announce:
            notify(f"'{label}' is still being processed.")

    def change_page(self, delta):
        if self.current:
            self.page += delta
            self._show_page()

    def toggle_view(self, *_):
        self.view_mode = (self.view_mode + 1) % len(VIEW_MODES)
        self.view_text.text = f"View: {VIEW_MODES[self.view_mode]}"
        self._show_page(reset=False)  # keep zoom/position when only the view changes

    def _on_zoom(self, zoom):
        self.zoom_label.text = f"{round(zoom * 100)}%"

    def _show_page(self, reset=True):
        pages = self.docs.get(self.current)
        if not pages:
            return
        self.page = max(0, min(self.page, len(pages) - 1))
        self.viewer.show(pages[self.page][VIEW_INDEX[VIEW_MODES[self.view_mode]]], reset=reset)
        self.page_label.text = f"Page {self.page + 1} / {len(pages)}"
        self.prev_btn.disabled = self.page == 0
        self.next_btn.disabled = self.page >= len(pages) - 1

    def go_evaluate(self, *_):
        self.manager.transition.direction = "left"
        self.manager.current = "evaluation"

    def go_back(self, *_):
        self.manager.transition.direction = "right"
        self.manager.current = "upload"


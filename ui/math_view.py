"""MathView: a label that typesets its LaTeX.

Text with no math is just an MDLabel. Text with math is shown as a plain-text label
at first, then swapped for a KaTeX-rendered image once the background renderer finishes
(and re-rendered when the widget's width changes). If rendering is unavailable the
plain label simply stays, so the screen never breaks.
"""
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.uix.image import Image
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel

from ui.math_render import SETUP_HINT, has_math, plain_text, renderer
from ui.widgets import notify

_FONT_SP = {"large": 16, "medium": 14, "small": 12}
_warned = False


def _warn_once(err):
    global _warned
    if _warned:
        return
    _warned = True
    notify(f"Math rendering unavailable: {str(err)[:90]}  [{SETUP_HINT}]"[:200])


class MathView(MDBoxLayout):
    def __init__(self, text="", role="large", only_math=False, **kwargs):
        """only_math=True shows nothing at all unless the text contains math
        (used for the live preview under the feedback box)."""
        super().__init__(orientation="vertical", adaptive_height=True, **kwargs)
        self.role = role
        self.only_math = only_math
        self._text = ""
        self._req = 0
        self._ev = None
        self._image = None
        self._label = MDLabel(font_style="Body", role=role, adaptive_height=True)
        self.add_widget(self._label)
        self.bind(width=self._schedule)
        self.set_text(text)

    # ---- public ----
    def set_text(self, text):
        self._text = text or ""
        self._req += 1
        math = has_math(self._text)
        if math and renderer.available():
            # keep the previous image on screen until the new one is ready
            self._schedule()
            if self._image is None:
                self._label.text = plain_text(self._text)
            return
        self._drop_image()
        self._label.text = "" if (self.only_math and not math) else plain_text(self._text)
        if math:  # math present but no renderer: tell the user once how to enable it
            Clock.schedule_once(lambda dt: _warn_once(renderer.error or "playwright is not installed"))

    # ---- internals ----
    def _schedule(self, *_):
        if self._ev:
            self._ev.cancel()
            self._ev = None
        if has_math(self._text) and renderer.available() and self.width > dp(60):
            self._ev = Clock.schedule_once(self._request, 0.25)

    def _request(self, *_):
        self._ev = None
        req = self._req = self._req + 1
        px = int(sp(_FONT_SP.get(self.role, 16)))

        def done(path, err):  # worker thread -> hop back to the Kivy thread
            if err or not path:
                Clock.schedule_once(lambda dt: _warn_once(err))
                return
            Clock.schedule_once(lambda dt: self._show(req, path))

        renderer.render_async(self._text, int(self.width), px, done)

    def _show(self, req, path):
        if req != self._req:
            return  # a newer request superseded this one
        if self._image is None:
            self._image = Image(size_hint=(1, None), fit_mode="fill", height=dp(24))
            self._image.bind(texture=self._fit)
            self.bind(width=self._fit)
            if self._label.parent is self:
                self.remove_widget(self._label)
            self.add_widget(self._image)
        self._image.source = path
        self._image.reload()
        self._fit()

    def _fit(self, *_):
        img = self._image
        if img is not None and img.texture is not None and img.texture.width:
            img.height = self.width * img.texture.height / img.texture.width

    def _drop_image(self):
        if self._image is not None:
            self.remove_widget(self._image)
            self.unbind(width=self._fit)
            self._image = None
        if self._label.parent is not self:
            self.add_widget(self._label)

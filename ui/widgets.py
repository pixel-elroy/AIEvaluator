"""Shared UI helpers so every screen looks consistent."""
import os

from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.widget import Widget
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDButton, MDButtonText
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import (
    MDDialog,
    MDDialogButtonContainer,
    MDDialogContentContainer,
    MDDialogHeadlineText,
)
from kivymd.uix.label import MDLabel
from kivymd.uix.snackbar import MDSnackbar, MDSnackbarText
from kivymd.uix.textfield import MDTextField, MDTextFieldHintText
from kivymd.uix.appbar import (
    MDActionTopAppBarButton,
    MDTopAppBar,
    MDTopAppBarLeadingButtonContainer,
    MDTopAppBarTitle,
)

# --- KivyMD 2.x fix -----------------------------------------------------------
# Ripple behaviour (MDCard, MDSnackbar, MDDialog, ...) creates an off-screen buffer
# the size of the widget when it is built. Self-sizing widgets start at height 0, and a
# 0-sized buffer crashes with "FBO Initialization failed: Incomplete attachment".
# Clamp the buffer to >= 1px. Runs at import time, before any widget is created.
from kivymd.uix.behaviors import ripple_behavior as _ripple

_RealFbo = _ripple.Fbo


def _safe_fbo(*args, **kwargs):
    size = kwargs.get("size")
    if size is not None:
        kwargs["size"] = (max(1, int(size[0])), max(1, int(size[1])))
    return _RealFbo(*args, **kwargs)


_ripple.Fbo = _safe_fbo
# ------------------------------------------------------------------------------

MAX_CONTENT_WIDTH = dp(820)

# --- Theme: white screens, soft green surfaces, black text (readable) ---------
# All greens stay light so black labels keep high contrast.
BLACK = (0.08, 0.12, 0.08, 1)           # near-black with a hint of green
WHITE = (1, 1, 1, 1)
PALE_GREEN = (0.93, 0.97, 0.91, 1)      # card / surface background
MINT_GREEN = (0.88, 0.95, 0.86, 1)      # secondary surface (alt cards)
LIGHT_GREEN = (0.72, 0.88, 0.68, 1)     # filled buttons, app bar
SOFT_GREEN = (0.80, 0.91, 0.76, 1)      # tonal / secondary buttons
LINE_GREEN = (0.55, 0.78, 0.50, 1)      # outlines, borders
ACCENT_GREEN = (0.45, 0.72, 0.42, 1)    # subtle accent (icons, highlights)


def muted_label(text, **kwargs):
    """Secondary-colour body text (black for readability on pale green)."""
    return MDLabel(
        text=text,
        font_style="Body",
        role="medium",
        adaptive_height=True,
        theme_text_color="Custom",
        text_color=BLACK,
        **kwargs,
    )


def fit_height(widget, floor=dp(8)):
    """Size a widget to its content WITHOUT ever letting its height hit 0.

    MDCard builds a GPU buffer the size of the card when it is created;
    a 0-height card (what adaptive_height gives before children exist)
    crashes with 'FBO Initialization failed'.
    """
    widget.size_hint_y = None
    widget.height = max(widget.minimum_height, floor)
    widget.bind(minimum_height=lambda w, h: setattr(w, "height", max(h, floor)))


def make_card(title, subtitle="", *, mint=False):
    """Outlined card that sizes itself to its content.

    mint=True uses a slightly deeper mint surface for visual rhythm between cards.
    """
    card = MDCard(
        style="outlined",
        orientation="vertical",
        padding=dp(20),
        spacing=dp(14),
        size_hint_y=None,
        height=dp(100),
        size_hint_x=1,
        size_hint_max_x=MAX_CONTENT_WIDTH,
        pos_hint={"center_x": 0.5},
        theme_bg_color="Custom",
        md_bg_color=MINT_GREEN if mint else PALE_GREEN,
        theme_line_color="Custom",
        line_color=LINE_GREEN,
    )
    fit_height(card)
    card.add_widget(
        MDLabel(
            text=title,
            font_style="Title",
            role="medium",
            adaptive_height=True,
            theme_text_color="Custom",
            text_color=BLACK,
        )
    )
    if subtitle:
        card.add_widget(muted_label(subtitle))
    return card


def make_path_field(hint):
    field = MDTextField(mode="outlined", readonly=True)
    field.add_widget(MDTextFieldHintText(text=hint))
    return field


def make_row(*widgets):
    row = MDBoxLayout(orientation="horizontal", spacing=dp(12), adaptive_height=True)
    for w in widgets:
        row.add_widget(w)
    return row


def make_button(text, style="filled", on_release=None, **kwargs):
    label = MDButtonText(text=text, theme_text_color="Custom", text_color=BLACK)
    if style in ("filled", "elevated"):
        kwargs.setdefault("theme_bg_color", "Custom")
        kwargs.setdefault("md_bg_color", LIGHT_GREEN)
    elif style == "tonal":
        kwargs.setdefault("theme_bg_color", "Custom")
        kwargs.setdefault("md_bg_color", SOFT_GREEN)
    elif style == "outlined":
        kwargs.setdefault("theme_line_color", "Custom")
        kwargs.setdefault("line_color", LINE_GREEN)
    btn = MDButton(label, style=style, **kwargs)
    if on_release:
        btn.bind(on_release=on_release)
    return btn


def make_app_bar(title, on_back=None):
    bar = MDTopAppBar(type="small", theme_bg_color="Custom", md_bg_color=LIGHT_GREEN)
    if on_back:
        container = MDTopAppBarLeadingButtonContainer()
        back = MDActionTopAppBarButton(
            icon="arrow-left", theme_icon_color="Custom", icon_color=BLACK)
        back.bind(on_release=on_back)
        container.add_widget(back)
        bar.add_widget(container)
    bar.add_widget(MDTopAppBarTitle(text=title, theme_text_color="Custom", text_color=BLACK))
    return bar


def notify(message):
    MDSnackbar(
        MDSnackbarText(text=message),
        y=dp(24),
        pos_hint={"center_x": 0.5},
        size_hint_x=0.6,
    ).open()


class ChooserDialog:
    """File / folder picker built with the KivyMD 2.x dialog API."""

    def __init__(self, title, on_select, extensions=None, dirselect=False, multiselect=False):
        self.on_select = on_select
        self.dirselect = dirselect

        filters = []
        if extensions:
            exts = tuple(e.lower() for e in extensions)
            filters = [lambda folder, name: name.lower().endswith(exts)]

        self.chooser = FileChooserListView(
            path=os.path.expanduser("~"),
            filters=filters,
            dirselect=dirselect,
            multiselect=multiselect,
            size_hint_y=None,
        )
        self._fit_height()
        Window.bind(on_resize=self._on_window_resize)

        cancel = make_button("Cancel", style="text", on_release=lambda *_: self.dismiss())
        select = make_button(
            "Select Folder" if dirselect else "Select", on_release=self._confirm
        )

        self.dialog = MDDialog(
            MDDialogHeadlineText(text=title),
            MDDialogContentContainer(self.chooser, orientation="vertical"),
            MDDialogButtonContainer(Widget(), cancel, select, spacing=dp(8)),
        )

    def open(self):
        self.dialog.open()

    def dismiss(self):
        Window.unbind(on_resize=self._on_window_resize)
        self.dialog.dismiss()

    def _fit_height(self):
        # between 200dp and 340dp, ~45% of the window height
        self.chooser.height = max(dp(200), min(dp(340), Window.height * 0.45))

    def _on_window_resize(self, *_):
        self._fit_height()

    def _confirm(self, *_):
        """Calls on_select with a LIST of chosen paths."""
        selection = list(self.chooser.selection)
        if not selection:
            if self.dirselect:
                selection = [self.chooser.path]  # nothing highlighted: use the folder being viewed
            else:
                notify("Please pick a file first.")
                return
        self.dismiss()
        self.on_select(selection)


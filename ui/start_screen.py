from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDIcon, MDLabel
from kivymd.uix.screen import MDScreen

from ui.widgets import fit_height, make_button, muted_label


class StartScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        card = MDCard(
            style="elevated",
            orientation="vertical",
            padding=dp(40),
            spacing=dp(20),
            size_hint_y=None,
            height=dp(100),
            size_hint_x=0.85,
            size_hint_max_x=dp(560),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        fit_height(card)

        theme = MDApp.get_running_app().theme_cls
        icon = MDIcon(
            icon="text-box-check-outline",
            theme_font_size="Custom",
            font_size=dp(64),
            theme_text_color="Custom",
            text_color=theme.primaryColor,
            halign="center",
            adaptive_height=True,
        )

        title = MDLabel(
            text="AI Automated Answer Script Evaluation",
            font_style="Headline",
            role="small",
            halign="center",
            adaptive_height=True,
        )
        subtitle = muted_label(
            "Rubric-based marking of handwritten answer scripts, with teacher review", halign="center"
        )

        buttons = MDBoxLayout(
            orientation="vertical",
            spacing=dp(8),
            adaptive_height=True,
            padding=[0, dp(12), 0, 0],
        )
        proceed = make_button(
            "Proceed to Evaluate", on_release=self.go_to_upload, pos_hint={"center_x": 0.5}
        )
        results_btn = make_button(
            "Previous results", style="outlined", on_release=self.go_to_results,
            pos_hint={"center_x": 0.5},
        )
        scores_btn = make_button(
            "Student scores table", style="outlined", on_release=self.go_to_scores,
            pos_hint={"center_x": 0.5},
        )
        settings_btn = make_button(
            "Settings", style="outlined", on_release=self.go_to_settings,
            pos_hint={"center_x": 0.5},
        )
        exit_btn = make_button(
            "Exit", style="text", on_release=self.exit_app, pos_hint={"center_x": 0.5}
        )
        buttons.add_widget(proceed)
        buttons.add_widget(results_btn)
        buttons.add_widget(scores_btn)
        buttons.add_widget(settings_btn)
        buttons.add_widget(exit_btn)

        for w in (icon, title, subtitle, buttons):
            card.add_widget(w)
        self.add_widget(card)

    def go_to_upload(self, *_):
        self.manager.transition.direction = "left"
        self.manager.current = "upload"

    def go_to_results(self, *_):
        self.manager.get_screen("review").back_to = "start"
        self.manager.transition.direction = "left"
        self.manager.current = "review"

    def go_to_settings(self, *_):
        self.manager.transition.direction = "left"
        self.manager.current = "settings"

    def go_to_scores(self, *_):
        self.manager.transition.direction = "left"
        self.manager.current = "scores"

    def exit_app(self, *_):
        MDApp.get_running_app().stop()


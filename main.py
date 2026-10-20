from kivy.config import Config

# Window settings must be applied BEFORE kivy.core.window is imported.
# (Changing Window.size inside build() can break the GL context on some Linux drivers.)
Config.set("graphics", "width", "1000")
Config.set("graphics", "height", "720")
Config.set("graphics", "resizable", 1)

from kivy.core.window import Window
from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager

from ui.widgets import WHITE
from ui.start_screen import StartScreen
from ui.upload_screen import UploadScreen
from ui.inspection_screen import InspectionScreen
from ui.settings_screen import SettingsScreen
from ui.evaluation_screen import EvaluationScreen
from ui.review_screen import ReviewScreen
from ui.results_screen import ResultsScreen
from ui.scores_screen import ScoresScreen


class AnswerEvaluatorApp(MDApp):
    def build(self):
        self.title = "Answer Script Evaluator"
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Lightgreen"

        sm = MDScreenManager()
        sm.add_widget(StartScreen(name="start"))
        sm.add_widget(UploadScreen(name="upload"))
        sm.add_widget(InspectionScreen(name="inspection"))
        sm.add_widget(EvaluationScreen(name="evaluation"))
        sm.add_widget(ReviewScreen(name="review"))
        sm.add_widget(ResultsScreen(name="results"))
        sm.add_widget(ScoresScreen(name="scores"))
        sm.add_widget(SettingsScreen(name="settings"))
        for screen in sm.screens:  # white screens; light-green cards/buttons come from widgets.py
            screen.theme_bg_color = "Custom"
            screen.md_bg_color = WHITE
        return sm

    def on_stop(self):
        self.root.get_screen("inspection").cleanup()

    def on_start(self):
        # Safe to touch the window once the app is running
        Window.minimum_width = dp(560)
        Window.minimum_height = dp(520)


if __name__ == "__main__":
    AnswerEvaluatorApp().run()

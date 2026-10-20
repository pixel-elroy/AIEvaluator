"""Results: class marksheet, per-student PDF reports, and agreement with faculty marks."""
import os
import threading

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView

from core import evaluator
from core.ocr_engine import safe_name
from ui.evaluation_screen import get_store
from ui.math_render import renderer
from ui.widgets import ChooserDialog, make_app_bar, make_button, make_card, make_row, muted_label, notify

AGREEMENT_HELP = (
    "Compare the AI's marks with marks given by faculty. Import a CSV with the columns "
    "student, question, marks (student = roll number or script label; question = 1, 2a, ...)."
)


def _f(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


class ResultsScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._report = None

        root = MDBoxLayout(orientation="vertical")
        root.add_widget(make_app_bar("Results & Export", self.go_back))
        scroll = MDScrollView(do_scroll_x=False)
        body = MDBoxLayout(orientation="vertical", padding=[dp(24), dp(24), dp(24), dp(32)],
                           spacing=dp(20), adaptive_height=True)

        table_card = make_card("Class results")
        self.table = MDBoxLayout(orientation="vertical", spacing=dp(6), adaptive_height=True)
        table_card.add_widget(self.table)
        body.add_widget(table_card)

        export_card = make_card("Export", "Choose a folder; files are written there.")
        export_card.add_widget(make_row(
            make_button("Class marksheet (CSV)", style="elevated", on_release=self.export_marksheet),
            make_button("Student reports (PDF)", style="elevated", on_release=self.export_reports)))
        body.add_widget(export_card)

        agree_card = make_card("Agreement with faculty", AGREEMENT_HELP)
        agree_card.add_widget(make_row(
            make_button("Import faculty marks (CSV)", style="elevated", on_release=self.import_faculty),
            make_button("Export comparison (CSV)", style="text", on_release=self.export_comparison)))
        self.agree_label = MDLabel(text="", font_style="Body", role="medium", adaptive_height=True)
        agree_card.add_widget(self.agree_label)
        body.add_widget(agree_card)

        nav_card = make_card(
            "Next steps",
            "Start another evaluation or leave the app. Scores stay saved in the database.",
        )
        nav_card.add_widget(make_row(
            make_button("New evaluation (Upload)", style="elevated", on_release=self.go_upload),
            make_button("View scores table", style="tonal", on_release=self.go_scores),
            make_button("Exit", style="text", on_release=self.exit_app),
        ))
        body.add_widget(nav_card)

        scroll.add_widget(body)
        root.add_widget(scroll)
        self.add_widget(root)

    # ------------------------------------------------------- lifecycle
    def on_pre_enter(self, *args):
        self.store = get_store()
        self._render_table()
        self._render_agreement()

    def _render_table(self):
        self.table.clear_widgets()
        students = self.store.list_students()
        if not students:
            self.table.add_widget(muted_label("No results yet."))
            return
        for s in students:
            evs = self.store.get_evaluations(s.script_key)
            final, _ai, mx = self.store.totals(s.script_key)
            if evs and all(e.approved for e in evs):
                status = "approved"
            elif s.status == "done":
                status = "awaiting review"
            else:
                status = s.status
            self.table.add_widget(MDLabel(
                text=f"{s.display}   ·   {final:g} / {mx:g}   ·   {status}",
                font_style="Body", role="large", adaptive_height=True))

    # ---------------------------------------------------------- exports
    def export_marksheet(self, *_):
        ChooserDialog("Choose a folder for the marksheet", self._write_marksheet, dirselect=True).open()

    def _write_marksheet(self, paths):
        target = os.path.join(paths[0], "marksheet.csv")
        try:
            evaluator.export_marksheet_csv(self.store, target)
        except (ValueError, OSError) as e:
            notify(f"Could not write the marksheet: {e}")
            return
        notify(f"Saved {target}")

    def export_reports(self, *_):
        ChooserDialog("Choose a folder for the PDF reports", self._write_reports, dirselect=True).open()

    def _write_reports(self, paths):
        keys = [s.script_key for s in self.store.list_students()
                if self.store.get_evaluations(s.script_key)]
        if not keys:
            notify("No evaluated scripts to export.")
            return
        notify(f"Writing {len(keys)} report(s)...")
        threading.Thread(target=self._report_worker, args=(keys, paths[0]), daemon=True).start()

    def _report_worker(self, keys, folder):
        """Runs off the UI thread. Math is typeset by KaTeX; if that is unavailable we fall back
        to the original text-only report so exporting never stops working."""
        from ui.report_html import write_student_report_pdf

        store = get_store()
        done, failed, fallback = 0, [], 0
        for key in keys:
            target = os.path.join(folder, safe_name(key) + ".pdf")
            try:
                try:
                    write_student_report_pdf(store, key, target)
                except Exception:
                    evaluator.write_student_report(store, key, target)  # old text-only report
                    fallback += 1
                done += 1
            except Exception as e:  # keep going; report at the end
                failed.append(f"{key} ({type(e).__name__})")
        msg = f"Wrote {done} report(s) to {folder}"
        if fallback:
            why = (renderer.error or "math rendering failed")[:70]
            msg += f"; {fallback} without typeset math ({why})"
        if failed:
            msg += f"; failed: {', '.join(failed[:3])}"
        Clock.schedule_once(lambda dt: notify(msg))

    # --------------------------------------------------------- agreement
    def import_faculty(self, *_):
        ChooserDialog("Choose the faculty marks CSV", self._read_faculty, [".csv"]).open()

    def _read_faculty(self, paths):
        try:
            count, warnings = evaluator.import_faculty_csv(self.store, paths[0])
        except (ValueError, OSError) as e:
            notify(f"Could not read the CSV: {e}")
            return
        notify(f"Imported {count} mark(s)" + (f"; {len(warnings)} row(s) skipped" if warnings else ""))
        self._render_agreement()

    def _render_agreement(self):
        if not self.store.get_faculty_marks():
            self._report = None
            self.agree_label.text = "No faculty marks imported yet."
            return
        r = self._report = evaluator.agreement_report(self.store)
        q, t = r["question"], r["total"]
        if not q.get("n"):
            self.agree_label.text = "No faculty marks matched the evaluated scripts."
        else:
            lines = [
                f"Compared {q['n']} question mark(s).",
                f"Mean absolute error: {_f(q['mae'])} marks   ·   bias: {q['bias']:+.2f} (positive = AI more generous)",
                f"Within ±0.5 mark: {q['within_half_pct']:.0f}%   ·   within ±1 mark: {q['within_one_pct']:.0f}%   ·   exact: {q['exact_pct']:.0f}%",
                f"Pearson r: {_f(q['pearson'])}   ·   Spearman: {_f(q['spearman'])}   ·   quadratic weighted kappa: {_f(q['qwk'])}",
            ]
            if t.get("n"):
                lines.append(f"Total marks ({t['n']} script(s)): MAE {_f(t['mae'])}, Pearson r {_f(t['pearson'])}")
            self.agree_label.text = "\n".join(lines)
        if r["unmatched"]:
            self.agree_label.text += "\nNot matched to any script: " + ", ".join(r["unmatched"][:8])

    def export_comparison(self, *_):
        if not self._report or not self._report["pairs"]:
            notify("Import faculty marks first.")
            return
        ChooserDialog("Choose a folder for the comparison", self._write_comparison, dirselect=True).open()

    def _write_comparison(self, paths):
        target = os.path.join(paths[0], "agreement.csv")
        try:
            evaluator.export_agreement_csv(self._report, target)
        except OSError as e:
            notify(f"Could not write the file: {e}")
            return
        notify(f"Saved {target}")

    def go_back(self, *_):
        self.manager.transition.direction = "right"
        self.manager.current = "review"

    def go_upload(self, *_):
        self.manager.transition.direction = "right"
        self.manager.current = "upload"

    def go_scores(self, *_):
        self.manager.transition.direction = "left"
        self.manager.current = "scores"

    def exit_app(self, *_):
        from kivymd.app import MDApp
        MDApp.get_running_app().stop()

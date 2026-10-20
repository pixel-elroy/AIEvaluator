"""Teacher review: page image beside each question's extracted answer, AI marks and rationale.
The teacher can change marks and feedback and approve each answer."""
from functools import partial

from kivy.metrics import dp
from kivy.uix.widget import Widget
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDButton, MDButtonIcon, MDButtonText, MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.textfield import MDTextField, MDTextFieldHintText

from ui.evaluation_screen import get_store
from ui.inspection_screen import ZoomableImage
from ui.math_render import has_math
from ui.math_view import MathView
from ui.widgets import make_app_bar, make_button, make_card, make_row, muted_label, notify

WIDE_LAYOUT_DP = 900  # side-by-side below this width the image goes on top instead


def _field(hint, **kwargs):
    field = MDTextField(mode="outlined", **kwargs)
    field.add_widget(MDTextFieldHintText(text=hint))
    return field


class ReviewScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.back_to = "evaluation"
        self.store = None
        self.paper = None
        self.students = []
        self.key = None
        self.pages = []
        self.page = 0
        self.refs = {}
        self.menu = None

        root = MDBoxLayout(orientation="vertical")
        root.add_widget(make_app_bar("Review & Marks", self.go_back))
        content = MDBoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))

        # --- student picker ---
        self.student_text = MDButtonText(text="Select student")
        self.student_btn = MDButton(
            MDButtonIcon(icon="account-outline"), self.student_text, style="outlined")
        self.student_btn.bind(on_release=self.open_menu)
        prev_s = MDIconButton(icon="chevron-left")
        prev_s.bind(on_release=lambda *_: self.step_student(-1))
        next_s = MDIconButton(icon="chevron-right")
        next_s.bind(on_release=lambda *_: self.step_student(1))
        toolbar = MDBoxLayout(orientation="horizontal", spacing=dp(8), adaptive_height=True)
        for w in (prev_s, self.student_btn, next_s, Widget(),
                  make_button("Results & export", style="tonal", on_release=self.go_results)):
            toolbar.add_widget(w)
        content.add_widget(toolbar)

        # --- identity (read by the model; the teacher confirms or corrects) ---
        self.name_input = _field("Student name")
        self.roll_input = _field("Roll number")
        content.add_widget(make_row(self.name_input, self.roll_input))
        self.total_label = MDLabel(text="", font_style="Title", role="medium", adaptive_height=True)
        self.note_label = muted_label("")
        content.add_widget(self.total_label)
        content.add_widget(self.note_label)

        # --- body: page image + question cards ---
        self.body = MDBoxLayout(orientation="horizontal", spacing=dp(12))
        self.viewer_box = MDBoxLayout(orientation="vertical", size_hint=(0.45, 1), spacing=dp(4))
        self.viewer = ZoomableImage(size_hint_y=1)
        self.viewer_box.add_widget(self.viewer)

        self.page_label = MDLabel(text="", halign="center", size_hint_x=None, width=dp(96),
                                  adaptive_height=True, pos_hint={"center_y": 0.5})
        controls = MDBoxLayout(orientation="horizontal", adaptive_height=True, spacing=dp(2))
        for w in (self._icon("chevron-left", lambda *_: self.change_page(-1)), self.page_label,
                  self._icon("chevron-right", lambda *_: self.change_page(1)), Widget(),
                  self._icon("magnify-minus-outline", lambda *_: self.viewer.zoom_by(1 / ZoomableImage.STEP)),
                  self._icon("magnify-plus-outline", lambda *_: self.viewer.zoom_by(ZoomableImage.STEP)),
                  self._icon("fit-to-screen-outline", lambda *_: self.viewer.fit())):
            controls.add_widget(w)
        self.viewer_box.add_widget(controls)

        self.cards_scroll = MDScrollView(do_scroll_x=False, size_hint=(0.55, 1))
        self.cards = MDBoxLayout(orientation="vertical", adaptive_height=True,
                                 spacing=dp(12), padding=[dp(4), dp(4), dp(4), dp(16)])
        self.cards_scroll.add_widget(self.cards)
        self.body.add_widget(self.viewer_box)
        self.body.add_widget(self.cards_scroll)
        content.add_widget(self.body)

        actions = make_row(make_button("Save changes", on_release=lambda *_: self._save_current(quiet=False)),
                           make_button("Approve all", style="outlined", on_release=self.approve_all))
        actions.adaptive_size = True
        actions.pos_hint = {"center_x": 0.5}
        content.add_widget(actions)

        root.add_widget(content)
        self.add_widget(root)
        self.bind(size=self._adapt_layout)

    @staticmethod
    def _icon(name, callback):
        btn = MDIconButton(icon=name)
        btn.bind(on_release=callback)
        return btn

    def _adapt_layout(self, *_):
        wide = self.width >= dp(WIDE_LAYOUT_DP)
        self.body.orientation = "horizontal" if wide else "vertical"
        self.viewer_box.size_hint = (0.45, 1) if wide else (1, 0.4)
        self.cards_scroll.size_hint = (0.55, 1) if wide else (1, 0.6)

    # ------------------------------------------------------- lifecycle
    def on_pre_enter(self, *args):
        self._adapt_layout()
        self.store = get_store()
        self.paper = self.store.load_paper()
        self.students = self.store.list_students()
        keys = [s.script_key for s in self.students]
        if not self.paper or not self.students:
            self._show_empty()
        elif self.key in keys:
            self._load_student(self.key)
        else:
            self._load_student(keys[0])

    def on_leave(self, *args):
        self._save_current(quiet=True)

    def _show_empty(self):
        self.key = None
        self.refs = {}
        self.cards.clear_widgets()
        self.cards.add_widget(muted_label("No evaluated scripts yet. Run an evaluation first."))
        self.student_text.text = "Select student"
        self.total_label.text = ""
        self.note_label.text = ""
        self.name_input.text = self.roll_input.text = ""
        self.pages = []
        self.viewer.clear()
        self.page_label.text = ""

    # --------------------------------------------------- student picker
    def open_menu(self, *_):
        if not self.students:
            return
        items = []
        for s in self.students:
            evs = self.store.get_evaluations(s.script_key)
            final, _ai, mx = self.store.totals(s.script_key)
            tick = "✓ " if evs and all(e.approved for e in evs) else ""
            items.append({"text": f"{tick}{s.display}  ·  {final:g}/{mx:g}",
                          "on_release": partial(self.select_student, s.script_key)})
        self.menu = MDDropdownMenu(caller=self.student_btn, items=items)
        self.menu.open()

    def select_student(self, key, *_):
        if self.menu:
            self.menu.dismiss()
        if key != self.key and not self._save_current(quiet=True):
            return  # invalid marks: stay put so the teacher can fix them
        self.students = self.store.list_students()
        self._load_student(key)

    def step_student(self, delta):
        keys = [s.script_key for s in self.students]
        if self.key in keys:
            self.select_student(keys[(keys.index(self.key) + delta) % len(keys)])

    # ------------------------------------------------------ show a script
    def _load_student(self, key):
        store, paper = self.store, self.paper
        self.key = key
        rec = store.get_student(key)
        self.student_text.text = rec.display
        self.name_input.text, self.roll_input.text = rec.name, rec.roll_no
        self.pages = store.get_pages(key)
        answers = {a.question_id: a for a in store.get_answers(key)}
        evs = {e.question_id: e for e in store.get_evaluations(key)}

        self.cards.clear_widgets()
        self.refs = {}
        for q in paper.questions:
            self.cards.add_widget(self._build_card(q, evs.get(q.id), answers.get(q.id)))

        notes = []
        if rec.status != "done":
            notes.append(f"This script is not fully evaluated (status: {rec.status})."
                         + (f" {rec.error}" if rec.error else "") + " Re-run the evaluation to finish it.")
        rubric = store.load_rubric(paper)
        if rubric and rubric.drafted:
            notes.append("The marking scheme was drafted by the AI, not supplied by a teacher: check the marks carefully.")
        self.note_label.text = "\n".join(notes)
        self._update_total()
        self._show_page(0)

    def _build_card(self, q, ev, ans):
        card = make_card(f"Question {q.id}  ·  {q.max_marks:g} marks")
        card.add_widget(MathView(q.text, role="medium"))
        if ev is None:
            card.add_widget(muted_label("Not evaluated yet."))
            return card
        if ev.needs_review:
            card.add_widget(MDLabel(
                text="⚠ Please check: " + (ev.review_reason or "flagged by the model"),
                theme_text_color="Error", font_style="Body", role="medium", adaptive_height=True))
        card.add_widget(muted_label("Student's answer (as extracted):"))
        card.add_widget(MathView(ans.text if ans and ans.text else "(no answer found)", role="large"))
        if ev.points:
            card.add_widget(muted_label("AI marking:"))
            card.add_widget(MathView(
                "\n".join(f"{p['awarded']:g}/{p['max']:g}   {p['description']}"
                          + (f" — {p['comment']}" if p.get("comment") else "") for p in ev.points),
                role="medium"))
        card.add_widget(MathView("Rationale: " + ev.rationale, role="medium"))

        marks = _field("Marks", input_filter="float")
        marks.text = f"{ev.marks:g}"
        marks.size_hint_x = None
        marks.width = dp(120)
        out_of = MDLabel(text=f"out of {q.max_marks:g}   (AI suggested {ev.ai_marks:g})",
                         adaptive_height=True, pos_hint={"center_y": 0.5})
        view = make_button("View page", style="text", on_release=partial(self._goto_answer_page, ans),
                           pos_hint={"center_y": 0.5})
        card.add_widget(make_row(marks, out_of, view))

        feedback = _field("Feedback to the student", multiline=True)
        feedback.max_height = dp(160)
        feedback.text = ev.feedback_text
        card.add_widget(feedback)
        # live typeset preview: only appears when the feedback contains LaTeX
        preview = MathView("", role="medium", only_math=True)
        feedback.bind(text=lambda _w, val, p=preview: p.set_text("Preview: " + val if has_math(val) else ""))
        preview.set_text("Preview: " + ev.feedback_text if has_math(ev.feedback_text) else "")
        card.add_widget(preview)

        approve = MDCheckbox(active=ev.approved, size_hint=(None, None), size=(dp(40), dp(40)))
        card.add_widget(make_row(approve, MDLabel(text="Approved", adaptive_height=True,
                                                  pos_hint={"center_y": 0.5})))
        self.refs[q.id] = {"marks": marks, "feedback": feedback, "approve": approve, "ev": ev}
        return card

    # ------------------------------------------------------- page viewer
    def _show_page(self, index, reset=True):
        if not self.pages:
            self.viewer.clear()
            self.page_label.text = "No page images"
            return
        self.page = max(0, min(index, len(self.pages) - 1))
        self.viewer.show(self.pages[self.page][1], reset=reset)  # the enhanced page
        self.page_label.text = f"Page {self.page + 1} / {len(self.pages)}"

    def change_page(self, delta):
        self._show_page(self.page + delta)

    def _goto_answer_page(self, ans, *_):
        if ans and ans.pages:
            self._show_page(ans.pages[0] - 1)
        else:
            notify("No page is recorded for this answer.")

    # ------------------------------------------------------------- save
    def _collect(self):
        edits = []
        for q in self.paper.questions:
            ref = self.refs.get(q.id)
            if not ref:
                continue
            try:
                marks = float(ref["marks"].text.strip())
            except ValueError:
                raise ValueError(f"Question {q.id}: marks must be a number.")
            if not 0 <= marks <= q.max_marks:
                raise ValueError(f"Question {q.id}: marks must be between 0 and {q.max_marks:g}.")
            edits.append((q.id, round(marks, 2), ref["feedback"].text.strip(), ref["approve"].active))
        return edits

    def _save_current(self, quiet=True):
        """Validate and store this script's edits. Returns False (and says why) if invalid."""
        if not self.key or not self.paper or not self.refs:
            return True
        try:
            edits = self._collect()
        except ValueError as e:
            notify(str(e))
            return False
        for qid, marks, feedback, approved in edits:
            ev = self.refs[qid]["ev"]
            keep_marks = abs(marks - ev.ai_marks) > 1e-9 or approved  # None = untouched AI marks
            self.store.update_teacher(self.key, qid, marks if keep_marks else None,
                                      feedback if feedback != ev.feedback else None, approved)
        self.store.upsert_student(self.key, name=self.name_input.text.strip(),
                                  roll_no=self.roll_input.text.strip(), id_checked=1)
        try:
            # force=True: restoring this student after a scores-table delete
            self.store.sync_student_score(self.key, force=True)
        except Exception:
            pass
        self._update_total()
        if not quiet:
            notify("Saved.")
        return True

    def approve_all(self, *_):
        for ref in self.refs.values():
            ref["approve"].active = True
        self._save_current(quiet=False)

    def _update_total(self):
        final, ai, mx = self.store.totals(self.key)
        self.total_label.text = f"Total: {final:g} / {mx:g}" + (f"   (AI suggested {ai:g})" if abs(final - ai) > 1e-9 else "")

    # ------------------------------------------------------ navigation
    def go_back(self, *_):
        self.manager.transition.direction = "right"
        self.manager.current = self.back_to

    def go_results(self, *_):
        if not self._save_current(quiet=True):
            return
        self.manager.transition.direction = "left"
        self.manager.current = "results"

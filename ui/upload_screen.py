import os
import re
from dataclasses import dataclass, field

from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.screen import MDScreen
from kivymd.uix.textfield import MDTextField, MDTextFieldHintText

from ui.widgets import (
    ChooserDialog,
    make_app_bar,
    make_button,
    make_card,
    make_path_field,
    make_row,
    muted_label,
    notify,
)


# ----------------------------------------------------------------------
# Answer-script data helpers (kept here so this file is self-contained)
# ----------------------------------------------------------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def natural_key(path):
    """Sort page1, page2, page10 in human order."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


@dataclass
class AnswerScript:
    """One student's answer script: a single PDF OR a group of page images."""

    student_id: str
    sources: list = field(default_factory=list)

    @property
    def is_pdf(self):
        return len(self.sources) == 1 and self.sources[0].lower().endswith(".pdf")

    @property
    def summary(self):
        if self.is_pdf:
            return f"PDF · {os.path.basename(self.sources[0])}"
        n = len(self.sources)
        return f"{n} image{'s' if n != 1 else ''}"


def validate_script_files(paths):
    """A script is ONE PDF or a GROUP of images. Returns an error message, or None if OK."""
    pdfs = [p for p in paths if p.lower().endswith(".pdf")]
    if pdfs and len(paths) > 1:
        return "Choose either ONE PDF or a group of images, not a mix."
    return None


def scan_folder(folder, existing=()):
    """One AnswerScript per PDF in `folder`.

    Returns (new_scripts, pdf_count, loose_image_count). PDFs already present in
    `existing` are skipped. Raises OSError if the folder cannot be read.
    """
    names = sorted(os.listdir(folder), key=natural_key)
    pdfs = [n for n in names if n.lower().endswith(".pdf")]
    n_images = sum(1 for n in names if n.lower().endswith(IMAGE_EXTS))
    known = {tuple(s.sources) for s in existing}

    new = []
    for name in pdfs:
        full = os.path.join(folder, name)
        if (full,) not in known:
            new.append(AnswerScript(os.path.splitext(name)[0], [full]))
    return new, len(pdfs), n_images


class UploadScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.answer_scripts = []  # list[AnswerScript]
        self.pending_files = []   # files chosen for the manual script

        root = MDBoxLayout(orientation="vertical")
        root.add_widget(make_app_bar("Document Upload & Management", self.go_back))

        scroll = MDScrollView(do_scroll_x=False)
        body = MDBoxLayout(
            orientation="vertical",
            padding=[dp(24), dp(24), dp(24), dp(32)],
            spacing=dp(20),
            adaptive_height=True,
        )

        # --- Question paper ---
        qp_card = make_card("Question Paper", "PDF or plain text file (.pdf / .txt).")
        self.qp_input = make_path_field("No file selected")
        qp_btn = make_button(
            "Browse", style="elevated",
            on_release=lambda *_: ChooserDialog(
                "Choose question paper", lambda p: setattr(self.qp_input, "text", p[0]),
                [".pdf", ".txt"],
            ).open(),
            pos_hint={"center_y": 0.5},
        )
        qp_card.add_widget(make_row(self.qp_input, qp_btn))
        body.add_widget(qp_card)

        # --- Subject (names the scores table in the database) ---
        subject_card = make_card(
            "Subject",
            "The app evaluates one subject at a time. This name becomes the scores table "
            "in the database (e.g. Mathematics, Physics).",
        )
        self.subject_input = MDTextField(mode="outlined")
        self.subject_input.add_widget(MDTextFieldHintText(text="Subject name (required)"))
        subject_card.add_widget(self.subject_input)
        body.add_widget(subject_card)

        # --- Marking scheme / rubric (optional) ---
        rubric_card = make_card(
            "Marking Scheme / Rubric (optional)",
            "PDF or text file. If you skip it, the AI drafts a scheme from the question paper "
            "and you should review the marks it gives.",
        )
        self.rubric_input = make_path_field("No file selected (the AI will draft one)")
        rubric_btn = make_button(
            "Browse", style="elevated",
            on_release=lambda *_: ChooserDialog(
                "Choose marking scheme", lambda p: setattr(self.rubric_input, "text", p[0]),
                [".pdf", ".txt"],
            ).open(),
            pos_hint={"center_y": 0.5},
        )
        rubric_clear = make_button(
            "Clear", style="text",
            on_release=lambda *_: setattr(self.rubric_input, "text", ""),
            pos_hint={"center_y": 0.5},
        )
        rubric_card.add_widget(make_row(self.rubric_input, rubric_btn, rubric_clear))
        body.add_widget(rubric_card)

        # --- Folder scan ---
        folder_card = make_card(
            "Automated Folder Scan",
            "Pick a folder of PDFs: one answer script is created per PDF. "
            "Students whose answers are images need to be added manually below.",
        )
        self.folder_input = make_path_field("No folder selected")
        folder_btn = make_button(
            "Browse", style="elevated",
            on_release=lambda *_: ChooserDialog(
                "Choose answer sheet folder", self._scan_folder, dirselect=True
            ).open(),
            pos_hint={"center_y": 0.5},
        )
        folder_card.add_widget(make_row(self.folder_input, folder_btn))
        body.add_widget(folder_card)

        # --- Manual entry ---
        manual_card = make_card(
            "Create Answer Script Manually",
            "One script = one PDF, or a group of page images (JPEG / PNG). "
            "Select several images at once with Ctrl/Shift-click.",
        )
        self.student_id_input = MDTextField(mode="outlined")
        self.student_id_input.add_widget(MDTextFieldHintText(text="Script label (student name or roll number)"))
        self.file_path_input = make_path_field("No files selected")
        file_btn = make_button(
            "Browse", style="elevated",
            on_release=lambda *_: ChooserDialog(
                "Choose a PDF or page images", self._set_script_files,
                [".pdf", *IMAGE_EXTS], multiselect=True,
            ).open(),
            pos_hint={"center_y": 0.5},
        )
        add_btn = make_button(
            "Add Script", on_release=self.create_manual_object, pos_hint={"right": 1}
        )
        manual_card.add_widget(self.student_id_input)
        manual_card.add_widget(make_row(self.file_path_input, file_btn))
        manual_card.add_widget(add_btn)
        body.add_widget(manual_card)

        # --- Summary list ---
        summary_card = make_card("Answer Script Objects")
        self.summary_label = muted_label("")
        self.list_box = MDBoxLayout(orientation="vertical", spacing=dp(6), adaptive_height=True)
        summary_card.add_widget(self.summary_label)
        summary_card.add_widget(self.list_box)
        body.add_widget(summary_card)

        body.add_widget(
            make_button(
                "Proceed to Document Inspection",
                on_release=self.proceed_to_inspection,
                pos_hint={"center_x": 0.5},
            )
        )

        scroll.add_widget(body)
        root.add_widget(scroll)
        self.add_widget(root)
        self.refresh_list()

    # ---- folder scan: one object per PDF ----
    def _scan_folder(self, paths):
        folder = paths[0]
        self.folder_input.text = folder
        try:
            new, n_pdfs, n_images = scan_folder(folder, self.answer_scripts)
        except OSError as e:
            notify(f"Cannot read folder: {e.strerror}")
            return

        self.answer_scripts.extend(new)
        self.refresh_list()

        if not n_pdfs:
            msg = "No PDFs found in that folder. Add scripts manually below."
        else:
            msg = f"Created {len(new)} script(s) from {n_pdfs} PDF(s)."
        if n_images:
            msg += f" {n_images} loose image(s) found: add those manually below."
        notify(msg)

    # ---- manual creation ----
    def _set_script_files(self, paths):
        error = validate_script_files(paths)
        if error:
            notify(error)
            return
        self.pending_files = sorted(paths, key=natural_key)
        n = len(self.pending_files)
        self.file_path_input.text = (
            os.path.basename(self.pending_files[0]) if n == 1 else f"{n} files selected"
        )

    def create_manual_object(self, *_):
        s_id = self.student_id_input.text.strip()
        if not s_id or not self.pending_files:
            notify("Enter a label (name or roll number) and choose file(s).")
            return
        if any(s.student_id == s_id for s in self.answer_scripts):
            notify(f"'{s_id}' already exists. Use a different name or number.")
            return
        self.answer_scripts.append(AnswerScript(s_id, list(self.pending_files)))
        self.student_id_input.text = ""
        self.file_path_input.text = ""
        self.pending_files = []
        self.refresh_list()

    def remove_script(self, index):
        if 0 <= index < len(self.answer_scripts):
            del self.answer_scripts[index]
            self.refresh_list()

    def refresh_list(self):
        self.list_box.clear_widgets()
        count = len(self.answer_scripts)
        self.summary_label.text = (
            f"{count} answer script object(s) created." if count else "Nothing created yet."
        )
        for i, script in enumerate(self.answer_scripts):
            delete = MDIconButton(icon="delete-outline", pos_hint={"center_y": 0.5})
            delete.bind(on_release=lambda _w, idx=i: self.remove_script(idx))
            self.list_box.add_widget(
                make_row(
                    MDLabel(
                        text=f"{script.student_id}  ·  {script.summary}",
                        font_style="Body",
                        role="large",
                        adaptive_height=True,
                        pos_hint={"center_y": 0.5},
                    ),
                    delete,
                )
            )

    def go_back(self, *_):
        self.manager.transition.direction = "right"
        self.manager.current = "start"

    def proceed_to_inspection(self, *_):
        if not self.qp_input.text:
            notify("Please select a question paper first.")
            return
        if not (self.subject_input.text or "").strip():
            notify("Please enter the subject name (it names the scores table).")
            return
        if not self.answer_scripts:
            notify("Create at least one answer script object.")
            return
        self.manager.transition.direction = "left"
        self.manager.current = "inspection"


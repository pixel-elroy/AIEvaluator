"""View and manage student scores in a KivyMD data table.

Rows come from the subject-named scores table in the database.
Teachers can select rows to delete, or edit a total mark manually.
"""
from functools import partial

from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.dialog import (
    MDDialog,
    MDDialogButtonContainer,
    MDDialogContentContainer,
    MDDialogHeadlineText,
)
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivymd.uix.textfield import MDTextField, MDTextFieldHintText
from kivy.uix.widget import Widget

from ui.evaluation_screen import get_store
from ui.widgets import make_app_bar, make_button, make_row, muted_label, notify


class SafeDataTable(MDDataTable):
    """MDDataTable that never auto-selects a row (KivyMD's default selection crashes).

    KivyMD schedules ``set_default_first_row`` for the next frame and indexes
    ``recycle_data[selected_row]`` without checking length. When the table was just
    built, replaced on Refresh, or has zero rows, that raises IndexError and kills
    the app. Skipping the default selection is harmless: the user can still tick rows.
    """

    def set_default_first_row(self, *args):
        # Do not call super(): even the try/except path can race when the table is
        # torn down mid-frame after clear_widgets() on Refresh/delete.
        return

    def select_all(self, *args, **kwargs):
        try:
            return super().select_all(*args, **kwargs)
        except (IndexError, KeyError, AttributeError, TypeError):
            return None


class ScoresScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table = None
        self._selected = set()  # row indices currently checked
        self._rows = []         # parallel list of score dicts

        root = MDBoxLayout(orientation="vertical")
        root.add_widget(make_app_bar("Student Scores", self.go_back))

        content = MDBoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))
        self.subject_label = MDLabel(
            text="", font_style="Title", role="medium", adaptive_height=True
        )
        self.hint = muted_label(
            "Tick rows to delete. Select a single row and press Edit total to change marks."
        )
        content.add_widget(self.subject_label)
        content.add_widget(self.hint)

        self.table_box = MDBoxLayout(orientation="vertical", size_hint=(1, 1))
        content.add_widget(self.table_box)

        actions = make_row(
            make_button("Refresh", style="outlined", on_release=self.refresh),
            make_button("Delete selected", style="tonal", on_release=self.delete_selected),
            make_button("Edit total", style="elevated", on_release=self.edit_total),
        )
        actions.adaptive_size = True
        actions.pos_hint = {"center_x": 0.5}
        content.add_widget(actions)

        root.add_widget(content)
        self.add_widget(root)

    def on_pre_enter(self, *args):
        self.refresh()

    def refresh(self, *_):
        store = get_store()
        subject = store.get_subject() or "(no subject set)"
        table_name = store.scores_table() or "—"
        self.subject_label.text = f"Subject: {subject}   ·   table: {table_name}"

        # Re-sync from evaluations so teacher edits in Review are reflected.
        # Sync respects excluded (deleted) rows and manual totals, so deletes
        # and Edit total survive this step.
        try:
            store.sync_all_scores()
        except Exception:
            pass

        self._rows = store.list_scores()
        self._selected.clear()
        self._rebuild_table()

    def _status_text(self, r):
        status = (r.get("status") or "").strip()
        if r.get("manual"):
            status = (status + " · manual").strip(" ·")
        return status or ("manual" if r.get("manual") else "")

    def _rebuild_table(self):
        # Drop the previous table first so any pending Clock callbacks on it cannot
        # touch a half-destroyed widget tree.
        if self.table is not None:
            try:
                self.table.unbind(on_check_press=self._on_check)
            except Exception:
                pass
            self.table = None
        self.table_box.clear_widgets()
        if not self._rows:
            self.table_box.add_widget(
                muted_label("No scores yet. Finish an evaluation for this subject first.")
            )
            self.table = None
            return

        column_data = [
            ("Roll no.", dp(100)),
            ("Name", dp(140)),
            ("Total", dp(80)),
            ("Max", dp(80)),
            ("Status", dp(120)),
        ]
        row_data = [
            (
                r["roll_no"] or "—",
                r["name"] or "—",
                f"{r['total']:g}",
                f"{r['max_total']:g}",
                self._status_text(r),
            )
            for r in self._rows
        ]
        self.table = SafeDataTable(
            size_hint=(1, 1),
            use_pagination=True,
            rows_num=12,
            check=True,
            column_data=column_data,
            row_data=row_data,
        )
        self.table.bind(on_check_press=self._on_check)
        self.table_box.add_widget(self.table)

    def _row_index_for(self, current_row):
        """Map a checked table row (list/tuple of cell strings) to self._rows index."""
        if not current_row:
            return None
        cells = list(current_row)
        roll_display = cells[0] if cells else None
        # Prefer exact match against the parallel _rows list by roll number.
        for i, r in enumerate(self._rows):
            if (r["roll_no"] or "—") == roll_display:
                return i
        # Fall back: match against the table's own row_data.
        if self.table is not None:
            try:
                as_tuple = tuple(cells)
                return self.table.row_data.index(as_tuple)
            except ValueError:
                try:
                    return self.table.row_data.index(cells)
                except ValueError:
                    pass
        return None

    def _on_check(self, instance_table, current_row):
        idx = self._row_index_for(current_row)
        if idx is None:
            return
        if idx in self._selected:
            self._selected.discard(idx)
        else:
            self._selected.add(idx)

    def delete_selected(self, *_):
        if not self._selected:
            notify("Tick one or more rows first.")
            return
        rolls = []
        for i in sorted(self._selected):
            if i < len(self._rows):
                roll = self._rows[i].get("roll_no")
                if roll is not None and str(roll) != "":
                    rolls.append(str(roll))
        if not rolls:
            notify("No valid rows selected.")
            return
        store = get_store()
        try:
            n = store.delete_scores(rolls)
        except Exception as e:
            notify(f"Could not delete: {e}")
            return
        notify(f"Deleted {n} score row(s).")
        # Rebuild without relying on a full re-entry so the empty state shows immediately.
        self._selected.clear()
        self._rows = store.list_scores()
        self._rebuild_table()

    def edit_total(self, *_):
        if len(self._selected) != 1:
            notify("Select exactly one row to edit its total.")
            return
        idx = next(iter(self._selected))
        if idx >= len(self._rows):
            notify("Selection is out of date; press Refresh.")
            return
        row = self._rows[idx]
        roll = row["roll_no"]
        field = MDTextField(mode="outlined", text=f"{row['total']:g}", input_filter="float")
        field.add_widget(MDTextFieldHintText(
            text=f"New total for {roll or '—'} (max {row['max_total']:g})"
        ))

        def _save(*_a):
            try:
                value = float(field.text.strip())
            except ValueError:
                notify("Enter a valid number.")
                return
            if value < 0 or value > (row["max_total"] or 1e9):
                notify(f"Total must be between 0 and {row['max_total']:g}.")
                return
            try:
                get_store().update_score_total(roll, value)
            except Exception as e:
                notify(f"Could not update: {e}")
                return
            dialog.dismiss()
            notify("Total updated (saved as manual).")
            self.refresh()

        cancel = make_button("Cancel", style="text", on_release=lambda *_: dialog.dismiss())
        save = make_button("Save", on_release=_save)
        dialog = MDDialog(
            MDDialogHeadlineText(text=f"Edit total — {roll or '—'}"),
            MDDialogContentContainer(field, orientation="vertical"),
            MDDialogButtonContainer(Widget(), cancel, save, spacing=dp(8)),
        )
        dialog.open()

    def go_back(self, *_):
        self.manager.transition.direction = "right"
        self.manager.current = "start"

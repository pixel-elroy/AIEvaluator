"""Evaluation: grading, results storage, the end-to-end pipeline, reports and agreement metrics.

Sections
  1. Grading one answer against the rubric (the model judges; plain code validates and sums)
  2. Store: SQLite database holding every stage's output (so runs can resume)
  3. EvaluationPipeline: preprocess -> transcribe -> read ID -> segment -> grade
  4. Reports: class marksheet (CSV) and per-student report (PDF)
  5. Agreement between AI marks and faculty marks
No Kivy imports.
"""
import csv
import hashlib
import html
import json
import math
import os
import sqlite3
import threading
import re
import time
from contextlib import closing
from dataclasses import dataclass, field
from typing import Optional

from core import ocr_engine, parser
from core.ocr_engine import AuthError, Cancelled, LLMClient, LLMError, QuotaExceeded, safe_name
from core.parser import Answer, ParseError, QuestionPaper, Rubric


MAX_BATCH_IMAGES = 6   # page images attached to one batched grading call
MAX_QUESTIONS_PER_CALL = 20  # all of a script's questions are graded in ONE call, up to this many
CALLS_PER_SCRIPT = 4   # hard cap on model calls for one script (normally 3: read, match + ID, grade)


def _clean_math(text):
    if not text:
        return ""
    s = str(text)
    # Remove LaTeX math delimiters
    s = s.replace(r"\(", "").replace(r"\)", "")
    s = s.replace(r"\[", "").replace(r"\]", "")

    # Convert fractions \frac{a}{b} -> (a / b)
    s = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\1 / \2)", s)

    # Convert square roots \sqrt{x} -> √(x)
    s = re.sub(r"\\sqrt\s*\{([^}]*)\}", r"√(\1)", s)
    s = s.replace(r"\sqrt", "√")

    # Clean up superscripts and subscripts
    s = re.sub(r"\^\{([^}]*)\}", r"^\1", s)
    s = re.sub(r"_\{([^}]*)\}", r"_\1", s)

    # Mathematical operators and symbols
    s = s.replace(r"\int", "∫")
    s = s.replace(r"\sum", "∑")
    s = s.replace(r"\pi", "π")
    s = s.replace(r"\omega", "ω")
    s = s.replace(r"\delta", "δ")
    s = s.replace(r"\cos", "cos")
    s = s.replace(r"\sin", "sin")
    s = s.replace(r"\tan^{-1}", "arctan")
    s = s.replace(r"\cdot", "·")
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")

    # Strip any remaining standalone backslash commands
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace(r"\\", " ")
    return s

# =====================================================================
# 1. Grading
# =====================================================================
@dataclass
class Evaluation:
    question_id: str
    max_marks: float
    ai_marks: float = 0.0
    points: list = field(default_factory=list)   # [{"description","awarded","max","comment"}]
    rationale: str = ""
    feedback: str = ""
    needs_review: bool = False
    review_reason: str = ""
    final_marks: Optional[float] = None          # set when a teacher confirms or changes the marks
    final_feedback: Optional[str] = None
    approved: bool = False

    @property
    def marks(self):
        return self.ai_marks if self.final_marks is None else self.final_marks

    @property
    def feedback_text(self):
        return self.feedback if self.final_feedback is None else self.final_feedback


def _clamp(value, low, high):
    return max(low, min(high, value))


def _number(value, default=0.0):
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rubric_text(qr, drafted):
    lines = []
    if drafted:
        lines.append("(This marking scheme was drafted by an AI and is unverified.)")
    if qr.model_answer:
        lines.append("Model answer: " + qr.model_answer)
    if qr.points:
        lines.append("Marking points:")
        lines += [f"{i}. ({p.marks:g} marks) {p.description}" for i, p in enumerate(qr.points, start=1)]
    else:
        lines.append("(No itemised marking points were provided.)")
    if qr.notes:
        lines.append("Notes (partial credit / alternatives): " + qr.notes)
    return "\n".join(lines)


def build_grading_prompt(question, qr, answer_text, drafted=False, with_images=False):
    itemised = bool(qr.points)
    schema = (
        '{"points": [{"description": "<scheme point copied>", "awarded": <number from 0 to that point\'s marks>, '
        '"comment": "<one short sentence>"}], '
        if itemised else
        '{"marks": <number from 0 to %g>, "points": [], ' % question.max_marks
    ) + (
        '"rationale": "<2-4 sentences explaining the marks>", '
        '"feedback": "<2-3 sentences addressed to the student: what was good and what to improve>", '
        '"needs_review": <true or false>, "review_reason": "<why a teacher should check, or empty string>"}'
    )
    return (
        "TASK: evaluate_answer\n"
        "You are an experienced, fair examiner marking one answer from a student's handwritten exam script. "
        "The answer was transcribed by software, so it may contain reading errors and [unclear] markers. "
        "Never penalise a student for a transcription artefact; if the transcription is too unreliable to judge, "
        "say so and set needs_review to true.\n\n"
        f"QUESTION ({question.max_marks:g} marks):\n{question.text}\n\n"
        f"MARKING SCHEME:\n{_rubric_text(qr, drafted)}\n\n"
        f"STUDENT'S ANSWER (verbatim transcription):\n<<<\n{answer_text}\n>>>\n\n"
        "Mark against the marking scheme, not by keywords. Consider conceptual correctness, key points covered, "
        "mathematical reasoning and method, relevance to the question, completeness and correctness of the final answer. "
        "Award partial credit for correct method or partly correct points. Accept valid alternative methods and equivalent forms.\n"
        + ("The page image(s) of the answer are attached: use them to resolve any [unclear] parts.\n" if with_images else "")
        + ("Include exactly one entry in \"points\" per marking point, in the same order.\n" if itemised else
           "Because there are no itemised points, give one overall mark in \"marks\" and leave \"points\" empty.\n")
        + "Return ONLY JSON: " + schema
    )


def _validate_evaluation(question, qr, answer, reply):
    """Turn the model's reply into an Evaluation. All arithmetic is done here, not by the model."""
    if not isinstance(reply, dict):
        raise LLMError("The model's grading reply was not a JSON object.")
    notes = []
    points, model_points = [], reply.get("points") if isinstance(reply.get("points"), list) else []
    if qr.points:
        if len(model_points) != len(qr.points):
            notes.append("the model returned the wrong number of marking points")
        for i, rp in enumerate(qr.points):
            mp = model_points[i] if i < len(model_points) and isinstance(model_points[i], dict) else {}
            awarded = _clamp(_number(mp.get("awarded")), 0.0, rp.marks)
            points.append({"description": rp.description, "awarded": round(awarded, 2),
                           "max": rp.marks, "comment": str(mp.get("comment", "") or "").strip()})
        marks = _clamp(sum(p["awarded"] for p in points), 0.0, question.max_marks)
        claimed = reply.get("marks")
        if claimed is not None and abs(_number(claimed, marks) - marks) > 0.5:
            notes.append("the model's own total differed from the sum of its points")
    else:
        marks = _clamp(_number(reply.get("marks")), 0.0, question.max_marks)
    if answer.confidence < 0.5:
        notes.append("the answer may not belong to this question (low matching confidence)")
    model_reason = str(reply.get("review_reason", "") or "").strip()
    flagged = bool(reply.get("needs_review")) or bool(notes)
    reason = "; ".join(([model_reason] if model_reason else []) + notes)
    return Evaluation(
        question_id=question.id, max_marks=question.max_marks, ai_marks=round(marks, 2), points=points,
        rationale=str(reply.get("rationale", "") or "").strip() or "No rationale was returned.",
        feedback=str(reply.get("feedback", "") or "").strip(),
        needs_review=flagged, review_reason=reason if flagged else "",
    )


def evaluate_answer(client, question, qr, answer, rubric_drafted=False, page_images=()):
    """Grade one answer. An empty answer scores 0 without calling the model."""
    if not answer.text.strip():
        return Evaluation(
            question_id=question.id, max_marks=question.max_marks, ai_marks=0.0,
            rationale="No answer was found for this question.",
            feedback="No answer was found for this question. If you attempted it, ask your teacher to check.",
            needs_review=True, review_reason="no answer found (could be an OCR or segmentation miss)",
        )
    prompt = build_grading_prompt(question, qr, answer.text, rubric_drafted, bool(page_images))
    images = [ocr_engine.image_part(p) for p in page_images]

    # Some local models occasionally return a valid JSON array/string instead of the
    # required top-level object. Retry once with an explicit format reminder rather
    # than failing the entire script immediately.
    reply = client.ask_json(prompt, images=images)
    if not isinstance(reply, dict):
        retry_prompt = (
            prompt
            + "\n\nFORMAT ERROR: Your previous response was not a JSON object. "
              "Return one top-level JSON object beginning with { and ending with }. "
              "Do not return an array, a quoted JSON string, Markdown fences, or commentary. "
              "The object must contain the keys specified in the schema above."
        )
        try:
            reply = client.ask_json(retry_prompt, images=images)
        except Exception as exc:
            raise LLMError(
                "The model did not return a grading JSON object after retry. "
                "Try a smaller/JSON-capable Ollama model or reduce the answer length."
            ) from exc
    if not isinstance(reply, dict):
        raise LLMError(
            "The model's grading reply was not a JSON object after retry. "
            "The model may be ignoring the requested output format."
        )
    return _validate_evaluation(question, qr, answer, reply)


def _norm_id(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def build_batch_grading_prompt(items, drafted=False, attached_pages=(), needs_pages=None):
    """One prompt that asks for the marks of several questions at once.
    items = [(question, qr, answer), ...]; needs_pages = {question_id: [page numbers]} for answers
    whose transcription is unclear and whose page images are attached."""
    needs_pages = needs_pages or {}
    blocks = []
    for question, qr, answer in items:
        pages = needs_pages.get(question.id)
        blocks.append(
            f"##### QUESTION {question.id} ({question.max_marks:g} marks)\n{question.text}\n\n"
            f"MARKING SCHEME:\n{_rubric_text(qr, drafted)}\n\n"
            + ("(Marking points are itemised: give one entry in \"points\" per marking point, in the same order.)\n"
               if qr.points else
               "(No itemised points: give one overall \"marks\" value and leave \"points\" empty.)\n")
            + (f"(Unclear transcription: check page(s) {', '.join(str(p) for p in pages)} of the attached images.)\n"
               if pages else "")
            + f"STUDENT'S ANSWER (verbatim transcription):\n<<<\n{answer.text}\n>>>"
        )
    ids = ", ".join(str(q.id) for q, _qr, _a in items)
    schema = (
        '{"results": [{"question_id": "<id exactly as given>", '
        '"points": [{"description": "<scheme point copied>", "awarded": <number from 0 to that point\'s marks>, '
        '"comment": "<one short sentence>"}], '
        '"marks": <overall mark, ONLY for questions without itemised points>, '
        '"rationale": "<2-4 sentences explaining the marks>", '
        '"feedback": "<2-3 sentences addressed to the student: what was good and what to improve>", '
        '"needs_review": <true or false>, "review_reason": "<why a teacher should check, or empty string>"}, ...]}'
    )
    return (
        "TASK: evaluate_answers_batch\n"
        "You are an experienced, fair examiner marking several answers from ONE student's handwritten exam script. "
        "The answers were transcribed by software, so they may contain reading errors and [unclear] markers. "
        "Never penalise a student for a transcription artefact; if the transcription is too unreliable to judge, "
        "say so and set needs_review to true.\n"
        "Mark each question independently, against ITS OWN marking scheme, not by keywords. Consider conceptual "
        "correctness, key points covered, mathematical reasoning and method, relevance to the question, completeness and "
        "correctness of the final answer. Award partial credit for correct method or partly correct points. Accept valid "
        "alternative methods and equivalent forms.\n"
        + (f"Page images of the script are attached (in this order: pages {', '.join(str(p) for p in attached_pages)}); "
           "use them only to resolve [unclear] parts.\n" if attached_pages else "")
        + f"\nThere are {len(items)} questions: {ids}. Return exactly one entry per question, in this order.\n"
        + "Return ONLY JSON: " + schema + "\n\n" + "\n\n".join(blocks)
    )


def _batch_entries(reply):
    """List of per-question dicts from a batched reply (tolerates a few common shapes)."""
    if isinstance(reply, dict):
        for key in ("results", "evaluations", "grades", "answers", "questions"):
            if isinstance(reply.get(key), list):
                reply = reply[key]
                break
        else:  # {"1": {...}, "2": {...}}
            keyed = [(k, v) for k, v in reply.items() if isinstance(v, dict)]
            reply = [dict(v, question_id=v.get("question_id", k)) for k, v in keyed]
    return [e for e in reply if isinstance(e, dict)] if isinstance(reply, list) else []


def evaluate_batch(client, items, rubric_drafted=False, page_images=(), attached_pages=(), needs_pages=None,
                   attempts=1):
    """Grade several answers with ONE model call. items = [(question, qr, answer)], all with text.

    Returns {question_id: Evaluation} for the questions that came back usable; the caller grades any
    others one by one. The model judges, plain code validates and sums, exactly as for a single answer."""
    prompt = build_batch_grading_prompt(items, rubric_drafted, attached_pages, needs_pages)
    images = [ocr_engine.image_part(p) for p in page_images]
    reply = client.ask_json(prompt, images=images, max_tokens=min(16000, 1500 + 800 * len(items)),
                            attempts=attempts)
    entries = _batch_entries(reply)

    by_id = {}
    for e in entries:
        by_id.setdefault(_norm_id(e.get("question_id", "")), e)
    if not any(_norm_id(q.id) in by_id for q, _qr, _a in items) and len(entries) == len(items):
        by_id = {_norm_id(q.id): e for (q, _qr, _a), e in zip(items, entries)}  # ids missing: trust the order

    results = {}
    for question, qr, answer in items:
        entry = by_id.get(_norm_id(question.id))
        if entry is None:
            continue
        try:
            results[question.id] = _validate_evaluation(question, qr, answer, entry)
        except LLMError:
            continue
    return results


# =====================================================================
# 2. Store (SQLite)
# =====================================================================
@dataclass
class StudentRecord:
    script_key: str
    name: str = ""
    roll_no: str = ""
    id_confidence: float = 0.0
    id_checked: bool = False
    sources_sig: str = ""
    status: str = "new"      # new | processing | done | paused | failed
    error: str = ""

    @property
    def display(self):
        return self.name or self.roll_no or self.script_key


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS students(
    script_key TEXT PRIMARY KEY, name TEXT DEFAULT '', roll_no TEXT DEFAULT '',
    id_confidence REAL DEFAULT 0, id_checked INTEGER DEFAULT 0, sources_sig TEXT DEFAULT '',
    status TEXT DEFAULT 'new', error TEXT DEFAULT '', updated REAL);
CREATE TABLE IF NOT EXISTS pages(
    script_key TEXT, page INTEGER, orig_path TEXT, enh_path TEXT, bin_path TEXT,
    PRIMARY KEY(script_key, page));
CREATE TABLE IF NOT EXISTS transcripts(
    script_key TEXT, engine TEXT, page INTEGER, text TEXT, PRIMARY KEY(script_key, engine, page));
CREATE TABLE IF NOT EXISTS answers(
    script_key TEXT, question_id TEXT, answer_text TEXT, confidence REAL, pages_json TEXT,
    PRIMARY KEY(script_key, question_id));
CREATE TABLE IF NOT EXISTS evaluations(
    script_key TEXT, question_id TEXT, max_marks REAL, ai_marks REAL, points_json TEXT,
    rationale TEXT, feedback TEXT, needs_review INTEGER, review_reason TEXT,
    final_marks REAL, final_feedback TEXT, approved INTEGER DEFAULT 0,
    PRIMARY KEY(script_key, question_id));
CREATE TABLE IF NOT EXISTS faculty_marks(
    student_ref TEXT, question_id TEXT, marks REAL, PRIMARY KEY(student_ref, question_id));
"""


class Store:
    """Every connection is opened per call, so the GUI thread and the worker thread can share it."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _exec(self, sql, args=()):
        with closing(self._connect()) as conn, conn:
            conn.execute(sql, args)

    def _query(self, sql, args=()):
        with closing(self._connect()) as conn:
            return conn.execute(sql, args).fetchall()

    # ---- meta ----
    def get_meta(self, key, default=None):
        rows = self._query("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set_meta(self, key, value):
        self._exec("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, json.dumps(value)))

    def save_paper(self, paper):
        self.set_meta("paper", paper.to_dict())

    def load_paper(self):
        d = self.get_meta("paper")
        return QuestionPaper.from_dict(d) if d else None

    def save_rubric(self, rubric):
        self.set_meta("rubric", rubric.to_dict())

    def load_rubric(self, paper):
        d = self.get_meta("rubric")
        return Rubric.from_dict(d, paper, drafted=bool(d.get("drafted"))) if d else None

    # ---- students ----
    def get_student(self, key):
        rows = self._query("SELECT * FROM students WHERE script_key=?", (key,))
        return self._student(rows[0]) if rows else None

    @staticmethod
    def _student(row):
        return StudentRecord(row["script_key"], row["name"], row["roll_no"], row["id_confidence"],
                             bool(row["id_checked"]), row["sources_sig"], row["status"], row["error"])

    def list_students(self):
        return [self._student(r) for r in self._query("SELECT * FROM students ORDER BY rowid")]

    def upsert_student(self, key, **fields):
        self._exec("INSERT OR IGNORE INTO students(script_key) VALUES(?)", (key,))
        allowed = {"name", "roll_no", "id_confidence", "id_checked", "sources_sig", "status", "error"}
        for name, value in fields.items():
            if name not in allowed:
                raise ValueError(name)
            self._exec(f"UPDATE students SET {name}=? WHERE script_key=?", (value, key))
        self._exec("UPDATE students SET updated=? WHERE script_key=?", (time.time(), key))

    def delete_script(self, key):
        for table in ("students", "pages", "transcripts", "answers", "evaluations"):
            self._exec(f"DELETE FROM {table} WHERE script_key=?", (key,))

    def clear_evaluations(self):
        self._exec("DELETE FROM evaluations")
        self._exec("UPDATE students SET status='new' WHERE status='done'")

    def archive_and_reset(self):
        """Keep the old database under a new name and start empty. Returns the backup path."""
        backup = f"{self.path}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
        os.replace(self.path, backup)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)
        return backup

    # ---- pages / transcripts / answers ----
    def save_pages(self, key, pages):
        self._exec("DELETE FROM pages WHERE script_key=?", (key,))
        for i, (orig, enh, binary) in enumerate(pages, start=1):
            self._exec("INSERT INTO pages VALUES(?,?,?,?,?)", (key, i, orig, enh, binary))

    def get_pages(self, key):
        rows = self._query("SELECT * FROM pages WHERE script_key=? ORDER BY page", (key,))
        return [(r["orig_path"], r["enh_path"], r["bin_path"]) for r in rows]

    def save_transcript(self, key, engine, page, text):
        self._exec("INSERT OR REPLACE INTO transcripts VALUES(?,?,?,?)", (key, engine, page, text))

    def get_transcripts(self, key, engine):
        rows = self._query("SELECT page, text FROM transcripts WHERE script_key=? AND engine=?", (key, engine))
        return {r["page"]: r["text"] for r in rows}

    def save_answers(self, key, answers):
        self._exec("DELETE FROM answers WHERE script_key=?", (key,))
        for a in answers:
            self._exec("INSERT INTO answers VALUES(?,?,?,?,?)",
                       (key, a.question_id, a.text, a.confidence, json.dumps(a.pages)))

    def get_answers(self, key):
        rows = self._query("SELECT * FROM answers WHERE script_key=? ORDER BY rowid", (key,))
        return [Answer(r["question_id"], r["answer_text"], r["confidence"], json.loads(r["pages_json"]))
                for r in rows]

    # ---- evaluations ----
    def save_evaluation(self, key, ev):
        self._exec(
            "INSERT OR REPLACE INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, ev.question_id, ev.max_marks, ev.ai_marks, json.dumps(ev.points), ev.rationale,
             ev.feedback, int(ev.needs_review), ev.review_reason, ev.final_marks, ev.final_feedback,
             int(ev.approved)))

    def get_evaluations(self, key):
        rows = self._query("SELECT * FROM evaluations WHERE script_key=? ORDER BY rowid", (key,))
        return [Evaluation(
            r["question_id"], r["max_marks"], r["ai_marks"], json.loads(r["points_json"]), r["rationale"],
            r["feedback"], bool(r["needs_review"]), r["review_reason"], r["final_marks"],
            r["final_feedback"], bool(r["approved"])) for r in rows]

    def update_teacher(self, key, question_id, final_marks, final_feedback, approved):
        self._exec(
            "UPDATE evaluations SET final_marks=?, final_feedback=?, approved=? WHERE script_key=? AND question_id=?",
            (final_marks, final_feedback, int(approved), key, question_id))

    def totals(self, key):
        """(final_total, ai_total, max_total) for one student."""
        evs = self.get_evaluations(key)
        return (round(sum(e.marks for e in evs), 2), round(sum(e.ai_marks for e in evs), 2),
                round(sum(e.max_marks for e in evs), 2))

    # ---- faculty marks (agreement study) ----
    def set_faculty_marks(self, rows):
        for ref, qid, marks in rows:
            self._exec("INSERT OR REPLACE INTO faculty_marks VALUES(?,?,?)", (ref, qid, marks))

    def get_faculty_marks(self):
        return [(r["student_ref"], r["question_id"], r["marks"])
                for r in self._query("SELECT * FROM faculty_marks")]

    # ---- subject scores table (one subject per database) ----
    @staticmethod
    def _safe_table_name(subject):
        """Turn a subject name into a valid SQLite table identifier."""
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", (subject or "").strip())
        cleaned = cleaned.strip("_") or "subject"
        if cleaned[0].isdigit():
            cleaned = "s_" + cleaned
        return cleaned[:60]

    def get_subject(self):
        return self.get_meta("subject") or ""

    def set_subject(self, subject):
        subject = (subject or "").strip()
        if not subject:
            raise ValueError("Subject name cannot be empty.")
        self.set_meta("subject", subject)
        self.ensure_scores_table(subject)

    def ensure_scores_table(self, subject=None):
        """Create the scores table named after the subject if it does not exist.

        Includes a ``manual`` flag so teacher edits of the total are not
        overwritten by a later sync from evaluations.
        """
        subject = subject or self.get_subject()
        if not subject:
            return None
        table = self._safe_table_name(subject)
        self.set_meta("scores_table", table)
        self._exec(
            f'CREATE TABLE IF NOT EXISTS "{table}" ('
            "roll_no TEXT PRIMARY KEY, "
            "name TEXT DEFAULT '', "
            "script_key TEXT DEFAULT '', "
            "total REAL DEFAULT 0, "
            "max_total REAL DEFAULT 0, "
            "status TEXT DEFAULT '', "
            "details_json TEXT DEFAULT '{}', "
            "manual INTEGER DEFAULT 0, "
            "updated REAL)"
        )
        # Migrate older tables that were created without the manual column.
        try:
            cols = {r["name"] for r in self._query(f'PRAGMA table_info("{table}")')}
            if "manual" not in cols:
                self._exec(f'ALTER TABLE "{table}" ADD COLUMN manual INTEGER DEFAULT 0')
        except sqlite3.OperationalError:
            pass
        return table

    def scores_table(self):
        table = self.get_meta("scores_table")
        subject = self.get_subject()
        if subject:
            # Always run ensure so older DBs get the `manual` column migration.
            table = self.ensure_scores_table(subject) or table
        return table

    def _excluded_scores(self):
        """Roll numbers the teacher removed from the scores table (persist across sync)."""
        raw = self.get_meta("excluded_scores") or []
        return {str(x) for x in raw} if isinstance(raw, list) else set()

    def _set_excluded_scores(self, excluded):
        self.set_meta("excluded_scores", sorted(str(x) for x in excluded))

    def sync_student_score(self, key, force=False):
        """Upsert one student's totals into the subject scores table.

        Respects teacher deletions (excluded roll numbers) and manual total edits
        (``manual=1`` rows keep their total; other fields still refresh).
        Pass force=True after a fresh evaluation so a previously deleted row returns.
        """
        table = self.scores_table()
        if not table:
            return
        student = self.get_student(key)
        if student is None:
            return
        roll = student.roll_no or student.script_key
        if not roll:
            return
        excl = self._excluded_scores()
        if str(roll) in excl:
            if not force:
                return  # teacher deleted this row; do not re-insert on ordinary sync
            excl.discard(str(roll))
            self._set_excluded_scores(excl)
        final, _ai, mx = self.totals(key)
        details = {
            e.question_id: {"marks": e.marks, "max": e.max_marks, "approved": e.approved}
            for e in self.get_evaluations(key)
        }
        # Preserve a manually edited total if the row already exists with manual=1.
        existing = self._query(
            f'SELECT total, manual FROM "{table}" WHERE roll_no=?', (str(roll),)
        )
        if existing and int(existing[0]["manual"] or 0):
            total = existing[0]["total"]
            manual = 1
        else:
            total = final
            manual = 0
        self._exec(
            f'INSERT OR REPLACE INTO "{table}" '
            "(roll_no, name, script_key, total, max_total, status, details_json, manual, updated) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                str(roll),
                student.name,
                student.script_key,
                total,
                mx,
                student.status,
                json.dumps(details),
                manual,
                time.time(),
            ),
        )

    def sync_all_scores(self):
        """Refresh the scores table from every evaluated student."""
        table = self.scores_table()
        if not table:
            return 0
        count = 0
        for s in self.list_students():
            if self.get_evaluations(s.script_key):
                self.sync_student_score(s.script_key)
                count += 1
        return count

    def list_scores(self):
        """Rows from the subject scores table, ordered by roll number."""
        table = self.scores_table()
        if not table:
            return []
        try:
            rows = self._query(f'SELECT * FROM "{table}" ORDER BY roll_no')
        except sqlite3.OperationalError:
            return []
        out = []
        for r in rows:
            keys = r.keys() if hasattr(r, "keys") else []
            out.append({
                "roll_no": r["roll_no"],
                "name": r["name"],
                "script_key": r["script_key"],
                "total": r["total"],
                "max_total": r["max_total"],
                "status": r["status"],
                "details": json.loads(r["details_json"] or "{}"),
                "manual": int(r["manual"] if "manual" in keys else 0) if "manual" in keys else 0,
                "updated": r["updated"],
            })
        return out

    def update_score_total(self, roll_no, total):
        """Manually set the total marks for a roll number (persists across sync)."""
        table = self.scores_table()
        if not table:
            raise ValueError("No scores table has been created yet (set a subject first).")
        # Clear exclusion if the teacher is editing a previously deleted roll.
        excl = self._excluded_scores()
        roll = str(roll_no)
        if roll in excl:
            excl.discard(roll)
            self._set_excluded_scores(excl)
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                f'UPDATE "{table}" SET total=?, manual=1, updated=? WHERE roll_no=?',
                (float(total), time.time(), roll),
            )
            if cur.rowcount == 0:
                raise ValueError(f"No score row found for roll number '{roll}'.")

    def delete_scores(self, roll_nos):
        """Remove one or more rows from the scores table by roll_no.

        Also records them as excluded so a later sync from evaluations does not
        put the rows back (this was the bug that made a single-row delete appear
        to do nothing after Refresh).
        """
        table = self.scores_table()
        if not table or not roll_nos:
            return 0
        rolls = [str(r) for r in roll_nos if r is not None and str(r) != ""]
        if not rolls:
            return 0
        placeholders = ",".join("?" for _ in rolls)
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                f'DELETE FROM "{table}" WHERE roll_no IN ({placeholders})',
                rolls,
            )
            n = cur.rowcount
        excl = self._excluded_scores()
        excl.update(rolls)
        self._set_excluded_scores(excl)
        return n


# =====================================================================
# 3. Pipeline
# =====================================================================
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sources_signature(sources):
    parts = []
    for p in sources:
        st = os.stat(p)
        parts.append(f"{p}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


class EvaluationPipeline:
    """preprocess -> transcribe -> read ID -> segment -> grade, with every stage saved to the Store.

    Re-running resumes: finished stages are skipped, so a quota pause costs nothing."""

    def __init__(self, settings, store, data_dir, cancel=None, progress=None):
        self.settings = settings
        self.store = store
        self.data_dir = data_dir
        self.cancel = cancel or threading.Event()
        self._progress = progress or (lambda message, fraction: None)
        self._client = None
        self._index, self._count = 0, 1
        # reasoning effort (e.g. "none"/"low"): reading and matching rarely need it, grading may
        self.read_effort = (settings.get("reasoning_read") or "").strip()
        self.grade_effort = (settings.get("reasoning_grade") or "").strip()

    @property
    def client(self):
        if self._client is None:
            self._client = LLMClient.from_settings(self.settings, cancel=self.cancel)
            self._client.reasoning = self.read_effort
            self._client.notice = lambda m: self._say(m)  # "rate limited: waiting 30s..." shows in the log
        return self._client

    def _check(self):
        if self.cancel.is_set():
            raise Cancelled()

    def _say(self, message, fraction_in_script=None):
        overall = None if fraction_in_script is None else (self._index + fraction_in_script) / self._count
        self._progress(message, overall)

    # ---- question paper + rubric ----
    def prepare(self, qp_path, rubric_path=None):
        """Parse the question paper and marking scheme (cached by file content)."""
        store = self.store
        paper_hash = file_sha256(qp_path)
        rubric_hash = file_sha256(rubric_path) if rubric_path else "draft"
        if store.get_meta("paper_hash") not in (None, paper_hash) and store.list_students():
            backup = store.archive_and_reset()
            self._say(f"A different question paper was detected: earlier results were kept as {os.path.basename(backup)}.")

        cache_key = "cache:paper:" + paper_hash
        cached = store.get_meta(cache_key)
        paper = None
        if cached:
            try:
                paper = QuestionPaper.from_dict(cached)
                self._say("Question paper: using the saved parse.")
            except ParseError:
                # Old/corrupt cache entries must not permanently block a new parse.
                self._say("Saved question parse is invalid; rebuilding it from the PDF.")
        if paper is None:
            self._say("Reading the question paper with PaddleOCR...")
            # The project explicitly requires PaddleOCR for the question-paper PDF,
            # including PDFs that also happen to contain a native text layer.
            text = parser.read_document_text(
                qp_path, self.client, lambda m: self._say(m),
                engine="paddleocr", force_ocr=True,
            )
            self._check()
            paper = parser.parse_question_paper(self.client, text)
            store.set_meta(cache_key, paper.to_dict())

        rkey = f"cache:rubric:{rubric_hash}:{paper_hash}"
        cached = store.get_meta(rkey)
        if cached:
            rubric = Rubric.from_dict(cached, paper, drafted=bool(cached.get("drafted")))
            self._say("Marking scheme: using the saved parse.")
        else:
            self._check()
            self.client.reasoning = self.read_effort
            if rubric_path:
                self._say("Reading the marking scheme...")
                text = parser.read_document_text(rubric_path, self.client, lambda m: self._say(m))
                rubric = parser.parse_rubric(self.client, paper, text)
            else:
                self._say("No marking scheme supplied: drafting one with the model...")
                self.client.reasoning = self.grade_effort  # solving the questions benefits from reasoning
                rubric = parser.draft_rubric(self.client, paper)
            store.set_meta(rkey, rubric.to_dict())

        if store.get_meta("rubric_key") not in (None, rubric_hash):
            store.clear_evaluations()  # different marking scheme: earlier marks no longer valid
            self._say("The marking scheme changed: earlier marks will be recalculated.")
        for w in (paper.warnings + rubric.warnings)[:8]:
            self._say("Note: " + w)
        store.set_meta("paper_hash", paper_hash)
        store.set_meta("rubric_key", rubric_hash)
        store.save_paper(paper)
        store.save_rubric(rubric)
        self._say(f"Question paper and marking scheme ready. API calls so far: {self.client.calls}.")
        return paper, rubric

    # ---- one script ----
    def run_script(self, key, sources, paper, rubric, force=False):
        store = self.store
        sig = sources_signature(sources)
        student = store.get_student(key)
        if student and student.sources_sig != sig:
            store.delete_script(key)  # the files changed: start this script over
            student = None
        if force and student:
            store.delete_script(key)
            student = None
        if student and student.status == "done":
            # Cached result: still refresh the scores table so a previously deleted
            # row comes back after the teacher re-runs evaluation on the same scripts.
            try:
                store.sync_student_score(key, force=True)
            except Exception:
                pass
            self._say(f"{key}: already evaluated.", 1.0)
            return student
        store.upsert_student(key, sources_sig=sig, status="processing", error="")
        self.client.begin_budget(CALLS_PER_SCRIPT)
        try:
            self._run_stages(key, sources, paper, rubric)
        except (Cancelled, QuotaExceeded):
            store.upsert_student(key, status="paused")
            raise
        except AuthError:
            store.upsert_student(key, status="paused")
            raise
        except (LLMError, ParseError, ocr_engine.OCRUnavailable, ocr_engine.PageError) as e:
            store.upsert_student(key, status="failed", error=str(e))
            raise
        finally:
            used = self.client.end_budget()
        store.upsert_student(key, status="done", error="")
        try:
            store.sync_student_score(key, force=True)  # re-show even if previously deleted
        except Exception:
            pass  # scores table is best-effort; do not fail the run
        self._say(f"{key}: finished using {used} model call(s). Requests so far this run: {self.client.calls}.", 1.0)
        return store.get_student(key)

    def _run_stages(self, key, sources, paper, rubric):
        store = self.store
        method = "vlm"  # answer scripts are always read by the Gemini vision model
        primary = ocr_engine.PRIMARY_ENGINE[method]
        self.client.reasoning = self.read_effort

        # 1. preprocess (cached in the data folder)
        pages = store.get_pages(key)
        if not pages:
            self._say(f"{key}: preparing page images...", 0.02)
            page_dir = os.path.join(self.data_dir, "pages", safe_name(key))
            os.makedirs(page_dir, exist_ok=True)
            pages = ocr_engine.process_document(sources, page_dir, "p")
            store.save_pages(key, pages)
        n = len(pages)
        self._check()

        # 2. CALL 1 - read every page of the script in one request
        existing = store.get_transcripts(key, primary)
        todo = [p for p in range(1, n + 1) if p not in existing]
        if todo:
            self._check()
            label = f"page {todo[0]}" if len(todo) == 1 else f"pages {todo[0]}-{todo[-1]}"
            self._say(f"{key}: reading {label} of {n}...", 0.15)
            texts = ocr_engine.read_pages(self.client, [pages[p - 1][1] for p in todo],
                                          note=lambda m: self._say(f"{key}: {m}"))
            unread = []
            for page_no, text in zip(todo, texts):
                if text is None:
                    unread.append(page_no)       # not saved, so a resume reads just these again
                else:
                    store.save_transcript(key, primary, page_no, text)
            if unread:
                raise LLMError("The model's reply stopped before page(s) "
                               + ", ".join(map(str, unread)) + ". Press Start to read the rest.")
        page_texts = [store.get_transcripts(key, primary).get(i, "") for i in range(1, n + 1)]
        self._check()

        # 3. CALL 2 - match answers to questions AND read the student's name / roll number together
        student = store.get_student(key)
        answers = store.get_answers(key)
        need_answers = len(answers) != len(paper.questions)
        if need_answers or not student.id_checked:
            self._say(f"{key}: matching answers to questions and reading the student's name...", 0.55)
            found_id = None
            if need_answers:
                answers, found_id = parser.segment_answers(self.client, paper, page_texts, want_student=True)
                store.save_answers(key, answers)
            if not student.id_checked:
                if found_id is None and page_texts and page_texts[0].strip():
                    try:   # the model gave no identity (e.g. one-question paper): a small extra call, if any are left
                        found_id = ocr_engine.extract_student_id(self.client, page_texts[0])
                    except (AuthError, QuotaExceeded, Cancelled):
                        raise
                    except LLMError:
                        found_id = None
                if found_id:
                    store.upsert_student(key, name=found_id["name"], roll_no=found_id["roll_no"],
                                         id_confidence=found_id["confidence"], id_checked=1)
                else:
                    store.upsert_student(key, id_checked=1)  # leave blank; the teacher can type it
        self._check()

        # 4. CALL 3 - grade every question in one request (one repair call for anything the model skipped)
        self.client.reasoning = self.grade_effort
        done_ids = {e.question_id for e in store.get_evaluations(key)}
        by_id = {a.question_id: a for a in answers}
        pending = []
        for q in paper.questions:
            if q.id in done_ids:
                continue
            answer = by_id[q.id]
            if not answer.text.strip():  # nothing written: scores 0 without a model call
                store.save_evaluation(key, evaluate_answer(self.client, q, rubric.for_question(q.id), answer))
            else:
                pending.append((q, rubric.for_question(q.id), answer))

        for start in range(0, len(pending), MAX_QUESTIONS_PER_CALL):  # one chunk unless the paper is huge
            self._check()
            chunk = pending[start:start + MAX_QUESTIONS_PER_CALL]
            ids = ", ".join(str(q.id) for q, _qr, _a in chunk)
            self._say(f"{key}: grading question{'s' if len(chunk) > 1 else ''} {ids}...", 0.7)
            graded, missing = self._grade_chunk(chunk, rubric, pages, n)
            for ev in graded:
                store.save_evaluation(key, ev)  # saved even if others failed, so a resume only redoes the rest
            if missing:
                raise LLMError("The model did not grade question(s) " + ", ".join(str(q.id) for q, _r, _a in missing)
                               + ". Press Start to grade just those.")

    def _page_numbers(self, answer, n):
        """Pages whose images should accompany an answer the model could not read cleanly."""
        if answer.text.strip() and ("[unclear]" in answer.text or answer.confidence < 0.6):
            return [p for p in answer.pages[:3] if 1 <= p <= n]
        return []

    def _grade_chunk(self, chunk, rubric, pages, n):
        """Grade a list of (question, qr, answer) with ONE call, plus ONE repair call for any question the
        reply skipped or garbled. Returns (evaluations, items_still_ungraded); never makes more calls."""
        attached, needs = [], {}
        for q, _qr, answer in chunk:  # attach the unclear pages that fit; the rest are graded from text alone
            extra = [p for p in self._page_numbers(answer, n) if p not in attached]
            wanted = self._page_numbers(answer, n)
            if wanted and len(attached) + len(extra) <= MAX_BATCH_IMAGES:
                attached += extra
                needs[q.id] = wanted
        images = [pages[p - 1][1] for p in attached]

        got, remaining = {}, list(chunk)
        for _attempt in range(2):  # the call, then (if needed) the repair call
            if not remaining:
                break
            try:
                reply = evaluate_batch(self.client, remaining, rubric.drafted, images, attached,
                                       {q.id: needs[q.id] for q, _r, _a in remaining if q.id in needs})
            except (AuthError, QuotaExceeded, Cancelled):
                raise
            except LLMError:
                reply = {}  # unusable reply (or the call budget is used up)
            got.update(reply)
            remaining = [it for it in chunk if it[0].id not in got]
        return [got[q.id] for q, _r, _a in chunk if q.id in got], remaining

    # ---- many scripts ----
    def run_all(self, scripts, paper, rubric, force=False):
        """scripts = [(key, [source paths]), ...]. One failed script does not stop the rest;
        cancellation, auth errors and quota exhaustion do (progress stays saved)."""
        self._count = max(1, len(scripts))
        failed = []
        for i, (key, sources) in enumerate(scripts):
            self._index = i
            self._check()
            try:
                self.run_script(key, sources, paper, rubric, force)
            except (Cancelled, QuotaExceeded, AuthError):
                raise
            except (LLMError, ParseError, ocr_engine.OCRUnavailable, ocr_engine.PageError, OSError) as e:
                failed.append((key, str(e)))
                self._say(f"{key}: FAILED ({e})", 1.0)
        self._index = self._count - 1  # so "Finished" reports exactly 100%
        used = self._client.tokens_used if self._client else 0
        self._say(f"Finished. Model tokens used this run: {used:,}." if used else "Finished.", 1.0)
        return {"total": len(scripts), "failed": failed}


# =====================================================================
# 4. Reports
# =====================================================================
def _csv_safe(value):
    s = "" if value is None else str(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s  # stop spreadsheet formula injection


def export_marksheet_csv(store, path):
    paper = store.load_paper()
    if paper is None:
        raise ValueError("There are no results to export yet.")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["script", "name", "roll_no"] + [f"Q{q.id}" for q in paper.questions]
                   + ["total", "max_total", "ai_total", "status", "approved"])
        for s in store.list_students():
            evs = {e.question_id: e for e in store.get_evaluations(s.script_key)}
            final, ai, mx = store.totals(s.script_key)
            approved = ("yes" if evs and all(e.approved for e in evs.values())
                        else "partial" if any(e.approved for e in evs.values()) else "no")
            w.writerow([_csv_safe(s.script_key), _csv_safe(s.name), _csv_safe(s.roll_no)]
                       + [f"{evs[q.id].marks:g}" if q.id in evs else "" for q in paper.questions]
                       + [f"{final:g}", f"{mx:g}", f"{ai:g}", s.status, approved])


def _html_to_pdf(page_html, path):
    import fitz

    story = fitz.Story(html=page_html)
    writer = fitz.DocumentWriter(path)
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (36, 36, -36, -36)
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()


def _e(text):
    return html.escape(str(text or "")).replace("\n", "<br>")


def write_student_report(store, key, path):
    """PDF with identity, question-wise answers, marks, rationale, feedback and the total."""
    student, paper = store.get_student(key), store.load_paper()
    if student is None or paper is None:
        raise ValueError("No results for this student.")
    evs = {e.question_id: e for e in store.get_evaluations(key)}
    answers = {a.question_id: a for a in store.get_answers(key)}
    final, _ai, mx = store.totals(key)
    parts = [
        "<h1>Evaluation Report</h1>",
        f"<p><b>Student:</b> {_e(student.name or '-')} &nbsp; <b>Roll no.:</b> {_e(student.roll_no or '-')} "
        f"&nbsp; <b>Script:</b> {_e(student.script_key)}</p>",
        f"<h2>Total: {final:g} / {mx:g}</h2><hr>",
    ]
    for q in paper.questions:
        ev, ans = evs.get(q.id), answers.get(q.id)
        parts.append(f"<h3>Question {_e(q.id)} &mdash; {ev.marks:g} / {q.max_marks:g}</h3>" if ev
                     else f"<h3>Question {_e(q.id)} &mdash; not evaluated</h3>")
        parts.append(f"<p><i>{_e(_clean_math(q.text))}</i></p>")
        parts.append(f"<p><b>Extracted answer:</b><br>{_e(_clean_math(ans.text if ans and ans.text else '(no answer found)'))}</p>")
        if ev:
            for p in ev.points:
                parts.append(f"<p>&bull; {p['awarded']:g}/{p['max']:g} &nbsp;{_e(_clean_math(p['description']))}"
                             + (f" &mdash; {_e(_clean_math(p['comment']))}" if p.get("comment") else "") + "</p>")
            parts.append(f"<p><b>Rationale:</b> {_e(_clean_math(ev.rationale))}</p>")
            parts.append(f"<p><b>Feedback:</b> {_e(_clean_math(ev.feedback_text))}</p>")
            if ev.final_marks is not None and abs(ev.final_marks - ev.ai_marks) > 0.001:
                parts.append(f"<p><i>Marks adjusted by the teacher (AI suggested {ev.ai_marks:g}).</i></p>")
        # no blank gap between questions — only a thin rule, no extra block spacing
        parts.append("<hr style='margin:4px 0;border:none;border-top:1px solid #ccc'>")
    _html_to_pdf(
        "<html><body style='font-size:10pt;margin:0'>"
        "<style>h3{margin:8px 0 2px 0}p{margin:2px 0}</style>"
        + "".join(parts) + "</body></html>",
        path,
    )


# =====================================================================
# 5. Agreement with faculty marks
# =====================================================================
def _mean(xs):
    return sum(xs) / len(xs)


def _pearson(x, y):
    if len(x) < 2:
        return None
    mx, my = _mean(x), _mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def quadratic_weighted_kappa(a, b, step=0.5):
    """Agreement on marks grouped into `step`-sized bins, penalising big disagreements more."""
    ca, cb = [round(v / step) for v in a], [round(v / step) for v in b]
    low, high = min(ca + cb), max(ca + cb)
    k = high - low + 1
    if k == 1:
        return 1.0
    n = len(ca)
    observed = [[0.0] * k for _ in range(k)]
    for i, j in zip(ca, cb):
        observed[i - low][j - low] += 1
    row = [sum(r) for r in observed]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * observed[i][j]
            den += w * row[i] * col[j] / n
    return None if den == 0 else 1 - num / den


def agreement_metrics(ai, faculty):
    """Agreement statistics for two equally long lists of marks."""
    n = len(ai)
    if n == 0 or n != len(faculty):
        return {"n": 0}
    diffs = [a - f for a, f in zip(ai, faculty)]
    return {
        "n": n,
        "mae": _mean([abs(d) for d in diffs]),
        "rmse": math.sqrt(_mean([d * d for d in diffs])),
        "bias": _mean(diffs),  # positive = the AI is more generous than the faculty
        "pearson": _pearson(ai, faculty),
        "spearman": _pearson(_ranks(ai), _ranks(faculty)),
        "qwk": quadratic_weighted_kappa(ai, faculty),
        "exact_pct": 100 * sum(abs(d) < 1e-9 for d in diffs) / n,
        "within_half_pct": 100 * sum(abs(d) <= 0.5 + 1e-9 for d in diffs) / n,
        "within_one_pct": 100 * sum(abs(d) <= 1.0 + 1e-9 for d in diffs) / n,
    }


def import_faculty_csv(store, path):
    """CSV with columns: student (roll no. or script name), question, marks.
    Returns (rows_imported, warnings)."""
    rows, warnings = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        names = {(h or "").strip().lower(): h for h in (reader.fieldnames or [])}

        def pick(*options):
            return next((names[o] for o in options if o in names), None)

        c_student = pick("student", "roll_no", "roll", "script", "id")
        c_q = pick("question", "question_id", "q")
        c_marks = pick("marks", "mark", "score")
        if not (c_student and c_q and c_marks):
            raise ValueError("The CSV needs columns named student, question and marks.")
        for line, r in enumerate(reader, start=2):
            try:
                marks = float(r[c_marks])
            except (TypeError, ValueError):
                warnings.append(f"Line {line}: marks '{r[c_marks]}' is not a number (skipped).")
                continue
            rows.append((str(r[c_student]).strip(), str(r[c_q]).strip().lstrip("Qq"), marks))
    store.set_faculty_marks(rows)
    return len(rows), warnings


def agreement_report(store):
    """Compare AI marks with imported faculty marks.

    Returns {"pairs": [...], "question": metrics, "total": metrics, "per_question": {...}, "unmatched": [...]}."""
    students = store.list_students()
    ref_to_key = {}
    for s in students:
        for ref in (s.script_key, s.roll_no):
            if ref:
                ref_to_key[ref.strip().lower()] = s.script_key
    faculty, unmatched = {}, []
    for ref, qid, marks in store.get_faculty_marks():
        key = ref_to_key.get(ref.strip().lower())
        if key is None:
            if ref not in unmatched:
                unmatched.append(ref)
            continue
        faculty[(key, qid)] = marks
    pairs, totals_ai, totals_fac = [], [], []
    for s in students:
        evs = {e.question_id: e for e in store.get_evaluations(s.script_key)}
        got = [(q, e) for q, e in evs.items() if (s.script_key, q) in faculty]
        for qid, e in got:
            pairs.append({"script": s.script_key, "question": qid, "max": e.max_marks,
                          "ai": e.ai_marks, "faculty": faculty[(s.script_key, qid)]})
        if got and len(got) == len(evs):
            totals_ai.append(sum(e.ai_marks for _, e in got))
            totals_fac.append(sum(faculty[(s.script_key, q)] for q, _ in got))
    per_q = {}
    for qid in dict.fromkeys(p["question"] for p in pairs):
        sel = [p for p in pairs if p["question"] == qid]
        per_q[qid] = agreement_metrics([p["ai"] for p in sel], [p["faculty"] for p in sel])
    return {"pairs": pairs, "unmatched": unmatched, "per_question": per_q,
            "question": agreement_metrics([p["ai"] for p in pairs], [p["faculty"] for p in pairs]),
            "total": agreement_metrics(totals_ai, totals_fac)}


def export_agreement_csv(report, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["script", "question", "max", "ai_marks", "faculty_marks", "difference"])
        for p in report["pairs"]:
            w.writerow([_csv_safe(p["script"]), p["question"], f"{p['max']:g}", f"{p['ai']:g}",
                        f"{p['faculty']:g}", f"{p['ai'] - p['faculty']:g}"])

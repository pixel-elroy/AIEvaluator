"""Parsing: question paper, marking scheme (rubric) and answer segmentation.

Everything that turns documents or transcriptions into structured data lives here,
together with the small data classes the other stages share.
No Kivy imports.
"""
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field

from core import ocr_engine


class ParseError(Exception):
    """A document could not be turned into structured data."""


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _num(value):
    """float(value), or None when it is not a usable number. Accepts '2', '2.5', '2 marks', 'L7'."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        m = _NUM_RE.search(value)
        if not m:
            return None
        value = m.group(0)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _first(d, keys):
    """First non-None value stored under any of `keys` in dict d."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


def _norm_id(value):
    """'Q2(a)', 'Question 2a', '2 (a)', '2.a' -> '2a' so ids written differently still match."""
    s = str(value).strip().lower()
    s = re.sub(r"^(question|ques|qn|q)\s*[.:\-]?\s*", "", s)
    return re.sub(r"[\s().\[\]_\-:]", "", s)


class _IdMap:
    """Maps whatever id spelling a model used back to the paper's own question id."""

    def __init__(self, paper):
        self.exact = {q.id for q in paper.questions}
        groups = {}
        for q in paper.questions:
            groups.setdefault(_norm_id(q.id), []).append(q.id)
        self.norm = {k: v[0] for k, v in groups.items() if len(v) == 1}

    def resolve(self, raw):
        if raw is None:
            return None
        raw = str(raw).strip()
        return raw if raw in self.exact else self.norm.get(_norm_id(raw))


_ID_KEYS = ("question_id", "id", "question", "number", "qid", "q", "no", "question_no")
_RUBRIC_WRAPPERS = ("rubric", "rubrics", "marking_scheme", "scheme", "marking_rubric", "questions",
                    "items", "result", "data", "output", "response")
_SEG_WRAPPERS = ("answers", "segments", "segmentation", "answer_ranges", "mapping", "result",
                 "data", "output", "response", "questions")


def _entries(data, ids, wrappers):
    """Normalise the shapes a model may reply with into [(raw_id_or_None, entry_dict), ...].

    Handles: a bare list of entries; {wrapper: [entries]}; {wrapper: {id: entry}} and {id: entry}.
    A non-dict value under an id key (e.g. {"1": [[1, 3]]}) is returned as {"_value": value}."""
    def from_list(lst):
        return [(_first(e, _ID_KEYS), e) for e in lst if isinstance(e, dict)]

    def from_map(m):
        return [(k, v if isinstance(v, dict) else {"_value": v})
                for k, v in m.items() if ids.resolve(k) is not None]

    if isinstance(data, list):
        return from_list(data)
    if not isinstance(data, dict):
        return []
    for key in wrappers:
        v = data.get(key)
        if isinstance(v, list):
            found = from_list(v)
            if found:
                return found
        elif isinstance(v, dict):
            found = from_map(v) or _entries(v, ids, wrappers)
            if found:
                return found
    found = from_map(data)
    if found:
        return found
    for v in data.values():  # last resort: any list of objects
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return from_list(v)
    return []


def _resolve_entries(entries, paper, ids):
    """[(qid, entry)] using the paper's ids; falls back to ORDER when no id matches at all."""
    resolved = [(ids.resolve(rid), e) for rid, e in entries]
    if resolved and all(q is None for q, _ in resolved) and len(resolved) == len(paper.questions):
        return [(q.id, e) for q, (_, e) in zip(paper.questions, resolved)], True
    return resolved, False


# ---------------------------------------------------------------- data classes
@dataclass
class Question:
    id: str
    text: str
    max_marks: float


@dataclass
class RubricPoint:
    description: str
    marks: float


@dataclass
class QuestionRubric:
    question_id: str
    model_answer: str = ""
    points: list = field(default_factory=list)  # list[RubricPoint]
    notes: str = ""


@dataclass
class Answer:
    question_id: str
    text: str = ""
    confidence: float = 0.0
    pages: list = field(default_factory=list)  # 1-based page numbers


@dataclass
class QuestionPaper:
    questions: list = field(default_factory=list)  # list[Question]
    title: str = ""
    warnings: list = field(default_factory=list)

    @property
    def total_marks(self):
        return sum(q.max_marks for q in self.questions)

    def get(self, question_id):
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def to_dict(self):
        return {"title": self.title, "warnings": self.warnings,
                "questions": [{"id": q.id, "text": q.text, "max_marks": q.max_marks}
                              for q in self.questions]}

    @classmethod
    def from_dict(cls, d):
        """Build a question paper from model JSON or a previously saved dictionary.

        Some models return the questions array directly; others wrap it in a
        ``data``/``result`` object. Normalize those forms before validation.
        """
        if isinstance(d, list):
            q_list = d
            metadata = {}
        elif isinstance(d, dict):
            metadata = d
            q_list = d.get("questions")
            if not isinstance(q_list, list):
                # Handle common wrapper objects without mistaking warnings or
                # unrelated arrays for the question list.
                for key in ("data", "result", "output", "response", "content"):
                    nested = d.get(key)
                    if isinstance(nested, str):
                        try:
                            import json
                            nested = json.loads(nested)
                        except (ValueError, TypeError):
                            continue
                    if isinstance(nested, dict) and isinstance(nested.get("questions"), list):
                        metadata = {**nested, **{k: v for k, v in d.items() if k not in ("data", "result", "output", "response", "content")}}
                        q_list = nested["questions"]
                        break
                    if isinstance(nested, list):
                        q_list = nested
                        break
            if not isinstance(q_list, list):
                # Last-resort fallback: only accept a list whose entries look
                # like questions, rather than any arbitrary metadata list.
                for val in d.values():
                    if isinstance(val, list) and val and all(
                        isinstance(item, str) or
                        (isinstance(item, dict) and any(k in item for k in ("text", "question", "id", "number")))
                        for item in val
                    ):
                        q_list = val
                        break
        else:
            metadata, q_list = {}, None

        if not isinstance(q_list, list) or not q_list:
            raise ParseError("No questions were found in the question paper.")

        questions, warnings, seen = [], [], []
        for i, q in enumerate(q_list, start=1):
            if isinstance(q, str):
                questions.append(Question(str(i), q, 10.0))
                continue
            if not isinstance(q, dict):
                continue
            qid = str(q.get("id", "") or q.get("number", "")).strip() or str(i)
            if qid in seen:
                warnings.append(f"Duplicate question id '{qid}' was renamed.")
                qid = f"{qid}_{i}"
            seen.append(qid)

            raw_marks = next((q[k] for k in ("max_marks", "marks", "points")
                              if k in q and q[k] is not None and q[k] != ""), None)
            marks = _num(raw_marks)
            if marks is None:
                marks = 10.0  # placeholder so the paper still loads
                warnings.append(f"Question {qid}: no marks were found, 10 was assumed. Check the paper.")

            text = str(q.get("text", "") or q.get("question", "") or "").strip()
            if not text:
                warnings.append(f"Question {qid} was skipped because its text was empty.")
                continue
            questions.append(Question(qid, text, marks))

        if not questions:
            raise ParseError("No questions were found in the question paper.")

        saved_warnings = metadata.get("warnings", [])
        if not isinstance(saved_warnings, list):
            saved_warnings = []
        return cls(questions, str(metadata.get("title", "Exam") or "Exam").strip(),
                   list(saved_warnings) + warnings)


@dataclass
class Rubric:
    items: dict = field(default_factory=dict)  # question_id -> QuestionRubric
    drafted: bool = False                       # True = written by the AI, not by a teacher
    warnings: list = field(default_factory=list)

    def for_question(self, question_id):
        return self.items.get(question_id) or QuestionRubric(question_id)

    def to_dict(self):
        return {"drafted": self.drafted, "warnings": self.warnings,
                "rubric": [{"question_id": r.question_id, "model_answer": r.model_answer,
                            "notes": r.notes,
                            "points": [{"description": p.description, "marks": p.marks}
                                       for p in r.points]}
                           for r in self.items.values()]}

    @classmethod
    def from_dict(cls, d, paper, drafted=False):
        """Validate a model reply (or saved dict) against the question paper.

        Tolerant of id spellings (Q1 / 1 / 2(a) / 2a), of a missing {"rubric": ...} wrapper and
        of the rubric being keyed by question id."""
        warnings = [str(w) for w in d.get("warnings", [])] if isinstance(d, dict) and isinstance(d.get("warnings"), list) else []
        ids = _IdMap(paper)
        entries = _entries(d, ids, _RUBRIC_WRAPPERS)
        if not entries:
            raise ParseError("The marking scheme could not be structured.")
        resolved, positional = _resolve_entries(entries, paper, ids)
        if positional:
            warnings.append("Question ids in the marking scheme did not match the paper; they were matched by order.")
        items = {}
        for qid, e in resolved:
            if qid is None:
                warnings.append("Marking scheme mentions a question that is not on the paper (ignored).")
                continue
            if qid in items and items[qid].points:
                continue
            items[qid] = QuestionRubric(
                qid,
                str(_first(e, ("model_answer", "answer", "solution", "expected_answer")) or "").strip(),
                _parse_points(_first(e, ("points", "marking_points", "criteria", "steps", "marking_scheme", "scheme", "breakdown", "_value"))),
                str(_first(e, ("notes", "note", "partial_credit", "comments")) or "").strip())
        for q in paper.questions:
            item = items.get(q.id)
            if item is None or not item.points:
                items.setdefault(q.id, QuestionRubric(q.id))
                warnings.append(f"Question {q.id}: no itemised marking points.")
            elif abs(sum(p.marks for p in item.points) - q.max_marks) > 0.01:
                warnings.append(f"Question {q.id}: marking points add up to "
                                f"{sum(p.marks for p in item.points):g}, not {q.max_marks:g}.")
        return cls(items, drafted, warnings)


_MARKS_IN_TEXT = re.compile(r"\(?\s*(\d+(?:\.\d+)?)\s*(?:marks?|m)\b\s*\)?", re.I)


def _parse_points(raw):
    """Marking points from a list of {description, marks}, a {description: marks} map or strings."""
    if isinstance(raw, dict):
        raw = [{"description": k, "marks": v} for k, v in raw.items()]
    points = []
    for p in raw if isinstance(raw, list) else []:
        if isinstance(p, dict):
            marks = _num(_first(p, ("marks", "mark", "score", "points", "max_marks", "value")))
            desc = _first(p, ("description", "point", "criterion", "criteria", "step", "text", "title", "name"))
        elif isinstance(p, str):
            m = _MARKS_IN_TEXT.search(p)
            marks = float(m.group(1)) if m else None
            desc = _MARKS_IN_TEXT.sub("", p, count=1).strip(" -:") if m else None
        else:
            continue
        if marks is not None:
            points.append(RubricPoint(str(desc or "").strip(), marks))
    return points


# ------------------------------------------------------------- reading documents
def read_document_text(path, client=None, progress=None, engine="paddleocr", force_ocr=False):
    """Plain text of a .txt / .pdf / image file.

    A PDF with a real text layer is read directly unless force_ocr=True. Scanned PDFs and images use the selected local OCR backend."""
    low = path.lower()
    if low.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    
    # If it's a PDF with a native text layer, extract directly
    if low.endswith(".pdf") and not force_ocr:
        import fitz
        with fitz.open(path) as doc:
            texts = [page.get_text() for page in doc]
        if sum(len(t.strip()) for t in texts) >= 40 * max(1, len(texts)):
            return "\n\n".join(texts)

    if engine not in ("paddleocr", "vlm"):
        raise ParseError(f"Unknown document OCR engine: {engine}")
    
    if engine == "vlm" and client is None:
        raise ParseError(f"'{os.path.basename(path)}' needs the configured vision model to read it.")
        
    work = tempfile.mkdtemp(prefix="evaluator_doc_")
    try:
        pages = ocr_engine.process_document([path], work, "doc")
        out = []
        for i, (_orig, enhanced, _binary) in enumerate(pages, start=1):
            if progress:
                progress(f"OCR page {i}/{len(pages)} of {os.path.basename(path)} using {engine}")
            out.append(ocr_engine.transcribe_page(engine, enhanced, client))
        return "\n\n".join(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------- model stages
def _questions_listing(paper):
    return "\n".join(f"{q.id}: ({q.max_marks:g} marks) {q.text}" for q in paper.questions)


def parse_question_paper(client, text):
    """Question paper text -> QuestionPaper (ids, text, maximum marks)."""
    prompt = (
        "TASK: parse_question_paper\n"
        "Below is the full text of an exam question paper. Extract every question the students must answer.\n"
        "Return ONLY a JSON object of this form:\n"
        '{"title": "<exam title or empty string>", "questions": [{"id": "<question number as printed, e.g. 1, 2a, 3(b)>", '
        '"text": "<full question text>", "max_marks": <number or null>}]}\n'
        "Rules:\n"
        "- Keep the paper's own numbering for ids. Ids must be unique.\n"
        "- If sub-parts carry their own marks, list each sub-part as a separate question (for example 2a, 2b) "
        "and repeat the shared stem in its text. If only the whole question has marks, keep it as one question.\n"
        "- max_marks must be a number; use null when no marks are stated.\n"
        "- Copy question text faithfully. Do not answer the questions.\n"
        "- Ignore general instructions, headers and footers that are not questions.\n\n"
        "QUESTION PAPER:\n<<<\n" + text.strip() + "\n>>>"
    )
    if not (text or "").strip():
        raise ParseError("PaddleOCR did not extract any text from the question paper. Check the PDF pages or OCR installation.")
    try:
        data = client.ask_json(prompt)
        return QuestionPaper.from_dict(data)
    except ParseError as exc:
        raise ParseError(
            f"The model reply could not be converted into questions ({exc}). "
            "Check the OCR text and the model's JSON output."
        ) from exc


def parse_rubric(client, paper, rubric_text):
    """Teacher's marking scheme text -> structured Rubric."""
    prompt = (
        "TASK: parse_rubric\n"
        "Below are the exam questions and the teacher's marking scheme. Structure the marking scheme for every question.\n"
        "Return ONLY JSON: " + _RUBRIC_SCHEMA + "\n"
        "Rules:\n"
        "- Use exactly the question ids listed below.\n"
        "- Each point is one gradable item that earns marks; the marks of a question's points should add up to its maximum marks.\n"
        "- Copy the scheme's wording faithfully. Do NOT invent points the scheme does not contain.\n"
        "- Put partial-credit rules and accepted alternative methods in \"notes\".\n"
        "- If the scheme says nothing about a question, return it with an empty points list.\n\n"
        "QUESTIONS:\n" + _questions_listing(paper) + "\n\n"
        "MARKING SCHEME:\n<<<\n" + rubric_text.strip() + "\n>>>"
    )
    return Rubric.from_dict(client.ask_json(prompt), paper, drafted=False)


def _ask(client, prompt, max_tokens):
    """client.ask_json that also works with simple clients lacking max_tokens."""
    # While a per-script call budget is active, one attempt only: the next prompt is the retry.
    attempts = 1 if getattr(client, "call_budget", None) is not None else 3
    try:
        return client.ask_json(prompt, max_tokens=max_tokens, attempts=attempts)
    except TypeError as exc:
        if "max_tokens" not in str(exc):
            raise
        return client.ask_json(prompt)


def _snippet(client):
    """What the model actually said, so an error message shows the real cause."""
    reply = (getattr(client, "last_reply", "") or "").strip().replace("\n", " ")
    return f" The model replied: {reply[:300]!r}" if reply else ""


def _draft_prompts(paper):
    questions = _questions_listing(paper)
    return [
        (
            "TASK: draft_rubric\n"
            "Solve each exam question and create a concise marking rubric.\n"
            "Return one JSON object only, with this exact structure: " + _RUBRIC_SCHEMA + "\n"
            "Include every question_id exactly as listed. For each question, provide a model_answer, "
            "a points array of gradable steps, and notes. The point marks must add up to the question's "
            "maximum marks. Do not use Markdown fences or commentary.\n\nQUESTIONS:\n" + questions
        ),
        (
            "Return ONLY a valid JSON object. No prose, no Markdown, no code fences.\n"
            "The top-level key must be rubric. Its value must be an array with one object per question.\n"
            "Each object must have question_id (string), model_answer (string), points (array of objects "
            "with description (string) and marks (number)), and notes (string).\n"
            "Use these question IDs and maximum marks exactly; keep model answers and marking points concise.\n"
            + questions + "\nJSON schema example: " + _RUBRIC_SCHEMA
        ),
    ]


def _rescale_points(rubric, paper):
    """A drafted rubric whose points do not add up to the question's marks would make full marks
    unreachable (or too easy). Scale such questions so they total exactly the question's marks.
    Only used for AI-drafted rubrics: a teacher's own allocation is never changed."""
    for q in paper.questions:
        item = rubric.items.get(q.id)
        total = sum(p.marks for p in item.points) if item else 0
        if not item or not item.points or total <= 0 or abs(total - q.max_marks) <= 0.01:
            continue
        scaled = [round(p.marks * q.max_marks / total, 2) for p in item.points]
        scaled[-1] = round(q.max_marks - sum(scaled[:-1]), 2)  # make the sum exact
        for point, marks in zip(item.points, scaled):
            point.marks = max(0.0, marks)
        old = f"Question {q.id}: marking points add up to {total:g}, not {q.max_marks:g}."
        rubric.warnings = [w for w in rubric.warnings if w != old]
        rubric.warnings.append(f"Question {q.id}: the drafted points added up to {total:g}; "
                               f"they were rescaled to total {q.max_marks:g}.")


def draft_rubric(client, paper):
    """Ask the model to draft a rubric. Tolerates id spellings and wrapper differences, retries once
    with a simpler prompt, then re-asks only for questions that came back without marking points."""
    rubric, last_error = None, None
    for prompt in _draft_prompts(paper):
        try:
            candidate = Rubric.from_dict(_ask(client, prompt, 6000), paper, drafted=True)
        except (ParseError, ValueError, TypeError) as exc:
            last_error = exc
            continue
        if any(candidate.for_question(q.id).points for q in paper.questions):
            rubric = candidate
            break
        last_error = ParseError("the reply contained no marking points")
    if rubric is None:
        raise ParseError(
            "The model could not draft a usable marking scheme after two attempts "
            f"({last_error}).{_snippet(client)} Try again, or supply a marking scheme file.")

    # Re-ask once for just the questions that have no points yet.
    missing = [q for q in paper.questions if not rubric.for_question(q.id).points]
    if missing:
        sub = QuestionPaper(missing, paper.title)
        try:
            extra = Rubric.from_dict(_ask(client, _draft_prompts(sub)[1], 4000), sub, drafted=True)
            for q in missing:
                got = extra.for_question(q.id)
                if got.points:
                    rubric.items[q.id] = got
                    rubric.warnings = [w for w in rubric.warnings if w != f"Question {q.id}: no itemised marking points."]
        except (ocr_engine.AuthError, ocr_engine.QuotaExceeded):
            raise
        except (ParseError, ValueError, TypeError, ocr_engine.LLMError):
            pass  # these questions are graded holistically and flagged for review
    _rescale_points(rubric, paper)
    rubric.warnings.insert(
        0,
        "No instructor marking scheme was supplied: the model drafted this rubric. "
        "Review the model answers, point allocations, and partial-credit rules before relying on marks."
    )
    return rubric


_RUBRIC_SCHEMA = (
    '{"rubric": [{"question_id": "<id>", "model_answer": "<model answer or empty string>", '
    '"points": [{"description": "<what earns the marks>", "marks": <number>}], "notes": "<text or empty string>"}]}'
)


def _number_lines(page_texts):
    lines = []
    for page_no, text in enumerate(page_texts, start=1):
        for line in text.splitlines():
            if line.strip():
                lines.append((page_no, line.rstrip()))
    return lines


def _parse_ranges(value):
    """Line-number spans from the shapes models use -> [(first, last), ...].

    [[1, 3], [7, 8]] / [1, 3] / [4, 5, 9] (single lines) / "L1-L3, L7" / {"start": 1, "end": 3}."""
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [(int(value), int(value))]
    if isinstance(value, str):
        spans = []
        for part in re.split(r"[;,]", value):
            nums = [int(n) for n in re.findall(r"\d+", part)]
            if len(nums) >= 2:
                spans.append((nums[0], nums[1]))
            elif nums:
                spans.append((nums[0], nums[0]))
        return spans
    if isinstance(value, dict):
        a = _num(_first(value, ("start", "first", "from", "begin", "start_line")))
        b = _num(_first(value, ("end", "last", "to", "stop", "end_line")))
        return [(int(a), int(b if b is not None else a))] if a is not None else []
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            if len(value) == 2:
                return [(int(value[0]), int(value[1]))]
            return [(int(v), int(v)) for v in value]
        spans = []
        for v in value:
            spans.extend(_parse_ranges(v))
        return spans
    return []


def _answers_from_reply(data, paper, lines):
    """list[Answer] from a model reply, or None when the reply's structure is not understood."""
    ids = _IdMap(paper)
    entries = _entries(data, ids, _SEG_WRAPPERS)
    if not entries:
        return None
    resolved, _ = _resolve_entries(entries, paper, ids)
    by_id = {}
    for qid, e in resolved:
        if qid is None or qid in by_id:
            continue
        spans = _parse_ranges(_first(e, ("ranges", "line_ranges", "lines", "range", "line_numbers",
                                         "line_range", "spans", "_value")))
        chosen = set()
        for first, last in spans:
            if first > last:
                first, last = last, first
            first, last = max(1, first), min(len(lines), last)
            chosen.update(range(first, last + 1))
        picked = [lines[n - 1] for n in sorted(chosen)]
        conf = _num(e.get("confidence"))
        if conf is not None and conf > 1:
            conf /= 100.0  # a percentage
        conf = 0.7 if conf is None else min(1.0, conf)  # no confidence given: assume reasonable
        by_id[qid] = Answer(qid, "\n".join(t for _, t in picked), conf if picked else 0.0,
                            sorted({pg for pg, _ in picked}))
    if not by_id:
        return None
    return [by_id.get(q.id, Answer(q.id)) for q in paper.questions]


_MARKER = re.compile(
    r"^\s*(?:(?:q(?:uestion)?|ans(?:wer)?)\s*[.:\-]?\s*(?:no\.?)?\s*)?\(?"
    r"(\d+\s*(?:\(\s*[a-z]\s*\)|[a-z])?)\)?\s*(?:[.):\-]|\s|$)", re.I)


def _regex_segment(paper, lines):
    """Fallback when the model's reply cannot be used: split on question labels such as
    'Q2', '2(a)', 'Ans 3'. Marked with low confidence so the teacher is asked to check it."""
    ids = _IdMap(paper)
    order = {q.id: i for i, q in enumerate(paper.questions)}
    marks, last = [], -1  # (line index, question id)
    for n, (_pg, text) in enumerate(lines):
        m = _MARKER.match(text)
        qid = ids.resolve(m.group(1)) if m else None
        if qid is not None and order[qid] > last:
            marks.append((n, qid))
            last = order[qid]
    if not marks:
        return None
    found = {}
    for k, (start, qid) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        picked = lines[start:end]
        found[qid] = Answer(qid, "\n".join(t for _, t in picked), 0.45, sorted({pg for pg, _ in picked}))
    return [found.get(q.id, Answer(q.id)) for q in paper.questions]


_STUDENT_SCHEMA = '"student": {"name": "<name or empty string>", "roll_no": "<roll number or empty string>", "confidence": <0 to 1>}, '
_STUDENT_RULE = ("Also read the student's name and roll / registration number, usually written at the top of the "
                 "first page; copy them exactly as written, or use empty strings if absent.\n")


def _student_from_reply(data):
    """{"name", "roll_no", "confidence"} from a reply's "student" entry, or None if it has none."""
    raw = data.get("student") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return None

    def clean(v):
        return str(v).strip() if isinstance(v, (str, int, float)) and not isinstance(v, bool) else ""

    try:
        conf = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"name": clean(raw.get("name")), "roll_no": clean(raw.get("roll_no")), "confidence": conf}


def segment_answers(client, paper, page_texts, want_student=False):
    """Assign transcription lines to questions.

    With want_student=True the same model call also reads the student's name and roll number, and the
    result is (answers, student) where student is a dict or None (the model did not give one).

    Order of attempts: (1) one question -> the whole script; (2) the model, twice with different
    prompts, accepting any reasonable reply shape; (3) label-based splitting. Never returns silently
    wrong data without a low confidence (which makes the review screen flag it)."""
    def result(answers, student=None):
        return (answers, student) if want_student else answers

    blank = [Answer(q.id) for q in paper.questions]
    lines = _number_lines(page_texts)
    if not lines:
        return result(blank)
    if len(paper.questions) == 1:
        return result([Answer(paper.questions[0].id, "\n".join(t for _, t in lines), 0.9,
                              sorted({pg for pg, _ in lines}))])

    numbered, last_page = [], 0
    for n, (page_no, text) in enumerate(lines, start=1):
        if page_no != last_page:
            numbered.append(f"=== PAGE {page_no} ===")
            last_page = page_no
        numbered.append(f"L{n}: {text}")
    listing, transcription = _questions_listing(paper), "\n".join(numbered)
    prompts = [
        ("TASK: segment_answers\n"
         "Decide which lines are the student's answer to each question.\n"
         + (_STUDENT_RULE if want_student else "") +
         'Return ONLY JSON: {' + (_STUDENT_SCHEMA if want_student else "") +
         '"answers": [{"question_id": "<id>", "ranges": [[<first line number>, <last line number>]], '
         '"confidence": <number between 0 and 1>}]}\n'
         "Use the question ids exactly as listed and the numbers after L as line numbers.\n\n"
         "QUESTIONS:\n" + listing + "\n\nTRANSCRIPTION:\n" + transcription, 4096),
        ("TASK: segment_answers_retry\n"
         "Return one JSON object and nothing else. The object MUST have this exact shape: "
         + ('{"student": {"name": "", "roll_no": "", "confidence": 0.9}, ' if want_student else '{') +
         '"answers": [{"question_id": "EXACT_ID_FROM_LIST", "ranges": [[1, 3]], "confidence": 0.8}]}\n'
         "Each range is [first_line_number, last_line_number] using the L numbers below. "
         "Use an empty ranges list when no answer for that question is present. "
         "Do not add markdown, explanations, or extra keys. Include every question exactly once.\n\n"
         "QUESTIONS:\n" + listing + "\n\nTRANSCRIPTION:\n" + transcription, 3000),
    ]
    student, understood = None, None
    for prompt, tokens in prompts:
        try:
            data = _ask(client, prompt, tokens)
        except (ocr_engine.AuthError, ocr_engine.QuotaExceeded):
            raise
        except ocr_engine.LLMError:
            continue
        student = student or (_student_from_reply(data) if want_student else None)
        found = _answers_from_reply(data, paper, lines)
        if found is None:
            continue
        if any(a.text.strip() for a in found):
            return result(found, student)
        understood = found  # understood, but it claimed there is no answer anywhere

    fallback = _regex_segment(paper, lines)
    if fallback is not None:
        return result(fallback, student)
    if understood is not None:
        return result(understood, student)
    raise ParseError(
        "The model's answer-to-question matching could not be understood, and the transcription "
        f"has no question labels to fall back on.{_snippet(client)}")

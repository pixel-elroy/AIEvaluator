"""Student PDF report with typeset math.

Lives in ui/ on purpose: it only READS from the Store and never touches core/.
The report is built as HTML, KaTeX typesets the math, and headless Chromium prints it
to PDF (see ui/math_render.py).
"""
import html
import uuid

from ui.math_render import WEB_DIR, ensure_katex, renderer, to_html

CSS = """
@page { size: A4; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color: #000;
       font-size: 11pt; line-height: 1.45; margin: 0; background: #fff; }
h1 { font-size: 19pt; margin: 0 0 4px 0; }
.sub { color: #000; margin-bottom: 10px; }
.total { display: inline-block; background: #d7efcf; border: 1px solid #9bcf8c;
         border-radius: 8px; padding: 6px 14px; font-weight: 600; font-size: 13pt; margin: 6px 0 8px 0; }
.q { border: 1px solid #b9dcae; background: #f3faf0; border-radius: 0;
     padding: 8px 12px; margin: 0; border-bottom: none; break-inside: avoid; }
.q:first-of-type { border-radius: 8px 8px 0 0; }
.q:last-of-type { border-radius: 0 0 8px 8px; border-bottom: 1px solid #b9dcae; }
.q + .q { border-top: 1px solid #b9dcae; }
.q h2 { font-size: 12.5pt; margin: 0 0 4px 0; display: flex; justify-content: space-between; }
.lbl { font-weight: 600; margin-top: 6px; }
.body { white-space: pre-wrap; overflow-wrap: anywhere; }
.pts { margin: 2px 0 0 0; padding: 0; list-style: none; }
.pts li { white-space: pre-wrap; padding: 1px 0; }
.m { max-width: 100%; }
.m[data-d="1"] { display: block; }
.katex-display { margin: 0.35em 0; }
.bad { color: #c00; }
"""


def _h(text):
    """Escaped HTML with math spans (for model / student text)."""
    return to_html(text or "")


def _e(value):
    """Plain escaped text (names, ids)."""
    return html.escape(str(value if value is not None else ""))


def _question_block(q, ev, ans):
    head = f"<h2><span>Question {_e(q.id)}</span>"
    if ev is None:
        return (f'<div class="q">{head}<span>not evaluated</span></h2>'
                f'<div class="body">{_h(q.text)}</div></div>')
    head += f"<span>{ev.marks:g} / {q.max_marks:g}</span></h2>"
    parts = [f'<div class="q">{head}', f'<div class="body">{_h(q.text)}</div>']
    parts.append('<div class="lbl">Answer (as read from the script)</div>')
    parts.append(f'<div class="body">{_h(ans.text if ans and ans.text else "(no answer found)")}</div>')
    if ev.points:
        parts.append('<div class="lbl">Marking</div><ul class="pts">')
        for p in ev.points:
            line = f"{p['awarded']:g}/{p['max']:g}   {p['description']}"
            if p.get("comment"):
                line += f" — {p['comment']}"
            parts.append(f"<li>{_h(line)}</li>")
        parts.append("</ul>")
    if ev.rationale:
        parts.append('<div class="lbl">Rationale</div>')
        parts.append(f'<div class="body">{_h(ev.rationale)}</div>')
    if ev.feedback_text:
        parts.append('<div class="lbl">Feedback</div>')
        parts.append(f'<div class="body">{_h(ev.feedback_text)}</div>')
    parts.append("</div>")
    return "".join(parts)


def build_report_html(store, key):
    paper = store.load_paper()
    rec = store.get_student(key)
    answers = {a.question_id: a for a in store.get_answers(key)}
    evals = {e.question_id: e for e in store.get_evaluations(key)}
    final, _ai, mx = store.totals(key)
    try:
        subject = store.get_subject() or ""
    except Exception:
        subject = ""

    who = [f"Name: {_e(rec.name or '—')}", f"Roll no.: {_e(rec.roll_no or '—')}"]
    if subject:
        who.insert(0, f"Subject: {_e(subject)}")
    blocks = "".join(_question_block(q, evals.get(q.id), answers.get(q.id)) for q in paper.questions)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{_e(rec.display)}</title>
<link rel="stylesheet" href="katex/katex.min.css">
<style>{CSS}</style>
<script src="katex/katex.min.js"></script>
<script src="render.js"></script>
</head><body>
<div id="content">
<h1>Evaluation report</h1>
<div class="sub">{"  ·  ".join(who)}</div>
<div class="total">Total: {final:g} / {mx:g}</div>
{blocks}
</div>
<script>window.addEventListener("DOMContentLoaded", function () {{
  renderMathIn(document.getElementById("content"));
}});</script>
</body></html>"""


def write_student_report_pdf(store, key, out_path):
    """Write one student's PDF report with rendered math. Blocks until the file is written."""
    ensure_katex()
    doc = build_report_html(store, key)
    tmp = WEB_DIR / f"_report_{uuid.uuid4().hex}.html"  # must sit next to katex/ and render.js
    tmp.write_text(doc, encoding="utf-8")
    try:
        renderer.render_pdf(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)

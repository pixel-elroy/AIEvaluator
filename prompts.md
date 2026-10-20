# Prompts for Answer Script Evaluator

All model prompts used by the application. Edit carefully; the code expects the exact TASK names and JSON shapes described in the surrounding code.

## Transcription (ocr_engine.py)

### TRANSCRIBE_PROMPT
```
TASK: transcribe_page
This is one scanned page of a student's handwritten exam answer script. Transcribe EXACTLY what is written, in reading order.
- Be verbatim: do NOT correct spelling, grammar or mathematics, and do NOT solve or complete anything.
- Write mathematics in LaTeX between $ and $.
- Write [unclear] for any word or symbol you cannot read.
- Write crossed-out text as [struck: ...].
- If the student drew a graph, plot, diagram or geometric figure, insert a line [Graph Description: ...] that describes it (axes and labels, endpoints, slopes, intercepts, shapes, trends and any labelled values).
- Keep question numbers and labels exactly as written (for example Q1, 2(a), Ans 3).
- If the page is blank, reply with nothing.
Output ONLY the transcription, with no commentary.
```

### BATCH_TRANSCRIBE_PROMPT
```
TASK: transcribe_pages
These are {n} consecutive scanned pages of ONE student's handwritten exam answer script, attached in order. Transcribe EXACTLY what is written on each page, in reading order.
- Start each page with a line containing only === PAGE k === (k = 1 for the first attached image, 2 for the second, and so on). Output exactly {n} such lines, in order, even for a blank page (write nothing after it).
- Be verbatim: do NOT correct spelling, grammar or mathematics, and do NOT solve or complete anything.
- Write mathematics in LaTeX between $ and $.
- Write [unclear] for any word or symbol you cannot read.
- Write crossed-out text as [struck: ...].
- If the student drew a graph, plot, diagram or geometric figure, insert a line [Graph Description: ...] that describes it (axes and labels, endpoints, slopes, intercepts, shapes, trends and any labelled values).
- Keep question numbers and labels exactly as written (for example Q1, 2(a), Ans 3).
Output ONLY the page markers and the transcription, with no commentary.
```

### ID_PROMPT
```
TASK: read_student_id
Below is the transcription of the first page of a student's exam answer script. Find the student's name and roll number / registration number, usually written at the top.
Return ONLY JSON: {"name": "<name or empty string>", "roll_no": "<roll number or empty string>", "confidence": <number between 0 and 1>}
Copy the text exactly as written. If a field is not present or unreadable, use an empty string.

TRANSCRIPTION:
<<<
```

## Parsing (parser.py)

### parse_question_paper
```
TASK: parse_question_paper
Below is the full text of an exam question paper. Extract every question the students must answer.
Return ONLY a JSON object of this form:
{"title": "<exam title or empty string>", "questions": [{"id": "<question number as printed, e.g. 1, 2a, 3(b)>", "text": "<full question text>", "max_marks": <number or null>}]}
Rules:
- Keep the paper's own numbering for ids. Ids must be unique.
- If sub-parts carry their own marks, list each sub-part as a separate question (for example 2a, 2b) and repeat the shared stem in its text. If only the whole question has marks, keep it as one question.
- max_marks must be a number; use null when no marks are stated.
- Copy question text faithfully. Do not answer the questions.
- Ignore general instructions, headers and footers that are not questions.

QUESTION PAPER:
<<<
{text}
>>>
```

### parse_rubric
```
TASK: parse_rubric
Below are the exam questions and the teacher's marking scheme. Structure the marking scheme for every question.
Return ONLY JSON: {"rubric": [{"question_id": "<id>", "model_answer": "<model answer or empty string>", "points": [{"description": "<what earns the marks>", "marks": <number>}], "notes": "<text or empty string>"}]}
Rules:
- Use exactly the question ids listed below.
- Each point is one gradable item that earns marks; the marks of a question's points should add up to its maximum marks.
- Copy the scheme's wording faithfully. Do NOT invent points the scheme does not contain.
- Put partial-credit rules and accepted alternative methods in "notes".
- If the scheme says nothing about a question, return it with an empty points list.

QUESTIONS:
{questions_listing}

MARKING SCHEME:
<<<
{rubric_text}
>>>
```

### draft_rubric (first attempt)
```
TASK: draft_rubric
Solve each exam question and create a concise marking rubric.
Return one JSON object only, with this exact structure: {"rubric": [{"question_id": "<id>", "model_answer": "<model answer or empty string>", "points": [{"description": "<what earns the marks>", "marks": <number>}], "notes": "<text or empty string>"}]}
Include every question_id exactly as listed. For each question, provide a model_answer, a points array of gradable steps, and notes. The point marks must add up to the question's maximum marks. Do not use Markdown fences or commentary.

QUESTIONS:
{questions}
```

### draft_rubric (retry / simpler)
```
Return ONLY a valid JSON object. No prose, no Markdown, no code fences.
The top-level key must be rubric. Its value must be an array with one object per question.
Each object must have question_id (string), model_answer (string), points (array of objects with description (string) and marks (number)), and notes (string).
Use these question IDs and maximum marks exactly; keep model answers and marking points concise.
{questions}
JSON schema example: {"rubric": [{"question_id": "<id>", "model_answer": "<model answer or empty string>", "points": [{"description": "<what earns the marks>", "marks": <number>}], "notes": "<text or empty string>"}]}
```

### segment_answers
```
TASK: segment_answers
Decide which lines are the student's answer to each question.
Return ONLY JSON: {"answers": [{"question_id": "<id>", "ranges": [[<first line number>, <last line number>]], "confidence": <number between 0 and 1>}]}
Use the question ids exactly as listed and the numbers after L as line numbers.

QUESTIONS:
{listing}

TRANSCRIPTION:
{transcription}
```

### segment_answers_retry
```
TASK: segment_answers_retry
Return one JSON object and nothing else. The object MUST have this exact shape: {"answers": [{"question_id": "EXACT_ID_FROM_LIST", "ranges": [[1, 3]], "confidence": 0.8}]}
Each range is [first_line_number, last_line_number] using the L numbers below. Use an empty ranges list when no answer for that question is present. Do not add markdown, explanations, or extra keys. Include every question exactly once.

QUESTIONS:
{listing}

TRANSCRIPTION:
{transcription}
```

## Grading (evaluator.py)

### evaluate_answer (single)
Built by `build_grading_prompt`. Schema depends on whether the rubric has itemised points.

Core instruction:
```
TASK: evaluate_answer
You are an experienced, fair examiner marking one answer from a student's handwritten exam script. The answer was transcribed by software, so it may contain reading errors and [unclear] markers. Never penalise a student for a transcription artefact; if the transcription is too unreliable to judge, say so and set needs_review to true.

QUESTION ({max_marks} marks):
{question.text}

MARKING SCHEME:
{rubric_text}

STUDENT'S ANSWER (verbatim transcription):
<<<
{answer_text}
>>>

Mark against the marking scheme, not by keywords. Consider conceptual correctness, key points covered, mathematical reasoning and method, relevance to the question, completeness and correctness of the final answer. Award partial credit for correct method or partly correct points. Accept valid alternative methods and equivalent forms.
[optional: page images note]
[itemised or holistic instruction]
Return ONLY JSON: {schema}
```

### evaluate_answers_batch
Built by `build_batch_grading_prompt`. Returns a list under "results".

Core instruction:
```
TASK: evaluate_answers_batch
You are an experienced, fair examiner marking several answers from ONE student's handwritten exam script. ...
Mark each question independently, against ITS OWN marking scheme, not by keywords. ...
Return ONLY JSON: {"results": [{"question_id": "...", "points": [...], "marks": ..., "rationale": "...", "feedback": "...", "needs_review": ..., "review_reason": "..."}, ...]}
```

# Answer Script Evaluator

KivyMD desktop app for AI-assisted marking of handwritten exam scripts (Gemini vision + local PaddleOCR for question papers).

## New features in this version

1. **Subject / scores table**  
   On the Upload screen enter a **Subject** name. When evaluation finishes, totals are written to a SQLite table named after that subject (sanitised). Only one subject per database.

2. **Student Scores screen**  
   Start → *Student scores table*, or Results → *View scores table*.  
   - MDDataTable with checkboxes  
   - Delete selected rows  
   - Edit a single student's total mark

3. **Results navigation**  
   After export / review, the Results screen has:
   - *New evaluation (Upload)* — back to document upload  
   - *View scores table*  
   - *Exit*

4. **Prompts**  
   All model prompts are documented in `prompts.md`. Edit that file to change wording; some prompts are still embedded in code as fallbacks for dynamic construction (grading schema, etc.).

## Run

```bash
cd answer_evaluator
# install deps: kivy, kivymd, paddleocr, opencv-python-headless, PyMuPDF, numpy, ...
PYTHONPATH=. python main.py
```

Settings / API key: set `GEMINI_API_KEY` in a `.env` file next to `main.py`, or use the Settings screen.

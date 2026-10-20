"""Load model prompts from prompts.md so they stay outside the Python source."""
import os
import re

_PROMPTS = None
_PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts.md")


def _load():
    global _PROMPTS
    if _PROMPTS is not None:
        return _PROMPTS
    _PROMPTS = {}
    if not os.path.isfile(_PROMPTS_PATH):
        return _PROMPTS
    with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    # Sections are ### NAME then ``` ... ```
    for m in re.finditer(r"^###\s+(\S+)\s*\n```\n(.*?)\n```", text, re.M | re.S):
        _PROMPTS[m.group(1).strip()] = m.group(2).strip()
    return _PROMPTS


def get(name, default=""):
    """Return the prompt text for `name`, or `default` if missing."""
    return _load().get(name, default)


def format(name, **kwargs):
    """get(name).format(**kwargs). Raises KeyError if the prompt is missing."""
    tmpl = get(name)
    if not tmpl:
        raise KeyError(f"Prompt '{name}' not found in prompts.md")
    return tmpl.format(**kwargs)


"""The prompts this family sends: the bank judge shared by the basic families, plus its own
`FORBIDDEN_TEMPLATE`."""

from wsbench.basic.prompts import FORBIDDEN_TEMPLATE, PROMPT_VERSION, SCHEMA, SYSTEM, render_user
from wsbench.basic.prompts import PROMPTS as SHARED

PROMPTS: dict[str, str] = {**SHARED, "FORBIDDEN_TEMPLATE": FORBIDDEN_TEMPLATE}

__all__ = ["PROMPTS", "PROMPT_VERSION", "SCHEMA", "SYSTEM", "render_user"]

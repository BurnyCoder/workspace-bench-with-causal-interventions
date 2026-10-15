"""The prompts this family sends: the bank judge shared by the basic families, plus its own
`FAMILY_RULE_POETRY`."""

from wsbench.basic.prompts import FAMILY_RULE_POETRY, PROMPT_VERSION, SCHEMA, SYSTEM, render_user
from wsbench.basic.prompts import PROMPTS as SHARED

PROMPTS: dict[str, str] = {**SHARED, "FAMILY_RULE_POETRY": FAMILY_RULE_POETRY}

__all__ = ["PROMPTS", "PROMPT_VERSION", "SCHEMA", "SYSTEM", "render_user"]

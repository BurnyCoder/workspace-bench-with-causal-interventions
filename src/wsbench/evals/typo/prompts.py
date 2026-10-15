"""The prompts this family sends: the bank judge shared by the basic families, plus its own
`FAMILY_RULE_TYPO`."""

from wsbench.basic.prompts import FAMILY_RULE_TYPO, PROMPT_VERSION, SCHEMA, SYSTEM, render_user
from wsbench.basic.prompts import PROMPTS as SHARED

PROMPTS: dict[str, str] = {**SHARED, "FAMILY_RULE_TYPO": FAMILY_RULE_TYPO}

__all__ = ["PROMPTS", "PROMPT_VERSION", "SCHEMA", "SYSTEM", "render_user"]

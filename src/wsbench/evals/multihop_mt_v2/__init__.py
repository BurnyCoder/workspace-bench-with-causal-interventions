"""Multihop v2 (multi-token): the shortcut-gated successor of multihop_mt (plan 0013)."""

from wsbench.multitoken.family import mt_family
from wsbench.registry import register

SPEC = register(mt_family("multihop_mt_v2", "Multihop v2 (multi-token)", calls_per_arm="≈ 0.8k"))

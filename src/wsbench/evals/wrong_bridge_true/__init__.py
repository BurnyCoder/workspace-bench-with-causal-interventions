"""Wrong-bridge multihop contrast: the TRUE intermediate the model failed to use."""

from wsbench.basic.family import bank_family
from wsbench.registry import register

SPEC = register(bank_family("wrong_bridge_true", "Wrong-bridge multihop (true-bridge contrast)"))

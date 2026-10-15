"""Wrong-bridge multihop: the model's WRONG intermediate, scored on the shared bank judge."""

from wsbench.basic.family import bank_family
from wsbench.registry import register

SPEC = register(bank_family("wrong_bridge", "Wrong-bridge multihop"))

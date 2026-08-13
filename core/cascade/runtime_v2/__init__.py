"""cascade_runtime_v2 — runtime CASCADE com as correcoes da auditoria:
kernel C fundido no F0, gate calibrado, F1 lowrank AVX2 corrigido,
residencia real = bytes empacotados (4.5 bpw)."""
from .confidence_gate import GateCalibrator, GateConfig, decide_gate
from .q4k_linear import Q4KLinearModule, Q8RLinearModule, patch_block_linears
from .q4k_pack import pack_q4k, unpack_q4k

__all__ = [
    "GateCalibrator", "GateConfig", "decide_gate",
    "Q4KLinearModule", "Q8RLinearModule", "patch_block_linears",
    "pack_q4k", "unpack_q4k",
]

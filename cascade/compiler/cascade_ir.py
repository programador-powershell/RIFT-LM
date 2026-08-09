"""CASCADE-IR M0 — representação intermediária mínima."""
from __future__ import annotations

from typing import Any

IR_VERSION = 3

OPCODES = (
    "CASCADE_OP_EMBEDDING",
    "CASCADE_OP_LINEAR",
    "CASCADE_OP_RMSNORM",
    "CASCADE_OP_ROPE",
    "CASCADE_OP_ATTENTION",
    "CASCADE_OP_ACTIVATION",
    "CASCADE_OP_ADD",
    "CASCADE_OP_OUTPUT",
    "CASCADE_OP_CUSTOM",
)


def make_linear_ir(
    *,
    model_id: str,
    architecture_hint: str,
    target_layer: str,
    cascade_ref: int = 0,
) -> dict[str, Any]:
    return {
        "ir_version": IR_VERSION,
        "model_id": model_id,
        "architecture_hint": architecture_hint,
        "tensors": [
            {"id": 0, "name": "input", "dtype": "f32", "semantic_role": "activation"},
            {"id": 1, "name": "output", "dtype": "f32", "semantic_role": "activation"},
            {"id": 2, "name": target_layer, "dtype": "mixed", "semantic_role": "weight", "source_name": target_layer},
        ],
        "operations": [
            {
                "id": 0,
                "opcode": "CASCADE_OP_LINEAR",
                "inputs": [0],
                "outputs": [1],
                "weights": [2],
                "cascade_ref": cascade_ref,
                "attrs": {"target_layer": target_layer},
            }
        ],
        "input_ids_tensor": 0,
        "output_tensor": 1,
    }


def validate_cascade_ir(ir: dict[str, Any]) -> None:
    if int(ir.get("ir_version", -1)) != IR_VERSION:
        raise ValueError(f"ir_version inválido: {ir.get('ir_version')}")
    ops = ir.get("operations")
    if not isinstance(ops, list) or not ops:
        raise ValueError("operations[] vazio")
    for op in ops:
        if op.get("opcode") not in OPCODES:
            raise ValueError(f"opcode desconhecido: {op.get('opcode')}")
        if "cascade_ref" not in op:
            raise ValueError("operation sem cascade_ref")

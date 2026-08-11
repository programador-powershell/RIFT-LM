"""Limpeza de workspace Colab após baterias CASCADE."""
from __future__ import annotations

import gc
import glob as _glob
import os
import shutil
from pathlib import Path
from typing import List


def _colab_available() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _removal_allowed(path: Path, *, allow_all: bool) -> bool:
    """Remoção destrutiva permitida apenas quando:
    - google.colab é importável OU RIFT_ALLOW_LOCAL_CLEANUP=1 (allow_all), OU
    - o caminho está sob /content ou /tmp.
    Nunca remove diretórios relativos ao cwd em máquina local fora do Colab.
    """
    if allow_all:
        return True
    s = path.as_posix()
    for root in ("/content", "/tmp"):
        if s == root or s.startswith(root + "/"):
            return True
    return False


def cleanup_colab_workspace(*, label: str = "CASCADE", wipe_hf_cache: bool = False) -> None:
    """Libera artefatos temporários no Colab.

    wipe_hf_cache=False: mantém cache HF (entre C0/C1/C2 do mesmo modelo).
    wipe_hf_cache=True: limpa hub/transformers no final da série ou do modelo.

    Fora do Colab as remoções destrutivas viram no-op (os diretórios de saída
    locais sobrevivem), a menos que RIFT_ALLOW_LOCAL_CLEANUP=1.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    allow_all = _colab_available() or os.environ.get("RIFT_ALLOW_LOCAL_CLEANUP", "").strip() == "1"

    removed: List[str] = []
    skipped: List[str] = []
    if wipe_hf_cache:
        home = Path.home()
        targets = [
            home / ".cache" / "huggingface" / "hub",
            home / ".cache" / "huggingface" / "transformers",
            home / ".cache" / "huggingface" / "modules",
            home / ".cache" / "torch",
            Path("/content") / ".cache",
            Path("/root") / ".cache" / "huggingface" / "hub",
            Path("/root") / ".cache" / "huggingface" / "transformers",
        ]
        for path in targets:
            try:
                if not (path.is_dir() or path.is_file()):
                    continue
                if not _removal_allowed(path, allow_all=allow_all):
                    skipped.append(str(path))
                    continue
                if path.is_dir():
                    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    shutil.rmtree(path, ignore_errors=True)
                    removed.append(f"{path} (~{size / (1024**3):.2f} GiB)")
                elif path.is_file():
                    path.unlink(missing_ok=True)
                    removed.append(str(path))
            except Exception as exc:
                print(f"[cleanup] AVISO {path}: {exc}")

    patterns = [
        "/tmp/cascade_*",
        "/tmp/cascade_c0_*",
        "/tmp/cascade_c1_*",
        "/tmp/cascade_c2_*",
        "/content/cascade_c0_test_output",
        "/content/cascade_c1_test_output",
        "/content/cascade_c2_test_output",
        "/content/cascade_run/cascade_c0_test_output",
        "/content/cascade_run/cascade_c1_test_output",
        "/content/cascade_run/cascade_c2_test_output",
        "/content/*_launcher.py",
        "/content/cascade_launcher.py",
        "/content/rift_launcher.py",
        "/content/winner_launcher.py",
        "/content/aether_launcher.py",
        "/content/spectra_launcher.py",
        "/content/__pycache__",
        "/content/cascade_run/__pycache__",
        "/content/cascade_run/cascade/**/__pycache__",
    ]
    for pat in patterns:
        for match in _glob.glob(pat, recursive=True):
            p = Path(match)
            if not _removal_allowed(p, allow_all=allow_all):
                skipped.append(str(p))
                continue
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    removed.append(str(p))
                elif p.is_file():
                    p.unlink(missing_ok=True)
                    removed.append(str(p))
            except Exception:
                pass

    # output dirs relative — só removidos no Colab/opt-in (nunca no cwd local)
    for rel in ("cascade_c0_test_output", "cascade_c1_test_output", "cascade_c2_test_output"):
        for base in (Path("."), Path("/content"), Path("/content/cascade_run")):
            p = base / rel
            if not p.is_dir():
                continue
            if not _removal_allowed(p, allow_all=allow_all):
                skipped.append(str(p))
                continue
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))

    gc.collect()
    if skipped:
        # dedupe preserve order
        seen_s = set()
        uniq_s = []
        for x in skipped:
            if x not in seen_s:
                seen_s.add(x)
                uniq_s.append(x)
        print(
            f"[cleanup] {label}: {len(uniq_s)} caminho(s) preservado(s) fora do Colab "
            f"(defina RIFT_ALLOW_LOCAL_CLEANUP=1 para forçar remoção local)"
        )
    if removed:
        # dedupe preserve order
        seen = set()
        uniq = []
        for x in removed:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        print(f"[cleanup] {label}: espaço liberado ({len(uniq)} item(ns)):")
        for x in uniq[:20]:
            print(f"  - {x}")
        if len(uniq) > 20:
            print(f"  ... +{len(uniq)-20} outros")
    else:
        print(f"[cleanup] {label}: nada a remover")

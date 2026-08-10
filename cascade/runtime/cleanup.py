"""Limpeza de workspace Colab após baterias CASCADE."""
from __future__ import annotations

import gc
import glob as _glob
import shutil
from pathlib import Path
from typing import List


def cleanup_colab_workspace(*, label: str = "CASCADE", wipe_hf_cache: bool = False) -> None:
    """Libera artefatos temporários no Colab.

    wipe_hf_cache=False: mantém cache HF (entre C0/C1/C2 do mesmo modelo).
    wipe_hf_cache=True: limpa hub/transformers no final da série ou do modelo.
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

    removed: List[str] = []
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
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    removed.append(str(p))
                elif p.is_file():
                    p.unlink(missing_ok=True)
                    removed.append(str(p))
            except Exception:
                pass

    # output dirs relative
    for rel in ("cascade_c0_test_output", "cascade_c1_test_output", "cascade_c2_test_output"):
        for base in (Path("."), Path("/content"), Path("/content/cascade_run")):
            p = base / rel
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed.append(str(p))

    gc.collect()
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

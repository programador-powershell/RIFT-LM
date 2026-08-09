#!/usr/bin/env python3
"""WINNER.cpp v0.8 native self-test + real-tensor reference battery for Colab.

The native C++ suite validates the reviewed runtime. The primary dashboard
battery separately measures F0 + low-rank residual on a real Hugging Face
Linear tensor with PyTorch. The two latency scopes are never conflated.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


torch = None
F = None
AutoModel = None
AutoModelForCausalLM = None
AutoTokenizer = None

REPOSITORY_ARCHIVE_TEMPLATE = "https://github.com/programador-powershell/RIFT-LM/archive/{ref}.tar.gz"
MAX_ARCHIVE_BYTES = 25 * 1024 * 1024


class ResultsPublishError(RuntimeError):
    pass


def repository_archive() -> str:
    ref = os.environ.get("RIFT_SOURCE_REF", "main").strip()
    if ref != "main" and not re.fullmatch(r"[a-fA-F0-9]{40}", ref):
        raise RuntimeError("RIFT_SOURCE_REF inválido")
    return REPOSITORY_ARCHIVE_TEMPLATE.format(ref=ref)


def ensure_ml_dependencies() -> None:
    global torch, F, AutoModel, AutoModelForCausalLM, AutoTokenizer
    if torch is not None:
        return
    try:
        import torch as _torch
        import torch.nn.functional as _F
        from transformers import AutoModel as _AutoModel
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "PyTorch e Transformers são necessários. Instale com: "
            f"pip install torch transformers\nErro: {exc}"
        ) from exc
    torch = _torch
    F = _F
    AutoModel = _AutoModel
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer


def normalize_huggingface_model_id(value: str) -> str:
    raw = str(value or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme:
        if parsed.scheme != "https" or parsed.netloc.lower() not in {
            "huggingface.co", "www.huggingface.co",
        }:
            raise ValueError("Use um model ID org/modelo ou URL HTTPS do huggingface.co")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("URL do Hugging Face precisa conter org/modelo")
        raw = f"{parts[0]}/{parts[1]}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", raw):
        raise ValueError("Model ID inválido; use org/modelo")
    return raw


def resolve_linear_weight_name(model: Any, requested: str) -> str:
    state = model.state_dict()
    if requested != "auto":
        name = requested if requested.endswith(".weight") else f"{requested}.weight"
        tensor = state.get(name)
        if tensor is None or getattr(tensor, "ndim", 0) != 2:
            raise ValueError(f"Tensor Linear 2D não encontrado: {name}")
        return name
    preferred = (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "transformer.h.0.attn.c_attn.weight",
    )
    for name in preferred:
        if name in state and state[name].ndim == 2:
            return name
    for name, tensor in state.items():
        if name.endswith(".weight") and getattr(tensor, "ndim", 0) == 2:
            if min(tensor.shape) >= 32:
                return name
    raise ValueError("Nenhum tensor Linear 2D compatível foi encontrado")


def capture_activation(model: Any, tokenizer: Any, module_name: str, device: Any, prompt: str):
    modules = dict(model.named_modules())
    module = modules.get(module_name)
    if module is None:
        raise ValueError(f"Módulo não encontrado para capturar ativação: {module_name}")
    captured: list[Any] = []

    def hook(_module, inputs):
        if inputs and hasattr(inputs[0], "detach"):
            captured.append(inputs[0].detach())

    handle = module.register_forward_pre_hook(hook)
    try:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            model(**encoded)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("O forward não produziu ativação na camada selecionada")
    x = captured[0].reshape(-1, captured[0].shape[-1]).to(device=device, dtype=torch.float32)
    return x[-min(32, x.shape[0]):]


def load_model(model_id: str, *, device: Any, trust_remote_code: bool):
    token = os.environ.get("HF_TOKEN") or None
    common = {"trust_remote_code": trust_remote_code, "token": token}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    load_kwargs = {**common, "low_cpu_mem_usage": True, "dtype": torch.float16 if device.type == "cuda" else torch.float32}
    errors = []
    for cls in (AutoModelForCausalLM, AutoModel):
        try:
            model = cls.from_pretrained(model_id, **load_kwargs).to(device).eval()
            return tokenizer, model
        except Exception as exc:  # preserve the useful combined loader context
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError("Não foi possível carregar o modelo:\n" + "\n".join(errors))


def compute_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    ref = reference.float().reshape(-1)
    pred = candidate.float().reshape(-1)
    cosine = float(F.cosine_similarity(ref, pred, dim=0).item())
    error = torch.linalg.vector_norm(ref - pred)
    denominator = torch.linalg.vector_norm(ref).clamp_min(1e-12)
    return {"cosine": cosine, "nrmse": float((error / denominator).item())}


def benchmark_ms(fn, *, device: Any, iterations: int) -> dict[str, float]:
    for _ in range(3):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    values = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append((time.perf_counter_ns() - start) / 1_000_000)
    return {"median_ms": statistics.median(values), "min_ms": min(values), "iterations": iterations}


def fit_winner(weight: Any, maximum_rank: int):
    row_scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = weight / row_scale
    best_error = None
    best_threshold = None
    best_ternary = None
    for step in range(1, 19):
        threshold = step * 0.05
        ternary_codes = torch.where(
            normalized > threshold, torch.ones_like(weight),
            torch.where(normalized < -threshold, -torch.ones_like(weight), torch.zeros_like(weight)),
        )
        reconstructed = ternary_codes * row_scale
        error = float(torch.linalg.vector_norm(weight - reconstructed).item())
        if best_error is None or error < best_error:
            best_error, best_threshold, best_ternary = error, threshold, reconstructed
    residual = weight - best_ternary
    rank = min(maximum_rank, min(weight.shape))
    u, s, v = torch.pca_lowrank(residual, q=rank, center=False, niter=4)
    return best_ternary, u, s, v, float(best_threshold)


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError("Archive WINNER contém caminho fora do diretório de extração")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError("Archive WINNER contém link ou dispositivo não permitido")
        archive.extractall(destination, members=members, filter="data")


def download_with_limit(url: str, path: Path, maximum: int) -> None:
    request = Request(url, headers={"User-Agent": "winner-colab-benchmark/0.8"})
    with urlopen(request, timeout=90) as response, path.open("wb") as output:
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > maximum:
            raise RuntimeError("Archive WINNER excede o limite permitido")
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RuntimeError("Archive WINNER excede o limite permitido")
            output.write(chunk)


def compile_and_test_winner(work_dir: Path, *, profile_dim: int) -> dict[str, Any]:
    if not shutil.which("cmake") or not shutil.which("g++"):
        raise RuntimeError("cmake e g++ são necessários para validar WINNER.cpp")
    archive_path = work_dir / "rift-lm.tar.gz"
    source_parent = work_dir / "source"
    source_parent.mkdir(parents=True, exist_ok=True)
    archive_url = repository_archive()
    download_with_limit(archive_url, archive_path, MAX_ARCHIVE_BYTES)
    _safe_extract_tar(archive_path, source_parent)
    roots = list(source_parent.glob("RIFT-LM-*/winner_cpp/CMakeLists.txt"))
    if len(roots) != 1:
        raise RuntimeError("winner_cpp/CMakeLists.txt não encontrado no archive publicado")
    source_dir = roots[0].parent
    build_dir = work_dir / "build"
    started = time.perf_counter()
    subprocess.run([
        "cmake", "-S", str(source_dir), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release", "-DWINNER_NATIVE=ON", "-DBUILD_TESTING=ON",
    ], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel", "2"], check=True)
    binary = build_dir / "winner"
    self_test = subprocess.run([str(binary), "--self-test"], check=True, text=True, capture_output=True)
    profile_path = work_dir / "winner_native_profile.json"
    subprocess.run([
        str(binary), "--bench-kernels", "--dim", str(profile_dim),
        "--layers", "2", "--tokens", "4", "--output", str(profile_path),
    ], check=True)
    return {
        "source": archive_url,
        "build_seconds": time.perf_counter() - started,
        "self_test_pass": "[SELF-TEST] PASS" in self_test.stdout,
        "native_profile": json.loads(profile_path.read_text(encoding="utf-8")),
    }


class BatteryRecorder:
    def __init__(self, out_dir: Path, *, model_id: str):
        self.out_dir = out_dir
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.records: list[dict[str, Any]] = []
        self.json_path = out_dir / "winner_test_batteries.json"
        self.csv_path = out_dir / "winner_test_batteries.csv"

    def record(self, *, battery_id: str, status: str, measurement_scope: str,
               quality: dict[str, Any], metrics: dict[str, Any], notes: str,
               comparison_role: str | None = None, baseline_ram_bytes: int | None = None,
               candidate_ram_bytes: int | None = None, baseline_disk_bytes: int | None = None,
               candidate_disk_bytes: int | None = None) -> None:
        item = {
            "schema_version": 1, "timestamp_utc": self.timestamp, "run_id": self.run_id,
            "technology": "WINNER", "model_id": self.model_id, "battery_id": battery_id,
            "status": status, "measurement_scope": measurement_scope,
            "quality": quality, "metrics": metrics, "notes": notes,
            "baseline_ram_bytes": baseline_ram_bytes, "candidate_ram_bytes": candidate_ram_bytes,
            "baseline_disk_bytes": baseline_disk_bytes, "candidate_disk_bytes": candidate_disk_bytes,
            "comparison_role": comparison_role,
        }
        self.records = [record for record in self.records if record["battery_id"] != battery_id]
        self.records.append(item)
        self.records.sort(key=lambda record: record["battery_id"])
        self.json_path.write_text(json.dumps(self.records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        fields = ["timestamp_utc", "run_id", "technology", "model_id", "battery_id", "status",
                  "comparison_role", "baseline_ram_bytes", "candidate_ram_bytes",
                  "baseline_disk_bytes", "candidate_disk_bytes", "measurement_scope", "notes"]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: record.get(key) for key in fields})
        single = self.batteries_dir / f"{self.run_id}__{battery_id}.json"
        single.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[BATTERY] {battery_id}: gravada automaticamente -> {single}")


def read_setting(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def publish_to_vercel(path: Path, *, mode: str, endpoint: str | None) -> None:
    if mode == "off":
        print("[PUBLISH] Publicação remota desativada (--publish off).")
        return
    target = endpoint or read_setting("RIFT_RESULTS_ENDPOINT")
    token = read_setting("RIFT_INGEST_TOKEN")
    missing = []
    if not target:
        missing.append("RIFT_RESULTS_ENDPOINT")
    if not token:
        missing.append("RIFT_INGEST_TOKEN")
    if missing:
        message = "Configure " + " e ".join(missing)
        if mode == "required":
            raise ResultsPublishError(message)
        print(f"[PUBLISH] AVISO: {message}; resultados preservados localmente.")
        return
    if urlparse(target).scheme != "https" or len(token) < 32:
        raise ResultsPublishError("Endpoint precisa ser HTTPS e token deve ter ao menos 32 caracteres")
    records = json.loads(path.read_text(encoding="utf-8"))
    request = Request(
        target, data=json.dumps({"records": records}).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "User-Agent": "winner-colab-publisher/0.8"},
    )
    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ResultsPublishError(f"Vercel respondeu HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise ResultsPublishError(f"Falha de rede ao publicar: {exc.reason}") from exc
    publication = result.get("publication", {})
    print(f"[PUBLISH] {len(records)} registro(s) aceito(s); snapshot publicado com {publication.get('records', '?')} registro(s).")
    if publication.get("commit_url"):
        print(f"[PUBLISH] Commit: {publication['commit_url']}")


def run_phase1(args: argparse.Namespace, native: dict[str, Any]) -> Path:
    ensure_ml_dependencies()
    model_id = normalize_huggingface_model_id(args.model)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível; use --device cpu ou ative GPU no Colab")
    device = torch.device(args.device)
    print(f"[Phase1] Carregando {model_id} em {device}...")
    tokenizer, model = load_model(model_id, device=device, trust_remote_code=args.trust_remote_code)
    target_layer = resolve_linear_weight_name(model, args.target_layer)
    weight = model.state_dict()[target_layer].detach().to(device=device, dtype=torch.float32)
    try:
        x = capture_activation(model, tokenizer, target_layer.removesuffix(".weight"), device, args.prompt)
        activation_source = "real_model_activation"
    except Exception as exc:
        print(f"[Phase1] AVISO: ativação real indisponível ({exc}); usando fallback determinístico.")
        generator = torch.Generator(device=device).manual_seed(42)
        x = torch.randn((8, weight.shape[1]), generator=generator, device=device, dtype=torch.float32)
        activation_source = "synthetic_fallback"

    ternary, u, s, v, threshold = fit_winner(weight, args.maximum_rank)
    with torch.inference_mode():
        y_reference = F.linear(x, weight)
        y_base = F.linear(x, ternary)
        y_candidate = y_base + ((x @ v) * s) @ u.T
        reconstructed = ternary + (u * s) @ v.T
    q_weight_base = compute_metrics(weight, ternary)
    q_weight = compute_metrics(weight, reconstructed)
    q_base = compute_metrics(y_reference, y_base)
    q_candidate = compute_metrics(y_reference, y_candidate)
    quality_pass = q_candidate["cosine"] >= 0.98 and q_candidate["nrmse"] <= 0.10
    baseline_perf = benchmark_ms(lambda: F.linear(x, weight), device=device, iterations=args.iterations)
    candidate_perf = benchmark_ms(
        lambda: F.linear(x, ternary) + ((x @ v) * s) @ u.T,
        device=device, iterations=args.iterations,
    )
    base_perf = benchmark_ms(lambda: F.linear(x, ternary), device=device, iterations=args.iterations)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = BatteryRecorder(out_dir, model_id=model_id)
    numel = int(weight.numel())
    baseline_disk = numel * 4
    base_disk = (numel + 3) // 4 + int(weight.shape[0]) * 4
    residual_disk = int((u.numel() + s.numel() + v.numel()) * 4)
    candidate_disk = base_disk + residual_disk
    io_bytes = int((x.numel() + y_reference.numel()) * 4)
    baseline_ram = baseline_disk + io_bytes
    base_ram = numel + int(weight.shape[0]) * 4 + io_bytes
    candidate_ram = base_ram + residual_disk
    operation = {
        "metric": "linear_latency", "baseline_median_ms": baseline_perf["median_ms"],
        "candidate_median_ms": candidate_perf["median_ms"],
        "speedup_x": baseline_perf["median_ms"] / candidate_perf["median_ms"],
        "rows_processed": int(x.shape[0]), "device": str(device),
    }
    recorder.record(
        battery_id="B0_WINNER_CPP_BUILD_SELF_TEST", status="PASS",
        measurement_scope="Compilação nativa C++17 e --self-test; perfil nativo sintético separado da bateria do modelo.",
        quality={"full_local_gate_pass": bool(native["self_test_pass"])}, metrics={"winner": native},
        notes="Valida limites Q4, residual, árvore especulativa e page table; não mede o LLM end-to-end.",
    )
    recorder.record(
        battery_id="P1_WINNER_F0_TERNARY_2BIT",
        status="EXPERIMENTAL_PASS" if q_base["cosine"] >= 0.90 else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram, candidate_ram_bytes=base_ram,
        baseline_disk_bytes=baseline_disk, candidate_disk_bytes=base_disk,
        measurement_scope="Uma operação Linear real; disco F0 em 2 bits; RAM do path de referência usa códigos int8.",
        quality={"full_local_gate_pass": None, "weight": q_weight_base, "output": q_base},
        metrics={"operation": {**operation, "candidate_median_ms": base_perf["median_ms"],
                               "speedup_x": baseline_perf["median_ms"] / base_perf["median_ms"]},
                 "winner": {"threshold": threshold, "residual_rank": 0, "cpp_self_test_pass": True}},
        notes="O armazenamento é empacotado; o kernel PyTorch desta bateria pré-decodifica F0 em FP32.",
    )
    recorder.record(
        battery_id="P1_WINNER_F0_PLUS_LS", status="PASS" if quality_pass else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram, candidate_ram_bytes=candidate_ram,
        baseline_disk_bytes=baseline_disk, candidate_disk_bytes=candidate_disk,
        measurement_scope="Uma operação Linear real; F0 + residual low-rank; latência PyTorch e self-test C++ reportados separadamente; Tok/s não medido.",
        quality={"full_local_gate_pass": quality_pass, "weight": q_weight, "output": q_candidate,
                 "output_base": q_base},
        metrics={"operation": operation, "winner": {
            "threshold": threshold, "residual_rank": int(s.numel()),
            "stage_activation_rate": 1.0, "activation_source": activation_source,
            "cpp_self_test_pass": bool(native["self_test_pass"]),
            "native_profile": native["native_profile"], "native_lowbit_model_kernel": False,
        }},
        notes="O C++ validado ainda não carrega diretamente este tensor do Hugging Face; nenhum ganho end-to-end ou kernel low-bit nativo é reivindicado.",
        comparison_role="primary",
    )
    report = {
        "technology": "WINNER", "model_id": model_id, "target_layer": target_layer,
        "shape": list(weight.shape), "quality": q_candidate, "performance": operation,
        "storage": {"baseline": baseline_disk, "candidate": candidate_disk}, "native": native,
    }
    (out_dir / "winner_phase1_gain_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    del model, tokenizer, weight, ternary, u, s, v, reconstructed
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("\n" + "=" * 78)
    print("WINNER.cpp PHASE 1 — GAIN TRACKER")
    print("=" * 78)
    print(f"Modelo                  : {model_id}")
    print(f"Tensor                  : {target_layer}")
    print(f"Qualidade F0+LS         : cosine={q_candidate['cosine']:.6f} / NRMSE={q_candidate['nrmse']:.6f}")
    print(f"Disco                   : {baseline_disk:,} -> {candidate_disk:,} bytes")
    print(f"Linear PyTorch          : {baseline_perf['median_ms']:.4f} -> {candidate_perf['median_ms']:.4f} ms")
    print("C++ self-test           : PASS (latência nativa sintética separada)")
    print(f"Baterias JSON           : {recorder.json_path}")
    print("=" * 78)
    return recorder.json_path


def without_ipykernel_connection_args(argv: Iterable[str]) -> list[str]:
    values = list(argv)
    filtered = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "-f" and index + 1 < len(values):
            name = Path(values[index + 1]).name
            if name.startswith("kernel-") and name.endswith(".json"):
                index += 2
                continue
        if value.startswith("-f="):
            name = Path(value[3:]).name
            if name.startswith("kernel-") and name.endswith(".json"):
                index += 1
                continue
        filtered.append(value)
        index += 1
    return filtered


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="WINNER.cpp v0.8 native + real-tensor battery")
    parser.add_argument("--mode", choices=["self-test", "phase1"], default="phase1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--target-layer", default="auto")
    parser.add_argument("--prompt", default="Explique por que memória importa na inferência de modelos.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--maximum-rank", type=int, default=16)
    parser.add_argument("--cpp-profile-dim", type=int, default=128)
    parser.add_argument("--out", default="winner_m0_test_output")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--publish", choices=["auto", "required", "off"], default=os.environ.get("RIFT_PUBLISH_MODE", "auto"))
    parser.add_argument("--results-endpoint", default=None)
    values = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(without_ipykernel_connection_args(values))
    if args.iterations < 1 or args.maximum_rank < 1 or not 16 <= args.cpp_profile_dim <= 1024:
        parser.error("iterations/rank devem ser positivos e cpp-profile-dim deve estar entre 16 e 1024")
    try:
        with tempfile.TemporaryDirectory(prefix="winner_cpp_") as temporary:
            native = compile_and_test_winner(Path(temporary), profile_dim=args.cpp_profile_dim)
        if not native["self_test_pass"]:
            raise RuntimeError("WINNER.cpp compilou, mas o self-test não confirmou PASS")
        if args.mode == "self-test":
            print("WINNER.cpp SELF-TEST PASS")
            return
        batteries_path = run_phase1(args, native)
        publish_to_vercel(batteries_path, mode=args.publish, endpoint=args.results_endpoint)
    except ResultsPublishError as exc:
        raise SystemExit(f"[PUBLISH] ERRO: {exc}") from exc


if __name__ == "__main__":
    main()

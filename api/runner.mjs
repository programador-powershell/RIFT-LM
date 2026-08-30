// GET /runner.py → /api/runner (contrato §14.3): entrega o orquestrador
// COMPLETO da fila de baterias em text/plain, gerado no servidor com
// origin/repo/ref já resolvidos (§14.1). Sem autenticação: o script não
// contém segredo algum — os Secrets são lidos no kernel do Colab pela célula
// curta (bootstrap) gerada pelos dashboards.
//
// Contrato de globals da célula chamadora (a célula curta define e depois
// executa exec(urlopen(BASE + "/runner.py").read().decode("utf-8"))):
//   MODELS = ["org/modelo", ...]      (fallback: env RIFT_QUEUE_MODELS, csv)
//   TECHS  = ["all"] ou lista          (fallback: env RIFT_QUEUE_TECHS, csv)
//   BASE   = "https://<deploy>"        (fallback: env RIFT_QUEUE_BASE → origin baked)
import { rawBaseUrl, resolveRef, resolveRepo } from "./_lib/repo.mjs";

// Mesma sugestão fixa de quant da rota /gguf/ (contrato §11).
const GGUF_DEFAULT_QUANT = "UD-Q2_K_XL";
// Expansão de TECHS=["all"]: fila serial completa por modelo.
// VLB é uma tecnologia transversal de re-quantização/runtime e recebe o mesmo
// model id escolhido pelo usuário. GGUF continua fora do all normal e entra só
// quando o model id parece um repositório GGUF.
const ALL_TECHNOLOGIES = [
  "rift", "cascade", "aether", "spectra", "winner",
  "geyser", "microlm", "c-series", "c3", "final", "cap", "vlb",
];
const KNOWN_TECHNOLOGIES = [...ALL_TECHNOLOGIES, "gguf"];
// MicroLM (§22): o passo "microlm" não substitui modelo na URL — a rota
// /microlm é fixa e a bateria avalia sempre este modelo de referência.
const MICROLM_MODEL_ID = "microlm/MicroLM-22M-v0.2";
// Deps de tokenização pinadas (§14.3 — "aceita todo tipo de tokenização").
const TOKENIZER_DEPENDENCIES = [
  "transformers>=4.52.0",
  "accelerate>=0.33.0",
  "tokenizers>=0.20.0",
  "sentencepiece>=0.2,<1",
  "tiktoken>=0.7,<1",
];
// Limpeza prévia: saídas conhecidas da fila. VLB mantém seus artefatos sob
// /content/vlb_run e seu launcher sob /content/vlb_launcher.
const PRECLEAN_DIRECTORIES = [
  "/content/rift_m0_test_output",
  "/content/cascade_m0_test_output",
  "/content/aether_m0_test_output",
  "/content/spectra_m0_test_output",
  "/content/winner_m0_test_output",
  "/content/geyser_m0_test_output",
  "/content/cap_test_output",
  "/content/cascade_run",
  "/content/c3_run",
  "/content/final_run",
  "/content/gguf_run",
  "/content/gguf_test_output",
  "/content/final_test_output",
  "/content/microlm_run",
  "/content/microlm_m0_test_output",
  "/content/vlb_run",
  "/content/vlb_launcher",
];
const PRECLEAN_PATTERNS = [
  "/content/*_launcher.py",
  "/tmp/winner_cpp_*",
  "/tmp/winner_phase1_*",
  "/tmp/phase1_load_fail*",
  "/tmp/cascade_*",
  "/tmp/aether_*",
  "/tmp/spectra_*",
  "/tmp/geyser_*",
  "/tmp/cap_*",
  "/tmp/rift_*",
  "/tmp/vlb_*",
];
// Série C do CASCADE (C0 → C1 → C2): scripts + pacote cascade/ baixados do
// raw.githubusercontent no ref PINADO do deploy (espelha o painel principal).
// Árvore canônica (contrato §20, regra 2): pares [caminho_no_repo,
// caminho_local] — o repositório guarda os scripts em engines/cascade/ e o
// pacote python em core/cascade/, mas o layout LOCAL no Colab não muda
// (scripts no diretório de execução, pacote em cascade/...).
const CSERIES_SCRIPTS = [
  ["engines/cascade/cascade_c0_phase1_auto_batteries.py", "cascade_c0_phase1_auto_batteries.py"],
  ["engines/cascade/cascade_c1_block_auto_batteries.py", "cascade_c1_block_auto_batteries.py"],
  ["engines/cascade/cascade_c2_e2e_auto_batteries.py", "cascade_c2_e2e_auto_batteries.py"],
];
const CSERIES_PACKAGE_FILES = [
  ["core/cascade/__init__.py", "cascade/__init__.py"],
  ["core/cascade/compiler/__init__.py", "cascade/compiler/__init__.py"],
  ["core/cascade/compiler/cascade_ir.py", "cascade/compiler/cascade_ir.py"],
  ["core/cascade/compiler/decompose.py", "cascade/compiler/decompose.py"],
  ["core/cascade/compiler/bundle_writer.py", "cascade/compiler/bundle_writer.py"],
  ["core/cascade/compiler/block_decompose.py", "cascade/compiler/block_decompose.py"],
  ["core/cascade/kernels/__init__.py", "cascade/kernels/__init__.py"],
  ["core/cascade/kernels/int4.py", "cascade/kernels/int4.py"],
  ["core/cascade/kernels/lowrank.py", "cascade/kernels/lowrank.py"],
  ["core/cascade/kernels/fused_stage.py", "cascade/kernels/fused_stage.py"],
  ["core/cascade/runtime/__init__.py", "cascade/runtime/__init__.py"],
  ["core/cascade/runtime/confidence_gate.py", "cascade/runtime/confidence_gate.py"],
  ["core/cascade/runtime/reference.py", "cascade/runtime/reference.py"],
  ["core/cascade/runtime/block_runtime.py", "cascade/runtime/block_runtime.py"],
  ["core/cascade/runtime/cleanup.py", "cascade/runtime/cleanup.py"],
  ["core/cascade/converter/__init__.py", "cascade/converter/__init__.py"],
  ["core/cascade/converter/cascade_converter.py", "cascade/converter/cascade_converter.py"],
];

export function buildRunnerScript({
  origin,
  repo = resolveRepo(),
  ref = resolveRef(),
}) {
  const rawBase = rawBaseUrl(repo, ref);
  return `# -*- coding: utf-8 -*-
# =============================================================================
# Observatório LLM — orquestrador da fila de baterias (contrato §14.3)
# Gerado no servidor por ${origin} com repo/ref/origin resolvidos (§14.1).
# Executado via exec() pela célula curta do Colab; lê MODELS/TECHS/BASE do
# escopo chamador (globals) com fallback nas envs RIFT_QUEUE_MODELS /
# RIFT_QUEUE_TECHS (listas separadas por vírgula) e RIFT_QUEUE_BASE.
# Cada passo roda em subprocesso ISOLADO: falha não interrompe a fila.
# =============================================================================
import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from urllib.request import Request, urlopen

RIFT_RUNNER_ORIGIN = ${JSON.stringify(origin)}
RIFT_RUNNER_REPO = ${JSON.stringify(repo)}
RIFT_RUNNER_REF = ${JSON.stringify(ref)}
RIFT_RUNNER_RAW_BASE = ${JSON.stringify(rawBase)}
RIFT_GGUF_DEFAULT_QUANT = ${JSON.stringify(GGUF_DEFAULT_QUANT)}
RIFT_MICROLM_MODEL_ID = ${JSON.stringify(MICROLM_MODEL_ID)}
RIFT_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
RIFT_ALL_TECHS = json.loads(r'''${JSON.stringify(ALL_TECHNOLOGIES)}''')
RIFT_KNOWN_TECHS = set(json.loads(r'''${JSON.stringify(KNOWN_TECHNOLOGIES)}'''))
RIFT_TOKENIZER_DEPS = json.loads(r'''${JSON.stringify(TOKENIZER_DEPENDENCIES)}''')
RIFT_PRECLEAN_DIRS = json.loads(r'''${JSON.stringify(PRECLEAN_DIRECTORIES)}''')
RIFT_PRECLEAN_PATTERNS = json.loads(r'''${JSON.stringify(PRECLEAN_PATTERNS)}''')
RIFT_CSERIES_SCRIPTS = json.loads(r'''${JSON.stringify(CSERIES_SCRIPTS)}''')
RIFT_CSERIES_PKG_FILES = json.loads(r'''${JSON.stringify(CSERIES_PACKAGE_FILES)}''')

def _rift_env_list(name):
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]

# MODELS/TECHS/BASE do escopo chamador (célula curta), com fallback nas envs.
_rift_caller = globals()
RIFT_QUEUE_MODELS = (
    [str(model).strip() for model in (_rift_caller.get("MODELS") or []) if str(model).strip()]
    or _rift_env_list("RIFT_QUEUE_MODELS")
)
RIFT_QUEUE_TECHS = (
    [str(tech).strip().lower() for tech in (_rift_caller.get("TECHS") or []) if str(tech).strip()]
    or [tech.lower() for tech in _rift_env_list("RIFT_QUEUE_TECHS")]
    or ["all"]
)
RIFT_QUEUE_BASE = str(
    _rift_caller.get("BASE") or os.environ.get("RIFT_QUEUE_BASE") or RIFT_RUNNER_ORIGIN
).strip().rstrip("/")

if not RIFT_QUEUE_MODELS:
    raise SystemExit('[FILA] Defina MODELS = ["org/modelo", ...] na célula (ou a env RIFT_QUEUE_MODELS)')
_rift_bad_models = [model for model in RIFT_QUEUE_MODELS if not RIFT_MODEL_ID_RE.match(model)]
if _rift_bad_models:
    raise SystemExit("[FILA] Modelos inválidos (use org/modelo): " + ", ".join(_rift_bad_models))
_rift_base_lower = RIFT_QUEUE_BASE.lower()
if not (
    _rift_base_lower.startswith("https://")
    or _rift_base_lower.startswith("http://127.0.0.1")
    or _rift_base_lower.startswith("http://localhost")
):
    raise SystemExit("[FILA] BASE precisa usar HTTPS (http apenas para localhost)")

def _rift_import_colab_secrets():
    try:
        from google.colab import userdata
    except Exception:
        return
    for name in ("RIFT_INGEST_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            continue
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = str(value)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

_rift_import_colab_secrets()
if len(os.environ.get("RIFT_INGEST_TOKEN", "").strip()) < 32:
    raise SystemExit("[FILA] Configure o Secret RIFT_INGEST_TOKEN (>= 32 caracteres) no Colab")

# Repo/ref resolvidos no servidor (§14.2): os scripts Python nunca adivinham
# o repositório — estas envs valem para TODOS os subprocessos da fila.
os.environ["RIFT_GITHUB_REPOSITORY"] = RIFT_RUNNER_REPO
os.environ["RIFT_SOURCE_REF"] = RIFT_RUNNER_REF
os.environ.setdefault("RIFT_RESULTS_ENDPOINT", RIFT_QUEUE_BASE + "/api/results")

def _rift_is_gguf_model(model):
    # Mesma heurística da rota /gguf/ dos dashboards: "gguf" no NOME do modelo.
    return bool(re.search("gguf", (model.split("/", 1) + [""])[1], re.IGNORECASE))

def _rift_quote_model(model):
    return "/".join(urllib.parse.quote(part, safe="") for part in model.split("/"))

def _rift_step_url(tech, model):
    # URLs amigáveis dos launchers já existentes (vercel.json).
    path = _rift_quote_model(model)
    if tech == "microlm":
        return RIFT_QUEUE_BASE + "/microlm"
    if tech == "c3":
        return RIFT_QUEUE_BASE + "/c3/all/" + path
    if tech == "final":
        return RIFT_QUEUE_BASE + "/final/all/" + path
    if tech == "gguf":
        return (
            RIFT_QUEUE_BASE + "/gguf/" + path
            + "?quant=" + urllib.parse.quote(RIFT_GGUF_DEFAULT_QUANT, safe="")
        )
    # VLB and the ordinary technologies all use /<technology>/<org>/<model>.
    return RIFT_QUEUE_BASE + "/" + tech + "/" + path

def _rift_expand_techs(model):
    if "all" in RIFT_QUEUE_TECHS:
        return ["gguf"] if _rift_is_gguf_model(model) else list(RIFT_ALL_TECHS)
    techs = []
    for tech in RIFT_QUEUE_TECHS:
        if tech not in RIFT_KNOWN_TECHS:
            print("[FILA] AVISO: tecnologia desconhecida ignorada:", tech)
            continue
        techs.append(tech)
    return techs

def _rift_install_tokenizer_deps():
    print("[FILA] Instalando deps de tokenização pinadas:", ", ".join(RIFT_TOKENIZER_DEPS))
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *RIFT_TOKENIZER_DEPS])
    except Exception as exc:
        print("[FILA] AVISO: pip falhou; as baterias instalam as próprias dependências:", exc)

def _rift_cleanup_workspace(level="light"):
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if not os.path.isdir("/content"):
        return
    removed = 0
    for directory in RIFT_PRECLEAN_DIRS:
        path = Path(directory)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
            print("[cleanup-" + level + "] removido", directory)
    for pattern in RIFT_PRECLEAN_PATTERNS:
        for match in glob.glob(pattern):
            path = Path(match)
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                removed += 1
            except Exception as exc:
                print("[cleanup-" + level + "] AVISO", match, ":", exc)
    if level == "full":
        for path in [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface" / "transformers",
            Path("/root/.cache/huggingface/hub"),
            Path("/content/.cache"),
        ]:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
                print("[cleanup-full] removido", str(path))
    print("[cleanup-" + level + "] pronto (" + str(removed) + " item(ns))")

def _rift_gpu_memory_used_mb():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        return sum(values) if values else None
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        return None

def _rift_wait_for_resource_release(baseline_mb, timeout_s=180):
    gc.collect()
    if baseline_mb is None:
        print("[FILA] GPU não detectada; aguardando cooldown de CPU/RAM...")
        time.sleep(5)
        return
    deadline = time.monotonic() + timeout_s
    stable_reads = 0
    while time.monotonic() < deadline:
        current_mb = _rift_gpu_memory_used_mb()
        if current_mb is not None and current_mb <= baseline_mb + 128:
            stable_reads += 1
        else:
            stable_reads = 0
        print("[FILA] VRAM:", current_mb, "MB; alvo <=", baseline_mb + 128, "MB; estabilidade", stable_reads, "/3")
        if stable_reads >= 3:
            time.sleep(2)
            return
        time.sleep(2)
    print("[FILA] AVISO: a VRAM não retornou ao nível seguro em", timeout_s, "s; seguindo mesmo assim.")

def _rift_download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "rift-queue-runner/1.1"})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())

def _rift_run_cseries(model):
    root = Path("/content/cascade_run") if os.path.isdir("/content") else Path.cwd() / "cascade_run"
    root.mkdir(parents=True, exist_ok=True)
    for repo_path, local_path in RIFT_CSERIES_SCRIPTS + RIFT_CSERIES_PKG_FILES:
        _rift_download(RIFT_RUNNER_RAW_BASE + "/" + repo_path, root / local_path)
    worst = 0
    for _repo_path, script in RIFT_CSERIES_SCRIPTS:
        print("[FILA] série C >", script)
        return_code = subprocess.call(
            [sys.executable, str(root / script), "--model", model],
            cwd=str(root),
            env=os.environ.copy(),
        )
        if return_code != 0:
            print("[FILA] AVISO série C", script, "rc =", return_code)
            worst = worst or return_code
    shutil.rmtree(root, ignore_errors=True)
    return worst

def _rift_run_step(index, tech, model, work_dir):
    if tech == "c-series":
        return _rift_run_cseries(model)
    url = _rift_step_url(tech, model)
    if tech == "geyser":
        command = 'curl -fsSL "{0}" -o /content/geyser_launcher.py && python /content/geyser_launcher.py'.format(url)
        try:
            return subprocess.call(["bash", "-lc", command], env=os.environ.copy())
        finally:
            Path("/content/geyser_launcher.py").unlink(missing_ok=True)
    launcher_path = work_dir / ("launcher_%03d.py" % index)
    try:
        print("[FILA] baixando launcher:", url)
        _rift_download(url, launcher_path)
        return subprocess.call([sys.executable, str(launcher_path)], env=os.environ.copy())
    finally:
        launcher_path.unlink(missing_ok=True)

RIFT_QUEUE_STEPS = []
_rift_microlm_queued = False
for _rift_model in RIFT_QUEUE_MODELS:
    for _rift_tech in _rift_expand_techs(_rift_model):
        if _rift_tech == "microlm":
            if _rift_microlm_queued:
                continue
            _rift_microlm_queued = True
            RIFT_QUEUE_STEPS.append((RIFT_MICROLM_MODEL_ID, _rift_tech))
            continue
        RIFT_QUEUE_STEPS.append((_rift_model, _rift_tech))
if not RIFT_QUEUE_STEPS:
    raise SystemExit("[FILA] Nenhum passo válido na fila (verifique TECHS)")

print("[FILA] Base:", RIFT_QUEUE_BASE, "| repo:", RIFT_RUNNER_REPO, "| ref:", RIFT_RUNNER_REF)
print("[FILA]", len(RIFT_QUEUE_MODELS), "modelo(s) na fila —", len(RIFT_QUEUE_STEPS), "passo(s) no total")
_rift_install_tokenizer_deps()
print("[FILA] Limpeza prévia (saídas conhecidas de execuções anteriores)...")
_rift_cleanup_workspace("light")

_rift_work_dir = (
    Path("/content/rift_serial_queue") if os.path.isdir("/content") else Path.cwd() / "rift_serial_queue"
)
_rift_work_dir.mkdir(parents=True, exist_ok=True)
_rift_summary = []
for _rift_index, (_rift_model, _rift_tech) in enumerate(RIFT_QUEUE_STEPS, start=1):
    _rift_work_dir.mkdir(parents=True, exist_ok=True)
    _rift_cleanup_workspace("light")
    _rift_baseline_mb = _rift_gpu_memory_used_mb()
    print()
    print("=" * 78)
    print("[FILA] %d/%d — %s — %s" % (_rift_index, len(RIFT_QUEUE_STEPS), _rift_tech.upper(), _rift_model))
    print("=" * 78)
    _rift_started = time.monotonic()
    try:
        _rift_rc = _rift_run_step(_rift_index, _rift_tech, _rift_model, _rift_work_dir)
    except Exception as exc:
        print("[FILA] AVISO: passo falhou com exceção; a fila continua:", exc)
        _rift_rc = -1
    _rift_summary.append({
        "modelo": _rift_model,
        "tecnologia": _rift_tech,
        "rc": _rift_rc,
        "segundos": round(time.monotonic() - _rift_started, 1),
    })
    if _rift_rc != 0:
        print("[FILA] AVISO: passo terminou com código", _rift_rc, "— seguindo para o próximo item da fila.")
    else:
        print("[FILA] Passo concluído; benchmark publicado.")
    _rift_wait_for_resource_release(_rift_baseline_mb)

print()
print("[FILA] Resumo da fila:")
print("-" * 78)
print("%-4s %-40s %-11s %-11s %9s" % ("#", "MODELO", "TECNOLOGIA", "STATUS", "DURAÇÃO"))
for _rift_row_index, _rift_row in enumerate(_rift_summary, start=1):
    _rift_status = "OK" if _rift_row["rc"] == 0 else "FALHA(%s)" % _rift_row["rc"]
    print("%-4d %-40s %-11s %-11s %8.1fs" % (
        _rift_row_index,
        _rift_row["modelo"][:40],
        _rift_row["tecnologia"],
        _rift_status,
        _rift_row["segundos"],
    ))
print("-" * 78)
_rift_failures = [row for row in _rift_summary if row["rc"] != 0]
if _rift_failures:
    print("[FILA] Fila concluída com AVISOS —", len(_rift_failures), "passo(s) falharam (resumo acima).")
else:
    print("[FILA] Todos os passos foram concluídos e publicados em série.")
print("[FILA] Limpeza final (cache HF + temporários)...")
_rift_cleanup_workspace("full")
`;
}

export default {
  async fetch(request) {
    if (!["GET", "HEAD"].includes(request.method)) {
      return new Response("Método não permitido", {
        status: 405,
        headers: {
          Allow: "GET, HEAD",
          "Cache-Control": "no-store",
          "Content-Type": "text/plain; charset=utf-8",
          "X-Content-Type-Options": "nosniff",
        },
      });
    }
    const body = buildRunnerScript({ origin: new URL(request.url).origin });
    return new Response(request.method === "HEAD" ? null : body, {
      status: 200,
      headers: {
        "Cache-Control": "public, max-age=0, s-maxage=300",
        "Content-Disposition": 'inline; filename="runner.py"',
        "Content-Type": "text/plain; charset=utf-8",
        "X-Content-Type-Options": "nosniff",
      },
    });
  },
};

export const _test = {
  ALL_TECHNOLOGIES,
  CSERIES_PACKAGE_FILES,
  CSERIES_SCRIPTS,
  GGUF_DEFAULT_QUANT,
  KNOWN_TECHNOLOGIES,
  MICROLM_MODEL_ID,
  PRECLEAN_DIRECTORIES,
  PRECLEAN_PATTERNS,
  TOKENIZER_DEPENDENCIES,
  buildRunnerScript,
};

// Conversor CASCADE — runner local auto-contido (contrato §26.1).
// GET /converter.py (rewrite → /api/converter) serve um script Python gerado
// no servidor com repo/ref JÁ RESOLVIDOS (§14.1): o usuário baixa o arquivo
// (Content-Disposition: attachment) e roda no PC ou no Colab. O runner baixa
// o conversor pinado de core/cascade/converter/, opcionalmente baixa o modelo
// do Hugging Face (snapshot só de pesos+config+tokenizer), converte para
// CASCADE-DIR e, se pedido, envia o resultado ao HF Hub do usuário.
// Segurança (§26.4): nenhum token em código/URL/log — HF_TOKEN é lido apenas
// do ambiente pelo runner; o upload é ação do usuário no ambiente dele.
import { rawBaseUrl, resolveRef, resolveRepo } from "./_lib/repo.mjs";

const RUNNER_FILENAME = "cascade-converter-runner.py";
// Arquivos do conversor baixados pelo runner (cópia ÚNICA do §20 regra 1):
// caem em ./cascade/converter/ locais; ./cascade/__init__.py vazio completa o
// pacote 'cascade' mínimo para eventuais imports relativos.
const CONVERTER_REPO_FILES = [
  "core/cascade/converter/__init__.py",
  "core/cascade/converter/cascade_converter.py",
  "core/cascade/converter/CASCADE_DIR_FORMAT_v0.1.txt",
  "core/cascade/converter/requirements.txt",
];
const CONVERTER_LOCAL_DIR = "cascade/converter";
// Snapshot HF: apenas pesos + config/tokenizer (nada de PDFs/imagens do repo).
const HF_ALLOW_PATTERNS = ["*.safetensors", "*.json", "tokenizer*", "*.model"];
const PIP_INSTALL_HINT =
  "pip install 'torch' 'safetensors>=0.4' 'numpy>=1.26' 'huggingface_hub>=0.24,<1'";
const DEFAULT_DISK_BUDGET_GB = 75;

/**
 * Gera o runner local auto-contido (§26.1). Tudo que depende do servidor
 * (repo, ref, base raw) é BAKEADO como constante; o restante é CLI argparse:
 * --model OU --input (mutuamente exclusivos e um obrigatório), --output com
 * default derivado, --hf-repo, --publish on|off (default off),
 * --disk-budget-gb 75, --resume por padrão (desative com --no-resume),
 * --delete-source-shards (só com --model: download próprio do runner) e
 * passthrough de --group-size/--ranks.
 */
function buildRunner({ repo = resolveRepo(), ref = resolveRef(), origin = "" } = {}) {
  const repoBase = rawBaseUrl(repo, ref);
  return `#!/usr/bin/env python3
# Runner LOCAL do conversor CASCADE gerado por ${origin || "RIFT-LM"} (contrato §26.1).
# Fluxo: baixar modelo da HF (opcional) -> converter para CASCADE-DIR
# (pacote-por-pacote, orçamento de disco + retomada) -> enviar o resultado ao
# Hugging Face Hub (opcional, exige HF_TOKEN de ESCRITA no ambiente).
# Nenhum token é impresso ou embutido: HF_TOKEN/HUGGING_FACE_HUB_TOKEN são
# lidos somente de variáveis de ambiente.
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO_BASE = ${JSON.stringify(repoBase)}
GITHUB_REPOSITORY = ${JSON.stringify(repo)}
SOURCE_REF = ${JSON.stringify(ref)}
CONVERTER_REPO_FILES = ${JSON.stringify(CONVERTER_REPO_FILES)}
CONVERTER_LOCAL_DIR = ${JSON.stringify(CONVERTER_LOCAL_DIR)}
HF_ALLOW_PATTERNS = ${JSON.stringify(HF_ALLOW_PATTERNS)}
PIP_INSTALL_HINT = ${JSON.stringify(PIP_INSTALL_HINT)}
MODEL_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Runner local do conversor CASCADE: baixa o conversor pinado do "
            "GitHub, converte um checkpoint para CASCADE-DIR e opcionalmente "
            "envia o resultado ao Hugging Face Hub."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model", default=None,
        help="org/modelo do Hugging Face para baixar (snapshot apenas de "
             "*.safetensors + config/tokenizer)",
    )
    source.add_argument(
        "--input", default=None,
        help="pasta local com o checkpoint .safetensors já baixado",
    )
    parser.add_argument(
        "--output", default=None,
        help="diretório CASCADE-DIR de saída (default: <nome do modelo>-cascade)",
    )
    parser.add_argument(
        "--hf-repo", default=None,
        help="repo de destino no Hugging Face Hub (org/nome) para subir o "
             "resultado convertido — exige HF_TOKEN de ESCRITA no ambiente",
    )
    parser.add_argument(
        "--publish", choices=["on", "off"], default="off",
        help="repassa --publish ao conversor (dashboard; default off)",
    )
    parser.add_argument(
        "--disk-budget-gb", type=float, default=${DEFAULT_DISK_BUDGET_GB},
        help="orçamento máximo de disco (GB) da saída (default ${DEFAULT_DISK_BUDGET_GB}; "
             "<=0 desativa)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="desativa a retomada automática (--resume é o padrão do runner)",
    )
    parser.add_argument(
        "--delete-source-shards", action="store_true",
        help="apaga cada shard .safetensors de ORIGEM após todos os seus "
             "tensores serem convertidos e verificados — permitido apenas com "
             "--model (o download é do próprio runner)",
    )
    parser.add_argument("--group-size", type=int, default=None,
                        help="passthrough para o conversor (default do conversor: 64)")
    parser.add_argument("--ranks", default=None,
                        help="passthrough para o conversor, ex.: 8,16,32")
    parser.set_defaults(resume=True)
    return parser.parse_args()

def fail(message):
    raise SystemExit("[RUNNER] ERRO: " + message)

def check_dependencies(need_hub):
    missing = []
    for module, package in (("numpy", "numpy>=1.26"),
                            ("safetensors", "safetensors>=0.4"),
                            ("torch", "torch")):
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    if need_hub:
        try:
            __import__("huggingface_hub")
        except Exception:
            missing.append("huggingface_hub>=0.24,<1")
    if missing:
        fail(
            "dependências ausentes: " + ", ".join(missing)
            + ". Instale com: " + PIP_INSTALL_HINT
        )

def hf_token():
    # Lido SOMENTE do ambiente; nunca impresso e nunca colocado em URL.
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return token or None

def download_converter(base_dir):
    if not REPO_BASE.lower().startswith("https://"):
        fail("o download do conversor precisa usar HTTPS: " + REPO_BASE)
    package_root = base_dir / "cascade"
    target = base_dir / CONVERTER_LOCAL_DIR
    target.mkdir(parents=True, exist_ok=True)
    # Pacote 'cascade' mínimo: __init__.py vazio na raiz do pacote local.
    init_file = package_root / "__init__.py"
    if not init_file.exists():
        init_file.write_bytes(b"")
    for repo_path in CONVERTER_REPO_FILES:
        local_path = target / repo_path.rsplit("/", 1)[-1]
        url = REPO_BASE + "/" + repo_path
        request = Request(url, headers={"User-Agent": "rift-converter-runner/1.0"})
        local_path.write_bytes(urlopen(request, timeout=60).read())
        print("[RUNNER] download", repo_path, "->", str(local_path))
    return target / "cascade_converter.py"

def download_model(model, base_dir):
    from huggingface_hub import snapshot_download
    local_dir = base_dir / (model.split("/")[-1] + "-hf")
    token = hf_token()
    print("[RUNNER] HF_TOKEN:", "detectado no ambiente" if token else "não definido (repos públicos apenas)")
    print("[RUNNER] snapshot_download:", model, "->", str(local_dir))
    print("[RUNNER] padrões permitidos:", ", ".join(HF_ALLOW_PATTERNS))
    snapshot_download(
        repo_id=model,
        local_dir=str(local_dir),
        allow_patterns=HF_ALLOW_PATTERNS,
        token=token,
    )
    return local_dir

def upload_output(hf_repo, output_dir):
    token = hf_token()
    if not token:
        fail(
            "--hf-repo exige um HF_TOKEN de ESCRITA no ambiente "
            "(export HF_TOKEN=... ou Secret HF_TOKEN no Colab). "
            "Crie um token com permissão 'write' em huggingface.co/settings/tokens. "
            "O token nunca é impresso por este runner."
        )
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print("[RUNNER] criando (se preciso) o repo de destino:", hf_repo)
    api.create_repo(repo_id=hf_repo, exist_ok=True)
    print("[RUNNER] enviando", str(output_dir), "->", hf_repo)
    api.upload_folder(repo_id=hf_repo, folder_path=str(output_dir))
    print("[RUNNER] upload concluído: https://huggingface.co/" + hf_repo)

def main():
    args = parse_args()
    if args.model and not MODEL_RE.match(args.model):
        fail("--model inválido; use org/modelo (ex.: Qwen/Qwen2.5-7B-Instruct)")
    if args.hf_repo and not MODEL_RE.match(args.hf_repo):
        fail("--hf-repo inválido; use org/nome (ex.: seu-usuario/meu-modelo-cascade)")
    if args.delete_source_shards and not args.model:
        fail(
            "--delete-source-shards só é aceito com --model: nesse modo o "
            "download é do próprio runner. Com --input a origem é sua e o "
            "runner não a apaga."
        )
    base_dir = Path("/content") if os.path.isdir("/content") else Path.cwd()
    check_dependencies(need_hub=bool(args.model or args.hf_repo))

    if args.model:
        source_name = args.model.split("/")[-1]
    else:
        # resolve() contra o cwd de QUEM invocou o runner: o conversor roda em
        # subprocesso com cwd=base_dir (no Colab, /content) e um caminho
        # relativo seria resolvido contra o diretório errado — e o upload
        # final leria a pasta errada.
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            fail("--input não existe: " + str(input_path))
        source_name = input_path.name or "modelo"
    output_dir = (Path(args.output) if args.output else base_dir / (source_name + "-cascade")).resolve()
    print("[RUNNER] repo pinado:", GITHUB_REPOSITORY, "| ref:", SOURCE_REF)
    print("[RUNNER] saída:", str(output_dir))

    converter_script = download_converter(base_dir)
    if args.model:
        input_dir = download_model(args.model, base_dir)
    else:
        input_dir = input_path

    command = [
        sys.executable,
        str(converter_script),
        "convert",
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--disk-budget-gb", str(args.disk_budget_gb),
    ]
    if args.model:
        command += ["--model-id", args.model]
    if args.resume:
        command += ["--resume"]
    if args.delete_source_shards:
        command += ["--delete-source-shards"]
    if args.group_size is not None:
        command += ["--group-size", str(args.group_size)]
    if args.ranks:
        command += ["--ranks", args.ranks]
    if args.publish == "on":
        command += ["--publish"]

    env = os.environ.copy()
    env.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)
    env.setdefault("RIFT_SOURCE_REF", SOURCE_REF)
    if args.delete_source_shards and not os.path.isdir("/content"):
        # Download próprio do runner: liberar a guarda destrutiva SOMENTE para
        # esta execução (fora do Colab o conversor exige o opt-in explícito).
        env["RIFT_ALLOW_LOCAL_CLEANUP"] = "1"
        print("[RUNNER] limpeza local liberada apenas para os shards baixados por este runner.")
    print("[RUNNER] executando o conversor (pacote-por-pacote)...")
    return_code = subprocess.call(command, cwd=str(base_dir), env=env)
    print("[RUNNER] conversor finalizado rc =", return_code)
    if return_code != 0:
        raise SystemExit(return_code)

    if args.hf_repo:
        upload_output(args.hf_repo, output_dir)
    print("[RUNNER] concluído. Saída CASCADE-DIR em:", str(output_dir))

if __name__ == "__main__":
    main()
`;
}

export default {
  async fetch(request) {
    // GET only (contrato §26.1); HEAD é aceito como variante sem corpo,
    // como nos demais handlers de api/.
    if (!["GET", "HEAD"].includes(request.method)) {
      return new Response("Método não permitido", {
        status: 405,
        headers: { Allow: "GET, HEAD", "Content-Type": "text/plain; charset=utf-8" },
      });
    }
    const origin = new URL(request.url).origin;
    const body = buildRunner({ origin });
    return new Response(request.method === "HEAD" ? null : body, {
      status: 200,
      headers: {
        "Cache-Control": "public, max-age=0, s-maxage=300",
        // Botão de download do card do conversor (§26.3): attachment.
        "Content-Disposition": `attachment; filename="${RUNNER_FILENAME}"`,
        "Content-Type": "text/x-python; charset=utf-8",
        "X-Content-Type-Options": "nosniff",
      },
    });
  },
};

export const _test = {
  CONVERTER_LOCAL_DIR,
  CONVERTER_REPO_FILES,
  DEFAULT_DISK_BUDGET_GB,
  HF_ALLOW_PATTERNS,
  PIP_INSTALL_HINT,
  RUNNER_FILENAME,
  buildRunner,
};

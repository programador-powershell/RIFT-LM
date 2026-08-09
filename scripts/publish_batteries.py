#!/usr/bin/env python3
from pathlib import Path
import argparse, json, shutil, sys

def main():
    parser = argparse.ArgumentParser(
        description="Publica rift_test_batteries.json no dashboard Vercel."
    )
    parser.add_argument("source", help="Arquivo rift_test_batteries.json gerado pelos testes")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Arquivo não encontrado: {source}")

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"JSON inválido: {exc}")

    if not isinstance(payload, list):
        raise SystemExit("O histórico precisa ser um array JSON de baterias.")

    target = Path(__file__).resolve().parents[1] / "data" / "rift_test_batteries.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    print(f"Publicado: {len(payload)} bateria(s)")
    print(f"Destino: {target}")

if __name__ == "__main__":
    main()

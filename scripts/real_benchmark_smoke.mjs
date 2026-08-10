import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { _test as realApi } from "../api/real-test.mjs";

const launcher = realApi.buildLauncher({
  technology: "cascade",
  model: { modelId: "Qwen/Qwen2.5-0.5B", trustRemoteCode: false },
  targetLayer: "auto",
  device: "auto",
  publish: "off",
  trustRemoteCode: false,
  iterations: 50,
  warmup: 10,
  origin: "https://example.test",
  ref: "a".repeat(40),
});

assert.match(launcher, /real_benchmark_runner\.py/);
assert.match(launcher, /--iterations/);
assert.match(launcher, /50/);
assert.match(launcher, /--warmup/);
assert.match(launcher, /RIFT_INGEST_TOKEN/);
assert.doesNotMatch(launcher, /rows\/s.*Tok\/s/i);

const req = new Request(
  "https://example.test/api/real-test?technology=rift&model=Qwen%2FQwen2.5-0.5B&iterations=75&warmup=12",
);
const params = realApi.requestParameters(req);
assert.equal(params.technology, "rift");
assert.equal(params.model.modelId, "Qwen/Qwen2.5-0.5B");
assert.equal(params.iterations, 75);
assert.equal(params.warmup, 12);
assert.equal(realApi.normalizePositiveInt("9999", 50, 10, 500), 500);
assert.equal(realApi.normalizePositiveInt("x", 50, 10, 500), 50);

const vercel = JSON.parse(await readFile(new URL("../vercel.json", import.meta.url), "utf8"));
for (const technology of ["rift", "cascade", "aether", "spectra", "winner"]) {
  const route = vercel.rewrites.find((item) => item.source.startsWith(`/${technology}/`));
  assert.ok(route, `rewrite ausente: ${technology}`);
  assert.match(route.destination, /\/api\/real-test\?/);
}

const runnerPath = new URL("./real_benchmark_runner.py", import.meta.url);
const runnerSource = await readFile(runnerPath, "utf8");
for (const required of [
  "LINEAR_REAL_MEASURED_V3",
  "perf_counter_ns_with_cuda_sync_v1",
  "proc_rss_1ms_and_torch_cuda_peak_v1",
  "safetensors_data_offsets_v1",
  "candidate_exact_file_match",
  "INVALID_REAL_INPUT",
  "end_to_end_generation",
]) {
  assert.ok(runnerSource.includes(required), `runner sem marcador obrigatório: ${required}`);
}

const pyCompile = spawnSync("python3", ["-m", "py_compile", runnerPath.pathname], {
  encoding: "utf8",
});
assert.equal(pyCompile.status, 0, pyCompile.stderr || pyCompile.stdout);

const pythonCheck = String.raw`
import importlib.util
import tempfile
from pathlib import Path

path = ${JSON.stringify(new URL("./real_benchmark_runner.py", import.meta.url).pathname)}
spec = importlib.util.spec_from_file_location("real_runner", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

source = """class D:
    type = "cpu"
device = D()
def benchmark_ms(fn, *, device, iterations=2):
    return {"median_ms": 1.0}
def main():
    return benchmark_ms(lambda: 1 + 1, device=device, iterations=2)
if __name__ == "__main__":
    main()
"""
patched = mod.instrument_source(source)
compile(patched, "<patched>", "exec")
assert patched.count("def benchmark_ms(") == 1
assert patched.count("def _battery_original_benchmark_ms(") == 1

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    candidate = root / "candidate.bin"
    candidate.write_bytes(b"x" * 100)
    artifacts = mod.file_manifest(root)
    records = [{
        "timestamp_utc": "2026-08-09T00:00:00Z",
        "run_id": "run-1",
        "spec": "CASCADE v0.3",
        "technology": "CASCADE",
        "model_id": "Qwen/Qwen2.5-0.5B",
        "battery_id": "P1_CASCADE_GATED_F0_PLUS_F1",
        "status": "PASS",
        "comparison_role": "primary",
        "baseline_tok_s": 999,
        "candidate_tok_s": 999,
        "baseline_ram_bytes": 1000,
        "candidate_ram_bytes": 500,
        "baseline_disk_bytes": 200,
        "candidate_disk_bytes": 100,
        "gains": {"ram_reduction_pct": 50, "tok_s_gain_pct": 100},
        "metrics": {},
    }]
    sanitized = mod.sanitize_records(
        records,
        technology="cascade",
        model_id="Qwen/Qwen2.5-0.5B",
        target_layer=None,
        device="cpu",
        iterations=50,
        warmup=10,
        source_ref="main",
        probes={"baseline": {"memory": {}}, "candidate": {"memory": {}}},
        run_metrics={"peak_process_tree_rss_bytes": 1234},
        artifacts=artifacts,
        real_activation=True,
        activation_source="real_model_activation",
    )
    row = sanitized[0]
    assert row["baseline_tok_s"] is None and row["candidate_tok_s"] is None
    assert row["baseline_ram_bytes"] is None and row["candidate_ram_bytes"] is None
    assert row["comparison_role"] == "primary"
    assert row["implementation"]["eligible_for_primary_ranking"] is True
    assert row["metrics"]["storage_real"]["candidate_exact_file_match"] is True
    assert row["gains"]["ram_reduction_pct"] is None
    assert row["gains"]["tok_s_gain_pct"] is None

    invalid = mod.sanitize_records(
        records,
        technology="cascade",
        model_id="Qwen/Qwen2.5-0.5B",
        target_layer=None,
        device="cpu",
        iterations=50,
        warmup=10,
        source_ref="main",
        probes={},
        run_metrics={},
        artifacts=artifacts,
        real_activation=False,
        activation_source="synthetic_fallback",
    )[0]
    assert invalid["comparison_role"] == "diagnostic"
    assert invalid["status"] == "INVALID_REAL_INPUT"

print("real benchmark smoke: PASS")
`;

const pyCheck = spawnSync("python3", ["-c", pythonCheck], { encoding: "utf8" });
assert.equal(pyCheck.status, 0, pyCheck.stderr || pyCheck.stdout);

console.log("real benchmark smoke: PASS");

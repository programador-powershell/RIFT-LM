import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { _test as api } from "../api/results.mjs";

const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
const match = html.match(/<script>([\s\S]*?)<\/script>/);

assert.ok(match, "index.html precisa conter o script principal");
new vm.Script(match[1], { filename: "index-inline.js" });
assert.match(html, /America\/Sao_Paulo/);
assert.match(html, /Comparador RIFT × CASCADE/);
assert.match(html, /cascade_m0_phase1_test_v030_auto_batteries\.py/);
assert.doesNotMatch(html, /RIFT_GITHUB_TOKEN\s*=/);

const rift = { run_id: "run-rift", battery_id: "P1", technology: "RIFT" };
const cascade = { run_id: "run-cascade", battery_id: "P1", technology: "CASCADE" };
assert.equal(api.validateHistory([rift, cascade], "test").length, 2);
assert.equal(api.mergeHistories([rift], [cascade]).length, 2);

console.log("dashboard smoke test: PASS");

import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { requiredFiles, safePath, digest, verifyBundle, makeRequest } from "../extension/bundle.ts";

async function fixture() {
  const asset = new TextEncoder().encode("fixture only").buffer;
  const hash = await digest(asset);
  const files = [
    { path: "graph.onnx", role: "model_graph" },
    { path: "tokenizer/tokenizer.json", role: "tokenizer" },
    { path: "tokenizer/tokenizer_config.json", role: "tokenizer" },
  ].map((file) => ({ ...file, bytes: asset.byteLength, sha256: hash }));
  const manifestBytes = new TextEncoder().encode(JSON.stringify({ schema_version: "vons.bundle.manifest/v1", metadata: { backend: "direct" }, files })).buffer;
  return { manifestPath: "bundle-manifest-v1.json", manifestBytes, manifestHash: await digest(manifestBytes), backend: "direct", modelId: "test", assets: files.map((file) => ({ path: file.path, bytes: asset })) };
}
test("extension verifies exact manifest and asset bytes, including retained bundles", async () => {
  const bundle = await fixture(); await verifyBundle(bundle);
  await assert.rejects(verifyBundle({ ...bundle, manifestHash: "0".repeat(64) }), /integrity/);
  await assert.rejects(verifyBundle({ ...bundle, assets: bundle.assets.slice(1) }), /contents/);
  const corrupt = structuredClone(bundle); new Uint8Array(corrupt.assets[0].bytes)[0] = 0;
  await assert.rejects(verifyBundle(corrupt), /SHA-256/);
});
test("extension rejects escaped paths and incomplete model contracts", async () => {
  for (const value of ["../model", "https://evil.test/x", "%2e%2e/x", "a\\b", "/abs", "a?token", "a//b", "x\u0000y"]) assert.equal(safePath(value), false);
  assert.equal(safePath("tokenizer/tokenizer.json"), true);
  const bundle = await fixture(); const manifest = JSON.parse(new TextDecoder().decode(bundle.manifestBytes));
  manifest.metadata.graph = { external_data_locations: ["unknown.onnx.data"] };
  assert.throws(() => requiredFiles(new TextEncoder().encode(JSON.stringify(manifest)).buffer), /external-data/);
});
test("extension accepts bounded unique candidates and refuses empty or duplicate input", () => {
  assert.deepEqual(makeRequest(" ready ", "Choose", "A\r\nB\n").questions[0].options, ["A", "B"]);
  assert.throws(() => makeRequest("", "Choose", "A\nB"), /empty/);
  assert.throws(() => makeRequest("ready", "Choose", "A\nA"), /unique/);
  assert.throws(() => makeRequest("ready", "Choose", "A"), /2..32/);
});
test("extension has no page access, remote execution permission, or automatic action script", async () => {
  const manifest = JSON.parse(await readFile(new URL("../extension/manifest.json", import.meta.url), "utf8"));
  assert.equal(manifest.manifest_version, 3);
  assert.deepEqual(manifest.permissions, ["sidePanel"]);
  assert.equal(manifest.host_permissions, undefined);
  assert.equal(manifest.content_scripts, undefined);
  assert.equal(manifest.externally_connectable, undefined);
  assert.match(manifest.content_security_policy.extension_pages, /connect-src 'self';/);
  assert.doesNotMatch(manifest.content_security_policy.extension_pages, /'unsafe-eval'|https:|http:|\*/);
});

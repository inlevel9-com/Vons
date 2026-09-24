import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const onnx = await import("../src/onnx.ts");

test("structured state and the training text template are deterministic", () => {
  const question = { id: "q", type: "choice", prompt: "Pick", options: ["A"] };
  assert.equal(onnx.serializeState({ z: 1, a: ["x", 2] }), '{"a":["x",2],"z":1}');
  assert.equal(onnx.candidateText({ z: 1, a: ["x", 2] }, question, "A"), '{"a":["x",2],"z":1}\nQuestion: Pick\nCandidate: A');
});

test("default diffusion noise is deterministic Gaussian, not a bounded uniform surrogate", () => {
  const first = onnx.seededNoise(32, 7, 0);
  const second = onnx.seededNoise(32, 7, 0);
  assert.deepEqual(Array.from(first), Array.from(second));
  assert.ok(first.some((value) => value > 1 || value < -1));
  assert.ok(Array.from(first).every(Number.isFinite));
});

test("tokenizer ids, masks, and overflow behavior are explicit", async () => {
  const { Tokenizer } = await import("@huggingface/tokenizers");
  const tokenizer = new Tokenizer(
    JSON.parse(await readFile("../../artifacts/pilot/tokenizer/tokenizer.json", "utf8")),
    JSON.parse(await readFile("../../artifacts/pilot/tokenizer/tokenizer_config.json", "utf8")),
  );
  const question = { id: "q", type: "choice", prompt: "Pick", options: ["A/B?", "中文"] };
  const values = onnx.tokenizeCandidates(tokenizer, "ready", question, question.options, 512);
  assert.equal(values[0].inputIds[0], 101);
  assert.equal(values[0].inputIds.at(-1), 102);
  assert.equal(values[0].inputIds.length, values[0].attentionMask.length);
  assert.deepEqual(new Set(values[0].attentionMask), new Set([1]));
  assert.throws(() => onnx.tokenizeCandidates(tokenizer, "x", question, ["x".repeat(10000)], 4), /bundle limit/);
});

test("manifest integrity is checked before a runtime session is created", async () => {
  const manifest = JSON.parse(await readFile("../../artifacts/pilot/bundle-manifest-v1.json", "utf8"));
  const original = manifest.files.find((item) => item.role === "model_graph");
  original.sha256 = "0".repeat(64);
  const fetch = async () => new Response(new Uint8Array(original.bytes), { status: 200 });
  await assert.rejects(
    onnx.createOnnxWebBackend({ manifest, baseUrl: "https://example.test/bundle/", fetch }),
    /SHA-256 mismatch/,
  );
});

test("release runs can pin manifest bytes and encoded traversal is rejected", async () => {
  const manifest = JSON.parse(await readFile("../../artifacts/pilot/bundle-manifest-v1.json", "utf8"));
  await assert.rejects(
    onnx.createOnnxWebBackend({
      manifest,
      baseUrl: "https://example.test/bundle/",
      expectedManifestSha256: "0".repeat(64),
      fetch: async () => new Response(new Uint8Array(), { status: 200 }),
    }),
    /manifest SHA-256 mismatch/,
  );
  const escaped = structuredClone(manifest);
  escaped.files.find((item) => item.role === "model_graph").path = "%2e%2e/escape.onnx";
  await assert.rejects(
    onnx.createOnnxWebBackend({
      manifest: escaped,
      baseUrl: "https://example.test/bundle/",
      fetch: async () => new Response(new Uint8Array(), { status: 200 }),
    }),
    /bundle asset path must be relative/,
  );
});

test("malformed manifests are rejected before provider selection", async () => {
  await assert.rejects(
    onnx.createOnnxWebBackend({ manifest: { schema_version: "vons.bundle.manifest/v1" }, provider: "webgpu" }),
    /metadata.backend|baseUrl|manifestUrl/,
  );
});

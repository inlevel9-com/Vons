import assert from "node:assert/strict";
import test from "node:test";

const onnx = await import("../src/onnx.ts");

// Deliberately synthetic: public contract tests must not require pilot weights.
function testManifest() {
  const record = (path, role, bytes = 4) => ({
    path, role, bytes, sha256: "1".repeat(64), runtime_loaded: true, model_asset: true,
  });
  return {
    schema_version: "vons.bundle.manifest/v1",
    manifest: { path: "bundle-manifest.json", sha256: null, self_hash_excluded: true },
    metadata: { backend: "direct", option_count: 2, sequence_length: 32 },
    files: [
      record("model.onnx", "model_graph"),
      record("tokenizer.json", "tokenizer"),
      record("tokenizer_config.json", "tokenizer"),
    ],
    summary: { model_asset_bytes: 12, complete_download_bytes: 12 },
  };
}

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
    {
      version: "1.0",
      added_tokens: [],
      normalizer: { type: "BertNormalizer", clean_text: true, handle_chinese_chars: true, strip_accents: null, lowercase: true },
      pre_tokenizer: { type: "BertPreTokenizer" },
      post_processor: { type: "BertProcessing", sep: ["[SEP]", 2], cls: ["[CLS]", 1] },
      decoder: { type: "WordPiece", prefix: "##", cleanup: true },
      model: { type: "WordPiece", unk_token: "[UNK]", continuing_subword_prefix: "##", max_input_chars_per_word: 100,
        vocab: { "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, ready: 3, question: 4, pick: 5, candidate: 6, ":": 7, a: 8, b: 9, "/": 10, "?": 11, "中": 12, "文": 13 } },
    },
    { tokenizer_class: "BertTokenizer", unk_token: "[UNK]", cls_token: "[CLS]", sep_token: "[SEP]", model_max_length: 512 },
  );
  const question = { id: "q", type: "choice", prompt: "Pick", options: ["A/B?", "中文"] };
  const values = onnx.tokenizeCandidates(tokenizer, "ready", question, question.options, 512);
  assert.deepEqual(values[0].inputIds, [1, 3, 4, 7, 5, 6, 7, 8, 10, 9, 11, 2]);
  assert.deepEqual(values[1].inputIds, [1, 3, 4, 7, 5, 6, 7, 12, 13, 2]);
  assert.equal(values[0].inputIds.length, values[0].attentionMask.length);
  assert.deepEqual(new Set(values[0].attentionMask), new Set([1]));
  assert.throws(() => onnx.tokenizeCandidates(tokenizer, "x", question, ["x".repeat(10000)], 4), /bundle limit/);
});

test("manifest integrity is checked before a runtime session is created", async () => {
  const manifest = testManifest();
  const original = manifest.files.find((item) => item.role === "model_graph");
  original.sha256 = "0".repeat(64);
  const fetch = async () => new Response(new Uint8Array(original.bytes), { status: 200 });
  await assert.rejects(
    onnx.createOnnxWebBackend({ manifest, baseUrl: "https://example.test/bundle/", fetch }),
    /SHA-256 mismatch/,
  );
});

test("release runs can pin manifest bytes and encoded traversal is rejected", async () => {
  const manifest = testManifest();
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

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
  const { candidates: values, liveSequenceLength } = onnx.tokenizeCandidates(tokenizer, "ready", question, question.options, 512);
  assert.equal(liveSequenceLength, Math.max(values[0].inputIds.length, values[1].inputIds.length));
  assert.deepEqual(values[0].inputIds, [1, 3, 4, 7, 5, 6, 7, 8, 10, 9, 11, 2]);
  assert.deepEqual(values[1].inputIds, [1, 3, 4, 7, 5, 6, 7, 12, 13, 2]);
  assert.equal(values[0].inputIds.length, values[0].attentionMask.length);
  assert.deepEqual(new Set(values[0].attentionMask), new Set([1]));
  assert.throws(() => onnx.tokenizeCandidates(tokenizer, "x", question, ["x".repeat(10000)], 4), /bundle limit/);
});

test("tokenizeCandidates returns the longest live tokenized candidate length, bounded at 512", async () => {
  const { Tokenizer } = await import("@huggingface/tokenizers");
  const tokenizer = new Tokenizer(
    {
      version: "1.0",
      added_tokens: [],
      normalizer: { type: "BertNormalizer", clean_text: true, handle_chinese_chars: true, strip_accents: null, lowercase: true },
      pre_tokenizer: { type: "BertPreTokenizer" },
      post_processor: { type: "BertProcessing", sep: ["[SEP]", 2], cls: ["<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>", 1] },
      decoder: { type: "WordPiece", prefix: "##", cleanup: true },
      model: { type: "WordPiece", unk_token: "[UNK]", continuing_subword_prefix: "##", max_input_chars_per_word: 100,
        vocab: Object.assign({ "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "question": 101, "candidate": 102, ":": 103 },
          Object.fromEntries(Array.from({ length: 50 }, (_, i) => [`t${i}`, i + 3]))) },
    },
    { tokenizer_class: "BertTokenizer", unk_token: "[UNK]", cls_token: "[CLS]", sep_token: "[SEP]", model_max_length: 512 },
  );
  const shortQuestion = { id: "q", type: "choice", prompt: "Pick", options: ["t0 t1", "t0 t1 t2 t3"] };
  const short = onnx.tokenizeCandidates(tokenizer, "", shortQuestion, shortQuestion.options, 512);
  assert.ok(short.liveSequenceLength >= 1);
  assert.equal(short.liveSequenceLength, Math.max(short.candidates[0].inputIds.length, short.candidates[1].inputIds.length));
  assert.ok(short.liveSequenceLength <= 512);

  const capped = onnx.tokenizeCandidates(tokenizer, "", shortQuestion, shortQuestion.options, 64);
  assert.equal(capped.liveSequenceLength, short.liveSequenceLength);

  assert.throws(() => onnx.tokenizeCandidates(tokenizer, "", shortQuestion, shortQuestion.options, 0), /sequenceBudget/);
  assert.throws(() => onnx.tokenizeCandidates(tokenizer, "", shortQuestion, shortQuestion.options, 513), /sequenceBudget/);
});

test("variable sequence length regression: widths 64 and 128 produce identical live-token prefixes", async () => {
  const { Tokenizer } = await import("@huggingface/tokenizers");
  const tokenizer = new Tokenizer(
    {
      version: "1.0",
      added_tokens: [],
      normalizer: { type: "BertNormalizer", clean_text: true, handle_chinese_chars: true, strip_accents: null, lowercase: true },
      pre_tokenizer: { type: "BertPreTokenizer" },
      post_processor: { type: "BertProcessing", sep: ["[SEP]", 2], cls: ["<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>", 1] },
      decoder: { type: "WordPiece", prefix: "##", cleanup: true },
      model: { type: "WordPiece", unk_token: "[UNK]", continuing_subword_prefix: "##", max_input_chars_per_word: 100,
        vocab: { "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, ready: 3, question: 4, pick: 5, candidate: 6, ":": 7, a: 8, b: 9, c: 10, d: 11 } },
    },
    { tokenizer_class: "BertTokenizer", unk_token: "[UNK]", cls_token: "[CLS]", sep_token: "[SEP]", model_max_length: 512 },
  );
  const question = { id: "q", type: "choice", prompt: "Pick", options: ["a b", "c d"] };
  const result64 = onnx.tokenizeCandidates(tokenizer, "ready", question, question.options, 64);
  const result128 = onnx.tokenizeCandidates(tokenizer, "ready", question, question.options, 128);
  assert.equal(result64.liveSequenceLength, result128.liveSequenceLength);
  for (let index = 0; index < question.options.length; index += 1) {
    assert.deepEqual(result64.candidates[index].inputIds, result128.candidates[index].inputIds);
    assert.deepEqual(result64.candidates[index].attentionMask, result128.candidates[index].attentionMask);
  }
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

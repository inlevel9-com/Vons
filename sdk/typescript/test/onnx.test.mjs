import assert from "node:assert/strict";
import { createHash } from "node:crypto";
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

function testBackend(release) {
  return new onnx.OnnxWebBackend(
    {
      manifest: { schema_version: "vons.bundle.manifest/v1", metadata: { backend: "direct" } },
      manifestHash: "test-manifest-hash",
      model: new Uint8Array(),
      tokenizer: { token_to_id: () => 0 },
      externalData: [],
    },
    {},
    {
      provider: "wasm",
      ortVersion: null,
      wasmNumThreads: 1,
      webgpuAdapterRequested: false,
      webgpuDeviceRequested: false,
    },
    { run: async () => ({}), release, inputNames: [], outputNames: [] },
    {},
  );
}

function testDiffusionDecisionBackend(sequenceWidth, { beforeRun, onRelease } = {}) {
  let capturedFeeds;
  let runCalls = 0;
  const timings = [];
  const manifest = {
    schema_version: "vons.bundle.manifest/v1",
    manifest: { path: "bundle-manifest.json", sha256: null, self_hash_excluded: true },
    metadata: {
      backend: "diffusion",
      option_count: 32,
      sequence_length: 512,
      graph: { inputs: [{ name: "input_ids", shape: ["batch", 32, sequenceWidth] }] },
    },
    files: [],
    summary: { model_asset_bytes: 0, complete_download_bytes: 0 },
  };
  const runtime = {
    Tensor: class {
      constructor(type, data, dims) { return { type, data, dims }; }
    },
    env: { wasm: {} },
  };
  const tokenizer = {
    token_to_id: () => 0,
    encode: () => ({ ids: [1, 2, 3], attention_mask: [1, 1, 1], token_type_ids: [0, 0, 0] }),
  };
  const session = {
    inputNames: ["input_ids", "attention_mask", "token_type_ids", "option_mask", "initial_noise"],
    outputNames: ["scores", "answerability"],
    run: async (feeds) => {
      runCalls += 1;
      capturedFeeds = feeds;
      await beforeRun?.(feeds, runCalls);
      return {
        scores: { data: new Array(32).fill(0), dims: [1, 32] },
        answerability: { data: [5], dims: [1] },
      };
    },
    release: async () => { await onRelease?.(); },
  };
  const backend = new onnx.OnnxWebBackend(
    { manifest, manifestHash: "test-manifest-hash", model: new Uint8Array(), tokenizer, externalData: [] },
    runtime,
    { provider: "wasm", ortVersion: null, wasmNumThreads: 1, webgpuAdapterRequested: false, webgpuDeviceRequested: false },
    session,
    { onTiming: (sample) => timings.push(sample) },
  );
  return { backend, get capturedFeeds() { return capturedFeeds; }, get runCalls() { return runCalls; }, timings };
}

function testRuntimeBundle() {
  const payloads = new Map([
    ["model.onnx", new Uint8Array([1, 2, 3, 4])],
    ["tokenizer.json", new TextEncoder().encode(JSON.stringify({
      version: "1.0",
      added_tokens: [],
      normalizer: { type: "BertNormalizer", clean_text: true, handle_chinese_chars: true, strip_accents: null, lowercase: true },
      pre_tokenizer: { type: "BertPreTokenizer" },
      post_processor: { type: "BertProcessing", sep: ["[SEP]", 2], cls: ["[CLS]", 1] },
      decoder: { type: "WordPiece", prefix: "##", cleanup: true },
      model: { type: "WordPiece", unk_token: "[UNK]", continuing_subword_prefix: "##", max_input_chars_per_word: 100,
        vocab: { "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3 } },
    }))],
    ["tokenizer_config.json", new TextEncoder().encode(JSON.stringify({
      tokenizer_class: "BertTokenizer", unk_token: "[UNK]", cls_token: "[CLS]", sep_token: "[SEP]", model_max_length: 32,
    }))],
  ]);
  const files = [...payloads].map(([path, bytes]) => ({
    path,
    role: path === "model.onnx" ? "model_graph" : "tokenizer",
    bytes: bytes.byteLength,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    runtime_loaded: true,
    model_asset: true,
  }));
  const manifest = {
    schema_version: "vons.bundle.manifest/v1",
    manifest: { path: "bundle-manifest.json", sha256: null, self_hash_excluded: true },
    metadata: { backend: "direct", option_count: 2, sequence_length: 32 },
    files,
    summary: {
      model_asset_bytes: payloads.get("model.onnx").byteLength,
      complete_download_bytes: [...payloads.values()].reduce((sum, bytes) => sum + bytes.byteLength, 0),
    },
  };
  return {
    manifest,
    baseUrl: "https://example.test/bundle/",
    fetch: async (url) => {
      const path = new URL(url).pathname.split("/").at(-1);
      return new Response(payloads.get(path), { status: 200 });
    },
  };
}

function testSharedRuntimeBundle() {
  const tokenizer = new TextEncoder().encode(JSON.stringify({
    version: "1.0",
    added_tokens: [],
    normalizer: { type: "BertNormalizer", clean_text: true, handle_chinese_chars: true, strip_accents: null, lowercase: true },
    pre_tokenizer: { type: "BertPreTokenizer" },
    post_processor: { type: "BertProcessing", sep: ["[SEP]", 2], cls: ["[CLS]", 1] },
    decoder: { type: "WordPiece", prefix: "##", cleanup: true },
    model: { type: "WordPiece", unk_token: "[UNK]", continuing_subword_prefix: "##", max_input_chars_per_word: 100,
      vocab: { "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3 } },
  }));
  const config = new TextEncoder().encode(JSON.stringify({ max_tokens: 32, max_options: 2, diffusion_candidate_slots: 2,
    default_abstain_threshold: 0.55, default_answerability_threshold: 0.5 }));
  const payloads = new Map([
    ["encoder.onnx", new TextEncoder().encode("encoder graph")],
    ["encoder.onnx.data", new TextEncoder().encode("encoder weights")],
    ["direct.onnx", new TextEncoder().encode("direct graph")],
    ["direct.onnx.data", new TextEncoder().encode("direct weights")],
    ["diffusion.onnx", new TextEncoder().encode("diffusion graph")],
    ["diffusion.onnx.data", new TextEncoder().encode("diffusion weights")],
    ["tokenizer/tokenizer.json", tokenizer],
    ["tokenizer/tokenizer_config.json", new TextEncoder().encode(JSON.stringify({ tokenizer_class: "BertTokenizer" }))],
    ["config.json", config],
    ["calibration.json", new TextEncoder().encode("not loaded by the runtime")],
  ]);
  const roles = new Map([
    ["encoder.onnx", "encoder_graph"], ["encoder.onnx.data", "encoder_weights"],
    ["direct.onnx", "direct_graph"], ["direct.onnx.data", "direct_weights"],
    ["diffusion.onnx", "diffusion_graph"], ["diffusion.onnx.data", "diffusion_weights"],
    ["tokenizer/tokenizer.json", "tokenizer"], ["tokenizer/tokenizer_config.json", "tokenizer"],
    ["config.json", "config"], ["calibration.json", "calibration"],
  ]);
  const files = [...payloads].map(([path, bytes]) => ({ path, role: roles.get(path), bytes: bytes.byteLength,
    sha256: createHash("sha256").update(bytes).digest("hex"), runtime_loaded: true, model_asset: path !== "calibration.json" }));
  const manifest = {
    schema: "vons.shared-bundle/v1",
    model_id: "synthetic-shared-v1",
    encoder: { graph: "encoder.onnx", outputs: ["candidate_embeddings", "pooled"] },
    heads: {
      direct: { graph: "direct.onnx", inputs: ["candidate_embeddings", "pooled", "option_mask"] },
      diffusion: { graph: "diffusion.onnx", inputs: ["pooled", "option_mask", "initial_noise"] },
    },
    files,
  };
  return { manifest, payloads, baseUrl: "https://example.test/shared/" };
}

function testSharedDecisionBackend(backendName, { failHeadRelease = false } = {}) {
  const calls = { encoderFeeds: undefined, headFeeds: undefined, released: [] };
  const manifest = testSharedRuntimeBundle().manifest;
  const runtime = { Tensor: class { constructor(type, data, dims) { return { type, data, dims }; } }, env: { wasm: {} } };
  const tokenizer = {
    token_to_id: () => 0,
    encode: () => ({ ids: [1, 2, 3], attention_mask: [1, 1, 1], token_type_ids: [0, 0, 0] }),
  };
  const encoderSession = {
    inputNames: ["input_ids", "attention_mask", "token_type_ids", "option_mask"],
    outputNames: ["candidate_embeddings", "pooled"],
    run: async (feeds) => {
      calls.encoderFeeds = feeds;
      return {
        candidate_embeddings: { data: new Float32Array(2 * 256), dims: [1, 2, 256] },
        pooled: { data: new Float32Array(256), dims: [1, 256] },
      };
    },
    release: async () => { calls.released.push("encoder"); },
  };
  const headSession = {
    inputNames: backendName === "direct" ? ["candidate_embeddings", "pooled", "option_mask"] : ["pooled", "option_mask", "initial_noise"],
    outputNames: backendName === "direct" ? ["logits", "answerability"] : ["scores", "answerability"],
    run: async (feeds) => {
      calls.headFeeds = feeds;
      return {
        [backendName === "direct" ? "logits" : "scores"]: { data: new Float32Array(2), dims: [1, 2] },
        answerability: { data: new Float32Array([5]), dims: [1] },
      };
    },
    release: async () => {
      calls.released.push("head");
      if (failHeadRelease) throw new Error("synthetic head release failure");
    },
  };
  const backend = new onnx.OnnxWebBackend({
    manifest, manifestHash: "synthetic-shared-hash", backend: backendName, modelId: "synthetic-shared-v1",
    sequenceLength: 32, optionCount: 2, diffusionSlots: 2, tokenizer, model: new Uint8Array(), externalData: [],
  }, runtime, { provider: "wasm", ortVersion: null, wasmNumThreads: 1, webgpuAdapterRequested: false,
    webgpuDeviceRequested: false }, headSession, {}, encoderSession);
  return { backend, calls };
}

test("structured state and the training text template are deterministic", () => {
  const question = { id: "q", type: "choice", prompt: "Pick", options: ["A"] };
  assert.equal(onnx.serializeState({ z: 1, a: ["x", 2] }), '{"a":["x",2],"z":1}');
  assert.equal(onnx.candidateText({ z: 1, a: ["x", 2] }, question, "A"), '{"a":["x",2],"z":1}\nQuestion: Pick\nCandidate: A');
});

test("shared bundle loading verifies and fetches only the selected head assets", async () => {
  const fixture = testSharedRuntimeBundle();
  const requested = [];
  await assert.rejects(onnx.createOnnxWebBackend({
    manifest: fixture.manifest,
    baseUrl: fixture.baseUrl,
    head: "direct",
    fetch: async (url) => {
      const path = new URL(url).pathname.slice("/shared/".length);
      requested.push(path);
      return new Response(fixture.payloads.get(path), { status: 200 });
    },
  }));
  assert.deepEqual(requested.sort(), [
    "config.json", "direct.onnx", "direct.onnx.data", "encoder.onnx", "encoder.onnx.data",
    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
  ].sort());
  assert.equal(requested.includes("diffusion.onnx"), false);
  assert.equal(requested.includes("diffusion.onnx.data"), false);
  assert.equal(requested.includes("calibration.json"), false);
});

test("shared bundle loading requires a selected head before asset fetch", async () => {
  const fixture = testSharedRuntimeBundle();
  let fetches = 0;
  await assert.rejects(onnx.createOnnxWebBackend({
    manifest: fixture.manifest,
    baseUrl: fixture.baseUrl,
    fetch: async () => { fetches += 1; return new Response(); },
  }), /head: 'direct' or head: 'diffusion'/);
  assert.equal(fetches, 0);
});

test("shared bundle selected-head assets fail closed on hash mismatch", async () => {
  const fixture = testSharedRuntimeBundle();
  const manifest = structuredClone(fixture.manifest);
  manifest.files.find((file) => file.path === "direct.onnx.data").sha256 = "0".repeat(64);
  await assert.rejects(onnx.createOnnxWebBackend({
    manifest,
    baseUrl: fixture.baseUrl,
    head: "direct",
    fetch: async (url) => new Response(fixture.payloads.get(new URL(url).pathname.slice("/shared/".length)), { status: 200 }),
  }), /SHA-256 mismatch: direct.onnx.data/);
});

for (const backendName of ["direct", "diffusion"]) {
  test(`shared ${backendName} head receives verified encoder outputs and releases both sessions`, async () => {
    const instance = testSharedDecisionBackend(backendName);
    const response = await instance.backend.decide({
      state: "ready", seed: 7,
      questions: [{ id: "q", type: "choice", prompt: "Choose", options: ["A", "B"] }],
    });
    assert.deepEqual(Object.keys(instance.calls.encoderFeeds).sort(), ["attention_mask", "input_ids", "option_mask", "token_type_ids"]);
    assert.equal(instance.calls.encoderFeeds.input_ids.dims.join(","), "1,2,3");
    assert.equal(response.backend, backendName);
    if (backendName === "direct") {
      assert.deepEqual(Object.keys(instance.calls.headFeeds).sort(), ["candidate_embeddings", "option_mask", "pooled"]);
      assert.equal(instance.calls.headFeeds.candidate_embeddings.dims.join(","), "1,2,256");
    } else {
      assert.deepEqual(Object.keys(instance.calls.headFeeds).sort(), ["initial_noise", "option_mask", "pooled"]);
      assert.equal(instance.calls.headFeeds.initial_noise.dims.join(","), "1,2");
    }
    await instance.backend.dispose();
    assert.deepEqual(instance.calls.released.sort(), ["encoder", "head"]);
  });
}

test("shared session cleanup still releases the encoder when head release fails", async () => {
  const instance = testSharedDecisionBackend("direct", { failHeadRelease: true });
  await assert.rejects(instance.backend.dispose(), /synthetic head release failure/);
  assert.deepEqual(instance.calls.released.sort(), ["encoder", "head"]);
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

test("static diffusion input width pads feeds to the graph shape", async () => {
  const instance = testDiffusionDecisionBackend(512);
  await instance.backend.decide({
    state: "ready",
    seed: 7,
    questions: [{ id: "q", type: "choice", prompt: "Choose", options: ["first", "second"] }],
  });
  assert.deepEqual(instance.capturedFeeds.input_ids.dims, [1, 32, 512]);
  assert.equal(instance.capturedFeeds.input_ids.data.length, 32 * 512);
  assert.deepEqual(Array.from(instance.capturedFeeds.input_ids.data.slice(0, 3)), [1n, 2n, 3n]);
  assert.ok(Array.from(instance.capturedFeeds.input_ids.data.slice(3, 512)).every((value) => value === 0n));
  assert.equal(instance.timings[0].sequence_length, 512);
  assert.equal(instance.runCalls, 1);
  await instance.backend.dispose();
});

test("fixed diffusion input width shorter than tokenized content rejects before inference", async () => {
  const instance = testDiffusionDecisionBackend(2);
  await assert.rejects(
    instance.backend.decide({
      state: "ready",
      seed: 7,
      questions: [{ id: "q", type: "choice", prompt: "Choose", options: ["first", "second"] }],
    }),
    /model input width is 2/,
  );
  assert.equal(instance.runCalls, 0);
  await instance.backend.dispose();
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

test("manifest digest pins use raw URL bytes and stable JSON for in-memory manifests", async () => {
  const manifest = testManifest();
  const stableBytes = new TextEncoder().encode(onnx.serializeState(manifest));
  const formattedBytes = new TextEncoder().encode(JSON.stringify(manifest, null, 2));
  const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
  const stableHash = hash(stableBytes);
  const formattedHash = hash(formattedBytes);
  assert.notEqual(stableHash, formattedHash);

  let objectFetches = 0;
  await assert.rejects(
    onnx.createOnnxWebBackend({
      manifest,
      baseUrl: "https://example.test/bundle/",
      expectedManifestSha256: stableHash,
      fetch: async () => {
        objectFetches += 1;
        throw new Error("stop after manifest validation");
      },
    }),
    /stop after manifest validation/,
  );
  assert.equal(objectFetches, 1);

  await assert.rejects(
    onnx.createOnnxWebBackend({
      manifest,
      baseUrl: "https://example.test/bundle/",
      expectedManifestSha256: formattedHash,
      fetch: async () => {
        objectFetches += 1;
        throw new Error("unexpected asset fetch");
      },
    }),
    /manifest SHA-256 mismatch/,
  );
  assert.equal(objectFetches, 1);

  let urlFetches = 0;
  await assert.rejects(
    onnx.createOnnxWebBackend({
      manifestUrl: "https://example.test/bundle/bundle-manifest.json",
      expectedManifestSha256: formattedHash,
      fetch: async (url) => {
        urlFetches += 1;
        if (new URL(url).pathname.endsWith("/bundle-manifest.json")) {
          return new Response(formattedBytes, { status: 200 });
        }
        throw new Error("stop after manifest validation");
      },
    }),
    /stop after manifest validation/,
  );
  assert.equal(urlFetches, 2);
});

test("malformed manifests are rejected before provider selection", async () => {
  await assert.rejects(
    onnx.createOnnxWebBackend({ manifest: { schema_version: "vons.bundle.manifest/v1" }, provider: "webgpu" }),
    /metadata.backend|baseUrl|manifestUrl/,
  );
});

test("pre-aborted backend loading stops before fetching assets", async () => {
  const controller = new AbortController();
  controller.abort();
  let fetchCalls = 0;
  await assert.rejects(
    onnx.createOnnxWebBackend({
      manifest: testManifest(),
      baseUrl: "https://example.test/bundle/",
      signal: controller.signal,
      fetch: async () => {
        fetchCalls += 1;
        return new Response(new Uint8Array(), { status: 200 });
      },
    }),
    { name: "AbortError" },
  );
  assert.equal(fetchCalls, 0);
});

test("pre-aborted backend loading preserves an explicit null abort reason", async () => {
  const controller = new AbortController();
  controller.abort(null);
  let fetchCalls = 0;
  await assert.rejects(
    onnx.createOnnxWebBackend({
      manifest: testManifest(),
      baseUrl: "https://example.test/bundle/",
      signal: controller.signal,
      fetch: async () => {
        fetchCalls += 1;
        return new Response(new Uint8Array(), { status: 200 });
      },
    }),
    (reason) => reason === null,
  );
  assert.equal(fetchCalls, 0);
});

test("aborting an asset fetch cancels a late response body even if injected fetch ignores the signal", async () => {
  const controller = new AbortController();
  let fetchStarted;
  const started = new Promise((resolve) => { fetchStarted = resolve; });
  let completeFetch;
  let bodyCanceled;
  const loading = onnx.createOnnxWebBackend({
    manifest: testManifest(),
    baseUrl: "https://example.test/bundle/",
    signal: controller.signal,
    fetch: async (_url, init) => {
      assert.equal(init.signal, controller.signal);
      return new Promise((resolve) => {
        completeFetch = resolve;
        fetchStarted();
      });
    },
  });

  await started;
  controller.abort();
  completeFetch({
    ok: true,
    status: 200,
    body: { cancel: async (reason) => { bodyCanceled = reason; } },
  });
  await assert.rejects(loading, { name: "AbortError" });
  assert.equal(bodyCanceled, controller.signal.reason);
});

test("failed HTTP asset responses cancel their unread response body", async () => {
  let bodyCanceled;
  await assert.rejects(
    onnx.createOnnxWebBackend({
      manifestUrl: "https://example.test/bundle/bundle-manifest.json",
      fetch: async () => ({
        ok: false,
        status: 404,
        body: { cancel: async (reason) => { bodyCanceled = reason; } },
      }),
    }),
    /HTTP 404/,
  );
  assert.match(bodyCanceled.message, /HTTP 404/);
});

test("aborting a stalled asset body cancels its stream and stops loading promptly", async () => {
  const controller = new AbortController();
  let bodyReadStarted;
  const started = new Promise((resolve) => { bodyReadStarted = resolve; });
  let bodyCanceled;
  let bodyCancelStarted;
  const canceled = new Promise((resolve) => { bodyCancelStarted = resolve; });
  let resolveRead;
  let cancellationRequested = false;
  const reader = {
    read() {
      bodyReadStarted();
      return new Promise((resolve) => { resolveRead = resolve; });
    },
    cancel(reason) {
      if (!cancellationRequested) {
        cancellationRequested = true;
        bodyCanceled = reason;
        bodyCancelStarted();
      }
      resolveRead?.({ done: true, value: undefined });
      return Promise.resolve();
    },
    releaseLock() {},
  };
  const loading = onnx.createOnnxWebBackend({
    ...testRuntimeBundle(),
    signal: controller.signal,
    fetch: async (_url, init) => {
      assert.equal(init.signal, controller.signal);
      return { ok: true, status: 200, body: { getReader: () => reader } };
    },
  });

  await started;
  controller.abort(null);
  let timeout;
  const outcome = await Promise.race([
    loading.then(() => ({ kind: "resolved" }), (error) => ({ kind: "rejected", error })),
    new Promise((resolve) => { timeout = setTimeout(() => resolve({ kind: "timeout" }), 500); }),
  ]);
  clearTimeout(timeout);

  assert.equal(outcome.kind, "rejected", "loading should reject without waiting for the stalled body");
  assert.equal(outcome.error, null);
  await canceled;
  assert.equal(bodyCanceled, null);
});

test("WebGPU preflight devices are destroyed after success and abort", async () => {
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const webgpuRuntime = await import("onnxruntime-web/webgpu");
  const originalWasmPaths = webgpuRuntime.env.wasm.wasmPaths;
  const originalWasmNumThreads = webgpuRuntime.env.wasm.numThreads;
  const wasmPaths = "/__vons_webgpu_wasm_fallback__/";
  const destroyed = [];
  try {
    let adapterRequests = 0;
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: {
        gpu: {
          requestAdapter: async () => {
            adapterRequests += 1;
            if (adapterRequests > 1) return null;
            return { requestDevice: async () => ({ destroy: () => destroyed.push("completed") }) };
          },
        },
      },
    });
    await assert.rejects(onnx.createOnnxWebBackend({
      ...testRuntimeBundle(),
      provider: "webgpu",
      wasmPaths,
      wasmNumThreads: 2,
    }));
    assert.deepEqual(destroyed, ["completed"]);
    assert.equal(webgpuRuntime.env.wasm.wasmPaths, wasmPaths);
    assert.equal(webgpuRuntime.env.wasm.numThreads, 2);

    let adapterRequestStarted;
    const adapterStarted = new Promise((resolve) => { adapterRequestStarted = resolve; });
    let finishAdapterRequest;
    let deviceRequests = 0;
    const adapterController = new AbortController();
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: {
        gpu: {
          requestAdapter: () => new Promise((resolve) => {
            finishAdapterRequest = resolve;
            adapterRequestStarted();
          }),
        },
      },
    });
    const adapterLoading = onnx.createOnnxWebBackend({
      ...testRuntimeBundle(),
      provider: "webgpu",
      signal: adapterController.signal,
    });
    await adapterStarted;
    adapterController.abort();
    let adapterTimeout;
    const adapterOutcome = await Promise.race([
      adapterLoading.then(() => ({ kind: "resolved" }), (error) => ({ kind: "rejected", error })),
      new Promise((resolve) => { adapterTimeout = setTimeout(() => resolve({ kind: "timeout" }), 500); }),
    ]);
    clearTimeout(adapterTimeout);
    assert.equal(adapterOutcome.kind, "rejected", "adapter preflight should reject without waiting for requestAdapter");
    assert.equal(adapterOutcome.error.name, "AbortError");
    finishAdapterRequest({ requestDevice: async () => { deviceRequests += 1; return { destroy() {} }; } });
    await Promise.resolve();
    assert.equal(deviceRequests, 0);

    let deviceRequestStarted;
    const started = new Promise((resolve) => { deviceRequestStarted = resolve; });
    let finishDeviceRequest;
    const controller = new AbortController();
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: {
        gpu: {
          requestAdapter: async () => ({
            requestDevice: () => new Promise((resolve) => {
              finishDeviceRequest = resolve;
              deviceRequestStarted();
            }),
          }),
        },
      },
    });
    const loading = onnx.createOnnxWebBackend({
      ...testRuntimeBundle(),
      provider: "webgpu",
      signal: controller.signal,
    });
    await started;
    controller.abort();
    let deviceTimeout;
    const deviceOutcome = await Promise.race([
      loading.then(() => ({ kind: "resolved" }), (error) => ({ kind: "rejected", error })),
      new Promise((resolve) => { deviceTimeout = setTimeout(() => resolve({ kind: "timeout" }), 500); }),
    ]);
    clearTimeout(deviceTimeout);
    assert.equal(deviceOutcome.kind, "rejected", "device preflight should reject without waiting for requestDevice");
    assert.equal(deviceOutcome.error.name, "AbortError");
    finishDeviceRequest({ destroy: () => destroyed.push("aborted") });
    await Promise.resolve();
    assert.deepEqual(destroyed, ["completed", "aborted"]);
  } finally {
    webgpuRuntime.env.wasm.wasmPaths = originalWasmPaths;
    webgpuRuntime.env.wasm.numThreads = originalWasmNumThreads;
    if (originalNavigator) Object.defineProperty(globalThis, "navigator", originalNavigator);
    else delete globalThis.navigator;
  }
});

test("separately constructed backend sessions dispose independently and idempotently", async () => {
  const releases = [0, 0];
  const first = testBackend(async () => { releases[0] += 1; });
  const second = testBackend(async () => { releases[1] += 1; });

  await first.dispose();
  await first.dispose();
  assert.deepEqual(releases, [1, 0]);
  await assert.rejects(first.decide({}), /disposed/);

  await second.dispose();
  await second.dispose();
  assert.deepEqual(releases, [1, 1]);
});

test("dispose waits for active decisions and concurrent dispose callers share the release", async () => {
  let firstRunStarted;
  const started = new Promise((resolve) => { firstRunStarted = resolve; });
  let finishFirstRun;
  let releaseCalls = 0;
  const instance = testDiffusionDecisionBackend(512, {
    beforeRun: async (_feeds, runCalls) => {
      if (runCalls === 1) {
        firstRunStarted();
        await new Promise((resolve) => { finishFirstRun = resolve; });
      }
    },
    onRelease: () => { releaseCalls += 1; },
  });
  const request = {
    state: "ready",
    seed: 7,
    questions: [
      { id: "q1", type: "choice", prompt: "Choose", options: ["first", "second"] },
      { id: "q2", type: "choice", prompt: "Choose", options: ["first", "second"] },
    ],
  };
  const deciding = instance.backend.decide(request);
  await started;

  let disposalComplete = false;
  const firstDispose = instance.backend.dispose().then(() => { disposalComplete = true; });
  const secondDispose = instance.backend.dispose();
  await Promise.resolve();
  assert.equal(releaseCalls, 0);
  assert.equal(disposalComplete, false);
  await assert.rejects(instance.backend.decide(request), /disposed/);

  finishFirstRun();
  await deciding;
  await Promise.all([firstDispose, secondDispose]);
  assert.equal(instance.runCalls, 2, "the active multi-question decision should finish before release");
  assert.equal(releaseCalls, 1, "concurrent dispose calls should release exactly once");
});

test("WASM setup rejects promptly on abort and releases the late session", async (t) => {
  const { registerHooks } = await import("node:module");
  if (typeof registerHooks !== "function") {
    t.skip("requires Node.js module registerHooks");
    return;
  }

  const stateKey = "vons.synthetic.ort.lifecycle";
  const keyExpression = `Symbol.for(${JSON.stringify(stateKey)})`;
  const source = [
    `const state = globalThis[${keyExpression}];`,
    `export const env = { wasm: state.wasm, versions: { web: "synthetic" } };`,
    "export class Tensor {}",
    `export const InferenceSession = { create(model, options) { return state.create(model, options); } };`,
  ].join("\n");
  const moduleUrl = `data:text/javascript,${encodeURIComponent(source)}`;
  const hook = registerHooks({
    resolve(specifier, context, nextResolve) {
      if (specifier === "onnxruntime-web/wasm") return { url: moduleUrl, shortCircuit: true };
      return nextResolve(specifier, context);
    },
  });
  const key = Symbol.for(stateKey);
  const originalState = Object.getOwnPropertyDescriptor(globalThis, key);
  let sessionCreationStarted;
  const started = new Promise((resolve) => { sessionCreationStarted = resolve; });
  let resolveSessionCreation;
  let sessionReleased;
  const released = new Promise((resolve) => { sessionReleased = resolve; });
  let createCalls = 0;
  let releaseCalls = 0;
  const createSession = () => ({
    inputNames: ["input_ids", "attention_mask", "token_type_ids", "option_mask"],
    outputNames: ["logits", "answerability"],
    run: async () => ({}),
    release: async () => {
      releaseCalls += 1;
      sessionReleased();
    },
  });

  try {
    Object.defineProperty(globalThis, key, {
      configurable: true,
      value: {
        wasm: {},
        create: () => {
          createCalls += 1;
          if (createCalls > 1) return Promise.resolve(createSession());
          sessionCreationStarted();
          return new Promise((resolve) => { resolveSessionCreation = resolve; });
        },
      },
    });

    const controller = new AbortController();
    const creating = onnx.createOnnxWebBackend({
      ...testRuntimeBundle(),
      provider: "wasm",
      wasmPaths: "/synthetic/ort/",
      wasmNumThreads: 2,
      signal: controller.signal,
    });
    await started;
    controller.abort();
    await assert.rejects(creating, { name: "AbortError" });
    assert.equal(releaseCalls, 0);

    resolveSessionCreation(createSession());
    await released;
    assert.equal(releaseCalls, 1);

    const bundle = testRuntimeBundle();
    await assert.rejects(
      onnx.createOnnxWebBackend({ ...bundle, provider: "wasm", wasmNumThreads: 3 }),
      /WASM settings are page-global/,
    );
    await assert.rejects(
      onnx.createOnnxWebBackend({
        ...bundle,
        provider: "wasm",
        wasmPaths: "/synthetic/other-ort/",
        wasmNumThreads: 2,
      }),
      /WASM settings are page-global/,
    );
    assert.equal(createCalls, 1);
  } finally {
    hook.deregister();
    if (originalState) Object.defineProperty(globalThis, key, originalState);
    else delete globalThis[key];
  }
});

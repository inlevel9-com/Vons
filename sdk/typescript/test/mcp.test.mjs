import { test, describe, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, symlinkSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve, join } from "node:path";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { PassThrough } from "node:stream";

function syntheticManifest(overrides = {}) {
  const record = (path, role, bytes = 4) => ({
    path, role, bytes, sha256: "1".repeat(64), runtime_loaded: true, model_asset: true,
  });
  const base = {
    schema_version: "vons.bundle.manifest/v1",
    manifest: { path: "bundle-manifest-v1.json", sha256: null, self_hash_excluded: true },
    metadata: { backend: "direct", option_count: 2, sequence_length: 32 },
    files: [
      record("model.onnx", "model_graph"),
      record("tokenizer.json", "tokenizer"),
      record("tokenizer_config.json", "tokenizer"),
    ],
    summary: { model_asset_bytes: 12, complete_download_bytes: 12 },
  };
  return Object.assign({}, base, overrides);
}

function writeSyntheticBundleDir(manifest) {
  const dir = mkdtempSync(join(tmpdir(), "vons-mcp-bundle-"));
  for (const record of manifest.files) {
    const parts = record.path.split("/");
    if (parts.length > 1) {
      mkdirSync(join(dir, ...parts.slice(0, -1)), { recursive: true });
    }
    writeFileSync(join(dir, record.path), Buffer.alloc(record.bytes ?? 4));
  }
  const manifestJson = JSON.stringify(manifest, null, 2);
  const manifestBytes = Buffer.from(manifestJson, "utf8");
  const manifestSha256 = createHash("sha256").update(manifestBytes).digest("hex");
  writeFileSync(join(dir, "bundle-manifest-v1.json"), manifestBytes);
  return { dir, manifestSha256 };
}

class SyntheticOkBackend {
  constructor(choicesPerQuestion) {
    this.backend = "direct";
    this.choices = choicesPerQuestion;
  }
  async decide(request) {
    return {
      answers: this.choices.map((choice, idx) => {
        const options = request.questions[idx].options;
        const choiceIndex = options.indexOf(choice);
        const uniform = 1 / options.length;
        const bump = 0.3;
        const probabilities = options.map((o, i) => {
          if (i === choiceIndex) {
            return { option: o, probability: uniform + bump };
          }
          return { option: o, probability: uniform - (bump / Math.max(1, options.length - 1)) };
        });
        return {
          question_id: request.questions[idx].id,
          status: "ok",
          choice,
          probabilities,
          confidence: 0.72,
          seed: request.seed ?? null,
        };
      }),
      backend: "direct",
      model_id: "synthetic-ok-v0",
      metadata: { runtime: "synthetic", provenance: "mcp-pilot-test-fixture" },
    };
  }
}

class SyntheticAbstainBackend {
  constructor(reasons) { this.backend = "direct"; this.reasons = reasons; }
  async decide(request) {
    return {
      answers: request.questions.map((q, i) => ({
        question_id: q.id,
        status: "abstain",
        choice: null,
        abstain_reason: this.reasons[i] ?? "synthetic abstention",
        probabilities: [],
        confidence: 0,
      })),
      backend: "direct",
      model_id: "synthetic-abstain-v0",
    };
  }
}

class SyntheticThrowingBackend {
  constructor(msg) { this.backend = "direct"; this.msg = msg; }
  async decide(request) { void request; throw new Error(this.msg); }
}

describe("mcp.ts: isForbiddenBundlePath path safety", () => {
  test("allows relative, safe paths", async (t) => {
    void t;
    const { isForbiddenBundlePath } = await import("../src/mcp.ts");
    assert.equal(isForbiddenBundlePath("model.onnx"), false);
    assert.equal(isForbiddenBundlePath("a/b/c.json"), false);
    assert.equal(isForbiddenBundlePath("tokenizer/tokenizer.json"), false);
  });
  test("rejects URLs", async (t) => {
    void t;
    const { isForbiddenBundlePath } = await import("../src/mcp.ts");
    assert.equal(isForbiddenBundlePath("https://example.com/x.onnx"), true);
    assert.equal(isForbiddenBundlePath("http://evil/x"), true);
    assert.equal(isForbiddenBundlePath("file:///etc/passwd"), true);
  });
  test("rejects absolute and traversal forms", async (t) => {
    void t;
    const { isForbiddenBundlePath } = await import("../src/mcp.ts");
    assert.equal(isForbiddenBundlePath("/absolute/path"), true);
    assert.equal(isForbiddenBundlePath("../escape"), true);
    assert.equal(isForbiddenBundlePath("sub/../escape"), true);
    assert.equal(isForbiddenBundlePath("foo%2e%2ebar"), true);
    assert.equal(isForbiddenBundlePath("C:\\win\\path"), true);
    assert.equal(isForbiddenBundlePath("D:/win/path"), true);
    assert.equal(isForbiddenBundlePath("a/b/c?query"), true);
    assert.equal(isForbiddenBundlePath("a/b#frag"), true);
  });
});

describe("mcp-cli.ts createOnnxWebBackend integration: expectedManifestSha256 object-path mismatch regression", () => {
  test("pretty disk manifest bytes hash differs from stableJson() in-memory hash; omitting expectedManifestSha256 lets structural validation proceed", async (t) => {
    void t;
    const { createHash } = await import("node:crypto");
    const sdkMod = await import("../src/mcp.ts");
    const manifest = syntheticManifest();
    const prettyJson = JSON.stringify(manifest, null, 2);
    const diskBytes = Buffer.from(prettyJson, "utf8");
    const diskSha256 = createHash("sha256").update(diskBytes).digest("hex");
    const stableSerialize = (value) => {
      if (value === null || typeof value !== "object") {
        const prim = JSON.stringify(value);
        if (prim === undefined) throw new TypeError("undefined");
        return prim;
      }
      if (Array.isArray(value)) {
        return `[${value.map(stableSerialize).join(",")}]`;
      }
      const entries = Object.entries(value).sort(([a], [b]) => a.localeCompare(b));
      return `{${entries.map(([k, item]) => `${JSON.stringify(k)}:${stableSerialize(item)}`).join(",")}}`;
    };
    const stableSha256 = createHash("sha256")
      .update(Buffer.from(stableSerialize(manifest), "utf8"))
      .digest("hex");
    assert.notEqual(
      diskSha256,
      stableSha256,
      "regression requires pretty-disk hash and stable-json hash to actually differ",
    );
    const { loadBundleManifestStrict } = sdkMod;
    const { dir } = writeSyntheticBundleDir(manifest);
    const loaded = loadBundleManifestStrict(dir);
    assert.equal(loaded.manifestSha256, diskSha256);
    assert.equal(loaded.manifest.schema_version, manifest.schema_version);
    const onnxMod = await import("../src/onnx.ts");
    const localFetch = (async (input, init) => {
      void init;
      const u = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (!u.startsWith("vons-bundle-fs://local/")) {
        throw new Error(`refused ${u.slice(0, 64)}`);
      }
      const rel = decodeURIComponent(u.slice("vons-bundle-fs://local/".length));
      const path = await import("node:path");
      const fs2 = await import("node:fs");
      const abs = path.join(dir, ...rel.split("/"));
      const bytes = fs2.readFileSync(abs);
      return new Response(bytes.buffer, {
        status: 200,
        headers: { "content-type": "application/octet-stream", "content-length": String(bytes.byteLength) },
      });
    });
    let thrownMsg = null;
    try {
      await onnxMod.createOnnxWebBackend({
        manifest: loaded.manifest,
        baseUrl: "vons-bundle-fs://local/",
        provider: "wasm",
        fetch: localFetch,
      });
    } catch (e) {
      thrownMsg = e instanceof Error ? e.message : String(e);
    }
    assert.ok(thrownMsg !== null, "expected backend creation to fail for a synthetic bundle with invalid asset bytes");
    assert.equal(
      /SHA-256 mismatch/i.test(thrownMsg),
      false,
      `manifest SHA mismatch error must not appear; structural validation and asset loading must fail instead. Got: ${thrownMsg.slice(0, 180)}`,
    );
  });
});

describe("mcp.ts: loadBundleManifestStrict strict filesystem loader", () => {
  let bundleDir = null;
  let manifest = null;
  beforeEach((t) => {
    void t;
    manifest = syntheticManifest();
    const { dir } = writeSyntheticBundleDir(manifest);
    bundleDir = dir;
  });
  afterEach((t) => {
    void t;
    if (bundleDir) try { rmSync(bundleDir, { recursive: true }); } catch (_) { void 0; }
    bundleDir = null;
  });
  test("accepts a valid synthetic bundle", async (t) => {
    void t;
    const { loadBundleManifestStrict } = await import("../src/mcp.ts");
    const result = loadBundleManifestStrict(bundleDir);
    assert.equal(result.manifest.schema_version, "vons.bundle.manifest/v1");
    assert.equal(result.manifest.metadata.backend, "direct");
    assert.ok(result.manifestPath.endsWith("bundle-manifest-v1.json"), true);
    assert.ok(result.rootRealPath.length > 0);
  });
  test("rejects URL bundle directory", async (t) => {
    void t;
    const { loadBundleManifestStrict } = await import("../src/mcp.ts");
    assert.throws(() => loadBundleManifestStrict("https://evil.invalid/bundle"), /forbid|URL/);
    assert.throws(() => loadBundleManifestStrict("file:///tmp/nope"), /forbid|URL/);
  });
  test("rejects bundle manifest with traversal in declared file paths", async (t) => {
    void t;
    const { loadBundleManifestStrict } = await import("../src/mcp.ts");
    const badManifest = syntheticManifest();
    badManifest.files = [...badManifest.files];
    badManifest.files.push({ path: "../escape.pwn", role: "model_graph", bytes: 4, sha256: "1".repeat(64), model_asset: true, runtime_loaded: true });
    const { dir } = writeSyntheticBundleDir(badManifest);
    writeFileSync(join(dir, "escape.pwn"), Buffer.alloc(4));
    try {
      assert.throws(() => loadBundleManifestStrict(dir), /forbidden|traversal/);
    } finally { rmSync(dir, { recursive: true }); }
  });
  test("rejects manifest SHA-256 mismatch explicitly (no silent fallback)", async (t) => {
    void t;
    const { loadBundleManifestStrict } = await import("../src/mcp.ts");
    const expected = "a".repeat(64);
    assert.throws(() => loadBundleManifestStrict(bundleDir, expected), /mismatch/);
  });
  test("rejects malformed JSON manifest", async (t) => {
    void t;
    const { loadBundleManifestStrict } = await import("../src/mcp.ts");
    writeFileSync(join(bundleDir, "bundle-manifest-v1.json"), "{ not json ");
    assert.throws(() => loadBundleManifestStrict(bundleDir), /valid JSON/);
  });
  test("rejects wrong schema_version", async (t) => {
    void t;
    const { loadBundleManifestStrict } = await import("../src/mcp.ts");
    const badManifest = syntheticManifest();
    badManifest.schema_version = "vons.bundle.manifest/v0";
    const { dir } = writeSyntheticBundleDir(badManifest);
    try {
      assert.throws(() => loadBundleManifestStrict(dir), /schema_version/);
    } finally { rmSync(dir, { recursive: true }); }
  });
  test("rejects symlink escapes from declared files", async (t) => {
    void t;
    const { loadBundleManifestStrict } = await import("../src/mcp.ts");
    const outside = mkdtempSync(join(tmpdir(), "vons-mcp-outside-"));
    try {
      const evil = join(outside, "secret.txt");
      writeFileSync(evil, "should-never-be-read");
      const escape = join(bundleDir, "escape-link.onnx");
      try { symlinkSync(evil, escape); } catch (_) { void 0; }
      const badManifest = syntheticManifest();
      badManifest.files = [{ path: "escape-link.onnx", role: "model_graph", bytes: 42, sha256: "1".repeat(64), runtime_loaded: true, model_asset: true }];
      const manifestJson = JSON.stringify(badManifest, null, 2);
      writeFileSync(join(bundleDir, "bundle-manifest-v1.json"), manifestJson);
      assert.throws(() => loadBundleManifestStrict(bundleDir), /escapes bundle root|forbidden/);
    } finally { rmSync(outside, { recursive: true }); }
  });
  test("rejects a bundle manifest symlink that resolves outside the bundle root", async (t) => {
    void t;
    const { loadBundleManifestStrict } = await import("../src/mcp.ts");
    const outside = mkdtempSync(join(tmpdir(), "vons-mcp-outside-manifest-"));
    const externalManifestPath = join(outside, "bundle-manifest-v1.json");
    writeFileSync(externalManifestPath, JSON.stringify(syntheticManifest({ files: [] })));
    const manifestPath = join(bundleDir, "bundle-manifest-v1.json");
    try {
      rmSync(manifestPath);
      symlinkSync(externalManifestPath, manifestPath);
      assert.throws(() => loadBundleManifestStrict(bundleDir), /bundle manifest escapes bundle root via symlink/);
    } finally {
      rmSync(outside, { recursive: true });
    }
  });
});

describe("mcp.ts: createMcpServer synthetic backend validation", () => {
  test("rejects missing backend option", async (t) => {
    void t;
    const { createMcpServer } = await import("../src/mcp.ts");
    assert.throws(() => createMcpServer({}), /backend.*DecisionBackend/);
  });
  test("DecisionRequestZodSchema strict-rejects empty questions", async (t) => {
    void t;
    const { DecisionRequestZodSchema } = await import("../src/mcp.ts");
    const parsed = DecisionRequestZodSchema.safeParse({ state: "x", questions: [] });
    assert.equal(parsed.success, false);
  });
  test("DecisionRequestZodSchema accepts a boolean question with omitted options", async (t) => {
    void t;
    const { DecisionRequestZodSchema } = await import("../src/mcp.ts");
    const { validateRequest } = await import("../src/index.ts");
    const request = {
      state: "synthetic boolean request",
      questions: [{ id: "q-bool", type: "boolean", prompt: "Is the statement true?", options: [] }],
    };
    const parsed = DecisionRequestZodSchema.safeParse(request);
    assert.equal(parsed.success, true, "MCP schema must accept the shared boolean-options form");
    assert.doesNotThrow(() => validateRequest(request));
  });
  test("DecisionRequestZodSchema strict-rejects unknown extra keys", async (t) => {
    void t;
    const { DecisionRequestZodSchema } = await import("../src/mcp.ts");
    const parsed = DecisionRequestZodSchema.safeParse({
      state: "ok", questions: [{ id: "q", type: "choice", prompt: "x", options: ["A", "B"] }],
      evil_surprise: "should be rejected by strictObject",
    });
    assert.equal(parsed.success, false);
    assert.ok(!parsed.success, "strict schema rejected passthrough; no fake coerce or fallback allowed");
  });
  test("shared validateRequest rejects duplicated question IDs", async (t) => {
    void t;
    const { validateRequest } = await import("../src/index.ts");
    const dup = {
      state: "x",
      questions: [
        { id: "q1", type: "choice", prompt: "a", options: ["A", "B"] },
        { id: "q1", type: "choice", prompt: "b", options: ["A", "B"] },
      ],
    };
    assert.throws(() => validateRequest(dup));
  });
  test("abstention from backend passes validateResponse and preserves abstain_reason", async (t) => {
    void t;
    const { validateResponse } = await import("../src/index.ts");
    const backend = new SyntheticAbstainBackend(["confidence below threshold", "input unverifiable"]);
    const response = await backend.decide({
      state: "ok", questions: [
        { id: "q1", type: "choice", prompt: "a", options: ["A", "B"] },
        { id: "q2", type: "choice", prompt: "b", options: ["A", "B"] },
      ],
    });
    validateResponse(response);
    assert.equal(response.answers[0].status, "abstain");
    assert.equal(response.answers[0].choice, null);
    assert.equal(response.answers[0].abstain_reason, "confidence below threshold");
    assert.equal(response.answers[1].abstain_reason, "input unverifiable");
  });
  test("backend throwing propagates explicit error with no fallback decision", async (t) => {
    void t;
    const backend = new SyntheticThrowingBackend("synthetic backend error: weights not loaded");
    await assert.rejects(async () => backend.decide({
      state: "x", questions: [{ id: "q", type: "choice", prompt: "x", options: ["A", "B"] }],
    }), /synthetic backend error/);
  });
  test("SyntheticOkBackend produces a validatable response", async (t) => {
    void t;
    const { validateResponse } = await import("../src/index.ts");
    const backend = new SyntheticOkBackend(["A", "B"]);
    const response = await backend.decide({
      state: "ok", questions: [
        { id: "q1", type: "choice", prompt: "p1", options: ["A", "B"] },
        { id: "q2", type: "choice", prompt: "p2", options: ["A", "B"] },
      ],
    });
    validateResponse(response);
    assert.equal(response.answers[0].choice, "A");
    assert.equal(response.answers[1].choice, "B");
    assert.equal(response.answers.every((a) => a.status === "ok"), true);
  });
});

describe("mcp.ts: manifest resource exposes metadata only (no assets, no hashes)", () => {
  test("metadata resource omits files, summary, bundle_loaded_from absolute path, and file hashes", async (t) => {
    void t;
    const { bundleManifestToMetadata } = await import("../src/mcp.ts");
    const manifest = syntheticManifest();
    const md = bundleManifestToMetadata(manifest, "/tmp/synthetic-path");
    assert.equal(md.schema_version, "vons.bundle.metadata/v1");
    assert.equal(md.backend, "direct");
    assert.equal(md.option_count, 2);
    assert.equal(md.sequence_length, 32);
    assert.equal("bundle_loaded_from" in md, false);
    assert.equal("files" in md, false);
    assert.equal("summary" in md, false);
    assert.equal("sha256" in md, false);
  });
  test("optional metadata fields default to null when missing", async (t) => {
    void t;
    const { bundleManifestToMetadata } = await import("../src/mcp.ts");
    const manifest = syntheticManifest();
    manifest.metadata = { backend: "diffusion" };
    const md = bundleManifestToMetadata(manifest, "/tmp/x");
    assert.equal(md.option_count, null);
    assert.equal(md.sequence_length, null);
    assert.equal(md.model_id, null);
    assert.equal(md.backend, "diffusion");
  });
});

describe("mcp-cli.ts: CLI argument handling", () => {
  test("parseCliArgs captures --provider webgpu", async (t) => {
    void t;
    const { parseCliArgs } = await import("../src/mcp-cli.ts");
    const args = parseCliArgs(["--bundle", "/x", "--provider", "webgpu"]);
    assert.equal(args.provider, "webgpu");
    assert.equal(args.bundle, "/x");
  });
  test("parseCliArgs rejects unknown options with exit code 2", async (t) => {
    void t;
    const cliPath = resolve(import.meta.dirname, "..", "src", "mcp-cli.ts");
    const result = spawnSync(
      process.execPath,
      ["--experimental-strip-types", cliPath, "--bundle", "/x", "--unknown-flag"],
      { encoding: "utf8", timeout: 15000 },
    );
    assert.ok(result.status !== 0);
  });
  test("missing --bundle prints usage to stderr with 'no auto-discover' notice and non-zero status", async (t) => {
    void t;
    const cliPath = resolve(import.meta.dirname, "..", "src", "mcp-cli.ts");
    const result = spawnSync(
      process.execPath,
      ["--experimental-strip-types", cliPath],
      { encoding: "utf8", timeout: 15000 },
    );
    assert.ok(result.status !== 0, `expected non-zero exit, got ${result.status}`);
    assert.ok(/--bundle.*required/.test(result.stderr || ""), "stderr must mention required --bundle");
    assert.ok(/no auto-discover/.test(result.stderr || ""), "stderr must state that no auto-discovery is performed");
    assert.equal(/initialize|jsonrpc/i.test(result.stdout || ""), false, "no MCP stdout on usage error");
  });
  test("--help prints usage to stdout and exits 0 with constraint disclaimers", async (t) => {
    void t;
    const cliPath = resolve(import.meta.dirname, "..", "src", "mcp-cli.ts");
    const result = spawnSync(
      process.execPath,
      ["--experimental-strip-types", cliPath, "--help"],
      { encoding: "utf8", timeout: 15000 },
    );
    assert.equal(result.status, 0);
    assert.ok(/USAGE:/.test(result.stdout));
    assert.ok(/CONSTRAINTS/.test(result.stdout));
    assert.ok(/no auto-discovery|No HTTP listener|never authorizes|no silent fallback/.test(result.stdout));
  });
  test("--provider webgpu + --bundle (valid) exits with WebGPU rejection code", async (t) => {
    void t;
    const manifest = syntheticManifest();
    const { dir } = writeSyntheticBundleDir(manifest);
    try {
      const cliPath = resolve(import.meta.dirname, "..", "src", "mcp-cli.ts");
      const result = spawnSync(
        process.execPath,
        ["--experimental-strip-types", cliPath, "--bundle", dir, "--provider", "webgpu"],
        { encoding: "utf8", timeout: 15000 },
      );
      assert.equal(result.status, 4);
      assert.ok(/WebGPU adapter|cannot obtain a browser WebGPU/.test(result.stderr || ""));
    } finally { rmSync(dir, { recursive: true }); }
  });
  test("invalid bundle path exits with explicit error, no fallback, no MCP stdout traffic", async (t) => {
    void t;
    const cliPath = resolve(import.meta.dirname, "..", "src", "mcp-cli.ts");
    const notExist = join(tmpdir(), "vons-mcp-missing-bundle-12345");
    const result = spawnSync(
      process.execPath,
      ["--experimental-strip-types", cliPath, "--bundle", notExist],
      { encoding: "utf8", timeout: 15000 },
    );
    assert.equal(result.status, 3);
    assert.ok(/bundle load failed|not found/.test(result.stderr || ""));
    assert.ok(/No silent fallback/.test(result.stderr || ""));
    assert.equal(/jsonrpc|initialize/i.test(result.stdout || ""), false);
  });
  test("URI-scheme rejection truncates long --bundle values at 120 chars with (truncated) marker; sentinel tail beyond index 120 never leaks", async (t) => {
    void t;
    const cliPath = resolve(import.meta.dirname, "..", "src", "mcp-cli.ts");
    const TAIL_SENTINEL_1841 = "TAIL_SENTINEL_1841_zircon_trilobite_tip";
    const BOUNDARY_CHAR_119 = "\u2694";
    const httpsHeader = "https://example.com/prefix";
    const fillBeforeBoundary = "q".repeat(Math.max(0, 119 - httpsHeader.length));
    const tailAfterBoundary = "r".repeat(400);
    const longValue = httpsHeader + fillBeforeBoundary + BOUNDARY_CHAR_119 + tailAfterBoundary + TAIL_SENTINEL_1841;
    assert.equal(
      longValue[119],
      BOUNDARY_CHAR_119,
      `BOUNDARY_CHAR_119 must live at raw index 119 so the test exercises the exact truncation boundary. ` +
        `Actual longValue[119]=${JSON.stringify(longValue[119])}, ` +
        `fillBeforeBoundary.length=${fillBeforeBoundary.length}, ` +
        `httpsHeader.length=${httpsHeader.length}, ` +
        `longValue.length=${longValue.length}.`,
    );
    assert.ok(
      longValue.indexOf(TAIL_SENTINEL_1841) >= 120,
      `tail sentinel must begin at index >= 120 to test truncation; actual start=${longValue.indexOf(TAIL_SENTINEL_1841)} value.length=${longValue.length}`,
    );
    const longResult = spawnSync(
      process.execPath,
      ["--experimental-strip-types", cliPath, "--bundle", longValue],
      { encoding: "utf8", timeout: 15000 },
    );
    assert.equal(longResult.status, 3, `long https:// --bundle must exit=3 actual=${longResult.status} stderr=${JSON.stringify(longResult.stderr?.slice(0, 200))}`);
    assert.ok(
      (longResult.stderr || "").includes("URI scheme 'https://' is not supported"),
      `long https:// --bundle stderr must include the exact diagnostic: URI scheme 'https://' is not supported. stderr=${JSON.stringify(longResult.stderr?.slice(0, 220))}`,
    );
    assert.ok(/'https':\/\/|https:\/\/|scheme.*https/.test(longResult.stderr || ""), "stderr must mention https or scheme=https for long --bundle");
    assert.ok(/\(truncated\)/.test(longResult.stderr || ""), `stderr must contain literal (truncated) marker for long values. stderr=${JSON.stringify(longResult.stderr?.slice(0, 220))}`);
    assert.ok(
      (longResult.stderr || "").includes(BOUNDARY_CHAR_119),
      `stderr must include the distinctive BOUNDARY_CHAR_119 at raw index 119 '${BOUNDARY_CHAR_119}' so truncation is proven to include the boundary byte. stderr=${JSON.stringify(longResult.stderr?.slice(0, 220))}`,
    );
    assert.equal(
      (longResult.stdout || "").includes(TAIL_SENTINEL_1841),
      false,
      `long-value tail sentinel '${TAIL_SENTINEL_1841}' must NOT appear on stdout; got stdout=${JSON.stringify(longResult.stdout?.slice(0, 200))}`,
    );
    assert.equal(
      (longResult.stderr || "").includes(TAIL_SENTINEL_1841),
      false,
      `long-value tail sentinel '${TAIL_SENTINEL_1841}' must NOT appear on stderr even truncated form allowed 120 chars. stderr=${JSON.stringify((longResult.stderr || "").slice(0, 300))}`,
    );
    assert.equal(
      /bundle directory not found/i.test(longResult.stderr || ""),
      false,
      `long https:// --bundle must NOT produce misleading 'bundle directory not found' text`,
    );
    assert.equal(
      longResult.stdout,
      "",
      `stdout must be exactly empty for long-value URI rejection (no MCP no whitespace). actual stdout=${JSON.stringify(longResult.stdout)}`,
    );
    const shortPad = "d".repeat(Math.max(0, 120 - "https://example.com/s=".length));
    const exactly120 = "https://example.com/s=" + shortPad;
    assert.equal(
      exactly120.length,
      120,
      `this sub-case requires a raw --bundle value whose length is exactly 120 characters to prove (truncated) is omitted; actual length=${exactly120.length}`,
    );
    const exactResult = spawnSync(
      process.execPath,
      ["--experimental-strip-types", cliPath, "--bundle", exactly120],
      { encoding: "utf8", timeout: 15000 },
    );
    assert.equal(exactResult.status, 3, `exactly-120-char https:// --bundle must also exit=3 actual=${exactResult.status}`);
    assert.ok(
      (exactResult.stderr || "").includes("URI scheme 'https://' is not supported"),
      `exactly-120-char https:// --bundle stderr must include the exact diagnostic: URI scheme 'https://' is not supported. stderr=${JSON.stringify(exactResult.stderr?.slice(0, 220))}`,
    );
    assert.equal(
      /\(truncated\)/.test(exactResult.stderr || ""),
      false,
      `exactly-120-char value must NOT print '(truncated)' (only values strictly longer should get the marker). stderr=${JSON.stringify(exactResult.stderr?.slice(0, 220))}`,
    );
    assert.ok(
      /scheme|https|not supported|--bundle rejected/.test(exactResult.stderr || ""),
      `exactly-120-char value must still produce the explicit URI-scheme error. stderr=${JSON.stringify(exactResult.stderr?.slice(0, 220))}`,
    );
    assert.equal(
      exactResult.stdout,
      "",
      `stdout must be exactly empty for exact-120-char URI rejection (no MCP no whitespace). actual stdout=${JSON.stringify(exactResult.stdout)}`,
    );
  });
  test("rejects URI-scheme --bundle values (https://, file://, ftp://, data:) with exit 3, explicit scheme error, NEVER 'bundle directory not found'", async (t) => {
    void t;
    const cliPath = resolve(import.meta.dirname, "..", "src", "mcp-cli.ts");
    const cases = [
      ["https://example.com/bundle", "https"],
      ["file:///tmp/nonexistent-vons-mcp-xyz", "file"],
      ["ftp://files.internal/pub/bundle-v1", "ftp"],
      ["data:text/plain,rawbundlecontents", "data"],
    ];
    for (const [value, scheme] of cases) {
      const result = spawnSync(
        process.execPath,
        ["--experimental-strip-types", cliPath, "--bundle", value],
        { encoding: "utf8", timeout: 15000 },
      );
      assert.equal(result.status, 3, `exit=3 for scheme=${scheme} value=${value}`);
      assert.ok(
        /--bundle rejected|URI scheme|not supported/.test(result.stderr || ""),
        `stderr must contain explicit URI-scheme rejection for ${scheme}: got stderr=${JSON.stringify(result.stderr)}`,
      );
      assert.ok(
        new RegExp(`scheme.*'${scheme}://'|'${scheme}://'`).test(result.stderr || ""),
        `stderr must name the rejected scheme '${scheme}://' for value=${value}`,
      );
      assert.equal(
        /bundle directory not found/i.test(result.stderr || ""),
        false,
        `stderr must NOT contain the misleading 'bundle directory not found' string for scheme input ${value}`,
      );
      assert.equal(/jsonrpc|initialize/i.test(result.stdout || ""), false);
    }
  });
  test("supports shell-resolved ../ relative --bundle: synthetic bundle accessed via ../<basename>/./sibling/../<basename> succeeds: writes bundle-loaded banner to stderr, no scheme-rejection, no not-found error", async (t) => {
    void t;
    const manifest = syntheticManifest();
    const { dir } = writeSyntheticBundleDir(manifest);
    try {
      const cliPath = resolve(import.meta.dirname, "..", "src", "mcp-cli.ts");
      const parent = resolve(dir, "..");
      const basename = dir.slice(parent.length + 1) || "";
      const validDotDotRelative = `${parent}/./sibling/../${basename}`;
      const result = spawnSync(
        process.execPath,
        ["--experimental-strip-types", cliPath, "--bundle", validDotDotRelative, "--provider", "wasm"],
        { encoding: "utf8", timeout: 6000, input: "", maxBuffer: 2 * 1024 * 1024 },
      );
      assert.ok(
        result.status !== 3,
        `valid ../ bundle path must NOT exit with code=3 (scheme or missing-dir error). Got status=${result.status} stderr=${JSON.stringify(result.stderr?.slice(0, 300))}`,
      );
      assert.equal(
        /--bundle rejected|URI scheme|bundle directory not found/.test(result.stderr || ""),
        false,
        `valid ../ bundle must NOT be rejected as URI scheme or 'bundle directory not found'. Got stderr=${JSON.stringify(result.stderr?.slice(0, 300))}`,
      );
      assert.ok(
        /bundle loaded:|manifest\.sha256=|Starting stdio MCP server|provider=wasm/.test(result.stderr || ""),
        `valid ../ bundle must succeed and write the 'bundle loaded:' / manifest.sha256 / provider=wasm / Starting stdio MCP banner to stderr. Got stderr=${JSON.stringify(result.stderr?.slice(0, 400))}`,
      );
    } finally {
      rmSync(dir, { recursive: true });
    }
  });
});

describe("mcp.ts: MCP initialize / tools/list / tools/call stdio round trip (synthetic backend, injected)", () => {
  function createLoopbackStdio() {
    const serverStdin = new PassThrough();
    const serverStdout = new PassThrough();
    let outBuf = "";
    const outLines = [];
    let resolver = null;
    serverStdout.setEncoding("utf8");
    serverStdout.on("data", (chunk) => {
      outBuf += chunk;
      let idx;
      while ((idx = outBuf.indexOf("\n")) !== -1) {
        const line = outBuf.slice(0, idx);
        outBuf = outBuf.slice(idx + 1);
        if (line.length > 0) outLines.push(line);
        if (resolver) {
          const res = resolver;
          resolver = null;
          res(outLines.shift());
        }
      }
    });
    serverStdout.on("end", () => {
      if (resolver) {
        const res = resolver;
        resolver = null;
        res(null);
      }
    });
    const nextLine = (timeoutMs = 8000) => {
      if (outLines.length > 0) return Promise.resolve(outLines.shift());
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          resolver = null;
          reject(new Error(`timeout waiting for MCP message on stdout (buffer=${JSON.stringify(outBuf)})`));
        }, timeoutMs);
        resolver = (v) => {
          clearTimeout(timer);
          resolve(v);
        };
      });
    };
    const nextMessage = async (timeoutMs) => {
      const line = await nextLine(timeoutMs);
      if (!line) throw new Error("stream ended while waiting for message");
      return JSON.parse(line);
    };
    const writeLine = (line) => {
      serverStdin.write(line + "\n");
    };
    const end = () => {
      serverStdin.end();
    };
    return { serverStdin, serverStdout, writeLine, end, nextMessage, nextLine, serverStdoutLines: outLines };
  }

  function mcpRequest(method, params, id) {
    return JSON.stringify({ jsonrpc: "2.0", id, method, params });
  }

  test("initialize returns server capabilities with tool + resource, then list tools, then call vons_decide (SyntheticOkBackend)", async (t) => {
    void t;
    const { runStdioMcpServer } = await import("../src/mcp.ts");
    const manifest = syntheticManifest();
    const backend = new SyntheticOkBackend(["A", "B"]);
    const loop = createLoopbackStdio();
    const ac = new AbortController();
    const serverPromise = runStdioMcpServer({
      backend,
      manifest,
      manifestSha256: "a".repeat(64),
      bundleRoot: "/tmp/roundtrip-bundle",
      stdin: loop.serverStdin,
      stdout: loop.serverStdout,
      signal: ac.signal,
    }).catch((e) => e);

    loop.writeLine(
      mcpRequest(
        "initialize",
        {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "mcp-pilot-test", version: "0.1.0" },
        },
        1,
      ),
    );
    const initResp = await loop.nextMessage(8000);
    assert.equal(initResp.jsonrpc, "2.0");
    assert.equal(initResp.id, 1);
    assert.ok(initResp.result, "initialize result missing");
    assert.ok(initResp.result.capabilities?.tools, "initialize capabilities.tools missing");
    assert.equal(initResp.result.serverInfo?.name, "vons-local-mcp");

    loop.writeLine(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }));

    loop.writeLine(mcpRequest("tools/list", {}, 2));
    const listResp = await loop.nextMessage(6000);
    assert.equal(listResp.id, 2);
    assert.ok(Array.isArray(listResp.result?.tools));
    const toolNames = listResp.result.tools.map((tool) => tool.name);
    assert.ok(toolNames.includes("vons_decide"), `expected vons_decide in tools list, got ${toolNames.join(",")}`);

    const req = {
      state: "hello round trip",
      questions: [
        { id: "q1", type: "choice", prompt: "pick", options: ["A", "B"] },
        { id: "q2", type: "choice", prompt: "pick", options: ["A", "B"] },
      ],
      backend: "direct",
    };
    loop.writeLine(mcpRequest("tools/call", { name: "vons_decide", arguments: req }, 3));
    const callResp = await loop.nextMessage(8000);
    assert.equal(callResp.id, 3);
    assert.ok(callResp.result?.content && callResp.result.content.length > 0, "content missing");
    const parsed = JSON.parse(callResp.result.content[0].text);
    assert.equal(!!parsed.error, false, `unexpected MCP error: ${JSON.stringify(parsed)}`);
    assert.equal(parsed.answers.length, 2);
    assert.equal(parsed.answers[0].question_id, "q1");
    assert.equal(parsed.answers[0].choice, "A");
    assert.equal(parsed.answers[1].question_id, "q2");
    assert.equal(parsed.answers[1].choice, "B");

    ac.abort();
    loop.end();
    const shutdownOrError = await serverPromise;
    if (shutdownOrError instanceof Error && !/abort|close|ended|aborted/i.test(shutdownOrError.message)) {
      throw shutdownOrError;
    }
  });

  test("rejects backend answers whose question IDs do not exactly match the request", async (t) => {
    void t;
    const { runStdioMcpServer } = await import("../src/mcp.ts");
    const backend = {
      async decide() {
        return {
          answers: [
            { question_id: "q1", status: "ok", choice: "A", probabilities: [{ option: "A", probability: 1 }], confidence: 1 },
            { question_id: "unexpected", status: "ok", choice: "B", probabilities: [{ option: "B", probability: 1 }], confidence: 1 },
          ],
          backend: "direct",
          model_id: "synthetic-mismatched-question-ids",
        };
      },
    };
    const loop = createLoopbackStdio();
    const ac = new AbortController();
    const serverPromise = runStdioMcpServer({
      backend,
      stdin: loop.serverStdin,
      stdout: loop.serverStdout,
      signal: ac.signal,
    }).catch((error) => error);

    loop.writeLine(
      mcpRequest(
        "initialize",
        { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "mcp-id-mismatch", version: "0.1.0" } },
        401,
      ),
    );
    assert.equal((await loop.nextMessage(8000)).id, 401);
    loop.writeLine(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }));
    loop.writeLine(
      mcpRequest(
        "tools/call",
        {
          name: "vons_decide",
          arguments: {
            state: "synthetic request with two questions",
            questions: [
              { id: "q1", type: "choice", prompt: "first", options: ["A", "B"] },
              { id: "q2", type: "choice", prompt: "second", options: ["A", "B"] },
            ],
          },
        },
        402,
      ),
    );
    const callResp = await loop.nextMessage(8000);
    assert.equal(callResp.id, 402);
    assert.equal(callResp.result?.isError, true);
    assert.match(JSON.parse(callResp.result.content[0].text).error, /question IDs must exactly match/);

    ac.abort();
    loop.end();
    await serverPromise;
  });

  test("stderr logger messages never include request, response, or backend error content", async (t) => {
    void t;
    const { runStdioMcpServer } = await import("../src/mcp.ts");
    const questionSentinel = "QUESTION_SENTINEL_9242_gnevas";
    const stateSentinel = "STATE_SENTINEL_7161_zibex";
    const promptSentinel = "PROMPT_SENTINEL_8311_ambrom";
    const distinctiveErrorName = "ZeppelinRudderError_48291";
    const distinctiveErrorMessage = "kraken_barnacle_stern_918_wobble_fjord";
    const distinctiveErrorClass = class extends Error {
      constructor() {
        super(distinctiveErrorMessage);
        this.name = distinctiveErrorName;
      }
    };
    const sentinels = [questionSentinel, stateSentinel, promptSentinel, distinctiveErrorName, distinctiveErrorMessage];
    const logMessages = [];
    const backend = {
      async decide(request) {
        if (request.questions.some((q) => String(q.id) === questionSentinel || String(q.prompt) === promptSentinel)) {
          throw new distinctiveErrorClass();
        }
        throw new distinctiveErrorClass();
      },
    };
    const loop = createLoopbackStdio();
    const ac = new AbortController();
    const serverPromise = runStdioMcpServer({
      backend,
      stdin: loop.serverStdin,
      stdout: loop.serverStdout,
      signal: ac.signal,
      logger: (_level, message) => logMessages.push(message),
    }).catch((error) => error);

    loop.writeLine(
      mcpRequest(
        "initialize",
        { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "mcp-safe-logs", version: "0.1.0" } },
        501,
      ),
    );
    assert.equal((await loop.nextMessage(8000)).id, 501);
    loop.writeLine(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }));

    const call = async (arguments_, id) => {
      loop.writeLine(mcpRequest("tools/call", { name: "vons_decide", arguments: arguments_ }, id));
      return loop.nextMessage(8000);
    };
    const schemaValidationFailure = await call({
      state: "schema-fail-state",
      questions: [{ id: "q-schema", type: "choice", prompt: "synthetic", options: ["A", "B"], UNKNOWN_EXTRA_KEY_48291: true }],
      UNKNOWN_ROOT_KEY: true,
    }, 501);
    assert.equal(schemaValidationFailure.result?.isError ?? ("error" in schemaValidationFailure), true);

    const sharedContractFailure = await call({
      state: "valid synthetic state",
      questions: [
        { id: questionSentinel, type: "choice", prompt: "first", options: ["A", "B"] },
        { id: questionSentinel, type: "choice", prompt: "second", options: ["A", "B"] },
      ],
    }, 502);
    assert.equal(sharedContractFailure.result?.isError, true);

    const backendFailure = await call({
      state: stateSentinel,
      questions: [{ id: "q-backend", type: "choice", prompt: promptSentinel, options: ["A", "B"] }],
    }, 503);
    assert.equal(backendFailure.result?.isError, true);
    assert.match(
      backendFailure.result.content[0].text,
      /vons_decide backend raised a runtime error/,
      "backend failure → generic sanitized stdout error",
    );
    assert.equal(
      backendFailure.result.content[0].text.includes("synthetic failure echoed request"),
      false,
      "backend error must not include thrown message verbatim",
    );
    const allowedLogsExactly = new Set([
      "invalid tool arguments rejected by strict schema validation",
      "request rejected by shared contract validation",
      "backend inference failed; no response or fallback decision was produced",
    ]);
    for (const msg of logMessages) {
      assert.ok(allowedLogsExactly.has(msg), `log message must be EXACTLY one of the three constant strings; got: ${JSON.stringify(msg)}`);
    }
    const requiredLogsExactly = [
      "request rejected by shared contract validation",
      "backend inference failed; no response or fallback decision was produced",
    ];
    for (const required of requiredLogsExactly) {
      assert.ok(
        logMessages.some((m) => m === required),
        `missing required EXACT log string: ${required}. Actual log messages: ${JSON.stringify(logMessages)}`,
      );
    }
    for (const value of sentinels) {
      assert.equal(
        backendFailure.result.content[0].text.includes(value),
        false,
        `stdout error content must not leak sentinel ${value}`,
      );
      assert.equal(
        logMessages.join("\n").includes(value),
        false,
        `stderr log must not leak sentinel ${value}`,
      );
      assert.equal(
        sharedContractFailure.result.content[0].text.includes(value),
        false,
        `stdout shared-contract-failure content must not leak sentinel ${value}`,
      );
    }

    ac.abort();
    loop.end();
    await serverPromise;
  });

  test("malformed vons_decide arguments round trip → explicit MCP error content, no coerced choice", async (t) => {
    void t;
    const { runStdioMcpServer } = await import("../src/mcp.ts");
    const manifest = syntheticManifest();
    const backend = new SyntheticOkBackend(["X"]);
    const loop = createLoopbackStdio();
    const ac = new AbortController();
    const serverPromise = runStdioMcpServer({
      backend,
      manifest,
      manifestSha256: "b".repeat(64),
      bundleRoot: "/tmp/err-bundle",
      stdin: loop.serverStdin,
      stdout: loop.serverStdout,
      signal: ac.signal,
    }).catch((e) => e);

    loop.writeLine(
      mcpRequest(
        "initialize",
        {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "mcp-pilot-test-err", version: "0.1.0" },
        },
        1,
      ),
    );
    const initResp = await loop.nextMessage(8000);
    assert.equal(initResp.id, 1);

    loop.writeLine(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }));

    const badReq = {
      state: "x",
      questions: [
        { id: "q0", type: "choice", prompt: "p", options: ["A", "B"], EXTRA_KEY_SHOULD_BE_REJECTED: true },
      ],
    };
    loop.writeLine(mcpRequest("tools/call", { name: "vons_decide", arguments: badReq }, 2));
    const callResp = await loop.nextMessage(8000);
    assert.equal(callResp.jsonrpc, "2.0");
    assert.equal(callResp.id, 2);
    const hasErrorContent =
      (callResp.result && callResp.result.content && callResp.result.content.length > 0 && callResp.result.isError) ||
      typeof callResp.error?.message === "string";
    assert.ok(hasErrorContent, `expected an MCP error, got ${JSON.stringify(callResp).slice(0, 200)}`);

    ac.abort();
    loop.end();
    await serverPromise;
  });

  test("tools/list exposes exactly one tool (only vons_decide) and no KASI or extra tools leak via protocol", async (t) => {
    void t;
    const { runStdioMcpServer } = await import("../src/mcp.ts");
    const manifest = syntheticManifest();
    const backend = new SyntheticOkBackend(["X", "Y"]);
    const loop = createLoopbackStdio();
    const ac = new AbortController();
    const serverPromise = runStdioMcpServer({
      backend,
      manifest,
      manifestSha256: "c".repeat(64),
      bundleRoot: "/tmp/exact-tools-bundle",
      stdin: loop.serverStdin,
      stdout: loop.serverStdout,
      signal: ac.signal,
    }).catch((e) => e);

    loop.writeLine(
      mcpRequest(
        "initialize",
        {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "exact-tools-round-trip", version: "0.1.0" },
        },
        101,
      ),
    );
    const initResp = await loop.nextMessage(8000);
    assert.equal(initResp.id, 101);
    assert.equal(initResp.result.serverInfo.name, "vons-local-mcp");

    loop.writeLine(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }));

    loop.writeLine(mcpRequest("tools/list", {}, 102));
    const listResp = await loop.nextMessage(6000);
    assert.equal(listResp.id, 102);
    const tools = listResp.result.tools;
    assert.equal(Array.isArray(tools), true);
    assert.equal(
      tools.length,
      1,
      `exactly one tool must be exposed; got tools=[${tools.map((x) => x.name).join(",")}]`,
    );
    assert.equal(tools[0].name, "vons_decide");
    assert.ok(
      typeof tools[0].inputSchema === "object" && tools[0].inputSchema !== null,
      "vons_decide tool must advertise an inputSchema",
    );

    ac.abort();
    loop.end();
    await serverPromise;
  });

  test("resources/read returns manifest metadata without requiring a bundleRoot or leaking local paths", async (t) => {
    void t;
    const { runStdioMcpServer } = await import("../src/mcp.ts");
    const manifest = syntheticManifest();
    manifest.metadata.backend = "diffusion";
    manifest.metadata.model_id = "synthetic-regression-model-v0";
    manifest.metadata.option_count = 4;
    manifest.metadata.sequence_length = 2048;
    const backend = new SyntheticOkBackend(["A", "B", "C", "D"]);
    const loop = createLoopbackStdio();
    const ac = new AbortController();
    const serverPromise = runStdioMcpServer({
      backend,
      manifest,
      manifestSha256: "d".repeat(64),
      stdin: loop.serverStdin,
      stdout: loop.serverStdout,
      signal: ac.signal,
    }).catch((e) => e);

    loop.writeLine(
      mcpRequest(
        "initialize",
        {
          protocolVersion: "2024-11-05",
          capabilities: { resources: {} },
          clientInfo: { name: "resources-read-metadata", version: "0.1.0" },
        },
        201,
      ),
    );
    const initResp = await loop.nextMessage(8000);
    assert.equal(initResp.id, 201);
    assert.ok(initResp.result.capabilities?.resources, "server must declare resources capability");

    loop.writeLine(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }));

    loop.writeLine(
      mcpRequest("resources/read", { uri: "vons://bundle/manifest/metadata" }, 202),
    );
    const readResp = await loop.nextMessage(8000);
    assert.equal(readResp.jsonrpc, "2.0");
    assert.equal(readResp.id, 202);
    assert.ok(Array.isArray(readResp.result?.contents), "resources/read must return contents[] array");
    const entry = readResp.result.contents.find((c) => c.mimeType === "application/json");
    assert.ok(entry, `application/json entry must exist in metadata resource response`);
    const md = JSON.parse(entry.text);
    assert.equal(md.schema_version, "vons.bundle.metadata/v1");
    assert.equal(md.backend, "diffusion");
    assert.equal(md.model_id, "synthetic-regression-model-v0");
    assert.equal(md.option_count, 4);
    assert.equal(md.sequence_length, 2048);
    assert.equal("bundle_loaded_from" in md, false, "bundle_loaded_from absolute local path must not appear in metadata");
    assert.equal("files" in md, false);
    assert.equal("hashes" in md, false);

    ac.abort();
    loop.end();
    await serverPromise;
  });

  test("tools/call with SyntheticAbstainBackend preserves abstention choice=null + reason exact via round-trip protocol", async (t) => {
    void t;
    const { runStdioMcpServer } = await import("../src/mcp.ts");
    const manifest = syntheticManifest();
    const reasons = [
      "prompt does not contain sufficient grounding for an unambiguous pick",
      "backend abstains by synthetic injection: no options legible in question q2",
    ];
    const backend = new SyntheticAbstainBackend(reasons);
    const loop = createLoopbackStdio();
    const ac = new AbortController();
    const serverPromise = runStdioMcpServer({
      backend,
      manifest,
      manifestSha256: "e".repeat(64),
      bundleRoot: "/tmp/abstain-round-trip-bundle",
      stdin: loop.serverStdin,
      stdout: loop.serverStdout,
      signal: ac.signal,
    }).catch((e) => e);

    loop.writeLine(
      mcpRequest(
        "initialize",
        {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "abstain-round-trip", version: "0.1.0" },
        },
        301,
      ),
    );
    const initResp = await loop.nextMessage(8000);
    assert.equal(initResp.id, 301);
    assert.equal(initResp.result.serverInfo.name, "vons-local-mcp");

    loop.writeLine(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }));

    const req = {
      state: "abstain driver — synthetic backend must abstain exact",
      questions: [
        { id: "q1", type: "choice", prompt: "insufficient context 1", options: ["A", "B", "C"] },
        { id: "q2", type: "choice", prompt: "insufficient context 2", options: ["A", "B"] },
      ],
      backend: "direct",
    };
    loop.writeLine(mcpRequest("tools/call", { name: "vons_decide", arguments: req }, 302));
    const callResp = await loop.nextMessage(8000);
    assert.equal(callResp.id, 302);
    assert.ok(callResp.result?.content && callResp.result.content.length > 0, "abstention content missing");
    assert.notEqual(callResp.result.isError, true, "abstention is a valid decision response, not isError");
    const parsed = JSON.parse(callResp.result.content[0].text);
    assert.equal(!!parsed.error, false, `must not return .error for abstention: ${JSON.stringify(parsed)}`);
    assert.equal(parsed.answers.length, 2);
    assert.equal(parsed.answers[0].question_id, "q1");
    assert.equal(parsed.answers[0].status, "abstain");
    assert.equal(parsed.answers[0].choice, null);
    assert.equal(parsed.answers[0].abstain_reason, reasons[0]);
    assert.deepEqual(parsed.answers[0].probabilities, []);
    assert.equal(parsed.answers[1].question_id, "q2");
    assert.equal(parsed.answers[1].status, "abstain");
    assert.equal(parsed.answers[1].choice, null);
    assert.equal(parsed.answers[1].abstain_reason, reasons[1]);
    assert.deepEqual(parsed.answers[1].probabilities, []);

    ac.abort();
    loop.end();
    await serverPromise;
  });
});

describe("TUI findings pass: M2 positional IDs, M3 manifest realpath, M4 docs/diag", () => {
  test("M4 README provider wording says wasm is supported; webgpu explicitly rejected; no stale 'accepted as metadata' wording; client examples do not include --manifest-sha256", async (t) => {
    void t;
    const { readFileSync } = await import("node:fs");
    const { join } = await import("node:path");
    const readme = readFileSync(join(process.cwd(), "README.md"), "utf-8");
    assert.equal(
      /accepted as metadata/.test(readme),
      false,
      "README must not contain stale 'accepted as metadata' wording for --provider webgpu",
    );
    assert.ok(
      /wasm[\s\S]{0,60}supported value/.test(readme) || /wasm must be explicit/.test(readme),
      "README must describe --provider wasm as the supported value",
    );
    assert.ok(
      /webgpu[\s\S]{0,60}explicitly rejected/.test(readme) || /webgpu[\s\S]{0,60}exit code 4/.test(readme),
      "README must describe --provider webgpu as explicitly rejected with exit code 4",
    );
    assert.equal(
      /--manifest-sha256\s+[0-9a-f]{64}|manifestSha256:\s*["'][0-9a-f]{64}["']/.test(readme),
      false,
      "README client examples never include a full 64-char --manifest-sha256 hash literal",
    );
  });

  test("M4 source scan: no unrelated sorts in owned MCP sources; no request/response json echoes in stderr log writes; manifest sha256 banner uses short slice only", async (t) => {
    void t;
    const { readFileSync } = await import("node:fs");
    const { join } = await import("node:path");
    const mcpSource = readFileSync(join(process.cwd(), "src", "mcp.ts"), "utf-8");
    const cliSource = readFileSync(join(process.cwd(), "src", "mcp-cli.ts"), "utf-8");
    const allOwned = mcpSource + "\n" + cliSource;

    assert.equal(
      /\.(sort|toSorted)\s*\(/.test(allOwned),
      false,
      "No unrelated .sort() / .toSorted() calls in MCP-owned sources",
    );
    assert.equal(
      /log\s*\([^)]*JSON\s*\.stringify\s*\(\s*(req|request|res|response)\s*\)/.test(allOwned),
      false,
      "No log() calls ever JSON.stringify the request or response object (would leak to stderr)",
    );
    assert.ok(
      /manifest_sha256.*slice\s*\(\s*0\s*,\s*12\s*\)|banner.*slice\s*\(\s*0\s*,\s*12\s*\)|sha256.*slice\s*\(\s*0\s*,\s*12\s*\)/.test(cliSource),
      "mcp-cli banner uses manifest.sha256.slice(0,12) short digest only, never full",
    );
  });
});


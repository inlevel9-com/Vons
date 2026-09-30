import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { requiredFiles, safePath, digest, verifyBundle, importBundle, makeRequest } from "../extension/bundle.ts";
import { modelStorage } from "../extension/storage.ts";

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
async function waitForState(predicate) {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("Timed out waiting for the panel restore state.");
}
async function sharedFixture() {
  const payloads = new Map([
    ["encoder.onnx", new TextEncoder().encode("encoder")], ["encoder.onnx.data", new TextEncoder().encode("encoder weights")],
    ["direct.onnx", new TextEncoder().encode("direct")], ["direct.onnx.data", new TextEncoder().encode("direct weights")],
    ["diffusion.onnx", new TextEncoder().encode("diffusion")], ["diffusion.onnx.data", new TextEncoder().encode("diffusion weights")],
    ["tokenizer/tokenizer.json", new TextEncoder().encode("tokenizer")],
    ["tokenizer/tokenizer_config.json", new TextEncoder().encode("tokenizer config")],
    ["config.json", new TextEncoder().encode(JSON.stringify({ max_tokens: 512, max_options: 32, diffusion_candidate_slots: 32,
      default_abstain_threshold: 0.55, default_answerability_threshold: 0.5 }))],
    ["calibration.json", new TextEncoder().encode("calibration is not loaded")],
  ]);
  const roles = new Map([
    ["encoder.onnx", "encoder_graph"], ["encoder.onnx.data", "encoder_weights"],
    ["direct.onnx", "direct_graph"], ["direct.onnx.data", "direct_weights"],
    ["diffusion.onnx", "diffusion_graph"], ["diffusion.onnx.data", "diffusion_weights"],
    ["tokenizer/tokenizer.json", "tokenizer"], ["tokenizer/tokenizer_config.json", "tokenizer"],
    ["config.json", "config"], ["calibration.json", "calibration"],
  ]);
  const records = [];
  const localFiles = [];
  for (const [path, bytes] of payloads) {
    const value = { path, role: roles.get(path), bytes: bytes.byteLength,
      sha256: await digest(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)),
      runtime_loaded: true, model_asset: path !== "calibration.json" };
    records.push(value);
    localFiles.push({ name: path.split("/").at(-1), size: bytes.byteLength, webkitRelativePath: `shared/${path}`,
      arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) });
  }
  const manifest = {
    schema: "vons.shared-bundle/v1", model_id: "synthetic-shared-v1",
    encoder: { graph: "encoder.onnx", outputs: ["candidate_embeddings", "pooled"] },
    heads: {
      direct: { graph: "direct.onnx", inputs: ["candidate_embeddings", "pooled", "option_mask"] },
      diffusion: { graph: "diffusion.onnx", inputs: ["pooled", "option_mask", "initial_noise"] },
    },
    files: records,
  };
  const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
  localFiles.push({ name: "manifest.json", size: manifestBytes.byteLength, webkitRelativePath: "shared/manifest.json",
    arrayBuffer: async () => manifestBytes.buffer.slice(manifestBytes.byteOffset, manifestBytes.byteOffset + manifestBytes.byteLength) });
  return { files: localFiles, manifestBytes, manifest };
}
function createPanelHarness(storedModel, failingActions = []) {
  const ids = [
    "bundle-files", "remember", "run", "cancel", "forget", "provider", "model-head", "head-choice", "decision-form",
    "status", "model-badge", "model-description", "state", "question", "options", "example",
    "result", "raw", "copy",
  ];
  const elements = new Map(ids.map((id) => [id, {
    listeners: new Map(),
    dataset: {},
    files: [],
    addEventListener(name, callback) { this.listeners.set(name, callback); },
    setAttribute() {},
    replaceChildren() {},
  }]));
  elements.get("provider").value = "wasm";
  elements.get("model-head").value = "direct";

  const persisted = { value: storedModel };
  const failures = new Set(failingActions);
  const database = {
    close() {},
    transaction() {
      const transaction = {
        objectStore() {
          return {
            get() {
              const request = {};
              queueMicrotask(() => {
                request.result = persisted.value;
                persisted.readCompleted = true;
                transaction.oncomplete?.();
              });
              return request;
            },
            put(value) {
              const request = {};
              queueMicrotask(() => {
                if (failures.has("write")) transaction.onabort?.();
                else {
                  persisted.value = value;
                  transaction.oncomplete?.();
                }
              });
              return request;
            },
            clear() {
              const request = {};
              queueMicrotask(() => {
                if (failures.has("clear")) transaction.onabort?.();
                else {
                  persisted.value = undefined;
                  transaction.oncomplete?.();
                }
              });
              return request;
            },
          };
        },
      };
      return transaction;
    },
  };

  const originalGlobals = new Map(["document", "window", "indexedDB"].map((key) => [
    key,
    Object.getOwnPropertyDescriptor(globalThis, key),
  ]));
  globalThis.document = { getElementById: (id) => elements.get(id) ?? null };
  globalThis.window = { addEventListener() {} };
  globalThis.indexedDB = {
    open() {
      const request = {};
      queueMicrotask(() => {
        request.result = database;
        request.onsuccess?.();
      });
      return request;
    },
  };

  return {
    elements,
    persisted,
    get readCompleted() { return persisted.readCompleted === true; },
    restoreGlobals() {
      for (const [key, descriptor] of originalGlobals) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete globalThis[key];
      }
    },
  };
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
test("extension imports the shared encoder with only the chosen verified head assets", async () => {
  const source = await sharedFixture();
  const directPlan = requiredFiles(source.manifestBytes.buffer, "direct");
  const diffusionPlan = requiredFiles(source.manifestBytes.buffer, "diffusion");
  assert.equal(directPlan.head, "direct");
  assert.equal(diffusionPlan.head, "diffusion");
  assert.equal(directPlan.paths.includes("direct.onnx"), true);
  assert.equal(directPlan.paths.includes("diffusion.onnx"), false);
  assert.equal(diffusionPlan.paths.includes("diffusion.onnx"), true);
  assert.equal(diffusionPlan.paths.includes("direct.onnx"), false);

  const direct = await importBundle(source.files, "direct");
  assert.equal(direct.backend, "direct");
  assert.equal(direct.head, "direct");
  assert.equal(direct.assets.some((asset) => asset.path === "diffusion.onnx"), false);
  const corruptedDirect = structuredClone(direct);
  new Uint8Array(corruptedDirect.assets.find((asset) => asset.path === "direct.onnx").bytes)[0] ^= 1;
  await assert.rejects(verifyBundle(corruptedDirect), /SHA-256 verification: direct.onnx/);

  const bundle = await importBundle(source.files, "diffusion");
  assert.equal(bundle.backend, "diffusion");
  assert.equal(bundle.head, "diffusion");
  assert.deepEqual(bundle.assets.map((asset) => asset.path).sort(), diffusionPlan.paths.sort());
  assert.equal(bundle.assets.some((asset) => asset.path === "calibration.json"), false);
  const stored = structuredClone(bundle);
  await verifyBundle(stored);
  assert.equal(stored.head, "diffusion");
  await assert.rejects(verifyBundle({ ...bundle, backend: "direct", head: "direct" }), /Missing or wrong-sized model file/);
});
test("the selected shared head survives local model storage and restore verification", async () => {
  const source = await sharedFixture();
  const bundle = await importBundle(source.files, "diffusion");
  let persisted;
  const database = {
    close() {},
    transaction() {
      const transaction = {
        objectStore() {
          return {
            put(value) {
              queueMicrotask(() => { persisted = structuredClone(value); transaction.oncomplete?.(); });
              return {};
            },
            get() {
              const request = {};
              queueMicrotask(() => { request.result = structuredClone(persisted); transaction.oncomplete?.(); });
              return request;
            },
          };
        },
      };
      return transaction;
    },
  };
  const originalIndexedDb = Object.getOwnPropertyDescriptor(globalThis, "indexedDB");
  globalThis.indexedDB = { open() {
    const request = {};
    queueMicrotask(() => { request.result = database; request.onsuccess?.(); });
    return request;
  } };
  try {
    await modelStorage("write", bundle);
    const restored = await modelStorage("read");
    assert.equal(restored.head, "diffusion");
    assert.equal(restored.backend, "diffusion");
    await verifyBundle(restored);
  } finally {
    if (originalIndexedDb) Object.defineProperty(globalThis, "indexedDB", originalIndexedDb);
    else delete globalThis.indexedDB;
  }
});
test("a verified bundle stays available for the session when its first save fails", async () => {
  const source = await sharedFixture();
  const harness = createPanelHarness(undefined, ["write"]);
  try {
    await import("../extension/panel.ts?first-save-failure");
    await waitForState(() => harness.readCompleted);

    const files = harness.elements.get("bundle-files");
    files.files = source.files;
    harness.elements.get("remember").checked = true;
    await files.listeners.get("change")();

    assert.equal(harness.persisted.value, undefined);
    assert.equal(harness.elements.get("model-badge").textContent, "direct · shared");
    assert.equal(harness.elements.get("run").disabled, false);
    assert.equal(harness.elements.get("remember").checked, false);
    assert.equal(harness.elements.get("remember").disabled, false);
    assert.equal(harness.elements.get("forget").disabled, false);
    assert.match(harness.elements.get("status").textContent, /ready for this session, but could not be saved/);
    assert.equal(harness.elements.get("status").dataset.error, "true");
  } finally {
    harness.restoreGlobals();
  }
});
test("failed save and clear keep the verified session model and previous saved model accurate", async () => {
  const source = await sharedFixture();
  const saved = await importBundle(source.files, "direct");
  const harness = createPanelHarness(saved, ["write", "clear"]);
  try {
    await import("../extension/panel.ts?retained-model-storage-failures");
    await waitForState(() => harness.elements.get("status").textContent === "Your saved model is ready. Context and decisions were not retained.");

    const files = harness.elements.get("bundle-files");
    files.files = source.files;
    await files.listeners.get("change")();
    assert.match(harness.elements.get("status").textContent, /previous saved model remains on this device/);
    assert.equal(harness.persisted.value, saved);

    const head = harness.elements.get("model-head");
    harness.elements.get("remember").checked = true;
    head.value = "diffusion";
    await head.listeners.get("change")();
    assert.equal(harness.elements.get("model-badge").textContent, "diffusion · shared");
    assert.equal(harness.elements.get("run").disabled, false);
    assert.equal(harness.elements.get("remember").checked, false);
    assert.match(harness.elements.get("status").textContent, /could not replace the saved model/);
    assert.match(harness.elements.get("status").textContent, /previous saved model remains on this device/);
    assert.equal(harness.persisted.value, saved);

    head.value = "direct";
    await head.listeners.get("change")();
    assert.equal(harness.elements.get("model-badge").textContent, "direct · shared");
    assert.equal(harness.elements.get("run").disabled, false);
    assert.equal(harness.elements.get("remember").checked, false);
    assert.match(harness.elements.get("status").textContent, /previous saved model could not be removed and remains on this device/);
    assert.equal(harness.persisted.value, saved);
  } finally {
    harness.restoreGlobals();
  }
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

test("a retained bundle that fails restore verification can still be removed", async () => {
  const source = await sharedFixture();
  const saved = await importBundle(source.files, "diffusion");
  const ids = [
    "bundle-files", "remember", "run", "cancel", "forget", "provider", "model-head", "head-choice", "decision-form",
    "status", "model-badge", "model-description", "state", "question", "options", "example",
    "result", "raw", "copy",
  ];
  const makeElements = () => {
    const elements = new Map(ids.map((id) => [id, {
      listeners: new Map(),
      dataset: {},
      addEventListener(name, callback) { this.listeners.set(name, callback); },
      setAttribute() {},
      replaceChildren() {},
    }]));
    elements.get("provider").value = "wasm";
    elements.get("model-head").value = "direct";
    return elements;
  };
  let elements = makeElements();

  const originalGlobals = new Map(["document", "window", "indexedDB"].map((key) => [
    key,
    Object.getOwnPropertyDescriptor(globalThis, key),
  ]));
  const persisted = { value: saved };
  const database = {
    close() {},
    transaction() {
      const transaction = {
        objectStore() {
          return {
            get() {
              const request = {};
              queueMicrotask(() => {
                request.result = persisted.value;
                transaction.oncomplete?.();
              });
              return request;
            },
            clear() {
              const request = {};
              queueMicrotask(() => {
                persisted.value = undefined;
                transaction.oncomplete?.();
              });
              return request;
            },
          };
        },
      };
      return transaction;
    },
  };

  globalThis.document = { getElementById: (id) => elements.get(id) ?? null };
  globalThis.window = { addEventListener() {} };
  globalThis.indexedDB = {
    open() {
      const request = {};
      queueMicrotask(() => {
        request.result = database;
        request.onsuccess?.();
      });
      return request;
    },
  };

  try {
    await import("../extension/panel.ts?restore-shared-head");
    await waitForState(() => elements.get("status").textContent === "Your saved model is ready. Context and decisions were not retained.");

    assert.equal(elements.get("remember").checked, true);
    assert.equal(elements.get("remember").disabled, false);
    assert.equal(elements.get("forget").disabled, false);
    assert.equal(elements.get("model-head").value, "diffusion");
    assert.equal(elements.get("model-badge").textContent, "diffusion · shared");

    persisted.value = { manifestPath: "../corrupt-bundle" };
    elements = makeElements();
    globalThis.document = { getElementById: (id) => elements.get(id) ?? null };
    await import("../extension/panel.ts?remove-corrupt-saved-bundle");
    await waitForState(() => /integrity check/.test(elements.get("status").textContent ?? ""));

    assert.equal(elements.get("remember").checked, true);
    assert.equal(elements.get("remember").disabled, true);
    assert.equal(elements.get("forget").disabled, false);
    assert.match(elements.get("status").textContent, /integrity check/);

    await elements.get("forget").listeners.get("click")();

    assert.equal(persisted.value, undefined);
    assert.equal(elements.get("forget").disabled, true);
    assert.equal(elements.get("remember").checked, false);
    assert.equal(elements.get("status").textContent, "Model removed from this device.");
  } finally {
    for (const [key, descriptor] of originalGlobals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
});

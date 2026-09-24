import assert from "node:assert/strict";
import test from "node:test";

const sdk = await import("../src/index.ts");

test("host policy refuses unregistered tools and requires exact high-risk consent", () => {
  const adapter = new sdk.KASIAdapter({ unlock: "high" });
  const first = { name: "unlock", arguments: { target: "lab" } };
  const second = { name: "unlock", arguments: { target: "vault" } };
  assert.equal(adapter.decide({ calls: [second], confidence: 0.9, risk: "low" }).action, "confirm");
  assert.equal(adapter.decide({ calls: [second], confidence: 0.9, risk: "low" }, new Set([sdk.callFingerprint(first)])).action, "confirm");
  assert.equal(adapter.decide({ calls: [second], confidence: 0.9, risk: "low" }, new Set([sdk.callFingerprint(second)])).action, "call");
  assert.equal(new sdk.KASIAdapter({}).decide({ calls: [{ name: "read", arguments: {} }], confidence: 0.9, risk: "low" }).action, "refuse");
});

test("request and response validators enforce the shared contract", () => {
  assert.doesNotThrow(() => sdk.validateRequest({
    state: "ready",
    questions: [{ id: "q", type: "choice", prompt: "Pick", options: ["a", "b"] }],
  }));
  assert.throws(() => sdk.validateResponse({
    answers: [{ question_id: "q", choice: "a", probabilities: [{ option: "a", probability: 0.5 }], confidence: 0.9, status: "ok" }],
    backend: "direct",
    model_id: "test",
  }));
  assert.throws(() => sdk.validateRequest({
    state: "ready",
    questions: [{ id: "q", type: "choice", prompt: "Pick", options: ["a", "b"], rubric: ["bad"] }],
  }), /only score question/);
  assert.doesNotThrow(() => sdk.validateRequest({
    state: "x".repeat(2000),
    questions: [{ id: "q", type: "choice", prompt: "Pick", options: ["a", "b"] }],
  }));
  assert.doesNotThrow(() => sdk.validateRequest({
    state: "x".repeat(200),
    questions: Array.from({ length: 8 }, (_, index) => ({
      id: `q-${index}`,
      type: "choice",
      prompt: "Pick",
      options: ["a", "b"],
    })),
  }));
  assert.doesNotThrow(() => sdk.validateRequest({
    state: "ready",
    questions: [{ id: "q", type: "choice", prompt: "Pick", options: ["x".repeat(2000), "b"] }],
  }));
  assert.throws(() => sdk.validateRequest({
    state: "ready",
    questions: [{ id: "q", type: "choice", prompt: "Pick", options: ["", "b"] }],
  }), /options must be strings/);
  assert.doesNotThrow(() => sdk.validateRequest({
    state: "ready",
    questions: [{ id: "q", type: "choice", prompt: "Pick", options: ["a", "b"] }],
  }, 8, 512));
});

test("runtime selection is explicit", () => {
  assert.equal(sdk.selectRuntime("wasm"), "wasm");
  assert.throws(() => sdk.selectRuntime("webgpu"), /WebGPU is unavailable/);
  const previousGpu = globalThis.gpu;
  globalThis.gpu = {};
  assert.equal(sdk.selectRuntime("webgpu"), "webgpu");
  if (previousGpu === undefined) delete globalThis.gpu;
  else globalThis.gpu = previousGpu;
});

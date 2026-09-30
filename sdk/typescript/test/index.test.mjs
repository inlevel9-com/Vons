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

test("call fingerprints follow RFC 8785 UTF-16 property ordering", () => {
  const argumentsWithRfcKeys = {
    "€": "Euro Sign",
    "\r": "Carriage Return",
    "\ufb33": "Hebrew Letter Dalet With Dagesh",
    "1": "One",
    "😀": "Emoji: Grinning Face",
    "\u0080": "Control",
    "ö": "Latin Small Letter O With Diaeresis",
  };
  const fingerprint = sdk.callFingerprint({ name: "read", arguments: argumentsWithRfcKeys });
  const reversed = sdk.callFingerprint({
    name: "read",
    arguments: Object.fromEntries(Object.entries(argumentsWithRfcKeys).reverse()),
  });

  assert.equal(fingerprint, String.raw`{"arguments":{"\r":"Carriage Return","1":"One","":"Control","ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign","😀":"Emoji: Grinning Face","דּ":"Hebrew Letter Dalet With Dagesh"},"name":"read"}`);
  assert.equal(reversed, fingerprint);
});

test("call fingerprints use ECMAScript number serialization and reject invalid JSON values", () => {
  const fingerprint = sdk.callFingerprint({
    name: "read",
    arguments: {
      numbers: [333333333.33333329, 4.5, 2e-3, 1e-6, 1e-7, 1424953923781206.25, -0],
      string: "€$\u000f\nA'B\"\\\\\"/",
      literals: [null, true, false],
    },
  });
  const expected = String.raw`{"arguments":{"literals":[null,true,false],"numbers":[333333333.3333333,4.5,0.002,0.000001,1e-7,1424953923781206.2,0],"string":"€$\u000f\nA'B\"\\\\\"/"},"name":"read"}`;

  assert.equal(fingerprint, expected);
  assert.throws(() => sdk.callFingerprint({ name: "read", arguments: { when: new Date(1) } }), /JSON-compatible/);
  assert.throws(() => sdk.callFingerprint({ name: "read", arguments: { value: Number.NaN } }), /JSON-compatible/);
  assert.throws(() => sdk.callFingerprint({ name: "read", arguments: { value: new Array(1) } }), /JSON-compatible/);
  assert.throws(() => sdk.callFingerprint({ name: "read", arguments: { value: Number.MAX_SAFE_INTEGER + 1 } }), /JSON-compatible/);
  assert.throws(() => sdk.callFingerprint({ name: "read", arguments: { value: "\ud800" } }), /JSON-compatible/);
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

test("request-aware response validation preserves question order and bounded options", () => {
  const request = {
    state: "ready",
    questions: [
      { id: "q1", type: "choice", prompt: "First", options: ["a", "b"] },
      { id: "q2", type: "choice", prompt: "Second", options: ["x", "y"] },
    ],
  };
  const answer = (question_id, choice, options = [{ option: choice, probability: 1 }]) => ({
    question_id,
    choice,
    probabilities: options,
    confidence: 1,
    status: "ok",
  });
  const response = (answers) => ({ answers, backend: "direct", model_id: "test" });

  assert.doesNotThrow(() => sdk.validateResponseForRequest(
    response([answer("q1", "a"), answer("q2", "x")]),
    request,
  ));
  assert.throws(() => sdk.validateResponseForRequest(
    response([answer("q1", "a")]),
    request,
  ), /exactly one item per request question/);
  assert.throws(() => sdk.validateResponseForRequest(
    response([answer("q2", "x"), answer("q1", "a")]),
    request,
  ), /IDs and order must match/);
  assert.throws(() => sdk.validateResponseForRequest(
    response([answer("unknown", "a"), answer("q2", "x")]),
    request,
  ), /IDs and order must match/);
  assert.throws(() => sdk.validateResponseForRequest(
    response([answer("q1", "outside"), answer("q2", "x")]),
    request,
  ), /choice must be one of the request options/);
  assert.throws(() => sdk.validateResponseForRequest(
    response([answer("q1", "a", [
      { option: "a", probability: 0.5 },
      { option: "outside", probability: 0.5 },
    ]), answer("q2", "x")]),
    request,
  ), /probability options must be a subset/);
});

test("request-aware validation treats implicit boolean options as true and false", () => {
  const request = {
    state: "ready",
    questions: [{ id: "q", type: "boolean", prompt: "Is this true?", options: [] }],
  };
  const response = (choice) => ({
    answers: [{
      question_id: "q",
      choice,
      probabilities: [{ option: choice, probability: 1 }],
      confidence: 1,
      status: "ok",
    }],
    backend: "direct",
    model_id: "test",
  });

  assert.doesNotThrow(() => sdk.validateResponseForRequest(response("true"), request));
  assert.doesNotThrow(() => sdk.validateResponseForRequest(response("false"), request));
  assert.throws(() => sdk.validateResponseForRequest(response("yes"), request), /request options/);
  assert.throws(() => sdk.validateResponseForRequest({
    answers: [{
      question_id: "q",
      choice: "true",
      probabilities: [
        { option: "true", probability: 0.5 },
        { option: "yes", probability: 0.5 },
      ],
      confidence: 1,
      status: "ok",
    }],
    backend: "direct",
    model_id: "test",
  }, request), /probability options must be a subset/);

  assert.doesNotThrow(() => sdk.validateResponseForRequest({
    answers: [{ question_id: "q", choice: null, probabilities: [], confidence: 0, status: "abstain", abstain_reason: "uncertain" }],
    backend: "direct",
    model_id: "test",
  }, request));
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

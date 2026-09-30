import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

const { buildParityFixtures, canonicalJson, matchesExpectedFailure, PARITY_MANIFEST_HASHES } = await import("../demo/parity.ts");
const { validateRequest } = await import("../src/index.ts");

for (const [backend, url] of [
  ["direct", new URL("../../../artifacts/pilot/bundle-manifest-v1.json", import.meta.url)],
  ["diffusion", new URL("../../../artifacts/pilot-diffusion/bundle-manifest-v1.json", import.meta.url)],
]) {
  test(`pinned ${backend} parity manifest hash matches local manifest bytes`, async (t) => {
    let bytes;
    try {
      bytes = await readFile(url);
    } catch (error) {
      if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
        t.skip(`${backend} model manifest is not included in this source-only checkout`);
        return;
      }
      throw error;
    }

    const digest = createHash("sha256").update(bytes).digest("hex");
    assert.equal(digest, PARITY_MANIFEST_HASHES[backend], `${backend} manifest hash`);
  });
}

test("parity matrix covers candidate counts, question types, and explicit overflow", () => {
  const fixtures = buildParityFixtures();
  assert.ok(fixtures.length >= 20);
  assert.deepEqual(new Set(fixtures.map((item) => item.matrix.candidate_count)), new Set([2, 4, 8, 16, 32]));
  assert.ok(fixtures.some((item) => item.request.questions[0]?.type === "boolean"));
  assert.ok(fixtures.some((item) => item.request.questions[0]?.type === "score"));
  assert.ok(fixtures.some((item) => item.expectedFailure));
  assert.equal(new Set(fixtures.map((item) => item.id)).size, fixtures.length);
  for (const fixture of fixtures.filter((item) => !item.expectedFailure)) {
    assert.doesNotThrow(() => validateRequest(fixture.request), fixture.id);
  }
});

test("parity input JSON uses locale-independent code-unit key ordering", () => {
  const first = canonicalJson({ z: "🌐", a: 3, nested: { z: "é", y: 4 } });
  const second = canonicalJson({ nested: { y: 4, z: "é" }, a: 3, z: "🌐" });

  assert.equal(first, '{"a":3,"nested":{"y":4,"z":"é"},"z":"🌐"}');
  assert.equal(second, first);
  assert.throws(() => canonicalJson({ "ä": 2 }), /lowercase ASCII/);
  assert.throws(() => canonicalJson({ omitted: undefined }), /JSON-compatible/);
  assert.throws(() => canonicalJson({ fraction: 1.25 }), /safe integers/);
  assert.throws(() => canonicalJson({ invalid: Number.NaN }), /safe integers/);
});

test("expected parity failures require the declared error category", () => {
  assert.equal(
    matchesExpectedFailure(
      "unsupported_score_question",
      new Error("score questions are not supported by this candidate-selection ONNX head"),
    ),
    true,
  );
  assert.equal(matchesExpectedFailure("unsupported_score_question", new Error("network request failed")), false);
  assert.equal(
    matchesExpectedFailure(
      "aggregate_input_budget_exceeded",
      new RangeError("question overflow-q exceeds the 512-token aggregate input budget (tokens=700)"),
    ),
    true,
  );
  assert.equal(
    matchesExpectedFailure(
      "aggregate_input_budget_exceeded",
      new RangeError("question overflow-q has unsupported candidate count"),
    ),
    false,
  );
  assert.equal(
    matchesExpectedFailure(
      "aggregate_input_budget_exceeded",
      new RangeError("question overflow-q exceeds 512-token aggregate input budget (tokens=700)"),
    ),
    false,
  );
  assert.equal(
    matchesExpectedFailure(
      "aggregate_input_budget_exceeded",
      new TypeError("question overflow-q exceeds the 512-token aggregate input budget (tokens=700)"),
    ),
    false,
  );
});

import assert from "node:assert/strict";
import test from "node:test";

const { buildParityFixtures } = await import("../demo/parity.ts");
const { validateRequest } = await import("../src/index.ts");

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

import test from "node:test";
import assert from "node:assert/strict";
import { BrowserBenchmarkJournal } from "../demo/benchmark-capture.ts";

class MemoryStorage {
  values = new Map();
  keyReads = 0;
  getItemReads = 0;

  get length() {
    return this.values.size;
  }

  key(index) {
    this.keyReads += 1;
    return [...this.values.keys()][index] ?? null;
  }

  getItem(key) {
    this.getItemReads += 1;
    return this.values.get(key) ?? null;
  }

  setItem(key, value) {
    this.values.set(key, value);
  }

  removeItem(key) {
    this.values.delete(key);
  }
}

function header(runId, sessions = 2, repeats = 2) {
  return {
    schema: "vons.browser-benchmark/v1",
    run_id: runId,
    sessions,
    repeats_per_session: repeats,
    created_at: "2026-09-29T00:00:00.000Z",
    samples: [],
    session_records: [],
    summary: { successful_samples: 999 },
    capture: { mode: "localStorage_checkpoints" },
  };
}

function sample(runId, sessionStart, repeat) {
  return { run_id: runId, session_start: sessionStart, repeat, status: "ok" };
}

function session(runId, sessionStart) {
  return { run_id: runId, session_start: sessionStart, load_ms: 1, status: "ok" };
}

test("journal recovers ordered partial records without summary and clears only its namespace", () => {
  const storage = new MemoryStorage();
  storage.setItem("other-app:keep", "value");
  const journal = new BrowserBenchmarkJournal(storage);
  assert.equal(journal.writeHeader("run-1", header("run-1")), true);
  assert.equal(journal.writeSession("run-1", session("run-1", 2)), true);
  assert.equal(journal.writeSession("run-1", session("run-1", 1)), true);
  assert.equal(journal.writeSample("run-1", sample("run-1", 2, 1)), true);
  assert.equal(journal.writeSample("run-1", sample("run-1", 1, 0)), true);
  storage.setItem("vons.browser-benchmark.v1:run-1:sample:1:9", "{not json");
  assert.equal(journal.writeState("run-1", "pagehide"), true);

  const report = journal.recover("run-1");
  assert.ok(report);
  assert.deepEqual(report.samples.map((row) => [row.session_start, row.repeat]), [[1, 0], [2, 1]]);
  assert.deepEqual(report.session_records.map((row) => row.session_start), [1, 2]);
  assert.equal(report.capture.recovered, true);
  assert.equal(report.capture.recovery_complete, false);
  assert.equal(report.capture.saved_state, "pagehide");
  assert.equal(report.capture.dropped_records, 1);
  assert.equal("summary" in report, false);
  assert.deepEqual(journal.listRunIds(), ["run-1"]);

  assert.equal(journal.clearAll(), true);
  assert.equal(journal.listRunIds().length, 0);
  assert.equal(storage.getItem("other-app:keep"), "value");
});

test("journal marks a fully checkpointed report complete", () => {
  const journal = new BrowserBenchmarkJournal(new MemoryStorage());
  const runId = "run-complete";
  const reportHeader = header(runId, 1, 2);
  assert.equal(journal.writeHeader(runId, reportHeader), true);
  assert.equal(journal.writeSession(runId, session(runId, 1)), true);
  assert.equal(journal.writeSample(runId, sample(runId, 1, 0)), true);
  assert.equal(journal.writeSample(runId, sample(runId, 1, 1)), true);

  const report = journal.recover(runId);
  assert.ok(report);
  assert.equal(report.capture.recovery_complete, true);
  assert.equal("summary" in report, false);
});

test("recovery lists runs newest first in one storage pass", () => {
  const storage = new MemoryStorage();
  const journal = new BrowserBenchmarkJournal(storage);
  const entries = [
    ["run-old", "2026-09-28T00:00:00.000Z"],
    ["run-new", "2026-09-29T00:00:00.000Z"],
    ["run-tie-a", "2026-09-28T12:00:00.000Z"],
    ["run-tie-z", "2026-09-28T12:00:00.000Z"],
  ];
  for (const [runId, createdAt] of entries) {
    assert.equal(journal.writeHeader(runId, { ...header(runId), created_at: createdAt }), true);
    assert.equal(journal.writeSample(runId, sample(runId, 1, 0)), true);
  }
  storage.keyReads = 0;
  storage.getItemReads = 0;

  const runs = journal.listRunSummaries();
  assert.deepEqual(runs.map((run) => run.run_id), ["run-new", "run-tie-z", "run-tie-a", "run-old"]);
  assert.deepEqual(runs.map((run) => run.saved_samples), [1, 1, 1, 1]);
  assert.equal(storage.keyReads, storage.length);
  assert.equal(storage.getItemReads, storage.length);
});

test("a malformed journal record is counted in recovery metadata", () => {
  const storage = new MemoryStorage();
  const journal = new BrowserBenchmarkJournal(storage);
  assert.equal(journal.writeHeader("run-corrupt", header("run-corrupt", 1, 1)), true);
  assert.equal(journal.writeSession("run-corrupt", session("run-corrupt", 1)), true);
  assert.equal(journal.writeSample("run-corrupt", sample("run-corrupt", 1, 0)), true);
  storage.setItem("vons.browser-benchmark.v1:run-corrupt:sample:1:1", "{not json");

  const report = journal.recover("run-corrupt");
  assert.ok(report);
  assert.equal(report.capture.dropped_records, 1);
  assert.equal(report.capture.recovery_complete, true);
});

test("a checkpoint failure is reported and still permits one terminal state write", () => {
  const storage = new MemoryStorage();
  const setItem = storage.setItem.bind(storage);
  let writes = 0;
  let failNextWrite = false;
  storage.setItem = (key, value) => {
    writes += 1;
    if (failNextWrite) {
      failNextWrite = false;
      throw new Error("quota exceeded");
    }
    setItem(key, value);
  };
  const journal = new BrowserBenchmarkJournal(storage);
  assert.equal(journal.writeHeader("run-failed", header("run-failed", 1, 1)), true);
  assert.equal(journal.writeState("run-failed", "running"), true);

  failNextWrite = true;
  assert.equal(journal.writeSample("run-failed", sample("run-failed", 1, 0)), false);
  assert.equal(journal.writeSample("run-failed", sample("run-failed", 1, 0)), false);
  assert.equal(journal.writeState("run-failed", "write_failed"), true);

  const report = journal.recover("run-failed");
  assert.ok(report);
  assert.equal(report.capture.saved_state, "write_failed");
  assert.equal(report.capture.storage_status, "write_failed");
  assert.equal(writes, 4);
});

test("clearing a run prevents a live journal instance from recreating orphans", () => {
  const storage = new MemoryStorage();
  storage.setItem("other-app:keep", "value");
  const journal = new BrowserBenchmarkJournal(storage);
  assert.equal(journal.writeHeader("run-live", header("run-live")), true);
  assert.equal(journal.writeState("run-live", "running"), true);
  assert.equal(journal.clearAll(), true);

  assert.equal(journal.writeHeader("run-live", header("run-live")), false);
  assert.equal(journal.writeSample("run-live", sample("run-live", 1, 0)), false);
  assert.equal(journal.writeSession("run-live", session("run-live", 1)), false);
  assert.equal(journal.writeState("run-live", "pagehide"), false);
  assert.deepEqual([...storage.values.keys()], ["other-app:keep"]);
});

test("orphaned checkpoint keys remain clearable when no valid run header exists", () => {
  const storage = new MemoryStorage();
  storage.setItem("other-app:keep", "value");
  storage.setItem("vons.browser-benchmark.v1:orphan:sample:1:0", "{}");
  const journal = new BrowserBenchmarkJournal(storage);

  assert.deepEqual(journal.listRunSummaries(), []);
  assert.equal(journal.hasSavedCheckpoints(), true);
  assert.equal(journal.clearAll(), true);
  assert.equal(journal.hasSavedCheckpoints(), false);
  assert.equal(storage.getItem("other-app:keep"), "value");
});

test("a failed clear leaves its still-stored run readable", () => {
  const storage = new MemoryStorage();
  const journal = new BrowserBenchmarkJournal(storage);
  assert.equal(journal.writeHeader("run-clear-failure", header("run-clear-failure", 1, 1)), true);
  assert.equal(journal.writeSession("run-clear-failure", session("run-clear-failure", 1)), true);
  assert.equal(journal.writeSample("run-clear-failure", sample("run-clear-failure", 1, 0)), true);
  storage.removeItem = () => {
    throw new Error("storage unavailable");
  };

  assert.equal(journal.clearAll(), false);
  assert.deepEqual(journal.listRunIds(), ["run-clear-failure"]);
  assert.ok(journal.recover("run-clear-failure"));
});

test("a storage write failure is returned without throwing", () => {
  const storage = new MemoryStorage();
  storage.setItem = () => {
    throw new Error("quota exceeded");
  };
  const journal = new BrowserBenchmarkJournal(storage);

  assert.equal(journal.writeHeader("run-1", header("run-1")), false);
  assert.equal(journal.writeSample("run-1", sample("run-1", 1, 0)), false);
});

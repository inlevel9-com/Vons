export interface StorageLike {
  readonly length: number;
  key(index: number): string | null;
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

type JsonRecord = Record<string, unknown>;
type CheckpointStatus = "running" | "pagehide" | "completed" | "write_failed";

export interface SavedRunSummary {
  run_id: string;
  created_at: string | null;
  saved_samples: number;
  expected_samples: number | null;
  saved_state: string;
}

const PREFIX = "vons.browser-benchmark.v1:";

function isRecord(value: unknown): value is JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function parseRecord(value: string | null): JsonRecord | null {
  if (value === null) return null;
  try {
    const parsed: unknown = JSON.parse(value);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function recordSlot(value: JsonRecord, fields: string[]): boolean {
  return fields.every((field) => Number.isInteger(value[field]) && (value[field] as number) >= 0);
}

export class BrowserBenchmarkJournal {
  private writable = true;
  private readonly storage: StorageLike;
  private readonly seenRunIds = new Set<string>();

  constructor(storage: StorageLike) {
    this.storage = storage;
  }

  writeHeader(runId: string, header: JsonRecord): boolean {
    if (header.run_id !== runId || header.schema !== "vons.browser-benchmark/v1") return false;
    if (this.seenRunIds.has(runId) && !this.hasHeader(runId)) return false;
    const written = this.write(`${this.runPrefix(runId)}header`, header);
    if (written) this.seenRunIds.add(runId);
    return written;
  }

  writeSample<T extends { run_id: string; session_start: number; repeat: number }>(runId: string, sample: T): boolean {
    if (sample.run_id !== runId || !recordSlot(sample, ["session_start", "repeat"])) return false;
    if (!this.hasHeader(runId)) return false;
    const key = `${this.runPrefix(runId)}sample:${sample.session_start}:${sample.repeat}`;
    return this.write(key, sample);
  }

  writeSession(runId: string, session: JsonRecord): boolean {
    if (session.run_id !== runId || !recordSlot(session, ["session_start"])) return false;
    if (!this.hasHeader(runId)) return false;
    const key = `${this.runPrefix(runId)}session:${session.session_start}`;
    return this.write(key, session);
  }

  writeState(runId: string, status: CheckpointStatus): boolean {
    if (!this.hasHeader(runId)) return false;
    return this.write(`${this.runPrefix(runId)}state`, {
      status,
      updated_at: new Date().toISOString(),
    }, true);
  }

  listRunIds(): string[] {
    return this.listRunSummaries().map((run) => run.run_id);
  }

  hasSavedCheckpoints(): boolean {
    try {
      for (let index = 0; index < this.storage.length; index += 1) {
        if (this.storage.key(index)?.startsWith(PREFIX)) return true;
      }
      return false;
    } catch {
      return false;
    }
  }

  listRunSummaries(): SavedRunSummary[] {
    try {
      const runs = new Map<string, {
        header: JsonRecord | null;
        saved_samples: number;
        saved_state: string;
      }>();
      for (let index = 0; index < this.storage.length; index += 1) {
        const key = this.storage.key(index);
        if (!key?.startsWith(PREFIX)) continue;
        const remainder = key.slice(PREFIX.length);
        const separator = remainder.indexOf(":");
        if (separator < 1) continue;
        const runId = remainder.slice(0, separator);
        const kind = remainder.slice(separator + 1);
        const entry = runs.get(runId) ?? { header: null, saved_samples: 0, saved_state: "unknown" };
        const value = parseRecord(this.storage.getItem(key));
        if (kind === "header") {
          entry.header = value;
        } else if (kind.startsWith("sample:")) {
          if (value && value.run_id === runId && recordSlot(value, ["session_start", "repeat"])) {
            entry.saved_samples += 1;
          }
        } else if (kind === "state" && typeof value?.status === "string") {
          entry.saved_state = value.status;
        }
        runs.set(runId, entry);
      }
      return [...runs.entries()]
        .filter(([runId, entry]) => entry.header?.run_id === runId && entry.header.schema === "vons.browser-benchmark/v1")
        .map(([run_id, entry]) => {
          const header = entry.header as JsonRecord;
          const created_at = typeof header.created_at === "string" ? header.created_at : null;
          const sessions = header.sessions;
          const repeats = header.repeats_per_session;
          const validCounts = Number.isSafeInteger(sessions) && Number.isSafeInteger(repeats)
            && (sessions as number) > 0 && (repeats as number) > 0;
          return {
            run_id,
            created_at,
            saved_samples: entry.saved_samples,
            expected_samples: validCounts ? (sessions as number) * (repeats as number) : null,
            saved_state: entry.saved_state,
          };
        })
        .sort((left, right) => (right.created_at ?? "").localeCompare(left.created_at ?? "") || right.run_id.localeCompare(left.run_id));
    } catch {
      return [];
    }
  }

  recover(runId: string): JsonRecord | null {
    try {
      const header = this.readHeader(runId);
      if (!header || header.run_id !== runId || header.schema !== "vons.browser-benchmark/v1") return null;

      const { samples, sessionRecords, state, droppedRecords } = this.readRunRecords(runId);
      samples.sort((left, right) => (left.session_start as number) - (right.session_start as number) || (left.repeat as number) - (right.repeat as number));
      sessionRecords.sort((left, right) => (left.session_start as number) - (right.session_start as number));
      const sessions = header.sessions;
      const repeats = header.repeats_per_session;
      const validCounts = Number.isSafeInteger(sessions)
        && Number.isSafeInteger(repeats)
        && (sessions as number) > 0
        && (repeats as number) > 0;
      const expectedSamples = validCounts ? (sessions as number) * (repeats as number) : -1;
      const sessionStarts = sessionRecords.map((record) => record.session_start as number);
      const sampleSlots = samples.map((sample) => `${sample.session_start}:${sample.repeat}`);
      const complete = validCounts
        && sessionRecords.length === sessions
        && sessionRecords.every((record) => record.status === "ok")
        && new Set(sessionStarts).size === sessionRecords.length
        && sessionStarts.every((sessionStart) => sessionStart >= 1 && sessionStart <= (sessions as number))
        && samples.length === expectedSamples
        && new Set(sampleSlots).size === samples.length
        && samples.every((sample) => {
          const sessionStart = sample.session_start;
          const repeat = sample.repeat;
          return typeof sessionStart === "number"
            && typeof repeat === "number"
            && sessionStart >= 1
            && sessionStart <= (sessions as number)
            && repeat >= 0
            && repeat < (repeats as number);
        });
      const capture = isRecord(header.capture) ? header.capture : {};
      const savedState = typeof state?.status === "string" ? state.status : "unknown";
      const storageStatus = savedState === "write_failed"
        ? "write_failed"
        : savedState === "completed" || savedState === "pagehide"
          ? capture.storage_status ?? "unknown"
          : "unknown";
      const report: JsonRecord = {
        ...header,
        session_records: sessionRecords,
        samples,
        capture: {
          ...capture,
          storage_status: storageStatus,
          recovered: true,
          recovery_complete: complete,
          saved_state: savedState,
          dropped_records: droppedRecords,
          recovered_at: new Date().toISOString(),
        },
      };
      delete report.summary;
      return report;
    } catch {
      return null;
    }
  }

  clearAll(): boolean {
    try {
      const keys: string[] = [];
      for (let index = 0; index < this.storage.length; index += 1) {
        const key = this.storage.key(index);
        if (key?.startsWith(PREFIX)) keys.push(key);
      }
      for (const key of keys) this.storage.removeItem(key);
      return true;
    } catch {
      return false;
    }
  }

  private runPrefix(runId: string): string {
    return `${PREFIX}${runId}:`;
  }

  private readHeader(runId: string): JsonRecord | null {
    return parseRecord(this.storage.getItem(`${this.runPrefix(runId)}header`));
  }

  private hasHeader(runId: string): boolean {
    try {
      const header = this.readHeader(runId);
      return header?.run_id === runId && header.schema === "vons.browser-benchmark/v1";
    } catch {
      return false;
    }
  }

  private readRunRecords(runId: string): {
    samples: JsonRecord[];
    sessionRecords: JsonRecord[];
    state: JsonRecord | null;
    droppedRecords: number;
  } {
    const prefix = this.runPrefix(runId);
    const samples: JsonRecord[] = [];
    const sessionRecords: JsonRecord[] = [];
    let state: JsonRecord | null = null;
    let droppedRecords = 0;
    for (let index = 0; index < this.storage.length; index += 1) {
      const key = this.storage.key(index);
      if (!key?.startsWith(prefix)) continue;
      const kind = key.slice(prefix.length);
      if (kind === "state") {
        state = parseRecord(this.storage.getItem(key));
        continue;
      }
      if (!kind.startsWith("sample:") && !kind.startsWith("session:")) continue;
      const record = parseRecord(this.storage.getItem(key));
      const fields = kind.startsWith("sample:") ? ["session_start", "repeat"] : ["session_start"];
      if (!record || record.run_id !== runId || !recordSlot(record, fields)) {
        droppedRecords += 1;
      } else if (kind.startsWith("sample:")) {
        samples.push(record);
      } else {
        sessionRecords.push(record);
      }
    }
    return { samples, sessionRecords, state, droppedRecords };
  }

  private write(key: string, value: JsonRecord, allowAfterFailure = false): boolean {
    if (!this.writable && !allowAfterFailure) return false;
    try {
      this.storage.setItem(key, JSON.stringify(value));
      return true;
    } catch {
      this.writable = false;
      return false;
    }
  }
}

export function createBrowserBenchmarkJournal(): BrowserBenchmarkJournal | null {
  try {
    return new BrowserBenchmarkJournal(window.localStorage);
  } catch {
    return null;
  }
}

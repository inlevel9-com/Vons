import { createOnnxWebBackend, type OnnxTimingSample } from "../src/onnx.ts";
import { createBrowserBenchmarkJournal } from "./benchmark-capture.ts";

type BackendName = "direct" | "diffusion";
type ProviderName = "wasm" | "webgpu";

interface PerformanceWithMemory extends Performance {
  memory?: { usedJSHeapSize: number };
}

interface NavigatorWithDeviceMemory extends Navigator {
  deviceMemory?: number;
}

interface BenchmarkSample extends Omit<OnnxTimingSample, "live_candidates" | "allocated_candidates" | "live_tokens" | "sequence_length" | "tokenizeMs" | "prepareFeedMs" | "inferenceAndReadbackMs" | "postprocessMs" | "totalRequestMs"> {
  run_id: string;
  session_start: number;
  repeat: number;
  case_id: string;
  backend: BackendName;
  requested_provider: ProviderName;
  observed_provider: ProviderName | null;
  observed_provider_status: string;
  live_candidates: number | null;
  allocated_candidates: number | null;
  live_tokens: number | null;
  sequence_length: number | null;
  tokenizeMs: number | null;
  prepareFeedMs: number | null;
  inferenceAndReadbackMs: number | null;
  postprocessMs: number | null;
  totalRequestMs: number | null;
  bundle_hash: string | null;
  tokenizer_hash: string | null;
  input_hash: string;
  noise_hash: string | null;
  js_heap_used_bytes_before: number | null;
  js_heap_used_bytes_after: number | null;
  preceding_checkpoint_write_ms: number | null;
  status: "ok" | "error";
  error_kind?: string;
}

interface SessionRecord {
  session_start: number;
  load_ms: number;
  status: "ok" | "error";
  error_kind?: string;
}

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`benchmark element is missing: ${selector}`);
  return element;
}

const output = requiredElement<HTMLElement>("#output");
const downloadButton = requiredElement<HTMLButtonElement>("#download-report");
const recoveryControls = requiredElement<HTMLElement>("#recovery-controls");
const savedRunSelect = requiredElement<HTMLSelectElement>("#saved-run");
const recoverSavedRunButton = requiredElement<HTMLButtonElement>("#recover-saved-run");
const clearSavedRunsButton = requiredElement<HTMLButtonElement>("#clear-saved-runs");
const journal = createBrowserBenchmarkJournal();
const recoveryMode = new URLSearchParams(location.search).get("recover") === "1";

function showReport(report: unknown, fileName: string, notice?: string): void {
  const text = JSON.stringify(report, null, 2);
  output.textContent = notice ? `${notice}\n\n${text}` : text;
  downloadButton.hidden = false;
  downloadButton.onclick = () => {
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName;
    anchor.hidden = true;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
  };
  console.log(text);
}

function configureRecoveryView(): boolean {
  if (!recoveryMode) return false;
  recoveryControls.hidden = false;
  if (!journal) {
    output.textContent = "Saved-run recovery is unavailable because local browser storage could not be opened.";
    recoverSavedRunButton.disabled = true;
    clearSavedRunsButton.disabled = true;
    return true;
  }

  const renderSavedRuns = () => {
    const runs = journal.listRunSummaries();
    savedRunSelect.replaceChildren(...runs.map((run) => {
      const option = document.createElement("option");
      option.value = run.run_id;
      const expected = run.expected_samples ?? "?";
      option.textContent = `${run.run_id} — ${run.saved_samples}/${expected} samples · ${run.saved_state}`;
      return option;
    }));
    recoverSavedRunButton.disabled = runs.length === 0;
    clearSavedRunsButton.disabled = !journal.hasSavedCheckpoints();
    return runs;
  };
  renderSavedRuns();

  const showSelectedReport = (notice?: string): void => {
    const runId = savedRunSelect.value;
    const report = runId ? journal.recover(runId) : null;
    if (!report || !runId) {
      output.textContent = "No recoverable benchmark reports were found in this browser.";
      downloadButton.hidden = true;
      return;
    }
    showReport(report, `vons-recovered-${runId}.json`, notice);
  };
  recoverSavedRunButton.onclick = () => showSelectedReport();
  savedRunSelect.onchange = () => showSelectedReport();
  clearSavedRunsButton.onclick = () => {
    if (!window.confirm("Delete all saved Vons browser benchmark checkpoints from this browser? If a benchmark is still running, it will continue but cannot save more recovery checkpoints.")) return;
    const cleared = journal.clearAll();
    const remaining = renderSavedRuns();
    if (cleared) {
      output.textContent = "Saved benchmark checkpoints were cleared from this browser.";
      downloadButton.hidden = true;
    } else if (remaining.length > 0) {
      showSelectedReport("Some checkpoints could not be cleared. Remaining saved runs are still available below.");
    } else {
      output.textContent = "The saved-run list is empty; checkpoint removal could not be confirmed.";
      downloadButton.hidden = true;
    }
  };
  showSelectedReport();
  return true;
}

const manifestUrls = {
  direct: new URL("../../../artifacts/pilot/bundle-manifest-v1.json", import.meta.url),
  diffusion: new URL("../../../artifacts/pilot-diffusion/bundle-manifest-v1.json", import.meta.url),
};

const manifestHashes = {
  direct: "bd39eea2b8d5ed5f4d57daefdd4822c4e2c4817a68c4a42bef850b5edcbc97d",
  diffusion: "df0e7cae640c6604fb517a88d08dbdaa2454d6b3dbbf5708926349cf490b9607",
};

const heapBytes = (): number | null => (performance as PerformanceWithMemory).memory?.usedJSHeapSize ?? null;

function parameter(name: string, fallback: number): number {
  const raw = new URLSearchParams(location.search).get(name);
  if (raw === null) return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${name} must be a positive integer`);
  return value;
}

function textParameter(name: string, fallback: string): string {
  return new URLSearchParams(location.search).get(name) ?? fallback;
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
  return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
}

async function sha256(value: string | ArrayBufferLike): Promise<string> {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : new Uint8Array(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes as unknown as BufferSource);
  return Array.from(new Uint8Array(digest), (item) => item.toString(16).padStart(2, "0")).join("");
}

// Frozen from NumPy default_rng(7).standard_normal((1, 32), dtype=float32).
// The array, rather than a language-specific seed algorithm, is the parity input.
const FROZEN_NOISE_SEED_7 = Float32Array.from([
  1.52196932, -1.14410579, 1.15016162, -1.78465319, -0.77149874, 1.01916146,
  -0.926134586, -0.415745318, -0.0629121512, 0.513251185, -0.285995334,
  1.98480856, 1.30306911, -0.00464721443, 1.17548203, 1.34566987,
  -0.171809882, -2.4133966, -0.0993812904, -0.979565322, 0.942675471,
  0.452982962, 0.53610903, 0.494613141, 1.94684041, 0.207542002,
  -1.64390635, -0.542155981, -0.384397745, -0.410856962, 0.946411192,
  0.979099393,
]);

function percentile(values: number[], fraction: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * fraction;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

async function run(): Promise<void> {
  const backendName = textParameter("backend", "direct") as BackendName;
  const provider = textParameter("provider", "wasm") as ProviderName;
  if (!(["direct", "diffusion"] as string[]).includes(backendName)) throw new RangeError("backend must be direct or diffusion");
  if (!(["wasm", "webgpu"] as string[]).includes(provider)) throw new RangeError("provider must be wasm or webgpu");
  const sessions = parameter("sessions", 1);
  const warmup = parameter("warmup", 1);
  const repeats = parameter("repeats", 3);
  const caseCount = parameter("cases", 20);
  const runId = crypto.randomUUID();
  const cases = Array.from({ length: caseCount }, (_, index) => ({
    caseId: `case-${index + 1}`,
    state: `browser benchmark state ${index + 1}`,
  }));
  const request = (caseIndex: number) => ({
    state: cases[caseIndex % cases.length].state,
    questions: [{ id: cases[caseIndex % cases.length].caseId, type: "choice" as const, prompt: "Choose the next action", options: ["continue", "stop", "inspect", "defer"] }],
    seed: 7,
  });
  const inputHashes = new Map(cases.map((item, index) => [item.caseId, stableJson(request(index))]));
  const hashedInputs = new Map<string, string>();
  for (const [caseId, value] of inputHashes) hashedInputs.set(caseId, await sha256(value));
  const noise = backendName === "diffusion" ? FROZEN_NOISE_SEED_7 : null;
  const noiseHash = noise ? await sha256(noise.buffer) : null;
  const browserEnvironment = {
    user_agent: navigator.userAgent,
    platform: navigator.platform || null,
    hardware_concurrency: navigator.hardwareConcurrency ?? null,
    device_memory_gb: (navigator as NavigatorWithDeviceMemory).deviceMemory ?? null,
    cross_origin_isolated: globalThis.crossOriginIsolated,
  };
  const samples: BenchmarkSample[] = [];
  const sessionRecords: SessionRecord[] = [];
  let runtimeInfo: Awaited<ReturnType<typeof createOnnxWebBackend>>["runtimeInfo"] | null = null;
  let captureStatus: "active" | "unavailable" | "write_failed" = journal ? "active" : "unavailable";
  let runComplete = false;
  let precedingCheckpointWriteMs: number | null = null;
  const reportHeader: Record<string, unknown> = {
    schema: "vons.browser-benchmark/v1",
    run_id: runId,
    backend: backendName,
    requested_provider: provider,
    sessions,
    warmup_excluded_per_session: warmup,
    repeats_per_session: repeats,
    cases: caseCount,
    created_at: new Date().toISOString(),
    environment: browserEnvironment,
    runtime_info: runtimeInfo,
    session_start_semantics: "new ORT session on one page; module and browser caches may persist",
    execution_provider_evidence: "requested provider and session creation only; graph partition status is not exposed by this harness",
    session_records: [] as SessionRecord[],
    samples: [] as BenchmarkSample[],
    capture: {
      mode: journal ? "same_origin_localStorage_checkpoints" : "unavailable",
      storage_status: captureStatus,
      checkpoint_policy: "initial run header; runtime metadata header after each successful session load; one keyed write after each sample and session record",
      request_timing_note: "the current sample checkpoint runs after decide timing; synchronous storage and serialization can affect later intervals and heap observations",
      memory_note: "heap observations are not isolated from localStorage checkpoint allocations; preceding_checkpoint_write_ms sums journal write time since the prior sampled request, including setup writes before the first sample",
    },
  };
  const markWriteFailure = (): void => {
    captureStatus = "write_failed";
    if (journal) measureCheckpointWrite(() => journal.writeState(runId, "write_failed"));
  };
  const measureCheckpointWrite = (write: () => boolean): boolean => {
    const started = performance.now();
    const result = write();
    precedingCheckpointWriteMs = (precedingCheckpointWriteMs ?? 0) + performance.now() - started;
    return result;
  };
  if (journal && !measureCheckpointWrite(() => journal.writeHeader(runId, reportHeader))) markWriteFailure();
  if (captureStatus === "active" && journal && !measureCheckpointWrite(() => journal.writeState(runId, "running"))) markWriteFailure();
  window.addEventListener("pagehide", () => {
    if (journal) {
      const status = captureStatus === "active"
        ? runComplete ? "completed" : "pagehide"
        : "write_failed";
      if (!journal.writeState(runId, status)) captureStatus = "write_failed";
    }
  });

  const checkpointSample = (sample: BenchmarkSample): void => {
    sample.preceding_checkpoint_write_ms = precedingCheckpointWriteMs;
    precedingCheckpointWriteMs = null;
    if (captureStatus !== "active" || !journal) return;
    const written = measureCheckpointWrite(() => journal.writeSample(runId, sample));
    if (!written) markWriteFailure();
  };
  const checkpointSession = (record: SessionRecord): void => {
    if (captureStatus === "active" && journal
      && !measureCheckpointWrite(() => journal.writeSession(runId, { ...record, run_id: runId }))) markWriteFailure();
  };

  for (let sessionStart = 1; sessionStart <= sessions; sessionStart += 1) {
    let timing: OnnxTimingSample | undefined;
    const started = performance.now();
    let backend: Awaited<ReturnType<typeof createOnnxWebBackend>> | undefined;
    try {
      backend = await createOnnxWebBackend({
        manifestUrl: manifestUrls[backendName],
        expectedManifestSha256: manifestHashes[backendName],
        provider,
        wasmPaths: "/sdk/typescript/node_modules/onnxruntime-web/dist/",
        wasmNumThreads: 1,
        onTiming: (sample) => { timing = sample; },
        initialNoise: backendName === "diffusion" ? () => new Float32Array(noise as Float32Array) : undefined,
      });
      const loadMs = performance.now() - started;
      runtimeInfo = backend.runtimeInfo;
      reportHeader.runtime_info = runtimeInfo;
      if (captureStatus === "active" && journal
        && !measureCheckpointWrite(() => journal.writeHeader(runId, reportHeader))) markWriteFailure();
      const sessionRecord = { session_start: sessionStart, load_ms: loadMs, status: "ok" as const };
      sessionRecords.push(sessionRecord);
      checkpointSession(sessionRecord);
      for (let repeat = 0; repeat < warmup; repeat += 1) await backend.decide(request(repeat));
      for (let repeat = 0; repeat < repeats; repeat += 1) {
        const caseValue = cases[repeat % cases.length];
        const before = heapBytes();
        timing = undefined;
        try {
          await backend.decide(request(repeat));
          const measured = timing as OnnxTimingSample | undefined;
          if (!measured) throw new Error("timing hook did not produce a sample");
          const sample: BenchmarkSample = {
            ...measured,
            run_id: runId,
            session_start: sessionStart,
            repeat,
            case_id: caseValue.caseId,
            backend: backendName,
            requested_provider: provider,
            observed_provider: null,
            observed_provider_status: "session_created; execution_partition_not_exposed",
            live_candidates: measured.live_candidates,
            allocated_candidates: measured.allocated_candidates,
            live_tokens: measured.live_tokens,
            sequence_length: measured.sequence_length,
            bundle_hash: backend.manifestHash,
            tokenizer_hash: backend.manifest.files.find((item) => item.path.endsWith("tokenizer.json"))?.sha256 ?? null,
            input_hash: hashedInputs.get(caseValue.caseId) as string,
            noise_hash: noiseHash,
            js_heap_used_bytes_before: before,
            js_heap_used_bytes_after: heapBytes(),
            preceding_checkpoint_write_ms: precedingCheckpointWriteMs,
            status: "ok",
          };
          samples.push(sample);
          checkpointSample(sample);
        } catch (error) {
          const sample: BenchmarkSample = {
            questionId: caseValue.caseId,
            tokenizeMs: null,
            prepareFeedMs: null,
            inferenceAndReadbackMs: null,
            postprocessMs: null,
            totalRequestMs: null,
            run_id: runId,
            session_start: sessionStart,
            repeat,
            case_id: caseValue.caseId,
            backend: backendName,
            requested_provider: provider,
            observed_provider: null,
            observed_provider_status: "request_failed; execution_partition_not_exposed",
            live_candidates: null,
            allocated_candidates: null,
            live_tokens: null,
            sequence_length: null,
            bundle_hash: backend.manifestHash,
            tokenizer_hash: backend.manifest.files.find((item) => item.path.endsWith("tokenizer.json"))?.sha256 ?? null,
            input_hash: hashedInputs.get(caseValue.caseId) as string,
            noise_hash: noiseHash,
            js_heap_used_bytes_before: before,
            js_heap_used_bytes_after: heapBytes(),
            preceding_checkpoint_write_ms: precedingCheckpointWriteMs,
            status: "error",
            error_kind: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
          };
          samples.push(sample);
          checkpointSample(sample);
        }
      }
    } catch (error) {
      const sessionRecord = { session_start: sessionStart, load_ms: performance.now() - started, status: "error" as const, error_kind: error instanceof Error ? `${error.name}: ${error.message}` : String(error) };
      sessionRecords.push(sessionRecord);
      checkpointSession(sessionRecord);
    } finally {
      await backend?.dispose();
    }
  }

  const successful = samples.filter((sample) => sample.status === "ok");
  const memoryMaximum = (values: Array<number | null>): number | null => {
    const available = values.filter((value): value is number => value !== null);
    return available.length === 0 ? null : Math.max(...available);
  };
  runComplete = true;
  if (journal) {
    const written = journal.writeState(runId, captureStatus === "active" ? "completed" : "write_failed");
    if (!written) captureStatus = "write_failed";
  }
  const report = {
    ...reportHeader,
    runtime_info: runtimeInfo,
    capture: {
      ...(reportHeader.capture as Record<string, unknown>),
      storage_status: captureStatus,
    },
    session_records: sessionRecords,
    samples,
    summary: {
      successful_samples: successful.length,
      failed_samples: samples.length - successful.length,
      total_request_ms_p50: percentile(successful.flatMap((sample) => sample.totalRequestMs === null ? [] : [sample.totalRequestMs]), 0.5),
      total_request_ms_p95: percentile(successful.flatMap((sample) => sample.totalRequestMs === null ? [] : [sample.totalRequestMs]), 0.95),
      inference_and_readback_ms_p50: percentile(successful.flatMap((sample) => sample.inferenceAndReadbackMs === null ? [] : [sample.inferenceAndReadbackMs]), 0.5),
      inference_and_readback_ms_p95: percentile(successful.flatMap((sample) => sample.inferenceAndReadbackMs === null ? [] : [sample.inferenceAndReadbackMs]), 0.95),
      js_heap_before_max: memoryMaximum(successful.map((sample) => sample.js_heap_used_bytes_before)),
      js_heap_after_max: memoryMaximum(successful.map((sample) => sample.js_heap_used_bytes_after)),
      memory_note: "performance.memory before/after observations only; synchronous localStorage checkpoint allocations can affect later before readings; preceding_checkpoint_write_ms records intervening journal-write time; not a peak or GPU-memory measurement",
    },
  };
  showReport(report, `vons-${backendName}-${provider}-${runId}.json`);
}

if (!configureRecoveryView()) {
  void run().catch((error) => {
    showReport({ schema: "vons.browser-benchmark/v1", status: "error", error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) }, `vons-browser-benchmark-error-${crypto.randomUUID()}.json`);
  });
}

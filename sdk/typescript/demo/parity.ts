import { createOnnxWebBackend, type OnnxTimingSample, type OnnxTraceSample } from "../src/onnx.ts";
import type { DecisionRequest, DecisionResponse, QuestionType } from "../src/index.ts";

export type ParityBackend = "direct" | "diffusion";
export type ParityProvider = "wasm" | "webgpu";

export interface ParityFixture {
  id: string;
  request: DecisionRequest;
  expectedFailure?: string;
  matrix: { candidate_count: number; family: string; target_tokens: string };
}

export interface ParityRecord {
  fixture_id: string;
  input_hash: string;
  request: DecisionRequest;
  matrix: ParityFixture["matrix"];
  status: "ok" | "expected_error" | "unexpected_error";
  response?: DecisionResponse;
  timings?: OnnxTimingSample[];
  traces?: OnnxTraceSample[];
  error?: string;
}

const manifestUrls: Record<ParityBackend, URL> = {
  direct: new URL("../../../artifacts/pilot/bundle-manifest-v1.json", import.meta.url),
  diffusion: new URL("../../../artifacts/pilot-diffusion/bundle-manifest-v1.json", import.meta.url),
};

const manifestHashes: Record<ParityBackend, string> = {
  direct: "bd39eea2b8d5ed5f4d57daefdd4822c4e2c4817a68c4a42bef850b5edcbc97d",
  diffusion: "df0e7cae640c6604fb517a88d08dbdaa2454d6b3dbbf5708926349cf490b9607",
};

const FROZEN_NOISE_SEED_7 = Float32Array.from([
  1.52196932, -1.14410579, 1.15016162, -1.78465319, -0.77149874, 1.01916146,
  -0.926134586, -0.415745318, -0.0629121512, 0.513251185, -0.285995334, 1.98480856,
  1.30306911, -0.00464721443, 1.17548203, 1.34566987, -0.171809882, -2.4133966,
  -0.0993812904, -0.979565322, 0.942675471, 0.452982962, 0.53610903, 0.494613141,
  1.94684041, 0.207542002, -1.64390635, -0.542155981, -0.384397745, -0.410856962,
  0.946411192, 0.979099393,
]);

function options(count: number, label: string): string[] {
  return Array.from({ length: count }, (_, index) => `${label} candidate ${index + 1}`);
}

function question(
  id: string,
  type: QuestionType,
  prompt: string,
  values: string[],
): DecisionRequest["questions"][number] {
  return type === "score"
    ? { id, type, prompt, options: values, rubric: ["poor", "fair", "good", "very good", "excellent"] }
    : { id, type, prompt, options: values };
}

function fixture(
  id: string,
  family: string,
  targetTokens: string,
  candidateCount: number,
  state: DecisionRequest["state"],
  item: DecisionRequest["questions"][number],
  expectedFailure?: string,
): ParityFixture {
  return {
    id,
    request: { state, questions: [item], seed: 7 },
    expectedFailure,
    matrix: { candidate_count: candidateCount, family, target_tokens: targetTokens },
  };
}

export function buildParityFixtures(): ParityFixture[] {
  const result: ParityFixture[] = [];
  const counts = [2, 4, 8, 16, 32];
  for (const count of counts) {
    result.push(fixture(`short-${count}`, "choice", "short", count, "ready", question(`short-${count}-q`, "choice", "Choose the next action", options(count, "short"))));
    result.push(fixture(`unicode-${count}`, "choice", "unicode", count, { locale: "ko-KR", state: `검토 중인 화면 🌐 / пункт ${count}` }, question(`unicode-${count}-q`, "choice", "다음에 선택할 항목은 무엇입니까?", options(count, "unicode/다음"))));
    result.push(fixture(`score-${count}`, "score", "unsupported", count, { workflow: "quality", count, labels: ["α", "β", "γ"] }, question(`score-${count}-q`, "score", "Rate the candidate outcome", options(count, "score")), "score_head_unsupported"));
    const repeats = Math.max(2, Math.floor(60 / count));
    result.push(fixture(`long-${count}`, "choice", "near-aggregate-limit", count, "state ".repeat(50), question(`long-${count}-q`, "choice", "Select the most relevant detailed action", options(count, "detail ".repeat(repeats)))));
  }
  result.push(fixture("boolean-2", "boolean", "boolean", 2, "permission check", question("boolean-q", "boolean", "Should the host ask for confirmation?", [])));
  result.push({
    id: "multi-question",
    request: {
      state: { page: "settings", text: "Review before any external action" },
      questions: [
        question("multi-choice", "choice", "Which next step is safest?", options(4, "multi")),
        question("multi-boolean", "boolean", "Is confirmation required?", []),
      ],
      seed: 7,
    },
    matrix: { candidate_count: 4, family: "multi-question", target_tokens: "mixed" },
  });
  result.push(fixture(
    "overflow-explicit",
    "overflow",
    "over-limit",
    2,
    "x ".repeat(700),
    question("overflow-q", "choice", "This request must fail explicitly", options(2, "overflow")),
    "request_validation_or_tokenization_overflow",
  ));
  return result;
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
  return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
}

async function sha256(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), (item) => item.toString(16).padStart(2, "0")).join("");
}

export async function runParityCell(backendName: ParityBackend, provider: ParityProvider): Promise<Record<string, unknown>> {
  const fixtures = buildParityFixtures();
  const records: ParityRecord[] = [];
  let timings: OnnxTimingSample[] = [];
  let traces: OnnxTraceSample[] = [];
  const backend = await createOnnxWebBackend({
    manifestUrl: manifestUrls[backendName],
    expectedManifestSha256: manifestHashes[backendName],
    provider,
    wasmPaths: "/sdk/typescript/node_modules/onnxruntime-web/dist/",
    wasmNumThreads: 1,
    initialNoise: backendName === "diffusion" ? () => new Float32Array(FROZEN_NOISE_SEED_7) : undefined,
    onTiming: (sample) => { timings.push(sample); },
    onTrace: (sample) => { traces.push(sample); },
  });
  try {
    for (const item of fixtures) {
      const inputHash = await sha256(stableJson(item.request));
      timings = [];
      traces = [];
      try {
        const response = await backend.decide(item.request);
        records.push({ fixture_id: item.id, input_hash: inputHash, request: item.request, matrix: item.matrix, status: "ok", response, timings, traces });
      } catch (error) {
        const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
        records.push({
          fixture_id: item.id,
          input_hash: inputHash,
          request: item.request,
          matrix: item.matrix,
          status: item.expectedFailure ? "expected_error" : "unexpected_error",
          timings,
          traces,
          error: message,
        });
      }
    }
  } finally {
    await backend.dispose();
  }
  return {
    schema: "vons.browser-parity/v1",
    backend: backendName,
    requested_provider: provider,
    fixture_count: fixtures.length,
    runtime_info: backend.runtimeInfo,
    manifest_hash: backend.manifestHash,
    tokenizer_hash: backend.manifest.files.find((item) => item.path.endsWith("tokenizer.json"))?.sha256 ?? null,
    noise_seed: backendName === "diffusion" ? 7 : null,
    noise_values: backendName === "diffusion" ? Array.from(FROZEN_NOISE_SEED_7) : null,
    records,
    limitations: [
      "This harness checks contract-valid responses and reproducible fixture inputs; it does not establish task accuracy.",
      "A browser-vs-Python numeric parity claim requires separately saved CPU reference outputs for the same input and noise hashes.",
      "WebGPU provider selection does not expose graph partition evidence in this runtime harness.",
    ],
  };
}

if (typeof document !== "undefined") {
  const runButton = document.querySelector<HTMLButtonElement>("#run");
  const output = document.querySelector<HTMLElement>("#output");
  const backendSelect = document.querySelector<HTMLSelectElement>("#backend");
  const providerSelect = document.querySelector<HTMLSelectElement>("#provider");
  const downloadButton = document.querySelector<HTMLButtonElement>("#download");
  if (!runButton || !output || !backendSelect || !providerSelect || !downloadButton) throw new Error("parity harness controls are missing");
  let lastReport: Record<string, unknown> | null = null;
  const displayReport = (report: Record<string, unknown>): void => {
    const records = Array.isArray(report.records) ? report.records as Array<Record<string, unknown>> : [];
    const statusCounts = records.reduce<Record<string, number>>((counts, record) => {
      const status = String(record.status ?? "unknown");
      counts[status] = (counts[status] ?? 0) + 1;
      return counts;
    }, {});
    output.textContent = JSON.stringify({
      schema: report.schema,
      backend: report.backend,
      requested_provider: report.requested_provider,
      fixture_count: report.fixture_count,
      manifest_hash: report.manifest_hash,
      tokenizer_hash: report.tokenizer_hash,
      noise_seed: report.noise_seed,
      status_counts: statusCounts,
      errors: records.filter((record) => record.status !== "ok").map((record) => ({ fixture_id: record.fixture_id, status: record.status, error: record.error })),
      note: "Full parity JSON, including request hashes, timings, traces, and noise values, is available through Download report.",
    }, null, 2);
  };
  downloadButton.addEventListener("click", () => {
    if (!lastReport) return;
    const backend = backendSelect.value as ParityBackend;
    const provider = providerSelect.value as ParityProvider;
    const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `vons-parity-${backend}-${provider}.json`;
    link.click();
    URL.revokeObjectURL(url);
  });
  runButton.addEventListener("click", async () => {
    runButton.disabled = true;
    downloadButton.disabled = true;
    lastReport = null;
    output.textContent = "Running...";
    try {
      const report = await runParityCell(backendSelect.value as ParityBackend, providerSelect.value as ParityProvider);
      lastReport = report;
      downloadButton.disabled = false;
      displayReport(report);
    } catch (error) {
      output.textContent = JSON.stringify({ schema: "vons.browser-parity/v1", status: "error", error: String(error) }, null, 2);
    } finally {
      runButton.disabled = false;
    }
  });
}

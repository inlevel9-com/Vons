import { Tokenizer } from "@huggingface/tokenizers";

import {
  type Backend,
  type DecisionBackend,
  type DecisionRequest,
  type DecisionResponse,
  type Question,
  type QuestionAnswer,
  validateRequest,
  validateResponseForRequest,
} from "./index.ts";

export type OnnxExecutionProvider = "wasm" | "webgpu";

export interface BundleFileRecord {
  path: string;
  role: string;
  bytes: number;
  sha256: string;
  runtime_loaded: boolean;
  model_asset: boolean;
}

export interface BundleManifest {
  schema_version: "vons.bundle.manifest/v1";
  manifest: { path: string; sha256: string | null; self_hash_excluded: boolean };
  metadata: {
    backend: Backend;
    model_id?: string;
    option_count?: number;
    sequence_length?: number;
    input_names?: string[];
    output_names?: string[];
    graph?: {
      external_data_locations?: string[];
      inputs?: Array<{ name: string; dtype: string; shape: Array<number | string | null> | null }>;
      outputs?: Array<{ name: string; dtype: string; shape: Array<number | string | null> | null }>;
    };
    [key: string]: unknown;
  };
  files: BundleFileRecord[];
  summary: {
    model_asset_bytes: number;
    complete_download_bytes: number;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface SharedBundleManifest {
  schema: "vons.shared-bundle/v1";
  model_id: string;
  encoder: { graph: string; outputs: string[] };
  heads: Record<Backend, { graph: string; inputs: string[] }>;
  files: BundleFileRecord[];
  [key: string]: unknown;
}

export type OnnxBundleManifest = BundleManifest | SharedBundleManifest;

export interface TokenizedCandidate {
  inputIds: number[];
  attentionMask: number[];
  tokenTypeIds: number[];
}

export interface OnnxWebBackendOptions {
  /** Use a URL when the manifest is served beside the browser bundle. */
  manifestUrl?: string | URL;
  /** Supply a previously fetched manifest when the caller controls loading. */
  manifest?: OnnxBundleManifest;
  /** Select the head when loading a shared encoder bundle. */
  head?: Backend;
  /** Pin raw fetched bytes for manifestUrl, or stable JSON bytes for an in-memory manifest. */
  expectedManifestSha256?: string;
  /** Required with an in-memory manifest unless the browser has a useful base URI. */
  baseUrl?: string | URL;
  provider?: OnnxExecutionProvider;
  wasmPaths?: string | Record<string, string>;
  wasmNumThreads?: number;
  abstainThreshold?: number;
  answerabilityThreshold?: number;
  /** Override Direct's live candidate count for padding-cost experiments. */
  directOptionSlots?: number;
  /** Freeze the exact noise array when a cross-runtime comparison needs it. */
  initialNoise?: (length: number, seed: number | undefined, questionIndex: number) => Float32Array;
  /** Optional instrumentation hook; it is not part of the decision response contract. */
  onTiming?: (sample: OnnxTimingSample) => void;
  /** Optional local parity trace; disabled by default and not part of the response contract. */
  onTrace?: (sample: OnnxTraceSample) => void;
  fetch?: typeof fetch;
  /** Cancel loading/session startup; after creation, dispose the returned backend. */
  signal?: AbortSignal;
}

export interface OnnxRuntimeInfo {
  provider: OnnxExecutionProvider;
  ortVersion: string | null;
  wasmNumThreads: number | null;
  webgpuAdapterRequested: boolean;
  webgpuDeviceRequested: boolean;
}

export interface OnnxTimingSample {
  questionId: string;
  live_candidates: number;
  allocated_candidates: number;
  live_tokens: number;
  sequence_length: number;
  tokenizeMs: number;
  prepareFeedMs: number;
  inferenceAndReadbackMs: number;
  postprocessMs: number;
  totalRequestMs: number;
}

export interface OnnxTraceSample {
  questionId: string;
  inputIds: number[];
  attentionMask: number[];
  tokenTypeIds: number[];
  optionMask: number[];
  initialNoise: number[] | null;
  rawScores: number[];
  answerabilityLogit: number;
  probabilities: number[];
  answer: QuestionAnswer;
}

interface OrtTensor {
  data: ArrayLike<number>;
  dims: readonly number[];
}

interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>>;
  release(): Promise<void>;
  inputNames: readonly string[];
  outputNames: readonly string[];
}

interface OrtModule {
  Tensor: new (type: string, data: ArrayLike<number> | ArrayLike<bigint>, dims: readonly number[]) => unknown;
  InferenceSession: { create(model: ArrayBufferLike, options?: Record<string, unknown>): Promise<OrtSession> };
  env: {
    wasm: { wasmPaths?: string | Record<string, string>; numThreads?: number };
    versions?: { common?: string; web?: string };
  };
}

interface GpuDeviceLike {
  destroy(): void;
}

interface GpuAdapterLike {
  requestDevice(): Promise<GpuDeviceLike>;
}

interface GpuLike {
  requestAdapter(): Promise<GpuAdapterLike | null>;
}

interface LoadedAssets {
  manifest: OnnxBundleManifest;
  manifestHash: string;
  backend: Backend;
  modelId: string;
  sequenceLength: number;
  optionCount: number;
  diffusionSlots: number;
  defaultAbstainThreshold: number;
  defaultAnswerabilityThreshold: number;
  graphInputs?: GraphInputRecords;
  model: Uint8Array;
  encoderModel?: Uint8Array;
  tokenizer: Tokenizer;
  externalData: Array<{ path: string; data: Uint8Array }>;
  encoderExternalData?: Array<{ path: string; data: Uint8Array }>;
}

type GraphInputRecords = NonNullable<NonNullable<BundleManifest["metadata"]["graph"]>["inputs"]>;

const EXPECTED_SCHEMA = "vons.bundle.manifest/v1";
const EXPECTED_SHARED_SCHEMA = "vons.shared-bundle/v1";
const REQUIRED_DIRECT_INPUTS = ["input_ids", "attention_mask", "token_type_ids", "option_mask"];
const REQUIRED_DIFFUSION_INPUTS = [...REQUIRED_DIRECT_INPUTS, "initial_noise"];
const SHARED_ENCODER_OUTPUTS = ["candidate_embeddings", "pooled"];
const SHARED_HEAD_INPUTS: Record<Backend, string[]> = {
  direct: ["candidate_embeddings", "pooled", "option_mask"],
  diffusion: ["pooled", "option_mask", "initial_noise"],
};

function isFullGraphManifest(manifest: OnnxBundleManifest): manifest is BundleManifest {
  return Boolean(manifest && typeof manifest === "object" && (manifest as BundleManifest).schema_version === EXPECTED_SCHEMA);
}

function isSharedManifest(manifest: OnnxBundleManifest): manifest is SharedBundleManifest {
  return Boolean(manifest && typeof manifest === "object" && (manifest as SharedBundleManifest).schema === EXPECTED_SHARED_SCHEMA);
}

interface SharedBundleConfig {
  max_tokens: number;
  max_options: number;
  diffusion_candidate_slots: number;
  default_abstain_threshold: number;
  default_answerability_threshold: number;
}

function assertFiniteUnit(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new TypeError(`${name} must be finite and in [0, 1]`);
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    const primitive = JSON.stringify(value);
    if (primitive === undefined) throw new TypeError("state must be JSON-compatible");
    return primitive;
  }
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
  return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
}

interface WasmEnvironmentConfig {
  wasmPaths: string | null;
  numThreads: number | null;
}

// ONNX Runtime Web captures these module-global flags when its WASM session initializes.
const wasmEnvironmentConfigs = new WeakMap<object, WasmEnvironmentConfig>();

function wasmPathsKey(value: string | Record<string, string> | undefined): string | null {
  return value === undefined ? null : stableJson(value);
}

function configureWasmEnvironment(runtime: OrtModule, options: OnnxWebBackendOptions): void {
  const wasm = runtime.env.wasm;
  if (options.wasmNumThreads !== undefined && (!Number.isInteger(options.wasmNumThreads) || options.wasmNumThreads < 1)) {
    throw new TypeError("wasmNumThreads must be a positive integer");
  }

  const configured = wasmEnvironmentConfigs.get(wasm);
  if (configured) {
    const current: WasmEnvironmentConfig = {
      wasmPaths: wasmPathsKey(wasm.wasmPaths),
      numThreads: wasm.numThreads ?? null,
    };
    if (current.wasmPaths !== configured.wasmPaths || current.numThreads !== configured.numThreads) {
      throw new Error("ONNX Runtime Web WASM settings changed after this page configured them; reload the page to reconfigure");
    }
    if ((options.wasmPaths !== undefined && wasmPathsKey(options.wasmPaths) !== configured.wasmPaths)
      || (options.wasmNumThreads !== undefined && options.wasmNumThreads !== configured.numThreads)) {
      throw new Error("ONNX Runtime Web WASM settings are page-global; reload the page to use a different configuration");
    }
    return;
  }

  const requestedPaths = wasmPathsKey(options.wasmPaths);
  if (options.wasmPaths !== undefined && wasmPathsKey(wasm.wasmPaths) !== requestedPaths) {
    wasm.wasmPaths = options.wasmPaths;
  }
  if (options.wasmNumThreads !== undefined && wasm.numThreads !== options.wasmNumThreads) {
    wasm.numThreads = options.wasmNumThreads;
  }
  if ((options.wasmPaths !== undefined && wasmPathsKey(wasm.wasmPaths) !== requestedPaths)
    || (options.wasmNumThreads !== undefined && wasm.numThreads !== options.wasmNumThreads)) {
    throw new Error("ONNX Runtime Web did not apply the requested WASM settings; it may already be initialized");
  }
  wasmEnvironmentConfigs.set(wasm, {
    wasmPaths: wasmPathsKey(wasm.wasmPaths),
    numThreads: wasm.numThreads ?? null,
  });
}

export function serializeState(state: DecisionRequest["state"]): string {
  return typeof state === "string" ? state : stableJson(state);
}

export function candidateText(state: DecisionRequest["state"], question: Question, option: string): string {
  return `${serializeState(state)}\nQuestion: ${question.prompt}\nCandidate: ${option}`;
}

export function candidateListText(state: DecisionRequest["state"], question: Question, options: readonly string[]): string {
  return `${serializeState(state)}\nQuestion: ${question.prompt}\nCandidates:\n${options.join("\n")}`;
}

function assertFileRecords(files: BundleFileRecord[], label: string): void {
  if (!Array.isArray(files) || files.length === 0) throw new TypeError(`${label} has no files`);
  const paths = new Set<string>();
  for (const item of files) {
    if (!item || typeof item.path !== "string" || item.path.length === 0 || paths.has(item.path)) {
      throw new TypeError(`${label} contains an invalid or duplicate path: ${item?.path ?? "missing"}`);
    }
    if (item.path.startsWith("/") || item.path.includes("..") || /^[A-Za-z]:[\\/]/.test(item.path)) {
      throw new TypeError(`${label} path must be bundle-relative: ${item.path}`);
    }
    if (!Number.isSafeInteger(item.bytes) || item.bytes < 0 || !/^[0-9a-f]{64}$/.test(item.sha256)) {
      throw new TypeError(`${label} has invalid bytes or sha256: ${item.path}`);
    }
    paths.add(item.path);
  }
}

function assertManifest(manifest: BundleManifest): void {
  if (!manifest || manifest.schema_version !== EXPECTED_SCHEMA) {
    throw new TypeError(`unsupported bundle manifest schema: ${manifest?.schema_version ?? "missing"}`);
  }
  if (!manifest.metadata || (manifest.metadata.backend !== "direct" && manifest.metadata.backend !== "diffusion")) {
    throw new TypeError("bundle manifest metadata.backend must be direct or diffusion");
  }
  assertFileRecords(manifest.files, "bundle manifest");
}

function exactlyOneRecord(files: BundleFileRecord[], role: string): BundleFileRecord {
  const records = files.filter((item) => item.role === role);
  if (records.length !== 1) throw new TypeError(`shared bundle must contain exactly one ${role} file record`);
  return records[0];
}

function assertSharedManifest(manifest: SharedBundleManifest): void {
  if (!manifest || manifest.schema !== EXPECTED_SHARED_SCHEMA) {
    throw new TypeError(`unsupported bundle manifest schema: ${manifest?.schema ?? "missing"}`);
  }
  if (typeof manifest.model_id !== "string" || manifest.model_id.length === 0) {
    throw new TypeError("shared bundle model_id must be a non-empty string");
  }
  if (!manifest.encoder || typeof manifest.encoder.graph !== "string"
    || JSON.stringify(manifest.encoder.outputs) !== JSON.stringify(SHARED_ENCODER_OUTPUTS)) {
    throw new TypeError("shared bundle encoder contract is not supported");
  }
  if (!manifest.heads || !Array.isArray(manifest.heads.direct?.inputs)
    || !Array.isArray(manifest.heads.diffusion?.inputs)) {
    throw new TypeError("shared bundle must declare Direct and Diffusion heads");
  }
  for (const backend of ["direct", "diffusion"] as const) {
    const head = manifest.heads[backend];
    if (typeof head.graph !== "string" || JSON.stringify(head.inputs) !== JSON.stringify(SHARED_HEAD_INPUTS[backend])) {
      throw new TypeError(`shared bundle ${backend} head contract is not supported`);
    }
  }
  assertFileRecords(manifest.files, "shared bundle manifest");
  const required = [
    ["encoder_graph", manifest.encoder.graph],
    ["encoder_weights", `${manifest.encoder.graph}.data`],
    ["direct_graph", manifest.heads.direct.graph],
    ["direct_weights", `${manifest.heads.direct.graph}.data`],
    ["diffusion_graph", manifest.heads.diffusion.graph],
    ["diffusion_weights", `${manifest.heads.diffusion.graph}.data`],
  ] as const;
  for (const [role, path] of required) {
    const record = exactlyOneRecord(manifest.files, role);
    if (record.path !== path) throw new TypeError(`shared bundle ${role} path does not match its graph contract`);
  }
  exactlyOneRecord(manifest.files, "config");
  for (const name of ["tokenizer.json", "tokenizer_config.json"]) {
    const records = manifest.files.filter((item) => item.role === "tokenizer" && (item.path === name || item.path.endsWith(`/${name}`)));
    if (records.length !== 1) throw new TypeError(`shared bundle must include exactly one ${name}`);
  }
}

function assertBundleManifest(manifest: OnnxBundleManifest): void {
  if (isFullGraphManifest(manifest)) assertManifest(manifest);
  else assertSharedManifest(manifest as SharedBundleManifest);
}

function getBaseUrl(options: OnnxWebBackendOptions): URL {
  if (options.baseUrl) return new URL(options.baseUrl.toString(), globalThis.location?.href ?? "http://localhost/");
  if (options.manifestUrl) return new URL(options.manifestUrl.toString(), globalThis.location?.href ?? "http://localhost/");
  if (typeof globalThis.location?.href === "string") return new URL(globalThis.location.href);
  throw new TypeError("baseUrl or manifestUrl is required when running outside a browser");
}

function resolveAssetUrl(base: URL, path: string): URL {
  if (
    path.startsWith("/") ||
    path.includes("\\") ||
    path.includes("..") ||
    /%(?:2e|2f|5c)/i.test(path) ||
    /[?#]/.test(path) ||
    /^[A-Za-z]:[\\/]/.test(path) ||
    /^[A-Za-z][A-Za-z+.-]*:/.test(path)
  ) {
    throw new TypeError(`bundle asset path must be relative: ${path}`);
  }
  const resolved = new URL(path, base);
  const basePath = base.pathname.endsWith("/") ? base.pathname : `${base.pathname}/`;
  if (resolved.origin !== base.origin || !resolved.pathname.startsWith(basePath)) {
    throw new TypeError(`bundle asset path escapes its manifest directory: ${path}`);
  }
  return resolved;
}

async function sha256(bytes: Uint8Array): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error("Web Crypto SHA-256 is unavailable");
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes as unknown as BufferSource);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason === undefined ? new DOMException("The operation was aborted", "AbortError") : signal.reason;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw abortReason(signal);
}

async function awaitOrAbort<T>(promise: Promise<T>, signal?: AbortSignal, onLateValue?: (value: T) => void | Promise<void>): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) {
    if (onLateValue) void promise.then(onLateValue).catch(() => {});
    throw abortReason(signal);
  }
  let onAbort: (() => void) | undefined;
  const aborted = new Promise<never>((_, reject) => {
    onAbort = () => {
      try {
        throwIfAborted(signal);
      } catch (error) {
        reject(error);
      }
    };
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
  });

  try {
    return await Promise.race([promise, aborted]);
  } catch (error) {
    if (signal.aborted && onLateValue) void promise.then(onLateValue).catch(() => {});
    throw error;
  } finally {
    if (onAbort) signal.removeEventListener("abort", onAbort);
  }
}

async function awaitSessionOrAbort(sessionPromise: Promise<OrtSession>, signal?: AbortSignal): Promise<OrtSession> {
  return awaitOrAbort(sessionPromise, signal, (session) => session.release());
}

async function readResponseBytes(response: Response, signal?: AbortSignal): Promise<Uint8Array> {
  if (!signal) return new Uint8Array(await response.arrayBuffer());
  throwIfAborted(signal);

  const reader = response.body?.getReader();
  if (!reader) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    throwIfAborted(signal);
    return bytes;
  }

  const chunks: Uint8Array[] = [];
  let byteLength = 0;
  let finished = false;
  let pendingRead: Promise<ReadableStreamReadResult<Uint8Array>> | undefined;
  let cancelled = false;
  let onAbort: (() => void) | undefined;
  const cancelReader = (reason?: unknown) => {
    if (cancelled) return;
    cancelled = true;
    void reader.cancel(reason).catch(() => {});
  };
  const aborted = new Promise<never>((_, reject) => {
    onAbort = () => {
      const reason = abortReason(signal);
      reject(reason);
      cancelReader(reason);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
  });

  try {
    while (true) {
      throwIfAborted(signal);
      pendingRead = reader.read();
      const result = await Promise.race([pendingRead, aborted]);
      pendingRead = undefined;
      if (result.done) {
        finished = true;
        break;
      }
      if (result.value) {
        chunks.push(result.value);
        byteLength += result.value.byteLength;
      }
    }

    throwIfAborted(signal);
    const bytes = new Uint8Array(byteLength);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return bytes;
  } finally {
    if (onAbort) signal.removeEventListener("abort", onAbort);
    if (finished) {
      reader.releaseLock();
    } else if (pendingRead) {
      cancelReader();
      void pendingRead.then(() => reader.releaseLock(), () => reader.releaseLock());
    } else {
      cancelReader();
      reader.releaseLock();
    }
  }
}

async function fetchBytes(fetchImpl: typeof fetch, url: URL, signal?: AbortSignal): Promise<Uint8Array> {
  const response = await fetchImpl(url, { signal });
  if (signal?.aborted) {
    const reason = abortReason(signal);
    try {
      void response.body?.cancel(reason).catch(() => {});
    } catch {
      // Releasing a response body is best-effort; preserve the abort reason.
    }
    throw reason;
  }
  if (!response.ok) {
    const error = new Error(`failed to fetch ${url}: HTTP ${response.status}`);
    try {
      void response.body?.cancel(error).catch(() => {});
    } catch {
      // Releasing a response body is best-effort; preserve the HTTP error.
    }
    throw error;
  }
  return readResponseBytes(response, signal);
}

async function loadManifest(options: OnnxWebBackendOptions, fetchImpl: typeof fetch): Promise<{ manifest: OnnxBundleManifest; manifestHash: string; base: URL }> {
  throwIfAborted(options.signal);
  if (options.expectedManifestSha256 !== undefined && !/^[0-9a-f]{64}$/.test(options.expectedManifestSha256)) {
    throw new TypeError("expectedManifestSha256 must be a lowercase SHA-256 digest");
  }
  if (options.manifest) {
    assertBundleManifest(options.manifest);
    const manifestBytes = new TextEncoder().encode(stableJson(options.manifest));
    const manifestHash = await sha256(manifestBytes);
    throwIfAborted(options.signal);
    if (options.expectedManifestSha256 !== undefined && manifestHash !== options.expectedManifestSha256) {
      throw new Error(`bundle manifest SHA-256 mismatch: ${manifestHash}`);
    }
    return { manifest: options.manifest, base: getBaseUrl(options), manifestHash };
  }
  if (!options.manifestUrl) throw new TypeError("manifest or manifestUrl is required");
  const url = new URL(options.manifestUrl.toString(), globalThis.location?.href ?? "http://localhost/");
  const bytes = await fetchBytes(fetchImpl, url, options.signal);
  let manifest: OnnxBundleManifest;
  try {
    manifest = JSON.parse(new TextDecoder().decode(bytes)) as OnnxBundleManifest;
  } catch (error) {
    throw new TypeError(`bundle manifest is not valid JSON: ${String(error)}`);
  }
  assertBundleManifest(manifest);
  const manifestHash = await sha256(bytes);
  throwIfAborted(options.signal);
  if (options.expectedManifestSha256 !== undefined && manifestHash !== options.expectedManifestSha256) {
    throw new Error(`bundle manifest SHA-256 mismatch: ${manifestHash}`);
  }
  return { manifest, base: new URL(".", url), manifestHash };
}

async function loadAssets(options: OnnxWebBackendOptions): Promise<LoadedAssets> {
  throwIfAborted(options.signal);
  const fetchImpl = options.fetch ?? globalThis.fetch?.bind(globalThis);
  if (!fetchImpl) throw new Error("fetch is unavailable; provide options.fetch");
  const { manifest, manifestHash, base } = await loadManifest(options, fetchImpl);
  throwIfAborted(options.signal);
  if (isSharedManifest(manifest) && options.head !== "direct" && options.head !== "diffusion") {
    throw new TypeError("shared bundles require head: 'direct' or head: 'diffusion'");
  }
  const cache = new Map<string, Uint8Array>();
  const getFile = async (record: BundleFileRecord): Promise<Uint8Array> => {
    throwIfAborted(options.signal);
    const cached = cache.get(record.path);
    if (cached) return cached;
    const bytes = await fetchBytes(fetchImpl, resolveAssetUrl(base, record.path), options.signal);
    throwIfAborted(options.signal);
    if (bytes.byteLength !== record.bytes) throw new Error(`bundle byte count mismatch: ${record.path}`);
    const actual = await sha256(bytes);
    if (actual !== record.sha256) throw new Error(`bundle SHA-256 mismatch: ${record.path}`);
    cache.set(record.path, bytes);
    return bytes;
  };
  for (const record of manifest.files) resolveAssetUrl(base, record.path);
  const tokenizerJson = manifest.files.find((item) => item.role === "tokenizer" && item.path.endsWith("tokenizer.json"));
  const tokenizerConfig = manifest.files.find((item) => item.role === "tokenizer" && item.path.endsWith("tokenizer_config.json"));
  if (!tokenizerJson || !tokenizerConfig) throw new Error("bundle manifest must include tokenizer.json and tokenizer_config.json");
  const tokenizer = new Tokenizer(
    JSON.parse(new TextDecoder().decode(await getFile(tokenizerJson))) as object,
    JSON.parse(new TextDecoder().decode(await getFile(tokenizerConfig))) as object,
  );
  if (isFullGraphManifest(manifest)) {
    if (options.head !== undefined) throw new TypeError("head may only be selected for a shared bundle");
    const graph = manifest.files.find((item) => item.role === "model_graph");
    if (!graph) throw new Error("bundle manifest has no model graph");
    const model = await getFile(graph);
    const locations = manifest.metadata.graph?.external_data_locations ?? [];
    const externalData: Array<{ path: string; data: Uint8Array }> = [];
    for (const location of locations) {
      const record = manifest.files.find((item) => item.role === "model_external_data" && item.path === location);
      if (!record) throw new Error(`external-data location is missing from the manifest: ${location}`);
      externalData.push({ path: location, data: await getFile(record) });
    }
    return {
      manifest,
      manifestHash,
      backend: manifest.metadata.backend,
      modelId: manifest.metadata.model_id ?? `vons-${manifest.metadata.backend}-onnx-web`,
      sequenceLength: manifest.metadata.sequence_length ?? 512,
      optionCount: manifest.metadata.option_count ?? 32,
      diffusionSlots: manifest.metadata.option_count ?? 32,
      defaultAbstainThreshold: 0.55,
      defaultAnswerabilityThreshold: 0.5,
      graphInputs: manifest.metadata.graph?.inputs,
      model,
      tokenizer,
      externalData,
    };
  }

  const headName = options.head;
  if (headName !== "direct" && headName !== "diffusion") {
    throw new TypeError("shared bundles require head: 'direct' or head: 'diffusion'");
  }
  const shared = manifest as SharedBundleManifest;
  const configRecord = exactlyOneRecord(shared.files, "config");
  const configBytes = await getFile(configRecord);
  let parsedConfig: unknown;
  try {
    parsedConfig = JSON.parse(new TextDecoder().decode(configBytes)) as unknown;
  } catch (error) {
    throw new TypeError(`shared bundle config is not valid JSON: ${String(error)}`);
  }
  if (!parsedConfig || typeof parsedConfig !== "object" || Array.isArray(parsedConfig)) {
    throw new TypeError("shared bundle config is not a JSON object");
  }
  const config = parsedConfig as SharedBundleConfig;
  const { max_tokens: sequenceLength, max_options: optionCount, diffusion_candidate_slots: diffusionSlots,
    default_abstain_threshold: defaultAbstainThreshold,
    default_answerability_threshold: defaultAnswerabilityThreshold } = config;
  if (!Number.isSafeInteger(sequenceLength) || sequenceLength < 1 || sequenceLength > 512
    || !Number.isSafeInteger(optionCount) || optionCount < 2 || optionCount > 32
    || !Number.isSafeInteger(diffusionSlots) || diffusionSlots < optionCount || diffusionSlots > 32) {
    throw new TypeError("shared bundle config has invalid token or candidate limits");
  }
  assertFiniteUnit(defaultAbstainThreshold, "default_abstain_threshold");
  assertFiniteUnit(defaultAnswerabilityThreshold, "default_answerability_threshold");
  const encoderGraph = exactlyOneRecord(shared.files, "encoder_graph");
  const encoderWeights = exactlyOneRecord(shared.files, "encoder_weights");
  const headGraph = exactlyOneRecord(shared.files, `${headName}_graph`);
  const headWeights = exactlyOneRecord(shared.files, `${headName}_weights`);
  const encoderModel = await getFile(encoderGraph);
  const encoderExternalData = [{ path: encoderWeights.path, data: await getFile(encoderWeights) }];
  const model = await getFile(headGraph);
  const externalData = [{ path: headWeights.path, data: await getFile(headWeights) }];
  return {
    manifest,
    manifestHash,
    backend: headName,
    modelId: shared.model_id,
    sequenceLength,
    optionCount,
    diffusionSlots,
    defaultAbstainThreshold,
    defaultAnswerabilityThreshold,
    model,
    encoderModel,
    tokenizer,
    externalData,
    encoderExternalData,
  };
}

function encodeCandidate(tokenizer: Tokenizer, text: string, sequenceLength: number): TokenizedCandidate {
  const encoded = tokenizer.encode(text, { return_token_type_ids: true });
  const inputIds = [...encoded.ids];
  const attentionMask = [...encoded.attention_mask];
  const tokenTypeIds = [...encoded.token_type_ids];
  if (inputIds.length !== attentionMask.length || inputIds.length !== tokenTypeIds.length) {
    throw new Error("tokenizer returned arrays with different lengths");
  }
  if (inputIds.length > sequenceLength) {
    throw new RangeError(`tokenized candidate has ${inputIds.length} tokens; the bundle limit is ${sequenceLength}`);
  }
  if (inputIds.some((value) => !Number.isSafeInteger(value) || value < 0)) throw new Error("tokenizer returned an invalid token id");
  return { inputIds, attentionMask, tokenTypeIds };
}

export function tokenizeCandidates(
  tokenizer: Tokenizer,
  state: DecisionRequest["state"],
  question: Question,
  options: readonly string[],
  sequenceBudget: number,
): { candidates: TokenizedCandidate[]; liveSequenceLength: number } {
  if (!Number.isInteger(sequenceBudget) || sequenceBudget < 1 || sequenceBudget > 512) {
    throw new RangeError("sequenceBudget must be an integer in [1, 512]");
  }
  const candidates = options.map((option) => encodeCandidate(tokenizer, candidateText(state, question, option), sequenceBudget));
  let longest = 0;
  for (const candidate of candidates) {
    const length = candidate.inputIds.length;
    if (length > longest) longest = length;
  }
  if (longest < 1) throw new Error("each tokenized candidate must contain at least one token");
  const liveSequenceLength = Math.min(longest, sequenceBudget);
  return { candidates, liveSequenceLength };
}

function padCandidates(candidates: readonly TokenizedCandidate[], slots: number, sequenceLength: number, paddingId: number): {
  inputIds: BigInt64Array;
  attentionMask: BigInt64Array;
  tokenTypeIds: BigInt64Array;
  optionMask: Uint8Array;
} {
  if (!Number.isInteger(slots) || slots < candidates.length || slots < 1) throw new RangeError("invalid candidate slot count");
  const inputIds = new BigInt64Array(slots * sequenceLength);
  const attentionMask = new BigInt64Array(slots * sequenceLength);
  const tokenTypeIds = new BigInt64Array(slots * sequenceLength);
  const optionMask = new Uint8Array(slots);
  inputIds.fill(BigInt(paddingId));
  for (let optionIndex = 0; optionIndex < candidates.length; optionIndex += 1) {
    const candidate = candidates[optionIndex];
    optionMask[optionIndex] = 1;
    const offset = optionIndex * sequenceLength;
    for (let tokenIndex = 0; tokenIndex < candidate.inputIds.length; tokenIndex += 1) {
      inputIds[offset + tokenIndex] = BigInt(candidate.inputIds[tokenIndex]);
      attentionMask[offset + tokenIndex] = BigInt(candidate.attentionMask[tokenIndex]);
      tokenTypeIds[offset + tokenIndex] = BigInt(candidate.tokenTypeIds[tokenIndex]);
    }
  }
  return { inputIds, attentionMask, tokenTypeIds, optionMask };
}

function resolveFeedSequenceLength(
  graphInputs: GraphInputRecords | undefined,
  liveSequenceLength: number,
  sequenceBudget: number,
): number {
  const inputIds = graphInputs?.find((input) => input.name === "input_ids");
  if (!inputIds || inputIds.shape === null) return liveSequenceLength;
  if (!Array.isArray(inputIds.shape) || inputIds.shape.length !== 3) {
    throw new TypeError("manifest input_ids graph shape must have rank 3");
  }
  const declaredWidth = inputIds.shape[2];
  if (typeof declaredWidth === "string" || declaredWidth === null) return liveSequenceLength;
  if (!Number.isSafeInteger(declaredWidth) || declaredWidth < 1 || declaredWidth > sequenceBudget) {
    throw new TypeError("manifest input_ids graph width must be within the sequence budget");
  }
  if (declaredWidth < liveSequenceLength) {
    throw new RangeError(`tokenized candidate has ${liveSequenceLength} tokens; model input width is ${declaredWidth}`);
  }
  return declaredWidth;
}

export function seededNoise(length: number, seed: number | undefined, questionIndex: number): Float32Array {
  if (!Number.isInteger(length) || length < 1) throw new RangeError("noise length must be positive");
  let state = ((seed ?? 0) ^ (questionIndex + 1) * 0x9e3779b9) >>> 0;
  const result = new Float32Array(length);
  const nextUnit = (): number => {
    state = (Math.imul(state ^ (state >>> 16), 0x45d9f3b) + 0x27100001) >>> 0;
    // Open interval (0, 1) avoids log(0) while keeping the stream deterministic.
    return (state + 1) / 4294967297;
  };
  for (let index = 0; index < length; index += 1) {
    const radius = Math.sqrt(-2 * Math.log(nextUnit()));
    const angle = 2 * Math.PI * nextUnit();
    result[index] = radius * Math.cos(angle);
    if (index + 1 < length) {
      result[index + 1] = radius * Math.sin(angle);
      index += 1;
    }
  }
  return result;
}

function sigmoid(value: number): number {
  if (!Number.isFinite(value)) throw new Error("model returned a non-finite answerability logit");
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

function softmax(values: readonly number[]): number[] {
  if (values.length === 0 || values.some((value) => Number.isNaN(value))) throw new Error("model returned invalid option scores");
  const maximum = Math.max(...values);
  if (!Number.isFinite(maximum)) throw new Error("model returned no finite option score");
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) throw new Error("could not normalize model option scores");
  return exponentials.map((value) => value / total);
}

function tensor(runtime: OrtModule, type: string, data: ArrayLike<number> | ArrayLike<bigint>, dims: readonly number[]): unknown {
  return new runtime.Tensor(type, data, dims);
}

function tensorValues(output: OrtTensor | undefined, expectedLength: number, name: string): number[] {
  if (!output || !output.dims) throw new Error(`missing output tensor: ${name}`);
  const values = Array.from(output.data, Number);
  if (values.length !== expectedLength) throw new Error(`output ${name} has ${values.length} values; expected ${expectedLength}`);
  return values;
}

async function loadOrt(provider: OnnxExecutionProvider, options: OnnxWebBackendOptions): Promise<{ runtime: OrtModule; info: OnnxRuntimeInfo }> {
  throwIfAborted(options.signal);
  if (provider === "webgpu") {
    const gpu = (globalThis.navigator as Navigator & { gpu?: GpuLike } | undefined)?.gpu;
    if (!gpu) throw new Error("WebGPU is unavailable; choose wasm explicitly");
    const adapter = await awaitOrAbort(gpu.requestAdapter(), options.signal);
    throwIfAborted(options.signal);
    if (!adapter) throw new Error("WebGPU adapter request returned null");
    const device = await awaitOrAbort(adapter.requestDevice(), options.signal, (lateDevice) => lateDevice.destroy());
    try {
      throwIfAborted(options.signal);
    } finally {
      // This probe only checks availability; ONNX Runtime creates and owns its own device.
      device.destroy();
    }
  }
  const runtime = (provider === "webgpu"
    ? await import("onnxruntime-web/webgpu")
    : await import("onnxruntime-web/wasm")) as unknown as OrtModule;
  throwIfAborted(options.signal);
  if (provider === "wasm" || options.wasmPaths !== undefined || options.wasmNumThreads !== undefined) {
    configureWasmEnvironment(runtime, options);
  }
  return {
    runtime,
    info: {
      provider,
      ortVersion: runtime.env.versions?.web ?? runtime.env.versions?.common ?? null,
      wasmNumThreads: runtime.env.wasm.numThreads ?? null,
      webgpuAdapterRequested: provider === "webgpu",
      webgpuDeviceRequested: provider === "webgpu",
    },
  };
}

export class OnnxWebBackend implements DecisionBackend {
  readonly backend: Backend;
  readonly runtimeInfo: OnnxRuntimeInfo;
  readonly manifest: OnnxBundleManifest;
  readonly manifestHash: string;
  private readonly tokenizer: Tokenizer;
  private readonly runtime: OrtModule;
  private readonly session: OrtSession;
  private readonly encoderSession: OrtSession | undefined;
  private readonly sequenceLength: number;
  private readonly optionCount: number;
  private readonly diffusionSlots: number;
  private readonly graphInputs: GraphInputRecords | undefined;
  private readonly modelId: string;
  private readonly paddingId: number;
  private readonly abstainThreshold: number;
  private readonly answerabilityThreshold: number;
  private readonly directOptionSlots: number | undefined;
  private readonly initialNoise: OnnxWebBackendOptions["initialNoise"];
  private readonly onTiming: OnnxWebBackendOptions["onTiming"];
  private readonly onTrace: OnnxWebBackendOptions["onTrace"];
  private disposed = false;
  private activeDecisions = 0;
  private decisionsDrained: (() => void) | undefined;
  private releasePromise: Promise<void> | undefined;
  private constructor(
    assets: LoadedAssets,
    runtime: OrtModule,
    runtimeInfo: OnnxRuntimeInfo,
    session: OrtSession,
    options: OnnxWebBackendOptions,
    encoderSession?: OrtSession,
  ) {
  const legacyManifest = isFullGraphManifest(assets.manifest) ? assets.manifest : undefined;
    this.backend = assets.backend ?? legacyManifest?.metadata.backend ?? "direct";
    this.runtimeInfo = runtimeInfo;
    this.manifest = assets.manifest;
    this.manifestHash = assets.manifestHash;
    this.tokenizer = assets.tokenizer;
    this.runtime = runtime;
    this.session = session;
    this.encoderSession = encoderSession;
    this.sequenceLength = assets.sequenceLength ?? legacyManifest?.metadata.sequence_length ?? 512;
    this.optionCount = assets.optionCount ?? legacyManifest?.metadata.option_count ?? 32;
    this.diffusionSlots = assets.diffusionSlots ?? this.optionCount;
    this.graphInputs = assets.graphInputs ?? legacyManifest?.metadata.graph?.inputs;
    this.modelId = assets.modelId ?? legacyManifest?.metadata.model_id ?? `vons-${this.backend}-onnx-web`;
    this.paddingId = assets.tokenizer.token_to_id("[PAD]") ?? 0;
    this.abstainThreshold = options.abstainThreshold ?? assets.defaultAbstainThreshold ?? 0.55;
    this.answerabilityThreshold = options.answerabilityThreshold ?? assets.defaultAnswerabilityThreshold ?? 0.5;
    this.directOptionSlots = options.directOptionSlots;
    this.initialNoise = options.initialNoise;
    this.onTiming = options.onTiming;
    this.onTrace = options.onTrace;
  }

  static async create(options: OnnxWebBackendOptions = {}): Promise<OnnxWebBackend> {
    throwIfAborted(options.signal);
    const assets = await loadAssets(options);
    throwIfAborted(options.signal);
    const provider = options.provider ?? "wasm";
    assertFiniteUnit(options.abstainThreshold ?? 0.55, "abstainThreshold");
    assertFiniteUnit(options.answerabilityThreshold ?? 0.5, "answerabilityThreshold");
    const { runtime, info } = await loadOrt(provider, options);
    throwIfAborted(options.signal);
    const sharedManifest = isSharedManifest(assets.manifest) ? assets.manifest : undefined;
    const expectedInputs = sharedManifest ? SHARED_HEAD_INPUTS[assets.backend]
      : assets.backend === "diffusion" ? REQUIRED_DIFFUSION_INPUTS : REQUIRED_DIRECT_INPUTS;
    const declaredInputs = sharedManifest?.heads[assets.backend].inputs;
    const modelInputs = declaredInputs ?? (isFullGraphManifest(assets.manifest) ? assets.manifest.metadata.input_names : undefined) ?? expectedInputs;
    if (JSON.stringify(modelInputs) !== JSON.stringify(expectedInputs)) throw new Error("bundle input contract is not supported by this backend");
    const outputNames = (isFullGraphManifest(assets.manifest) ? assets.manifest.metadata.output_names : undefined)
      ?? (assets.backend === "diffusion" ? ["scores", "answerability"] : ["logits", "answerability"]);
    const sessions: OrtSession[] = [];
    try {
      let encoderSession: OrtSession | undefined;
      if (assets.encoderModel) {
        encoderSession = await awaitSessionOrAbort(runtime.InferenceSession.create(assets.encoderModel.buffer, {
          executionProviders: [provider],
          externalData: assets.encoderExternalData ?? [],
        }), options.signal);
        sessions.push(encoderSession);
        throwIfAborted(options.signal);
        if (encoderSession.inputNames.length !== REQUIRED_DIRECT_INPUTS.length
          || REQUIRED_DIRECT_INPUTS.some((name) => !encoderSession!.inputNames.includes(name))) {
          throw new Error(`loaded encoder inputs do not match the manifest: ${encoderSession.inputNames.join(",")}`);
        }
        if (encoderSession.outputNames.length !== SHARED_ENCODER_OUTPUTS.length
          || SHARED_ENCODER_OUTPUTS.some((name) => !encoderSession!.outputNames.includes(name))) {
          throw new Error("loaded encoder outputs do not match the manifest");
        }
      }
      const session = await awaitSessionOrAbort(runtime.InferenceSession.create(assets.model.buffer, {
        executionProviders: [provider],
        externalData: assets.externalData,
      }), options.signal);
      sessions.push(session);
      throwIfAborted(options.signal);
      if (session.inputNames.length !== expectedInputs.length || expectedInputs.some((name) => !session.inputNames.includes(name))) {
        throw new Error(`loaded session inputs do not match the manifest: ${session.inputNames.join(",")}`);
      }
      if (sharedManifest && session.outputNames.length !== outputNames.length) {
        throw new Error("loaded shared head outputs do not match the bundle contract");
      }
      if (outputNames.some((name) => !session.outputNames.includes(name))) {
        throw new Error(sharedManifest ? "loaded shared head outputs do not match the bundle contract" : "loaded session outputs do not match the manifest");
      }
      return new OnnxWebBackend(assets, runtime, info, session, options, encoderSession);
    } catch (error) {
      await Promise.allSettled(sessions.map((session) => session.release()));
      throw error;
    }
  }

  async decide(request: DecisionRequest): Promise<DecisionResponse> {
    if (this.disposed) throw new Error("ONNX Runtime session has been disposed");
    this.activeDecisions += 1;
    try {
      validateRequest(request);
      const answers: QuestionAnswer[] = [];
      const sequenceBudget = this.sequenceLength;
      for (const [questionIndex, question] of request.questions.entries()) {
        const requestStarted = performance.now();
        if (question.type === "score") {
          throw new Error("score questions are not supported by this candidate-selection ONNX head");
        }
        const options = question.type === "boolean" && question.options.length === 0 ? ["true", "false"] : question.options;
        const slots = this.backend === "diffusion" ? this.diffusionSlots : this.directOptionSlots ?? options.length;
        if (options.length === 0 || options.length > this.optionCount || slots > this.optionCount) {
          throw new RangeError(`question ${question.id} has unsupported candidate count`);
        }
        const aggregateTokenCount = this.tokenizer.encode(candidateListText(request.state, question, options)).ids.length;
        if (aggregateTokenCount > sequenceBudget) {
          throw new RangeError(`question ${question.id} exceeds the ${sequenceBudget}-token aggregate input budget (tokens=${aggregateTokenCount})`);
        }
        const tokenizeStarted = performance.now();
        const { candidates: encoded, liveSequenceLength } = tokenizeCandidates(this.tokenizer, request.state, question, options, sequenceBudget);
        const tokenized = performance.now();
        const feedSequenceLength = resolveFeedSequenceLength(this.graphInputs, liveSequenceLength, sequenceBudget);
        const padded = padCandidates(encoded, slots, feedSequenceLength, this.paddingId);
        let traceNoise: Float32Array | null = null;
        const feeds: Record<string, unknown> = {
          input_ids: tensor(this.runtime, "int64", padded.inputIds, [1, slots, feedSequenceLength]),
          attention_mask: tensor(this.runtime, "int64", padded.attentionMask, [1, slots, feedSequenceLength]),
          token_type_ids: tensor(this.runtime, "int64", padded.tokenTypeIds, [1, slots, feedSequenceLength]),
          option_mask: tensor(this.runtime, "bool", padded.optionMask, [1, slots]),
        };
        if (this.backend === "diffusion") {
          const noise = this.initialNoise?.(this.diffusionSlots, request.seed, questionIndex) ?? seededNoise(this.diffusionSlots, request.seed, questionIndex);
          if (!(noise instanceof Float32Array) || noise.length !== this.diffusionSlots) throw new TypeError("initialNoise must return a Float32Array with the diffusion slot count");
          traceNoise = noise;
          feeds.initial_noise = tensor(this.runtime, "float32", noise, [1, this.diffusionSlots]);
        }
        const inferenceStarted = performance.now();
        let headFeeds = feeds;
        if (this.encoderSession) {
          const encoderOutputs = await this.encoderSession.run({
            input_ids: feeds.input_ids,
            attention_mask: feeds.attention_mask,
            token_type_ids: feeds.token_type_ids,
            option_mask: feeds.option_mask,
          });
          headFeeds = {
            ...(this.backend === "direct" ? { candidate_embeddings: encoderOutputs.candidate_embeddings } : {}),
            pooled: encoderOutputs.pooled,
            option_mask: feeds.option_mask,
            ...(this.backend === "diffusion" ? { initial_noise: feeds.initial_noise } : {}),
          };
        }
        const outputs = await this.session.run(headFeeds);
        const readbackCompleted = performance.now();
        const scoreName = this.backend === "diffusion" ? "scores" : "logits";
        const scores = tensorValues(outputs[scoreName], slots, scoreName).slice(0, options.length);
        const probabilities = softmax(scores);
        const answerabilityLogit = tensorValues(outputs.answerability, 1, "answerability")[0];
        const answerability = sigmoid(answerabilityLogit);
        const choiceIndex = probabilities.reduce((best, value, index) => value > probabilities[best] ? index : best, 0);
        const confidence = probabilities[choiceIndex] * answerability;
        const abstainReason = answerability < this.answerabilityThreshold
          ? "answerability_below_threshold"
          : confidence < this.abstainThreshold ? "confidence_below_threshold" : null;
        const answer: QuestionAnswer = {
          question_id: question.id,
          choice: abstainReason ? null : options[choiceIndex],
          probabilities: options.map((option, index) => ({ option, probability: probabilities[index] })),
          confidence,
          status: abstainReason ? "abstain" : "ok",
          abstain_reason: abstainReason,
        };
        answers.push(answer);
        this.onTrace?.({
          questionId: question.id,
          inputIds: Array.from(padded.inputIds, Number),
          attentionMask: Array.from(padded.attentionMask, Number),
          tokenTypeIds: Array.from(padded.tokenTypeIds, Number),
          optionMask: Array.from(padded.optionMask, Number),
          initialNoise: traceNoise === null ? null : Array.from(traceNoise),
          rawScores: scores,
          answerabilityLogit,
          probabilities,
          answer,
        });
        const postprocessCompleted = performance.now();
        this.onTiming?.({
          questionId: question.id,
          live_candidates: encoded.length,
          allocated_candidates: slots,
          live_tokens: encoded.reduce((total, candidate) => total + candidate.inputIds.length, 0),
          sequence_length: feedSequenceLength,
          tokenizeMs: tokenized - tokenizeStarted,
          prepareFeedMs: inferenceStarted - tokenized,
          inferenceAndReadbackMs: readbackCompleted - inferenceStarted,
          postprocessMs: postprocessCompleted - readbackCompleted,
          totalRequestMs: postprocessCompleted - requestStarted,
        });
      }
      const response: DecisionResponse = {
        answers,
        backend: this.backend,
        model_id: this.modelId,
        metadata: {
          provider: this.runtimeInfo.provider,
          ort_version: this.runtimeInfo.ortVersion,
          sequence_length: this.sequenceLength,
          candidate_slots: this.backend === "diffusion" ? this.diffusionSlots : this.directOptionSlots ?? "live",
          tokenizer: "@huggingface/tokenizers@0.2.0",
        },
      };
      validateResponseForRequest(response, request);
      return response;
    } finally {
      this.activeDecisions -= 1;
      if (this.activeDecisions === 0) {
        const resolve = this.decisionsDrained;
        this.decisionsDrained = undefined;
        resolve?.();
      }
    }
  }

  async dispose(): Promise<void> {
    if (!this.releasePromise) {
      this.disposed = true;
      const decisionsDrained = this.activeDecisions === 0
        ? Promise.resolve()
        : new Promise<void>((resolve) => { this.decisionsDrained = resolve; });
      this.releasePromise = decisionsDrained.then(async () => {
        const sessions = [this.session, ...(this.encoderSession ? [this.encoderSession] : [])];
        const releases = await Promise.allSettled(sessions.map((session) => session.release()));
        const failures = releases.flatMap((result) => result.status === "rejected" ? [result.reason] : []);
        if (failures.length === 1) throw failures[0];
        if (failures.length > 1) throw new AggregateError(failures, "failed to release ONNX Runtime sessions");
      });
    }
    await this.releasePromise;
  }
}

export async function createOnnxWebBackend(options: OnnxWebBackendOptions = {}): Promise<OnnxWebBackend> {
  return OnnxWebBackend.create(options);
}

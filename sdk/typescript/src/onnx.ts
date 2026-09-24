import { Tokenizer } from "@huggingface/tokenizers";

import {
  type Backend,
  type DecisionBackend,
  type DecisionRequest,
  type DecisionResponse,
  type Question,
  type QuestionAnswer,
  validateRequest,
  validateResponse,
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

export interface TokenizedCandidate {
  inputIds: number[];
  attentionMask: number[];
  tokenTypeIds: number[];
}

export interface OnnxWebBackendOptions {
  /** Use a URL when the manifest is served beside the browser bundle. */
  manifestUrl?: string | URL;
  /** Supply a previously fetched manifest when the caller controls loading. */
  manifest?: BundleManifest;
  /** Pin the exact manifest bytes expected by a release or measurement run. */
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

interface GpuLike {
  requestAdapter(): Promise<{ requestDevice(): Promise<unknown> } | null>;
}

interface LoadedAssets {
  manifest: BundleManifest;
  manifestHash: string;
  model: Uint8Array;
  tokenizer: Tokenizer;
  externalData: Array<{ path: string; data: Uint8Array }>;
}

const EXPECTED_SCHEMA = "vons.bundle.manifest/v1";
const REQUIRED_DIRECT_INPUTS = ["input_ids", "attention_mask", "token_type_ids", "option_mask"];
const REQUIRED_DIFFUSION_INPUTS = [...REQUIRED_DIRECT_INPUTS, "initial_noise"];

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

export function serializeState(state: DecisionRequest["state"]): string {
  return typeof state === "string" ? state : stableJson(state);
}

export function candidateText(state: DecisionRequest["state"], question: Question, option: string): string {
  return `${serializeState(state)}\nQuestion: ${question.prompt}\nCandidate: ${option}`;
}

export function candidateListText(state: DecisionRequest["state"], question: Question, options: readonly string[]): string {
  return `${serializeState(state)}\nQuestion: ${question.prompt}\nCandidates:\n${options.join("\n")}`;
}

function assertManifest(manifest: BundleManifest): void {
  if (!manifest || manifest.schema_version !== EXPECTED_SCHEMA) {
    throw new TypeError(`unsupported bundle manifest schema: ${manifest?.schema_version ?? "missing"}`);
  }
  if (!manifest.metadata || (manifest.metadata.backend !== "direct" && manifest.metadata.backend !== "diffusion")) {
    throw new TypeError("bundle manifest metadata.backend must be direct or diffusion");
  }
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) throw new TypeError("bundle manifest has no files");
  const paths = new Set<string>();
  for (const item of manifest.files) {
    if (!item || typeof item.path !== "string" || item.path.length === 0 || paths.has(item.path)) {
      throw new TypeError(`bundle manifest contains an invalid or duplicate path: ${item?.path ?? "missing"}`);
    }
    if (item.path.startsWith("/") || item.path.includes("..") || /^[A-Za-z]:[\\/]/.test(item.path)) {
      throw new TypeError(`bundle manifest path must be bundle-relative: ${item.path}`);
    }
    if (!Number.isSafeInteger(item.bytes) || item.bytes < 0 || !/^[0-9a-f]{64}$/.test(item.sha256)) {
      throw new TypeError(`bundle manifest has invalid bytes or sha256: ${item.path}`);
    }
    paths.add(item.path);
  }
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

async function fetchBytes(fetchImpl: typeof fetch, url: URL, signal?: AbortSignal): Promise<Uint8Array> {
  const response = await fetchImpl(url, { signal });
  if (!response.ok) throw new Error(`failed to fetch ${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function loadManifest(options: OnnxWebBackendOptions, fetchImpl: typeof fetch): Promise<{ manifest: BundleManifest; manifestHash: string; base: URL }> {
  if (options.expectedManifestSha256 !== undefined && !/^[0-9a-f]{64}$/.test(options.expectedManifestSha256)) {
    throw new TypeError("expectedManifestSha256 must be a lowercase SHA-256 digest");
  }
  if (options.manifest) {
    assertManifest(options.manifest);
    const manifestBytes = new TextEncoder().encode(stableJson(options.manifest));
    const manifestHash = await sha256(manifestBytes);
    if (options.expectedManifestSha256 !== undefined && manifestHash !== options.expectedManifestSha256) {
      throw new Error(`bundle manifest SHA-256 mismatch: ${manifestHash}`);
    }
    return { manifest: options.manifest, base: getBaseUrl(options), manifestHash };
  }
  if (!options.manifestUrl) throw new TypeError("manifest or manifestUrl is required");
  const url = new URL(options.manifestUrl.toString(), globalThis.location?.href ?? "http://localhost/");
  const bytes = await fetchBytes(fetchImpl, url, options.signal);
  let manifest: BundleManifest;
  try {
    manifest = JSON.parse(new TextDecoder().decode(bytes)) as BundleManifest;
  } catch (error) {
    throw new TypeError(`bundle manifest is not valid JSON: ${String(error)}`);
  }
  assertManifest(manifest);
  const manifestHash = await sha256(bytes);
  if (options.expectedManifestSha256 !== undefined && manifestHash !== options.expectedManifestSha256) {
    throw new Error(`bundle manifest SHA-256 mismatch: ${manifestHash}`);
  }
  return { manifest, base: new URL(".", url), manifestHash };
}

async function loadAssets(options: OnnxWebBackendOptions): Promise<LoadedAssets> {
  const fetchImpl = options.fetch ?? globalThis.fetch?.bind(globalThis);
  if (!fetchImpl) throw new Error("fetch is unavailable; provide options.fetch");
  const { manifest, manifestHash, base } = await loadManifest(options, fetchImpl);
  const cache = new Map<string, Uint8Array>();
  const getFile = async (record: BundleFileRecord): Promise<Uint8Array> => {
    const cached = cache.get(record.path);
    if (cached) return cached;
    const bytes = await fetchBytes(fetchImpl, resolveAssetUrl(base, record.path), options.signal);
    if (bytes.byteLength !== record.bytes) throw new Error(`bundle byte count mismatch: ${record.path}`);
    const actual = await sha256(bytes);
    if (actual !== record.sha256) throw new Error(`bundle SHA-256 mismatch: ${record.path}`);
    cache.set(record.path, bytes);
    return bytes;
  };
  const graph = manifest.files.find((item) => item.role === "model_graph");
  if (!graph) throw new Error("bundle manifest has no model graph");
  const tokenizerJson = manifest.files.find((item) => item.role === "tokenizer" && item.path.endsWith("tokenizer.json"));
  const tokenizerConfig = manifest.files.find((item) => item.role === "tokenizer" && item.path.endsWith("tokenizer_config.json"));
  if (!tokenizerJson || !tokenizerConfig) throw new Error("bundle manifest must include tokenizer.json and tokenizer_config.json");
  const model = await getFile(graph);
  const tokenizer = new Tokenizer(
    JSON.parse(new TextDecoder().decode(await getFile(tokenizerJson))) as object,
    JSON.parse(new TextDecoder().decode(await getFile(tokenizerConfig))) as object,
  );
  const locations = manifest.metadata.graph?.external_data_locations ?? [];
  const externalData: Array<{ path: string; data: Uint8Array }> = [];
  for (const location of locations) {
    const record = manifest.files.find((item) => item.role === "model_external_data" && item.path === location);
    if (!record) throw new Error(`external-data location is missing from the manifest: ${location}`);
    externalData.push({ path: location, data: await getFile(record) });
  }
  return { manifest, manifestHash, model, tokenizer, externalData };
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
  sequenceLength: number,
): TokenizedCandidate[] {
  return options.map((option) => encodeCandidate(tokenizer, candidateText(state, question, option), sequenceLength));
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
  if (provider === "webgpu") {
    const gpu = (globalThis.navigator as Navigator & { gpu?: GpuLike } | undefined)?.gpu;
    if (!gpu) throw new Error("WebGPU is unavailable; choose wasm explicitly");
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU adapter request returned null");
    await adapter.requestDevice();
  }
  const runtime = (provider === "webgpu"
    ? await import("onnxruntime-web/webgpu")
    : await import("onnxruntime-web/wasm")) as unknown as OrtModule;
  if (options.wasmPaths) runtime.env.wasm.wasmPaths = options.wasmPaths;
  if (provider === "wasm") {
    if (options.wasmNumThreads !== undefined) {
      if (!Number.isInteger(options.wasmNumThreads) || options.wasmNumThreads < 1) throw new TypeError("wasmNumThreads must be a positive integer");
      runtime.env.wasm.numThreads = options.wasmNumThreads;
    }
  }
  return {
    runtime,
    info: {
      provider,
      ortVersion: runtime.env.versions?.web ?? runtime.env.versions?.common ?? null,
      wasmNumThreads: provider === "wasm" ? runtime.env.wasm.numThreads ?? null : null,
      webgpuAdapterRequested: provider === "webgpu",
      webgpuDeviceRequested: provider === "webgpu",
    },
  };
}

export class OnnxWebBackend implements DecisionBackend {
  readonly backend: Backend;
  readonly runtimeInfo: OnnxRuntimeInfo;
  readonly manifest: BundleManifest;
  readonly manifestHash: string;
  private readonly tokenizer: Tokenizer;
  private readonly runtime: OrtModule;
  private readonly session: OrtSession;
  private readonly sequenceLength: number;
  private readonly optionCount: number;
  private readonly paddingId: number;
  private readonly abstainThreshold: number;
  private readonly answerabilityThreshold: number;
  private readonly directOptionSlots: number | undefined;
  private readonly initialNoise: OnnxWebBackendOptions["initialNoise"];
  private readonly onTiming: OnnxWebBackendOptions["onTiming"];
  private readonly onTrace: OnnxWebBackendOptions["onTrace"];
  private disposed = false;
  private constructor(assets: LoadedAssets, runtime: OrtModule, runtimeInfo: OnnxRuntimeInfo, session: OrtSession, options: OnnxWebBackendOptions) {
    this.backend = assets.manifest.metadata.backend;
    this.runtimeInfo = runtimeInfo;
    this.manifest = assets.manifest;
    this.manifestHash = assets.manifestHash;
    this.tokenizer = assets.tokenizer;
    this.runtime = runtime;
    this.session = session;
    this.sequenceLength = assets.manifest.metadata.sequence_length ?? 512;
    this.optionCount = assets.manifest.metadata.option_count ?? 32;
    this.paddingId = assets.tokenizer.token_to_id("[PAD]") ?? 0;
    this.abstainThreshold = options.abstainThreshold ?? 0.55;
    this.answerabilityThreshold = options.answerabilityThreshold ?? 0.5;
    this.directOptionSlots = options.directOptionSlots;
    this.initialNoise = options.initialNoise;
    this.onTiming = options.onTiming;
    this.onTrace = options.onTrace;
  }

  static async create(options: OnnxWebBackendOptions = {}): Promise<OnnxWebBackend> {
    const assets = await loadAssets(options);
    const provider = options.provider ?? "wasm";
    assertFiniteUnit(options.abstainThreshold ?? 0.55, "abstainThreshold");
    assertFiniteUnit(options.answerabilityThreshold ?? 0.5, "answerabilityThreshold");
    const { runtime, info } = await loadOrt(provider, options);
    const expectedInputs = assets.manifest.metadata.backend === "diffusion" ? REQUIRED_DIFFUSION_INPUTS : REQUIRED_DIRECT_INPUTS;
    const modelInputs = assets.manifest.metadata.input_names ?? expectedInputs;
    if (JSON.stringify(modelInputs) !== JSON.stringify(expectedInputs)) throw new Error("bundle input contract is not supported by this backend");
    const outputNames = assets.manifest.metadata.output_names ?? (assets.manifest.metadata.backend === "diffusion" ? ["scores", "answerability"] : ["logits", "answerability"]);
    const session = await runtime.InferenceSession.create(assets.model.buffer, {
      executionProviders: [provider],
      externalData: assets.externalData,
    });
    try {
      if (session.inputNames.length !== expectedInputs.length || expectedInputs.some((name) => !session.inputNames.includes(name))) {
        throw new Error(`loaded session inputs do not match the manifest: ${session.inputNames.join(",")}`);
      }
      if (outputNames.some((name) => !session.outputNames.includes(name))) throw new Error("loaded session outputs do not match the manifest");
    } catch (error) {
      await session.release();
      throw error;
    }
    return new OnnxWebBackend(assets, runtime, info, session, options);
  }

  async decide(request: DecisionRequest): Promise<DecisionResponse> {
    if (this.disposed) throw new Error("ONNX Runtime session has been disposed");
    validateRequest(request);
    const answers: QuestionAnswer[] = [];
    for (const [questionIndex, question] of request.questions.entries()) {
      const requestStarted = performance.now();
      if (question.type === "score") {
        throw new Error("score questions are not supported by this candidate-selection ONNX head");
      }
      const options = question.type === "boolean" && question.options.length === 0 ? ["true", "false"] : question.options;
      const slots = this.backend === "diffusion" ? this.optionCount : this.directOptionSlots ?? options.length;
      if (options.length === 0 || options.length > this.optionCount || slots > this.optionCount) {
        throw new RangeError(`question ${question.id} has unsupported candidate count`);
      }
      const aggregateTokenCount = this.tokenizer.encode(candidateListText(request.state, question, options)).ids.length;
      if (aggregateTokenCount > this.sequenceLength) {
        throw new RangeError(`question ${question.id} exceeds the ${this.sequenceLength}-token aggregate input budget (tokens=${aggregateTokenCount})`);
      }
      const tokenizeStarted = performance.now();
      const encoded = tokenizeCandidates(this.tokenizer, request.state, question, options, this.sequenceLength);
      const tokenized = performance.now();
      const padded = padCandidates(encoded, slots, this.sequenceLength, this.paddingId);
      let traceNoise: Float32Array | null = null;
      const feeds: Record<string, unknown> = {
        input_ids: tensor(this.runtime, "int64", padded.inputIds, [1, slots, this.sequenceLength]),
        attention_mask: tensor(this.runtime, "int64", padded.attentionMask, [1, slots, this.sequenceLength]),
        token_type_ids: tensor(this.runtime, "int64", padded.tokenTypeIds, [1, slots, this.sequenceLength]),
        option_mask: tensor(this.runtime, "bool", padded.optionMask, [1, slots]),
      };
      if (this.backend === "diffusion") {
        const noise = this.initialNoise?.(this.optionCount, request.seed, questionIndex) ?? seededNoise(this.optionCount, request.seed, questionIndex);
        if (!(noise instanceof Float32Array) || noise.length !== this.optionCount) throw new TypeError("initialNoise must return a Float32Array with option_count values");
        traceNoise = noise;
        feeds.initial_noise = tensor(this.runtime, "float32", noise, [1, this.optionCount]);
      }
      const inferenceStarted = performance.now();
      const outputs = await this.session.run(feeds);
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
        sequence_length: this.sequenceLength,
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
      model_id: this.manifest.metadata.model_id ?? `vons-${this.backend}-onnx-web`,
      metadata: {
        provider: this.runtimeInfo.provider,
        ort_version: this.runtimeInfo.ortVersion,
        sequence_length: this.sequenceLength,
        candidate_slots: this.backend === "diffusion" ? this.optionCount : this.directOptionSlots ?? "live",
        tokenizer: "@huggingface/tokenizers@0.2.0",
      },
    };
    validateResponse(response);
    return response;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.session.release();
  }
}

export async function createOnnxWebBackend(options: OnnxWebBackendOptions = {}): Promise<OnnxWebBackend> {
  return OnnxWebBackend.create(options);
}

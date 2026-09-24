export type QuestionType = "choice" | "boolean" | "score";
export type Backend = "direct" | "diffusion";
export type ResponseStatus = "ok" | "abstain";

const validQuestionTypes = new Set<QuestionType>(["choice", "boolean", "score"]);

export interface Question {
  id: string;
  type: QuestionType;
  prompt: string;
  options: string[];
  rubric?: string[];
}

export interface DecisionRequest {
  state: string | Record<string, unknown>;
  questions: Question[];
  backend?: Backend;
  seed?: number;
}

export interface OptionProbability {
  option: string;
  probability: number;
}

export interface QuestionAnswer {
  question_id: string;
  choice: string | null;
  probabilities: OptionProbability[];
  confidence: number;
  status: ResponseStatus;
  abstain_reason?: string | null;
}

export interface DecisionResponse {
  answers: QuestionAnswer[];
  backend: Backend;
  model_id: string;
  metadata?: Record<string, unknown>;
}

export type ToolRisk = "low" | "medium" | "high" | "critical";

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface KASIProposal {
  calls?: ToolCall[];
  confidence: number;
  risk?: string | null;
  response?: string | null;
}

export type KASIAction = "call" | "clarify" | "confirm" | "refuse" | "respond";

export interface KASIActionDecision {
  action: KASIAction;
  calls: ToolCall[];
  prompt: string | null;
  reason: string;
}

export interface DecisionBackend {
  readonly backend: Backend;
  decide(request: DecisionRequest): Promise<DecisionResponse>;
}

const toolNamePattern = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;
const validRisks = new Set<ToolRisk>(["low", "medium", "high", "critical"]);
const riskRank: Record<ToolRisk, number> = { low: 0, medium: 1, high: 2, critical: 3 };

function assertFiniteUnit(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new TypeError(`${name} must be finite and in [0, 1]`);
  }
}

function assertToolName(name: string): void {
  if (typeof name !== "string" || !toolNamePattern.test(name)) {
    throw new TypeError("tool name must match [A-Za-z0-9][A-Za-z0-9_.:-]{0,127}");
  }
}

function normalizeRisk(value: string | null | undefined): ToolRisk | null {
  return value && validRisks.has(value.trim().toLowerCase() as ToolRisk)
    ? (value.trim().toLowerCase() as ToolRisk)
    : null;
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    const primitive = JSON.stringify(value);
    if (primitive === undefined) throw new TypeError("arguments must be JSON-compatible");
    return primitive;
  }
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
  return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
}

export function callFingerprint(call: ToolCall): string {
  assertToolName(call.name);
  // SHA-256 is supplied by the host in production. The portable SDK key is a
  // canonical consent key; callers may hash it before persistence.
  return stableJson({ name: call.name, arguments: call.arguments });
}

export function validateRequest(request: DecisionRequest, maxQuestions = 8, maxTokens = 512): void {
  if (!request || typeof request !== "object") throw new TypeError("request must be an object");
  if (!Number.isInteger(maxQuestions) || maxQuestions < 1) throw new TypeError("maxQuestions must be a positive integer");
  if (!Number.isInteger(maxTokens) || maxTokens < 1) throw new TypeError("maxTokens must be a positive integer");
  if (typeof request.state !== "string" && (!request.state || typeof request.state !== "object" || Array.isArray(request.state))) {
    throw new TypeError("state must be a string or object");
  }
  if (typeof request.state === "string" && request.state.trim().length === 0) {
    throw new TypeError("state must not be empty");
  }
  if (!Array.isArray(request.questions) || request.questions.length < 1 || request.questions.length > maxQuestions) {
    throw new TypeError(`questions must contain 1..${maxQuestions} items`);
  }
  const ids = new Set<string>();
  for (const question of request.questions) {
    if (!question || typeof question !== "object") throw new TypeError("questions must contain objects");
    if (typeof question.id !== "string" || !question.id || ids.has(question.id)) throw new TypeError("question ids must be unique and non-empty");
    ids.add(question.id);
    if (!validQuestionTypes.has(question.type)) throw new TypeError(`question ${question.id} has an invalid type`);
    if (typeof question.prompt !== "string" || !question.prompt.trim()) throw new TypeError(`question ${question.id} has an empty prompt`);
    if (!Array.isArray(question.options) || question.options.some((option) => typeof option !== "string" || !option.trim())) {
      throw new TypeError(`question ${question.id} options must be strings`);
    }
    if (question.type === "boolean") {
      if (question.options.length > 0 && JSON.stringify(question.options) !== JSON.stringify(["true", "false"])) {
        throw new TypeError("boolean questions must use true/false options");
      }
    } else if (!Array.isArray(question.options) || question.options.length < 2 || question.options.length > 32) {
      throw new TypeError(`question ${question.id} must contain 2..32 options`);
    }
    if (new Set(question.options).size !== question.options.length) throw new TypeError("question options must be unique");
    if (question.rubric !== undefined && !Array.isArray(question.rubric)) {
      throw new TypeError("question rubric must be a list");
    }
    if (question.type === "score" && (!question.rubric || question.rubric.length < 2 || question.rubric.length > 10)) {
      throw new TypeError("score questions must contain a 2..10 item rubric");
    }
    if (question.type === "score" && question.rubric?.some((item) => typeof item !== "string")) {
      throw new TypeError("score question rubric items must be strings");
    }
    if (question.type !== "score" && question.rubric?.length) {
      throw new TypeError(`only score question ${question.id} may provide rubric`);
    }
  }
  // Tokenizer-specific budget checks belong to the runtime adapter. A character
  // estimate here would reject valid WordPiece inputs and cannot account for
  // the aggregate state/question/all-candidate contract. Keep maxTokens in the
  // signature for API compatibility and let adapters enforce their exact limit.
  void maxTokens;
}

export function validateResponse(response: DecisionResponse): void {
  if (!response || typeof response !== "object") throw new TypeError("response must be an object");
  if (typeof response.model_id !== "string" || !response.model_id.trim()) throw new TypeError("model_id is required");
  if (response.backend !== "direct" && response.backend !== "diffusion") throw new TypeError("invalid backend");
  if (!Array.isArray(response.answers)) throw new TypeError("answers must be a list");
  if (response.metadata !== undefined && (!response.metadata || typeof response.metadata !== "object" || Array.isArray(response.metadata))) {
    throw new TypeError("metadata must be an object");
  }
  const ids = new Set<string>();
  for (const answer of response.answers) {
    if (!answer || typeof answer !== "object") throw new TypeError("answers must contain objects");
    if (typeof answer.question_id !== "string" || !answer.question_id || ids.has(answer.question_id)) throw new TypeError("answer question ids must be unique and non-empty");
    ids.add(answer.question_id);
    if (!Array.isArray(answer.probabilities)) throw new TypeError("probabilities must be a list");
    assertFiniteUnit(answer.confidence, "confidence");
    const options = new Set<string>();
    let total = 0;
    for (const item of answer.probabilities) {
      if (!item || typeof item !== "object" || typeof item.option !== "string" || !item.option) {
        throw new TypeError("probability options must be non-empty strings");
      }
      if (options.has(item.option)) throw new TypeError("probability options must be unique");
      options.add(item.option);
      assertFiniteUnit(item.probability, "probability");
      total += item.probability;
    }
    if (answer.probabilities.length > 0 && Math.abs(total - 1) > 1e-6) throw new TypeError("probabilities must sum to 1");
    if (answer.status !== "ok" && answer.status !== "abstain") throw new TypeError("invalid response status");
    if (answer.choice !== null && typeof answer.choice !== "string") throw new TypeError("choice must be a string or null");
    if (answer.abstain_reason !== undefined && answer.abstain_reason !== null && typeof answer.abstain_reason !== "string") {
      throw new TypeError("abstain_reason must be a string or null");
    }
    if (answer.status === "ok" && !answer.choice) throw new TypeError("ok answers require choice");
    if (answer.status === "abstain" && (answer.choice !== null || !answer.abstain_reason?.trim())) {
      throw new TypeError("abstain answers require a reason and no choice");
    }
    if (answer.status === "ok" && answer.probabilities.length > 0 && !options.has(answer.choice!)) {
      throw new TypeError("choice must be present in probabilities");
    }
  }
}

export class KASIAdapter {
  private readonly policy: ReadonlyMap<string, ToolRisk>;
  private readonly threshold: number;

  constructor(policy: Record<string, ToolRisk>, threshold = 0.55) {
    if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) throw new TypeError("invalid threshold");
    this.threshold = threshold;
    const entries: [string, ToolRisk][] = [];
    for (const [name, risk] of Object.entries(policy)) {
      assertToolName(name);
      if (!validRisks.has(risk)) throw new TypeError(`invalid risk for ${name}`);
      entries.push([name, risk]);
    }
    this.policy = new Map(entries);
  }

  decide(proposal: KASIProposal, confirmedCalls: ReadonlySet<string> = new Set()): KASIActionDecision {
    assertFiniteUnit(proposal.confidence, "confidence");
    const calls = proposal.calls ?? [];
    if (calls.length > 0 && proposal.response) throw new TypeError("proposal cannot contain both calls and response");
    if (proposal.response && !(proposal.calls?.length)) {
      if (proposal.confidence >= this.threshold) return { action: "respond", calls: [], prompt: proposal.response, reason: "direct_response" };
    }
    if (proposal.confidence < this.threshold) return { action: "clarify", calls, prompt: "Please provide the missing information before acting.", reason: "low_confidence" };
    if (calls.length === 0) return { action: "refuse", calls, prompt: "No safe candidate action is available.", reason: "no_proposed_call" };
    const advisory = normalizeRisk(proposal.risk);
    if (proposal.risk !== null && proposal.risk !== undefined && !advisory) {
      return { action: "confirm", calls, prompt: "Confirm action with unknown model risk.", reason: "unknown_model_risk" };
    }
    for (const call of calls) {
      assertToolName(call.name);
      const hostRisk = this.policy.get(call.name);
      if (!hostRisk) return { action: "refuse", calls: [], prompt: "Tool is not registered by the host.", reason: "unregistered_tool" };
      const needsConsent = !advisory || Math.max(riskRank[hostRisk], riskRank[advisory]) >= riskRank.high;
      if (needsConsent && !confirmedCalls.has(callFingerprint(call))) {
        return { action: "confirm", calls, prompt: `Confirm action: ${call.name}(${stableJson(call.arguments)}).`, reason: "unconfirmed_or_unknown_risk" };
      }
    }
    return { action: "call", calls, prompt: null, reason: "approved_by_host_policy" };
  }
}

export function selectRuntime(preferred: "wasm" | "webgpu" = "wasm"): "wasm" | "webgpu" {
  if (preferred === "webgpu") {
    const scope = globalThis as { navigator?: { gpu?: unknown }; gpu?: unknown };
    const gpu = scope.navigator?.gpu ?? scope.gpu;
    if (!gpu) throw new Error("WebGPU is unavailable; choose wasm explicitly");
    return "webgpu";
  }
  if (typeof WebAssembly === "undefined") throw new Error("WebAssembly is unavailable");
  return "wasm";
}

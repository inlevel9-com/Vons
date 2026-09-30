import { readFileSync, realpathSync, statSync, existsSync } from "node:fs";
import { resolve, isAbsolute, sep, join } from "node:path";
import { createHash } from "node:crypto";
import type { Readable, Writable } from "node:stream";

import { McpServer } from "@modelcontextprotocol/server";
import type { Transport } from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";

import {
  type Backend,
  type DecisionBackend,
  type DecisionRequest,
  type DecisionResponse,
  validateRequest,
  validateResponse,
  validateResponseForRequest,
} from "./index.ts";
import type { BundleManifest } from "./onnx.ts";

const BUNDLE_MANIFEST_NAME = "bundle-manifest-v1.json";
const EXPECTED_BUNDLE_SCHEMA = "vons.bundle.manifest/v1";
const MANIFEST_RESOURCE_URI = "vons://bundle/manifest/metadata";
const MANIFEST_RESOURCE_NAME = "bundle-manifest-metadata";

const NonBlankZodString = z.string().regex(/\S/, {
  message: "value must contain at least one non-whitespace character",
});

const QuestionFields = {
  id: z.string().min(1),
  prompt: NonBlankZodString,
};
const NonScoreEmptyRubricZodSchema = z.array(z.string()).length(0);
const QuestionZodSchema = z.discriminatedUnion("type", [
  z.strictObject({
    ...QuestionFields,
    type: z.literal("choice"),
    options: z.array(NonBlankZodString).min(2).max(32),
    rubric: NonScoreEmptyRubricZodSchema.optional(),
  }),
  z.strictObject({
    ...QuestionFields,
    type: z.literal("boolean"),
    options: z.union([
      z.array(z.string()).length(0),
      z.tuple([z.literal("true"), z.literal("false")]),
    ]),
    rubric: NonScoreEmptyRubricZodSchema.optional(),
  }),
  z.strictObject({
    ...QuestionFields,
    type: z.literal("score"),
    options: z.array(NonBlankZodString).min(2).max(32),
    rubric: z.array(z.string()).min(2).max(10),
  }),
]);

export const DecisionRequestZodSchema = z.strictObject({
  state: z.union([
    NonBlankZodString,
    z.record(z.string(), z.unknown()).refine((value) => !Array.isArray(value), { message: "state must not be an array" }),
  ]),
  questions: z.array(QuestionZodSchema).min(1).max(8),
  backend: z.enum(["direct", "diffusion"]).optional(),
  seed: z.number().int().nonnegative().optional(),
});

export interface BundleMetadataResource {
  schema_version: "vons.bundle.metadata/v1";
  backend: Backend | null;
  model_id: string | null;
  option_count: number | null;
  sequence_length: number | null;
}

export interface McpServerOptions {
  backend: DecisionBackend;
  manifest?: BundleManifest | null;
  manifestSha256?: string | null;
  bundleRoot?: string | null;
  logger?: ((level: "debug" | "info" | "warn" | "error", message: string) => void) | null;
}

export function isForbiddenBundlePath(value: string): boolean {
  if (typeof value !== "string") return true;
  if (value.length === 0) return true;
  if (/^(https?|file|data|ftp):\/\//i.test(value.trim())) return true;
  if (value.startsWith("file://")) return true;
  if (isAbsolute(value)) return true;
  if (/^[A-Za-z]:[\\/]/.test(value)) return true;
  if (value.includes("\\")) return true;
  if (value.split("/").some((segment) => segment === "." || segment === "..")) return true;
  if (/%(?:2e|2f|5c)/i.test(value)) return true;
  if (/[?#]/.test(value)) return true;
  return false;
}

export interface StrictBundleLoaderResult {
  rootRealPath: string;
  manifest: BundleManifest;
  manifestSha256: string;
  manifestPath: string;
}

export function loadBundleManifestStrict(bundleDir: string, expectedSha256?: string | null): StrictBundleLoaderResult {
  if (typeof bundleDir !== "string" || bundleDir.length === 0) {
    throw new TypeError("--bundle directory path must be a non-empty string");
  }
  if (/^(https?|file):\/\//i.test(bundleDir.trim())) {
    throw new TypeError("--bundle must be a local directory path; URLs are forbidden");
  }
  if (!existsSync(bundleDir)) {
    throw new Error(`bundle directory not found: ${bundleDir}`);
  }
  const dirStat = statSync(bundleDir);
  if (!dirStat.isDirectory()) {
    throw new TypeError(`bundle path is not a directory: ${bundleDir}`);
  }
  const rootRealPath = realpathSync(bundleDir);
  const manifestPath = join(rootRealPath, BUNDLE_MANIFEST_NAME);
  if (!existsSync(manifestPath)) {
    throw new Error(
      `bundle manifest ${BUNDLE_MANIFEST_NAME} not found in bundle directory. ` +
        `The user must supply a hash-verified bundle; no auto-discovery is performed.`,
    );
  }
  let manifestPathReal: string | null = null;
  try {
    manifestPathReal = realpathSync(manifestPath);
  } catch {
    throw new Error(`bundle manifest ${BUNDLE_MANIFEST_NAME} cannot be resolved; check symlink targets and permissions`);
  }
  if (manifestPathReal !== join(rootRealPath, BUNDLE_MANIFEST_NAME)) {
    throw new Error(
      `bundle manifest escapes bundle root via symlink: ${BUNDLE_MANIFEST_NAME}`,
    );
  }
  // Re-resolve and read through this checked target so a replaced symlink cannot redirect the later read outside the bundle.
  const manifestRealPath = realpathSync(manifestPath);
  if (manifestRealPath === rootRealPath || !manifestRealPath.startsWith(rootRealPath + sep)) {
    throw new Error("bundle manifest escapes bundle root via symlink");
  }
  const manifestStat = statSync(manifestRealPath);
  if (!manifestStat.isFile()) {
    throw new TypeError(`bundle manifest is not a regular file: ${manifestPath}`);
  }
  const manifestBytes = readFileSync(manifestRealPath);
  const manifestSha256 = createHash("sha256").update(manifestBytes).digest("hex");
  if (expectedSha256 && typeof expectedSha256 === "string" && expectedSha256.length > 0) {
    if (!/^[0-9a-f]{64}$/.test(expectedSha256)) {
      throw new TypeError("--manifest-sha256 must be a 64-character lowercase hex digest");
    }
    if (manifestSha256 !== expectedSha256) {
      throw new Error(
        `bundle manifest SHA-256 mismatch. Expected ${expectedSha256}, got ${manifestSha256}. ` +
          `No silent fallback or auto-discovery is performed; verify the bundle contents.`,
      );
    }
  }
  let manifest: BundleManifest;
  try {
    manifest = JSON.parse(manifestBytes.toString("utf8")) as BundleManifest;
  } catch (cause) {
    throw new Error(`bundle manifest is not valid JSON: ${(cause as Error).message}`);
  }
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new TypeError("bundle manifest must be an object");
  }
  if (manifest.schema_version !== EXPECTED_BUNDLE_SCHEMA) {
    throw new TypeError(
      `bundle manifest schema_version must be ${EXPECTED_BUNDLE_SCHEMA}, got ${String(manifest.schema_version)}`,
    );
  }
  if (!manifest.metadata || typeof manifest.metadata !== "object" || Array.isArray(manifest.metadata)) {
    throw new TypeError("bundle manifest metadata must be an object");
  }
  if (manifest.metadata.backend !== "direct" && manifest.metadata.backend !== "diffusion") {
    throw new TypeError("bundle manifest metadata.backend must be 'direct' or 'diffusion'");
  }
  if (!Array.isArray(manifest.files)) {
    throw new TypeError("bundle manifest files must be an array");
  }
  const paths = new Set<string>();
  for (const record of manifest.files) {
    if (!record || typeof record !== "object" || typeof record.path !== "string" || record.path.length === 0) {
      throw new TypeError("bundle manifest file record must have a non-empty string path");
    }
    if (paths.has(record.path)) {
      throw new TypeError(`bundle manifest contains duplicate file path: ${record.path}`);
    }
    paths.add(record.path);
    if (isForbiddenBundlePath(record.path)) {
      throw new TypeError(`bundle manifest file path is forbidden (must be relative, no traversal): ${record.path}`);
    }
    const recordFullPath = resolve(rootRealPath, record.path);
    let recordFullRealPath: string | null = null;
    try {
      recordFullRealPath = realpathSync(recordFullPath);
    } catch {
      recordFullRealPath = null;
    }
    if (!recordFullRealPath) {
      throw new Error(`bundle declared file not found: ${record.path}`);
    }
    if (recordFullRealPath === rootRealPath || !recordFullRealPath.startsWith(rootRealPath + sep)) {
      throw new Error(`bundle declared file escapes bundle root via symlink: ${record.path}`);
    }
    if (!statSync(recordFullRealPath).isFile()) {
      throw new TypeError(`bundle declared path is not a regular file: ${record.path}`);
    }
  }
  return { rootRealPath, manifest, manifestSha256, manifestPath };
}

export function bundleManifestToMetadata(
  manifest: BundleManifest,
  _rootRealPath: string,
): BundleMetadataResource {
  const md = manifest.metadata ?? ({} as BundleManifest["metadata"]);
  return {
    schema_version: "vons.bundle.metadata/v1",
    backend: md.backend ?? null,
    model_id: typeof md.model_id === "string" ? md.model_id : null,
    option_count:
      Number.isInteger(md.option_count) && (md.option_count as number) >= 0 ? (md.option_count as number) : null,
    sequence_length:
      Number.isInteger(md.sequence_length) && (md.sequence_length as number) >= 0
        ? (md.sequence_length as number)
        : null,
  };
}

export function createMcpServer(options: McpServerOptions): McpServer {
  if (!options || typeof options !== "object") throw new TypeError("options is required");
  if (!options.backend || typeof options.backend.decide !== "function") {
    throw new TypeError("options.backend must implement DecisionBackend.decide");
  }
  const backend: DecisionBackend = options.backend;
  if (backend.backend !== "direct" && backend.backend !== "diffusion") {
    throw new TypeError("options.backend.backend must be 'direct' or 'diffusion'");
  }
  const manifest: BundleManifest | null = options.manifest ?? null;
  const bundleRoot: string | null = options.bundleRoot ?? null;
  const logger = options.logger ?? null;

  const log = (level: "debug" | "info" | "warn" | "error", message: string): void => {
    if (logger) {
      try {
        logger(level, message);
      } catch {
        void 0;
      }
    }
  };

  const server = new McpServer({
    name: "vons-local-mcp",
    version: "0.1.0",
  });

  const metadataResource: BundleMetadataResource | null = manifest
    ? bundleManifestToMetadata(manifest, bundleRoot ?? "")
    : null;

  server.registerResource(
    MANIFEST_RESOURCE_NAME,
    MANIFEST_RESOURCE_URI,
    {
      title: "Vons bundle manifest metadata",
      description:
        "Non-sensitive, non-asset metadata for the user-supplied hash-verified ONNX bundle. " +
        "Returns only schema_version, backend, model_id (if declared), option_count, and sequence_length. " +
        "No local filesystem path is emitted. Never includes file bytes, file hashes, tokenizer contents, graph data, or runtime asset digests.",
      mimeType: "application/json",
    },
    async (uri) => {
      if (uri.href !== MANIFEST_RESOURCE_URI) {
        throw new Error("unknown manifest resource URI");
      }
      if (!metadataResource) {
        throw new Error(
          "no manifest loaded. The server requires an explicit --bundle directory pointing to a " +
            "hash-verified ONNX bundle; no auto-discovery or implicit default is performed.",
        );
      }
      return {
        contents: [
          {
            uri: MANIFEST_RESOURCE_URI,
            mimeType: "application/json",
            text: JSON.stringify(metadataResource, null, 2),
          },
        ],
      };
    },
  );

  server.registerTool(
    "vons_decide",
    {
      description:
        "Call the user-supplied, hash-verified Vons local ONNX decision backend with a structured DecisionRequest. " +
        "Returns a DecisionResponse JSON with per-question probabilities and either a choice (status=ok) or explicit " +
        "abstention (status=abstain, choice=null, abstain_reason). The host remains responsible for validation, " +
        "consent, execution, and any KASI policy. This server never authorizes actions, never runs downstream tools, " +
        "never writes data, never logs request contents to stdout, and never performs silent fallback or auto-discovery.",
      inputSchema: DecisionRequestZodSchema,
    },
    async (args) => {
      const parseResult = DecisionRequestZodSchema.safeParse(args);
      if (!parseResult.success) {
        const firstIssue = parseResult.error.issues[0];
        const message =
          "vons_decide arguments failed strict schema validation: " +
          (firstIssue
            ? `${firstIssue.path.join(".") || "<root>"}: ${firstIssue.message}`
            : parseResult.error.message);
        log("warn", "invalid tool arguments rejected by strict schema validation");
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({ error: message, schema: "vons.mcp.tool.validation/v1" }),
            },
          ],
          isError: true,
        };
      }
      const zodValidated = parseResult.data as DecisionRequest;
      try {
        validateRequest(zodValidated);
      } catch (cause) {
        const message = "vons_decide request rejected by shared contract validation; no inference was run.";
        log("warn", "request rejected by shared contract validation");
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ error: message }) }],
          isError: true,
        };
      }
      if (zodValidated.backend !== undefined && zodValidated.backend !== backend.backend) {
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({
                error: "vons_decide requested backend does not match the configured backend; no inference was run.",
              }),
            },
          ],
          isError: true,
        };
      }
      let response: DecisionResponse;
      try {
        response = await backend.decide(structuredClone(zodValidated));
      } catch (cause) {
        void cause;
        const message =
          `vons_decide backend raised a runtime error. No response was produced, and no fallback decision was made. ` +
          `(Backend error details are intentionally not surfaced on stdout or stderr to avoid leaking internal request/asset contents.)`;
        log("error", "backend inference failed; no response or fallback decision was produced");
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ error: message }) }],
          isError: true,
        };
      }
      if (!response || typeof response !== "object") {
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({
                error: "vons_decide backend returned a non-object response. No fallback decision was substituted.",
              }),
            },
          ],
          isError: true,
        };
      }
      try {
        validateResponse(response);
      } catch (cause) {
        const message = `vons_decide backend response failed contract validation: ${(cause as Error).message}`;
        log("error", "backend response failed shared contract validation");
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ error: message }) }],
          isError: true,
        };
      }
      const expectedBackend = zodValidated.backend ?? backend.backend;
      if (response.backend !== expectedBackend) {
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({
                error: "vons_decide backend response backend does not match the configured backend; refusing to forward it.",
              }),
            },
          ],
          isError: true,
        };
      }
      const requestedQuestionIds = new Set(zodValidated.questions.map((question) => question.id));
      const answeredQuestionIds = new Set(response.answers.map((answer) => answer.question_id));
      if (
        requestedQuestionIds.size !== answeredQuestionIds.size ||
        [...requestedQuestionIds].some((questionId) => !answeredQuestionIds.has(questionId))
      ) {
        const message =
          "vons_decide backend response question IDs must exactly match the request; " +
          "refusing missing or unexpected answers.";
        log("error", "backend response question IDs did not match the request");
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ error: message }) }],
          isError: true,
        };
      }
      const requestQuestionsById = new Map(zodValidated.questions.map((question) => [question.id, question]));
      const includesUnrequestedOption = response.answers.some((answer) => {
        const question = requestQuestionsById.get(answer.question_id);
        if (!question) return true;
        const allowedOptions = new Set(
          question.type === "boolean" && question.options.length === 0
            ? ["true", "false"]
            : question.options,
        );
        return (
          (answer.choice !== null && !allowedOptions.has(answer.choice)) ||
          answer.probabilities.some((item) => !allowedOptions.has(item.option))
        );
      });
      if (includesUnrequestedOption) {
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({
                error: "vons_decide backend response includes an option outside the request candidate set; refusing to forward it.",
              }),
            },
          ],
          isError: true,
        };
      }
      try {
        validateResponseForRequest(response, zodValidated);
      } catch (cause) {
        const message = `vons_decide backend response failed request reconciliation: ${(cause as Error).message}`;
        log("error", "backend response failed request reconciliation");
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ error: message }) }],
          isError: true,
        };
      }
      const allAbstain =
        response.answers.length > 0 &&
        response.answers.every((a) => a.status === "abstain" && a.choice === null);
      if (allAbstain) {
        log("debug", "backend abstained; no fallback decision was applied");
      }
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify(response),
          },
        ],
        isError: false,
      };
    },
  );

  return server;
}

export interface RunStdioServerOptions extends McpServerOptions {
  stdin?: Readable | null;
  stdout?: Writable | null;
  signal?: AbortSignal | null;
}

export async function runStdioMcpServer(options: RunStdioServerOptions): Promise<void> {
  const server = createMcpServer(options);
  const stdin = (options.stdin ?? process.stdin) as Readable;
  const stdout = (options.stdout ?? process.stdout) as Writable;
  const transport: Transport = new StdioServerTransport(stdin, stdout);
  const signal = options.signal;
  let resolveTransportClosed: (() => void) | null = null;
  const transportClosed = new Promise<void>((resolve) => {
    resolveTransportClosed = resolve;
  });
  const previousOnClose = transport.onclose;
  transport.onclose = () => {
    try {
      previousOnClose?.();
    } finally {
      resolveTransportClosed?.();
    }
  };
  let onAbort: (() => void) | null = null;
  if (signal) {
    if (signal.aborted) return;
    onAbort = () => {
      void server.close().catch(() => void 0);
    };
    signal.addEventListener("abort", onAbort);
  }
  try {
    await server.connect(transport);
    await transportClosed;
  } finally {
    if (onAbort !== null && signal) {
      signal.removeEventListener("abort", onAbort);
    }
    await server.close().catch(() => void 0);
  }
}

export { StdioServerTransport };

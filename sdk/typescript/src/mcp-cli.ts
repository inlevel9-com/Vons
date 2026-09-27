import { parseArgs } from "node:util";
import { readFileSync, realpathSync } from "node:fs";
import { resolve, join, sep } from "node:path";
import { pathToFileURL } from "node:url";

import { loadBundleManifestStrict, runStdioMcpServer, isForbiddenBundlePath } from "./mcp.ts";

import { createOnnxWebBackend } from "./onnx.ts";
import type { BundleManifest } from "./onnx.ts";

interface CliArgs {
  bundle: string | undefined;
  manifestSha256: string | undefined;
  provider: "wasm" | "webgpu" | undefined;
  help: boolean;
}

function printUsage(stream: NodeJS.WritableStream = process.stderr): void {
  stream.write(
    [
      "vons-local-mcp stdio server (uses local ONNX Web backend via --bundle; no KASI adapter; no host tool execution)",
      "",
      "USAGE:",
      "  node --experimental-strip-types sdk/typescript/src/mcp-cli.ts --bundle <dir> [options]",
      "",
      "REQUIRED:",
      "  --bundle <dir>           Path to a user-supplied, hash-verified local ONNX bundle directory.",
      "                           Must contain bundle-manifest-v1.json declaring vons.bundle.manifest/v1.",
      "                           No URL, no auto-discovery, no implicit defaults. Shell-resolved relative paths (including ../..) are supported for the local --bundle directory after the scheme-rejection check.",
      "",
      "OPTIONS:",
      "  --manifest-sha256 <hex>  Optional 64-char lowercase SHA-256 to authenticate the bundle manifest.",
      "                           If declared and mismatched, the process aborts; no fallback or soft-error.",
      "  --provider <wasm|webgpu> Backend provider choice for the ONNX runtime (default: wasm).",
      "                           In the stdio CLI, --provider wasm is the only supported value.",
      "                           --provider webgpu is explicitly rejected with exit code 4 because a",
      "                           stdio process cannot obtain a browser WebGPU adapter. Use a browser",
      "                           or extension host with requestAdapter() explicitly to use WebGPU.",
      "  --help                   Print this help text and exit 0.",
      "",
      "CONSTRAINTS (explicit non-goals):",
      "  - No HTTP listener, no network, no remote connections.",
      "  - Writes MCP JSON-RPC messages ONLY to stdout.",
      "  - Diagnostic messages (errors, warning banners, runtime logs, and usage printed for invalid/missing flags) go ONLY to stderr.",
      "  - The --help usage text is intentionally printed to stdout (exit 0) so it may be piped or redirected by shell callers.",
      "  - Never executes downstream tools, never authorizes actions, never stores/persists/upload data.",
      "  - Never performs silent fallback decisions, never substitutes a fake choice on errors.",
      "  - CLI --bundle accepts local directory paths (including shell-resolved ../..); URL schemes are rejected; the chosen root is realpath-resolved before manifest and declared files are validated.",
      "  - Declared files inside bundle-manifest-v1.json must not use URLs, absolute paths, .. traversal, drive letters, query strings, or fragments; any symlink escape or realpath escape is rejected.",
      "  - No request or response contents are written to any log except the MCP stdout channel.",
      "",
    ].join("\n"),
  );
}

function parseCliArgs(argv: string[]): CliArgs {
  try {
    const parsed = parseArgs({
      args: argv,
      options: {
        bundle: { type: "string" },
        "manifest-sha256": { type: "string" },
        provider: { type: "string" },
        help: { type: "boolean", short: "h" },
      },
      strict: true,
      allowPositionals: false,
    });
    const providerRaw = parsed.values["provider"];
    let provider: CliArgs["provider"];
    if (providerRaw === undefined) {
      provider = undefined;
    } else if (providerRaw === "wasm" || providerRaw === "webgpu") {
      provider = providerRaw;
    } else {
      throw new Error(
        `--provider must be 'wasm' or 'webgpu', got '${String(providerRaw)}'. No auto-detection.`,
      );
    }
    return {
      bundle: parsed.values["bundle"],
      manifestSha256: parsed.values["manifest-sha256"],
      provider,
      help: Boolean(parsed.values.help),
    };
  } catch (cause) {
    printUsage();
    process.stderr.write(`\nargument error: ${(cause as Error).message}\n`);
    process.exit(2);
  }
}

function diag(level: "debug" | "info" | "warn" | "error", message: string): void {
  process.stderr.write(`[vons-mcp:${level}] ${message}\n`);
}

function makeBundleLocalFsFetch(
  bundleRootRealPath: string,
  manifest: BundleManifest,
): typeof fetch {
  const bundlePrefix = "vons-bundle-fs://local/";
  const validRelPaths = new Set(manifest.files.map((f) => f.path));
  const fetchImpl: typeof fetch = async (input, init) => {
    const urlStr = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (typeof urlStr !== "string") {
      throw new TypeError("invalid fetch input in local-fs bundle adapter");
    }
    if (init && "signal" in init && init.signal && init.signal.aborted) {
      throw new DOMException("Aborted", "AbortError");
    }
    if (!urlStr.startsWith(bundlePrefix)) {
      throw new Error(
        `stdios bundle fetch adapter only accepts ${bundlePrefix} URLs; refusing to fetch: ${urlStr.slice(0, 64)}`,
      );
    }
    const relEncoded = urlStr.slice(bundlePrefix.length);
    const rel = decodeURIComponent(relEncoded);
    if (isForbiddenBundlePath(rel) || !validRelPaths.has(rel)) {
      throw new Error(`bundle-local fetch refused: path not declared in manifest: ${rel}`);
    }
    const abs = join(bundleRootRealPath, ...rel.split("/"));
    let absRealPath: string | null = null;
    try {
      absRealPath = realpathSync(abs);
    } catch {
      throw new Error(`bundle-local fetch refused: declared file not found on disk (realpath failed): ${rel}`);
    }
    if (
      absRealPath !== bundleRootRealPath &&
      !absRealPath.startsWith(bundleRootRealPath + sep)
    ) {
      throw new Error(`bundle-local fetch refused: file escapes bundle root via symlink: ${rel}`);
    }
    const bytes = readFileSync(absRealPath);
    const body = new Uint8Array(bytes);
    return new Response(body.buffer as ArrayBuffer, {
      status: 200,
      headers: {
        "content-type": "application/octet-stream",
        "content-length": String(body.byteLength),
      },
    }) as Response;
  };
  return fetchImpl;
}

function bundleLocalBaseHref(bundleRootRealPath: string): string {
  return "vons-bundle-fs://local/";
}

async function main(argv: string[]): Promise<number> {
  const args = parseCliArgs(argv);
  if (args.help) {
    printUsage(process.stdout);
    return 0;
  }
  if (!args.bundle || args.bundle.length === 0) {
    printUsage();
    diag("error", "--bundle <dir> is required. No implicit or auto-discovered bundle path is used.");
    return 2;
  }
  if (args.provider !== undefined && args.provider !== "wasm" && args.provider !== "webgpu") {
    diag("error", `--provider must be 'wasm' or 'webgpu', got '${String(args.provider)}'. No auto-detection.`);
    return 2;
  }
  const provider = args.provider ?? "wasm";
  const URI_SCHEME_RE = /^(https?|file|ftp|data):(\/\/)?/i;
  if (URI_SCHEME_RE.test(args.bundle.trimStart())) {
    const trimmed = args.bundle.trim();
    const m = trimmed.match(URI_SCHEME_RE);
    const scheme = m ? m[1].toLowerCase() : "unknown";
    diag(
      "error",
      `--bundle rejected: URI scheme '${scheme}://' is not supported. ` +
        `The --bundle argument must be a local filesystem directory path (shell-resolved relative paths including ../.. are supported). ` +
        `Passed value: '${trimmed.slice(0, 120)}'${trimmed.length > 120 ? " (truncated)" : ""}. ` +
        `No silent fallback; no request contents were processed.`,
    );
    return 3;
  }
  const bundleDir = resolve(args.bundle);
  let loaded: Awaited<ReturnType<typeof loadBundleManifestStrict>>;
  try {
    loaded = loadBundleManifestStrict(bundleDir, args.manifestSha256);
  } catch (cause) {
    diag(
      "error",
      `bundle load failed: ${cause instanceof Error ? cause.message : String(cause)}. ` +
        `No silent fallback; the user must repair the bundle or path. No request contents were processed.`,
    );
    return 3;
  }
  const manifest: BundleManifest = loaded.manifest;
  if (provider === "webgpu") {
    diag(
      "error",
      "--provider webgpu was selected, but the stdio CLI cannot obtain a browser WebGPU adapter. " +
        "WebGPU is only supported inside browser/extension host contexts with an explicit requestAdapter() call. " +
        "Either re-run with --provider wasm, or invoke the SDK's browser host integration instead.",
    );
    return 4;
  }
  const md = manifest.metadata;
  diag(
    "info",
    `bundle loaded: backend=${md.backend}, option_count=${md.option_count ?? "null"}, ` +
      `sequence_length=${md.sequence_length ?? "null"}, model_id=${md.model_id ?? "(none declared)"}. ` +
      `manifest.sha256=${loaded.manifestSha256.slice(0, 12)}... (full digest omitted from logs). ` +
      `Starting stdio MCP server; MCP JSON-RPC only on stdout, diagnostics only on stderr.`,
  );
  diag(
    "info",
    `provider=${provider}. Wire ONNX Web backend with bundle-local fs fetch adapter; ` +
      `actual ONNX runtime loading only runs for real user bundles (not in synthetic protocol tests).`,
  );
  const bundleFetch = makeBundleLocalFsFetch(loaded.rootRealPath, manifest);
  let backend: Awaited<ReturnType<typeof createOnnxWebBackend>>;
  try {
    backend = await createOnnxWebBackend({
      manifest,
      baseUrl: bundleLocalBaseHref(loaded.rootRealPath),
      provider,
      fetch: bundleFetch,
    });
  } catch (cause) {
    diag(
      "error",
      `failed to create ONNX Web backend for provider=${provider}: ${cause instanceof Error ? cause.message : String(cause)}. ` +
        `No fallback backend or synthetic choice substituted; MCP server startup aborted.`,
    );
    return 5;
  }
  try {
    await runStdioMcpServer({
      backend,
      manifest,
      manifestSha256: loaded.manifestSha256,
      bundleRoot: loaded.rootRealPath,
      logger: (level, message) => diag(level, message),
    });
  } catch (cause) {
    diag("error", `stdio MCP server terminated with error: ${cause instanceof Error ? cause.message : String(cause)}`);
    return 1;
  } finally {
    try {
      await backend.dispose();
    } catch {
      void 0;
    }
  }
  return 0;
}

const _argv1 = typeof process !== "undefined" && process.argv && process.argv[1] ? process.argv[1] : null;
const _isDirectCliInvoke =
  _argv1 !== null &&
  typeof import.meta !== "undefined" &&
  import.meta.url !== null &&
  typeof import.meta.url === "string" &&
  pathToFileURL(resolve(_argv1)).href === import.meta.url;

if (_isDirectCliInvoke) {
  void main(process.argv.slice(2)).then((code) => {
    if (code !== 0) {
      process.exitCode = code;
    }
  });
}

export {
  parseCliArgs,
  printUsage,
  diag,
  main,
  isForbiddenBundlePath,
  loadBundleManifestStrict,
};

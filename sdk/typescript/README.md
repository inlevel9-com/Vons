# `@vons/decision-sdk`

The TypeScript package mirrors the Vons Python request/response contract and
keeps KASI policy and execution in the host. It contains no tool executor and
does not treat a model-produced risk or confidence value as permission.

The contract-only entry point remains small, while the optional `./onnx`
subpath uses pinned `onnxruntime-web@1.30.0` and
`@huggingface/tokenizers@0.2.0` for local browser inference. Applications
provide a model backend through `DecisionBackend`; the runtime helper verifies
bundle bytes, uses the declared WASM or WebGPU provider, and does not silently
fall back from an unavailable accelerator.
`callFingerprint` returns a canonical JSON consent key. Hosts that persist
consent for the Python adapter should SHA-256 this key before storage so the
cross-language fingerprint format remains identical.

Node 22+ can load the source directly with its TypeScript type stripping during
early development. A release build should type-check and bundle the package
with the consuming application's toolchain.

The local smoke harness covers both pilot bundles and both providers. Build it
and serve the repository root so the browser can fetch the ignored ORT runtime
files and the existing pilot artifacts:

```sh
npm ci
npm run build:browser
cd ../..
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/sdk/typescript/demo/`. The smoke page reports
session-load and one-request timings, optional Chromium JS-heap observations,
provider evidence, and the contract response. These are diagnostic smoke
values, not the repeated benchmark protocol.

## Source preview

For a UI that does not require serving the development workspace, build the
[Chrome side-panel extension](extension/README.md) with `npm run build:extension`.
It packages the runtime locally and accepts a user-selected, hash-verified model
folder. The guide explains installation and the separate model requirement.

## MCP stdio server (local-only, ONNX runtime backend)

This SDK ships a small, local-only Model Context Protocol (MCP) stdio server in
`src/mcp.ts` and `src/mcp-cli.ts`. It is strictly for hosts that already have a
user-supplied, hash-verified ONNX decision bundle on the local filesystem. The
server uses these exact, pinned npm dependencies:

- `@modelcontextprotocol/server` `2.1.0`
- `zod` `4.6.5`

### What it exposes

Exactly one MCP tool and one read-only MCP resource:

- **Tool:** `vons_decide` — takes a strict DecisionRequest JSON object mirroring
  the shared contract (`state`, `questions[]`, optional `backend`, optional
  non-negative integer `seed`), validates it through the shared
  `validateRequest` → `DecisionBackend.decide` → `validateResponse` pipeline,
  and returns a JSON-serialized DecisionResponse as a single MCP `text` content
  item. Malformed inputs raise an explicit MCP application error; no fake
  `choice`, no silent fallback, and no coerced success. Explicit abstentions
  (`status=abstain`, `choice=null`, `abstain_reason`) round-trip exactly as
  returned by the backend.
- **Resource:** `vons://bundle/manifest/metadata` — non-sensitive metadata only.
  Returns `schema_version`, `backend`, `model_id` (if declared), `option_count`,
  and `sequence_length`. No local filesystem path is emitted. Never includes file
  bytes, file hashes, tokenizer contents, graph data, or runtime asset digests.

### What it explicitly does NOT do

Per the Vons pilot scope and AGENTS.md host boundary:

- No HTTP listener, no network, no remote MCP service.
- Writes MCP JSON-RPC messages **only** to stdout.
- Diagnostic messages, errors, warning banners, and runtime logs go **only** to stderr. The `--help` usage text is intentionally written to stdout (exit code 0) so it may be piped or redirected by shell callers; when usage is printed for invalid/missing flags it is still written to stderr with a non-zero exit status.
- Never executes downstream tools, never authorizes actions, never runs
  host-side tooling.
- Never performs silent fallback decisions or substitutes a fake `choice`
  when the input is invalid or the backend fails.
- No auto-discovery of bundles, no search for manifests, no default paths.
  The user must pass an explicit `--bundle <dir>`.
- No request contents, response contents, or manifest asset hashes are ever
  written to any file, network, or stderr log line outside the MCP stdout
  JSON-RPC channel.

### CLI usage

```sh
node --experimental-strip-types sdk/typescript/src/mcp-cli.ts --bundle <dir> \
  [--manifest-sha256 <64-char-hex>] [--provider wasm|webgpu]
```

`--provider wasm` is the supported value. It is the only supported CLI value, and it is also the default when `--provider` is omitted entirely. `--provider webgpu` is explicitly rejected with exit code 4, because a stdio child process cannot obtain a browser WebGPU adapter. Hosts that want WebGPU must invoke the SDK directly from a browser/extension context that has called `requestAdapter()` explicitly.

### Codex project-local stdio setup example

Create or edit `config.toml` in the Codex project's `.codex/` directory (or its
equivalent config location for your workspace):

```toml
[mcp_servers.vons]
command = "node"
args = [
  "--experimental-strip-types",
  "/absolute/path/to/Vons/sdk/typescript/src/mcp-cli.ts",
  "--bundle",
  "/absolute/path/to/user-supplied-hash-verified-vons-bundle",
  # Optional: remove these two arguments to skip pinning; otherwise replace the placeholder.
  "--manifest-sha256",
  "<64-char-lowercase-manifest-sha256>"
]
```

The host is responsible for consent, validation, and execution around this
server.

### Claude Code project-local stdio setup example

Create or edit `.mcp.json` in the project root. Remove the optional flag/value
pair to skip pinning; otherwise replace the placeholder with the lowercase
SHA-256 of `bundle-manifest-v1.json` before use.

```json
{
  "mcpServers": {
    "vons": {
      "command": "node",
      "args": [
        "--experimental-strip-types",
        "/absolute/path/to/Vons/sdk/typescript/src/mcp-cli.ts",
        "--bundle",
        "/absolute/path/to/user-supplied-hash-verified-vons-bundle",
        "--manifest-sha256",
        "<64-char-lowercase-manifest-sha256>",
        "--provider",
        "wasm"
      ]
    }
  }
}
```

Both examples assume Node 22+ is on `PATH` and that the bundle directory was
previously verified by the host. The CLI starts in stdio protocol mode and
never forks an HTTP listener.

### Filesystem loader (strict)

The CLI `--bundle <dir>` argument accepts local directory paths (including
shell-resolved relative forms like `../my-bundle`), then realpath-resolves the
selected root before opening. URL schemes such as `https://`, `http://`, or
`file://` are rejected for `--bundle` itself. After the bundle root is
resolved, the manifest loader in `mcp.ts` applies the following strict
checks to each **manifest-declared relative file path** and to the bundle
manifest itself:

- URLs (`https://`, `http://`, `file://`, `data://`, `ftp://`)
- Absolute paths and `/`-rooted file paths
- Any occurrence of `..`, including URL-encoded `%2e%2e` or similar
- Windows drive letters and backslashes
- Query strings (`?`) or fragments (`#`)
- Symlinks or real paths that escape the `realpath()`-resolved bundle root

Declared files must also match entries inside a `vons.bundle.manifest/v1`
manifest named `bundle-manifest-v1.json`. Unknown, duplicate, or
non-existent declared files abort the server start with a clear error and
exit code 3.

This package is private in npm metadata until an npm release is requested.
Its `files` allowlist includes source, this guide and the software license. Models, tokenizers,
datasets, generated demo JavaScript and dependencies are intentionally excluded
from the public source export. Build the demos locally and supply your own
reviewed model bundles before opening the smoke, benchmark or parity pages.

The [Vons Community and Commercial License 1.0](LICENSE) permits qualifying
noncommercial use without a fee. Enterprise and other commercial use require a
separate written agreement; contact oswarld@inlevel9.com. This is source-available
software. Dependencies retain their own terms, and the research articles' CC BY
4.0 license does not grant commercial SDK rights.

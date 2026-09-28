<p align="center">
  <img src="docs/assets/vons-logo.png" alt="Vons" width="505">
</p>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/inlevel9-signature-dark.png">
    <img src="docs/assets/inlevel9-signature.png" alt="INLEVEL9" width="320">
  </picture>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0--preview-c5ff7a?style=flat-square&amp;labelColor=252525" alt="Version 0.1.0 preview">
  <img src="https://img.shields.io/badge/status-research_preview-8e9aaf?style=flat-square&amp;labelColor=252525" alt="Research preview">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-community%20%2F%20commercial-c5ff7a?style=flat-square&amp;labelColor=252525" alt="Noncommercial community use; commercial agreement required"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-3776ab?style=flat-square&amp;labelColor=252525" alt="Python 3.10 or later">
  <img src="https://img.shields.io/badge/node-%E2%89%A522-68a063?style=flat-square&amp;labelColor=252525" alt="Node 22 or later">
</p>
<h3 align="center">Plan globally. Decide locally. Keep the host in control.</h3>
<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="sdk/typescript/README.md">TypeScript SDK</a> ·
  <a href="sdk/typescript/extension/README.md">Chrome Extension</a> ·
  <a href="docs/MODEL_CARD.md">Model card</a> ·
  <a href="docs/publications/vons-paper-v1.1.pdf">Paper</a> ·
  <a href="docs/publications/vons-tech-report-v1.1.pdf">Tech Report</a> ·
  <a href="docs/COMMUNITY.md">Community</a> ·
  <a href="CONTRIBUTING.md">Contribute</a> ·
  <a href="docs/PUBLIC_RELEASE.md">Release guide</a>
</p>

---

Vons explores compact local decision models for frontier-agent workflows.
A planner supplies a state and a bounded set of candidates; Vons returns a
structured choice or abstention. The host owns execution, policy and consent.

The research code includes a one-pass **Direct** scorer and a conditional
**Diffusion** scorer with deterministic DDIM sampling. The English-first design
target is a model asset bundle under **64 MiB**, with CPU/WASM and optional
WebGPU execution. This is a target, not a complete browser-package guarantee.

This source preview includes the Python contract, training/evaluation/export
tools, a TypeScript SDK and tests. Pretrained weights, tokenizer assets, raw
benchmark data, private prompts and experiment logs are not distributed here.
There is no published npm package or hosted inference service implied by the
version badge.

## Read, try and participate

Start with the [community guide](docs/COMMUNITY.md): read the documentation, try a
deterministic example without weights, share an experience or reproduction
report, and contribute improvements. Failures and critical feedback are welcome.
The [contribution guide](CONTRIBUTING.md) explains review, privacy and attribution.

Destinations: [GitHub: inlevel9-com/Vons](https://github.com/inlevel9-com/Vons) ·
[Hugging Face: INLEVEL9/Vons](https://huggingface.co/INLEVEL9/Vons).
This release includes source, documentation and the reviewed research publications:

| Read the research | Version | Download |
| --- | --- | --- |
| Vons: A Compact, Host-Controlled Decision Component for Agent Workflows | Preprint v1.1 | [PDF · 21 pages](docs/publications/vons-paper-v1.1.pdf) |
| Vons: Compact Decision Models for Frontier-Agent Workflows | Technical Report v1.1 | [PDF · 29 pages](docs/publications/vons-tech-report-v1.1.pdf) |

Both articles are available under **CC BY 4.0**. See the
[publication index and verification scope](docs/publications/README.md).
The software remains a **0.1.0 research preview**. Article version numbers do
not establish software readiness. Version 1.1 preserves the frozen v1 evidence
and reports a separately labelled post-freeze remediation handoff; its bounded
synthetic, local-CPU and conditional-selection results do not establish external
generalization or production readiness. The paper uses arXiv **submission number
8127259**. The v1.1 PDF and metadata were processed successfully on
**2026-09-25** and submitted on **2026-09-26**. The arXiv account currently
reports **on hold** for moderation. This is a submission tracking number, not a
public arXiv article identifier. See the
[submission record](docs/publications/ARXIV.md); no public announcement or
peer-review acceptance is claimed.

## Quickstart

Python 3.10+ is sufficient for the contract and synthetic data commands:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m vons.cli generate-smoke --output data/generated/smoke.jsonl
python -m vons.cli validate-data --input data/generated/smoke.jsonl
```

The smoke generator creates local synthetic examples; it does not download a
benchmark or invoke a model. Generated files are ignored by Git.

A deterministic host-policy example:

```python
from vons import KASIAdapter, KASIProposal

adapter = KASIAdapter(tool_policy={"read_weather": "low"})
decision = adapter.decide(
    KASIProposal.from_mapping({
        "calls": [{"name": "read_weather"}],
        "confidence": 0.9,
        "risk": "low",
    })
)
```

This adapter returns a decision. It never runs the proposed tool, and an input
confidence value is not a calibrated probability or an authorization grant.
Unregistered tools are refused. See the [KASI contract](docs/KASI_INTEGRATION.md).

## TypeScript SDK

```sh
cd sdk/typescript
npm ci
npm test
npm run build:browser
npm run build:benchmark
npm run build:parity
```

The contract entry point is `src/index.ts`; the optional `src/onnx.ts` runtime
verifies model manifests and runs a supplied ONNX bundle with an explicit WASM
or WebGPU provider. The demo needs locally exported models and tokenizers.
A source checkout alone cannot run pretrained inference.

Read the [SDK guide](sdk/typescript/README.md) for local serving and runtime
requirements.

## Chrome Extension

The [Chrome extension](sdk/typescript/extension/README.md) provides a local side
panel: import a Vons model folder, enter a question and candidates, and inspect
the proposed choice or abstention. It supports both Direct and Diffusion
full-graph bundles. Models are supplied separately; the extension does not read
pages, execute actions or send prompts to a service.

```sh
cd sdk/typescript
npm ci
npm run build:extension
```

Load `sdk/typescript/dist/chrome-extension` through Chrome's **Load unpacked**
development workflow. The guide covers installation, local model storage,
limitations and packaging. Version **0.1.0** passed review and is publicly
available from the
[Chrome Web Store](https://chromewebstore.google.com/detail/vons-%E2%80%94-local-decisions/cbmjkokoojinfgafahncmjlfmicinide)
as verified on **2026-09-25**. Store availability does not establish model
quality, production readiness or publication of compatible model weights.

## Ollama native tool-call

Stage 1 Python host for Ollama's `/api/chat` tool-calling loop. Advertises
exactly one native function, `vons_decide`; the host validates every tool-call
argument against the `DecisionRequest` contract, runs only the verified local
ONNX bundle via `vons/onnx_runtime.py`, and reconciles exact question-ID
positions on the way out. Raw Ollama output is preserved separately and never
trusted as a decision.

```sh
# Dependencies for schema-only fixtures
python -m pip install -e '.[dev]'
python -m vons.cli ollama-tool-call --help

# Add the local CPU ONNX Runtime and bundled tokenizer for real local inference
python -m pip install -e '.[dev,inference]'
cat > /tmp/vons-messages.json <<'JSON'
{"messages":[{"role":"system","content":"Use vons_decide for one bounded choice."},{"role":"user","content":"Synthetic example: choose the safest next response when a request is underspecified."}]}
JSON
python -m vons.cli ollama-tool-call \
  --host http://127.0.0.1:11434 \
  --model qwen3.8:27b-mlx \
  --bundle ./verified-local-bundle \
  --input /tmp/vons-messages.json
```

Use `--synthetic choice-0` or `--synthetic abstain` instead of `--bundle` for
schema-only checks; those modes do not load or execute ONNX weights. The bundle
mode verifies the supplied manifest and files before opening the Ollama request.

Security gates applied before inference:
1. Host must resolve to loopback (`127.0.0.1`, `localhost`, `::1`).
2. `301/302/303/307/308` redirects that leave loopback are blocked by a custom
   `urllib` handler.
3. Only the `vons_decide` tool name is accepted; other tool calls raise a
   sanitized error before any backend runs.
4. Sanitized `OllamaToolCallError` messages use short fixed literals with a
   12-character hex `error_id`; request/prompt/response text and cause
   messages are never echoed to stdout or stderr.

Synthetic tests in `tests/test_ollama_tool_call.py` cover the local ONNX adapter
with fake sessions, injected backends, malformed/unknown arguments, wrong tool
names, response reconciliation, missing bundles, non-loopback endpoints, and
redirect escape paths. A local Ollama smoke is limited to synthetic input and
does not establish model quality or performance; see
[WORK_ALLOCATION.md](docs/WORK_ALLOCATION.md) for its evidence boundary.

## ChatGPT Developer Mode (private MCP test)

For a private, owner-operated ChatGPT test, OpenAI's [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
can forward a local Vons stdio MCP server without exposing an inbound public
endpoint. This is setup guidance only; a ChatGPT Developer Mode connection has
not been tested from this repository.

You need Node.js 22 or later, this source checkout, a user-supplied local Vons
bundle with its verified manifest, the official `tunnel-client`, a `tunnel_id`
and runtime API key, and access to ChatGPT Developer Mode. Tunnel permissions
and ChatGPT workspace permissions are separate: creating or editing a tunnel
requires Tunnels Read + Manage; running/selecting it requires Tunnels Read + Use;
some workspaces also require an administrator to enable Developer Mode.

Install the SDK dependencies once:

```sh
cd /absolute/path/to/Vons/sdk/typescript
npm ci
```

On a machine that can run the local MCP process, set `CONTROL_PLANE_API_KEY`
and `VONS_TUNNEL_ID` through your secret manager, install `tunnel-client` using
the [official guide](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels),
then create and run a stdio profile from the repository root. Replace the
bundle path with an absolute path to your own verified bundle:

```sh
cd /absolute/path/to/Vons
tunnel-client init \
  --sample sample_mcp_stdio_local \
  --profile vons-local \
  --tunnel-id "$VONS_TUNNEL_ID" \
  --mcp-command "node --experimental-strip-types $PWD/sdk/typescript/src/mcp-cli.ts --bundle /absolute/path/to/verified-local-bundle"
tunnel-client doctor --profile vons-local --explain
tunnel-client run --profile vons-local
```

Keep the tunnel client running, then in ChatGPT create a Developer Mode app,
choose **Tunnel** as its connection, and select or enter the same tunnel ID.
The CLI requires a local bundle; it does not download model weights. The
stdio server exposes `vons_decide` and performs local ONNX inference, but
ChatGPT conversation content and tool arguments still pass through OpenAI.
Local inference does not make a ChatGPT conversation private from ChatGPT.
Vons returns a decision only; the host remains responsible for policy,
consent, and any downstream execution.

Secure MCP Tunnel is for private use and Developer Mode testing. It does not
support public plugin submission or distribution; sharing a public ChatGPT
integration requires a separately hosted, stable HTTPS MCP endpoint. See
OpenAI's [MCP connection guide](https://developers.openai.com/api/docs/guides/agents-api/tools/mcp)
for the distinction between local stdio, session-environment, and OpenAI-hosted
connections.

## Research workflow

Install the optional dependencies when running your own experiments:

```sh
python -m pip install -e '.[train,export,ollama,dev]'
python -m vons.cli generate-synthetic --count 2000 --output data/generated/pilot.jsonl
python -m vons.cli --help
```

The pinned encoder and experiment settings are in
[`configs/pilot.json`](configs/pilot.json) and
[`configs/pilot-diffusion.json`](configs/pilot-diffusion.json).
Review upstream terms before downloading or redistributing assets. Reports
must preserve seeds, source revisions, failures, raw responses and measurement
scope. Never interpret a model's self-reported confidence as calibrated probability.

## Scope and limitations

- Synthetic pilot results do not establish external-task generalization.
- Direct and Diffusion results depend on training, candidate shape and padding;
  this preview does not establish a general method ranking.
- Browser smoke, repeated latency, numerical parity and memory measurement are
  separate checks. Missing measurements are unavailable, not zero.
- Python uses an approximate input-token estimate; the TypeScript ONNX path
  checks the exported tokenizer's aggregate budget.
- The general contract supports score questions, while the current candidate
  ONNX head reports them as unsupported.
- KASI is a host compatibility adapter, kept separate from the general decision
  contract. Model output cannot grant consent or authorize high-risk actions.

The [model card](docs/MODEL_CARD.md) describes intended use and release scope.
The [publication index](docs/publications/README.md) records manuscript availability.

## Development checks

```sh
python -m pytest -q
ruff check .
python tools/prepare_public_release.py
```

Some model/export tests require the optional training and export dependencies;
pytest reports those skips explicitly. The public-release audit checks an exact
file allowlist, known sensitive patterns, symlinks and reviewed binary digests.
It does not replace a complete secret or redistribution-rights review.

## Author

**Kwangseob Ahn**  
INLEVEL9 / SEJONG UNIV.  
[oswarld@inlevel9.com](mailto:oswarld@inlevel9.com)

## License and distribution

The [Vons Community and Commercial License 1.0](LICENSE) covers this
source release: qualifying **noncommercial use is free**; **enterprise and other
commercial use require a separate written agreement**. Contact
[oswarld@inlevel9.com](mailto:oswarld@inlevel9.com) for commercial terms.
This is source-available software with use restrictions.

The paper and Tech Report are separately distributed under
[CC BY 4.0](docs/publications/LICENSE.md), which permits commercial article reuse
with attribution. Their license does not grant commercial software rights.
Third-party components retain their original terms. See
[licensing scope](docs/LICENSING.md) and the [release guide](docs/PUBLIC_RELEASE.md).

# OpenAI Agents SDK with local Vons

This example connects the Python [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/mcp/)
to Vons through a local MCP stdio process. The OpenAI model plans and formats
the bounded-choice request; the Vons MCP process and selected ONNX bundle run
on the user's machine.

This is OpenAI API/Agents SDK integration. It does not connect the ChatGPT web
or desktop app, and it does not use a hosted or public Vons MCP server.

## Requirements

- Python 3.10+
- Node.js 22.6+ (or set `VONS_NODE` to its executable path)
- The Vons source checkout with SDK dependencies installed
- A reviewed local Vons full-graph bundle
- An OpenAI API key and an API model available to that project

The repository does not include pretrained model weights. This integration
does not download them.

## Setup and run

From the Vons repository root:

```sh
python -m pip install openai-agents
npm --prefix sdk/typescript ci

export OPENAI_API_KEY="<your OpenAI API key>"
export VONS_BUNDLE="/absolute/path/to/reviewed-vons-bundle"

# Optional settings
export OPENAI_MODEL="<model available to your API project>"
export VONS_MANIFEST_SHA256="<64-character-lowercase-sha256>"
# VONS_REPO defaults to this example's parent repository.
# VONS_NODE may specify an absolute Node.js executable path.

python examples/openai_agents_vons_stdio.py \
  "Compare these explicit options and explain any Vons abstention: keep the draft, revise it, or ask for clarification."
```

`VONS_BUNDLE` must be a local directory containing a `bundle-manifest-v1.json`
and the declared full-graph assets. The Vons CLI verifies the bundle before
starting inference. Supplying `VONS_MANIFEST_SHA256` also pins the manifest to
the expected lowercase SHA-256 digest.

## Data and execution boundary

- The Python Agents SDK sends the prompt to the configured OpenAI API. The
  model-generated `vons_decide` arguments and Vons tool result are also part of
  that API conversation. Do not send sensitive content unless this data flow
  is acceptable; API usage may incur charges.
- The Vons MCP server and ONNX inference run locally. The MCP child receives a
  minimal environment that excludes `OPENAI_API_KEY`.
- The agent exposes only `vons_decide`; it has no downstream action tools.
  `vons_decide` returns a proposal and cannot grant consent or authorize an
  action. Its scores are not calibrated probabilities.
- Agents SDK tracing is disabled in the example. The example does not write
  prompts, tool traffic, or results to local files.

## Verification status

The example is source-only. No OpenAI API call, billed run, or five-candidate
acceptance session has been performed. OpenAI Agents SDK runtime
interoperability and captured MCP traffic remain unverified. This guide is
separate from the private [ChatGPT Developer Mode tunnel instructions](README.md#chatgpt-developer-mode-private-mcp-test).

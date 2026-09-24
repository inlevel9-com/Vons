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

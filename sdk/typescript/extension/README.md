# Vons Chrome Extension

A Manifest V3 side panel for local candidate comparison. It supports full-graph
Direct and Diffusion Vons ONNX bundles, verifies exact manifest/asset hashes,
and keeps execution with the user. No model weight is included in this package.

## Build and install locally

Requires Node 22+ and Chrome 114+:

```sh
cd sdk/typescript
npm ci
npm test
npm run build:extension
```

Open `chrome://extensions`, enable Developer mode for your development profile,
choose **Load unpacked**, and select `sdk/typescript/dist/chrome-extension` from
the repository root. Pin **Vons — Local Decisions** in the toolbar and click it
to open the side panel. A store listing and automatic updates are not provided
by this local build. The generated folder includes all runtime JS/WASM, custom
software terms and third-party notices; it excludes model weights and local data.

For distribution, ZIP the *contents* of that generated directory so `manifest.json`
is at the archive root. Review the resulting package before Chrome Web Store
submission. Do not ZIP the workspace or your imported model folder.

## Chrome Web Store status

**Vons — Local Decisions**, version **0.1.0**, item ID
`cbmjkokoojinfgafahncmjlfmicinide`, was submitted to the Chrome Web Store on
**2026-09-25** after the owner approved the data-use declarations and final
submission. It subsequently passed review and the public listing was verified
on **2026-09-25**:
[install Vons — Local Decisions](https://chromewebstore.google.com/detail/vons-%E2%80%94-local-decisions/cbmjkokoojinfgafahncmjlfmicinide).
The package, English listing, icon, two screenshots, privacy disclosures and
reviewer instructions are included. The local unpacked installation remains
available above.

The store package contains runtime code and notices only. A compatible model
must be supplied by the user; the public source repository does not distribute
pretrained weights. Noncommercial use is free under the included software terms;
enterprise/commercial use requires a separate license.

## First comparison

1. Click **Import model folder** and choose the directory containing exactly one
   `bundle-manifest-v1.json`, its full ONNX graph, external-data files and tokenizer.
   Use a manifest produced by the Vons bundle tools. Head-only/shared-v1 partition
   manifests use a different runtime contract and are not supported here.
2. Optionally keep the model on this device. Nothing is uploaded. The original
   manifest and only the required runtime assets are retained, not checkpoint or
   report files that happen to share the folder. SHA-256 verifies consistency
   with the imported manifest, not the identity or trustworthiness of its author.
3. Enter context, a question and 2–32 unique candidates, one per line. **Use example**
   supplies an illustrative prompt; its outcome is not a quality test.
4. Click **Compare candidates**. Read the proposed choice or explicit abstention,
   uncalibrated model scores and single-run timings. No page action is executed.

WASM uses one thread and works without a WebGPU device. WebGPU is an explicit
experimental choice; failure is shown rather than silently falling back.
**Cancel** terminates the inference worker, and a subsequent comparison creates
a fresh runtime. A two-minute timeout also stops the worker. The model's actual
tokenizer enforces its aggregate input budget; long input is rejected rather than
silently truncated. Models larger than 128 MiB of selected runtime assets are
rejected before initialization. The paper's 64 MiB design target refers to model
assets, not the total extension plus runtime download.

## Privacy and host boundary

Only `sidePanel` is requested. There are no host permissions, content scripts,
page extraction, tool execution, network model fetches, analytics or account
requirements. Prompts/results are in memory only. IndexedDB stores a model only
when requested; **Remove model** deletes that retained copy. No Chrome Sync is
used. The bundled [privacy notice](privacy.html) describes actual handling.
The CSP restricts code and fetches to packaged extension resources and permits
WebAssembly compilation without permitting general JavaScript eval.

## Verification scope

Source tests cover exact file integrity, missing assets, path escapes, candidate
validation and unexpected extension permissions. On 2026-09-25, all 16 SDK tests,
strict extension type checking and the extension build passed. Real Chrome
testing of the generated interface over localhost verified Direct/WASM
inference, explicit token-budget rejection, cancellation followed by reuse,
optional model persistence and removal. The illustrative decision abstained;
this was a functionality check, not a model-quality benchmark.

The owner installed the unpacked extension; a subsequent independent check of
that actual side panel completed both Direct and Diffusion WASM inference using
the private pilot bundles. Both examples abstained. Direct took 326 ms for
inference and 212 ms for load/check; Diffusion took 3,238 ms and 159 ms. These are
single-run observations, not a latency benchmark or accuracy evidence. The
Diffusion ranking placed an unsuitable option first, reinforcing the need to
preserve abstention and human review. The imported models remain unpublished.

The installed-panel check exercises the packaged worker/WASM path under its
extension CSP. WebGPU, broader provider parity, external-task quality and
production readiness are not established by these checks. Store review
establishes listing-policy approval only, not scientific validity or safety.
Preserve the model manifest hash with each result.

Official references: [Side Panel API](https://developer.chrome.com/docs/extensions/reference/api/sidePanel),
[MV3 CSP](https://developer.chrome.com/docs/extensions/reference/manifest/content-security-policy),
[local development installation](https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world#load-unpacked).

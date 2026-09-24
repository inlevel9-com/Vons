import { createOnnxWebBackend, type OnnxWebBackend } from "../src/onnx.ts";
import type { DecisionRequest } from "../src/index.ts";
import { verifyBundle, type LocalBundle } from "./bundle.ts";

let backend: OnnxWebBackend | undefined;
let loadedKey = "";
self.onmessage = async (event: MessageEvent<{ bundle?: LocalBundle; provider: "wasm" | "webgpu"; request: DecisionRequest }>) => {
  const started = performance.now();
  try {
    const { bundle, provider, request } = event.data;
    if (bundle) {
      const key = `${bundle.manifestHash}:${provider}`;
      if (key !== loadedKey) {
        await backend?.dispose(); backend = undefined; loadedKey = "";
        self.postMessage({ type: "progress", text: "Checking model files and starting the local runtime…" });
        await verifyBundle(bundle);
        // This synthetic origin is only a lookup key. No model request can use network fetch.
        const base = "https://vons-bundle.invalid/";
        const bytes = new Map(bundle.assets.map((f) => [new URL(f.path, base).href, f.bytes]));
        bytes.set(new URL(bundle.manifestPath, base).href, bundle.manifestBytes);
        const localFetch: typeof fetch = async (input) => {
          const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
          const asset = bytes.get(url);
          if (!asset) throw new Error("Requested model asset is not in the selected local bundle.");
          return new Response(asset);
        };
        backend = await createOnnxWebBackend({ manifestUrl: new URL(bundle.manifestPath, base),
          expectedManifestSha256: bundle.manifestHash, fetch: localFetch, provider,
          wasmPaths: new URL("runtime/", self.location.href).href, wasmNumThreads: 1 });
        loadedKey = key;
      }
    }
    if (!backend || backend.runtimeInfo.provider !== provider) throw new Error("Import or reload the model before running.");
    const ready = performance.now();
    self.postMessage({ type: "progress", text: "Comparing your candidates on this device…" });
    const response = await backend.decide(request);
    self.postMessage({ type: "result", response, manifestHash: backend.manifestHash,
      runtime: backend.runtimeInfo, loadMs: ready - started, inferenceMs: performance.now() - ready });
  } catch (error) {
    self.postMessage({ type: "error", text: error instanceof Error ? error.message : "Local inference failed." });
  }
};

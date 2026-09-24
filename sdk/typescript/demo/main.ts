import { createOnnxWebBackend } from "../src/onnx.ts";

const outputElement = document.querySelector<HTMLElement>("#output");
if (!outputElement) throw new Error("smoke output element is missing");
const output = outputElement;

interface PerformanceWithMemory extends Performance {
  memory?: { usedJSHeapSize: number };
}

const heapBytes = (): number | null => (performance as PerformanceWithMemory).memory?.usedJSHeapSize ?? null;

const manifestUrls = {
  direct: new URL("../../../artifacts/pilot/bundle-manifest-v1.json", import.meta.url),
  diffusion: new URL("../../../artifacts/pilot-diffusion/bundle-manifest-v1.json", import.meta.url),
};

const manifestHashes = {
  direct: "bd39eea2b8d5ed5f4d57daefdd4822c4e2c4817a68c4a42bef850b5edcbc97d",
  diffusion: "df0e7cae640c6604fb517a88d08dbdaa2454d6b3dbbf5708926349cf490b9607",
};

function show(value: unknown): void {
  output.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

async function run(provider: "wasm" | "webgpu", backendName: "direct" | "diffusion"): Promise<void> {
  const started = performance.now();
  let backend: Awaited<ReturnType<typeof createOnnxWebBackend>> | undefined;
  try {
    backend = await createOnnxWebBackend({
      manifestUrl: manifestUrls[backendName],
      expectedManifestSha256: manifestHashes[backendName],
      provider,
      wasmPaths: "/sdk/typescript/node_modules/onnxruntime-web/dist/",
      wasmNumThreads: 1,
    });
    const initialized = performance.now();
    const beforeMemory = heapBytes();
    const response = await backend.decide({
      state: "browser smoke state",
      questions: [{ id: "smoke", type: "choice", prompt: "Choose the next action", options: ["continue", "stop"] }],
      seed: 7,
    });
    const completed = performance.now();
    show({
      pass: true,
      backend: backendName,
      runtime: backend.runtimeInfo,
      load_ms: initialized - started,
      inference_ms: completed - initialized,
      total_ms: completed - started,
      js_heap_used_bytes_before: beforeMemory,
      js_heap_used_bytes_after: heapBytes(),
      response,
    });
  } catch (error) {
    show({
      pass: false,
      backend: backendName,
      provider,
      elapsed_ms: performance.now() - started,
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
  } finally {
    await backend?.dispose();
  }
}

for (const button of Array.from(document.querySelectorAll<HTMLButtonElement>("button[data-backend]"))) {
  button.addEventListener("click", () => {
    const value = button.dataset.backend;
    if (value === "direct") void run("wasm", "direct");
    else if (value === "diffusion") void run("wasm", "diffusion");
    else if (value === "direct-webgpu") void run("webgpu", "direct");
    else if (value === "diffusion-webgpu") void run("webgpu", "diffusion");
  });
}

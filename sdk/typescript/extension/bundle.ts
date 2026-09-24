import { validateRequest, type DecisionRequest } from "../src/index.ts";
import type { BundleManifest } from "../src/onnx.ts";

export interface LocalAsset { path: string; bytes: ArrayBuffer; }
export interface LocalBundle {
  manifestPath: string;
  manifestBytes: ArrayBuffer;
  manifestHash: string;
  backend: "direct" | "diffusion";
  modelId: string;
  assets: LocalAsset[];
}
export const MAX_BUNDLE_BYTES = 128 * 1024 * 1024;
export function safePath(path: string): boolean {
  return path.length > 0 && !path.startsWith("/") && !path.includes("..")
    && !/[\\%?#:\u0000-\u001f]/.test(path)
    && path.split("/").every((part) => part.length > 0);
}
export async function digest(bytes: ArrayBuffer): Promise<string> {
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("");
}
export function requiredFiles(bytes: ArrayBuffer): { manifest: BundleManifest; paths: string[] } {
  if (bytes.byteLength > 2 * 1024 * 1024) throw new Error("Manifest exceeds 2 MiB.");
  const manifest = JSON.parse(new TextDecoder().decode(bytes)) as BundleManifest;
  if (manifest.schema_version !== "vons.bundle.manifest/v1" || !manifest.metadata
    || !["direct", "diffusion"].includes(manifest.metadata.backend)
    || !Array.isArray(manifest.files) || manifest.files.length > 2048) {
    throw new Error("Choose a Vons v1 manifest for a full Direct or Diffusion ONNX graph.");
  }
  const paths = new Set<string>();
  for (const file of manifest.files) {
    if (!file || typeof file.path !== "string" || !safePath(file.path) || paths.has(file.path)
      || !Number.isSafeInteger(file.bytes) || file.bytes < 0 || file.bytes > MAX_BUNDLE_BYTES
      || !/^[0-9a-f]{64}$/.test(file.sha256)) throw new Error("Invalid manifest file record.");
    paths.add(file.path);
  }
  const graphs = manifest.files.filter((f) => f.role === "model_graph");
  const tokenizers = ["tokenizer.json", "tokenizer_config.json"].map((name) =>
    manifest.files.filter((f) => f.role === "tokenizer" && (f.path === name || f.path.endsWith(`/${name}`))));
  if (graphs.length !== 1 || tokenizers.some((files) => files.length !== 1)) {
    throw new Error("The manifest must name one full graph and both tokenizer JSON files.");
  }
  const external = manifest.metadata.graph?.external_data_locations ?? [];
  if (!Array.isArray(external) || external.some((path) => typeof path !== "string" || !safePath(path)
    || !manifest.files.some((f) => f.path === path && f.role === "model_external_data"))) {
    throw new Error("The graph has an invalid external-data reference.");
  }
  return { manifest, paths: [...new Set([graphs[0].path, ...tokenizers.map((f) => f[0].path), ...external])] };
}
export async function verifyBundle(bundle: LocalBundle): Promise<void> {
  if (!safePath(bundle.manifestPath) || await digest(bundle.manifestBytes) !== bundle.manifestHash) {
    throw new Error("The saved manifest failed its integrity check. Import the model again.");
  }
  const { manifest, paths } = requiredFiles(bundle.manifestBytes);
  if (manifest.metadata.backend !== bundle.backend || bundle.assets.length !== paths.length) {
    throw new Error("Model bundle contents do not match the manifest.");
  }
  let total = bundle.manifestBytes.byteLength;
  for (const path of paths) {
    const matches = bundle.assets.filter((a) => a.path === path);
    const record = manifest.files.find((f) => f.path === path)!;
    if (matches.length !== 1 || matches[0].bytes.byteLength !== record.bytes) throw new Error(`Missing or wrong-sized model file: ${path}`);
    total += record.bytes;
    if (total > MAX_BUNDLE_BYTES) throw new Error("The selected runtime assets exceed 128 MiB.");
    if (await digest(matches[0].bytes) !== record.sha256) throw new Error(`Model file failed SHA-256 verification: ${path}`);
  }
}
export async function importBundle(files: File[]): Promise<LocalBundle> {
  if (files.length === 0 || files.length > 2048) throw new Error("Choose one model folder with at most 2048 files.");
  const manifests = files.filter((f) => f.name === "bundle-manifest-v1.json");
  if (manifests.length !== 1) throw new Error("Choose the folder containing exactly one bundle-manifest-v1.json.");
  const file = manifests[0];
  if (file.size > 2 * 1024 * 1024) throw new Error("Manifest exceeds 2 MiB.");
  const root = file.webkitRelativePath.slice(0, -file.name.length);
  const manifestBytes = await file.arrayBuffer();
  const { manifest, paths } = requiredFiles(manifestBytes);
  const assets: LocalAsset[] = [];
  let total = manifestBytes.byteLength;
  for (const path of paths) {
    const selected = files.filter((f) => f.webkitRelativePath === root + path);
    const expected = manifest.files.find((f) => f.path === path)!;
    if (selected.length !== 1 || selected[0].size !== expected.bytes) throw new Error(`Missing or wrong-sized model file: ${path}`);
    total += selected[0].size;
    if (total > MAX_BUNDLE_BYTES) throw new Error("The selected runtime assets exceed 128 MiB.");
    assets.push({ path, bytes: await selected[0].arrayBuffer() });
  }
  const bundle: LocalBundle = { manifestPath: file.name, manifestBytes,
    manifestHash: await digest(manifestBytes), backend: manifest.metadata.backend,
    modelId: manifest.metadata.model_id ?? `Vons ${manifest.metadata.backend}`, assets };
  await verifyBundle(bundle);
  return bundle;
}
export function makeRequest(state: string, prompt: string, candidates: string): DecisionRequest {
  const request: DecisionRequest = { state: state.trim(), seed: 7,
    questions: [{ id: "decision", type: "choice", prompt: prompt.trim(),
      options: candidates.split(/\r?\n/).map((s) => s.trim()).filter(Boolean) }] };
  validateRequest(request);
  return request;
}

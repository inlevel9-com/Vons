import { validateRequest, type Backend, type DecisionRequest } from "../src/index.ts";
import type { BundleManifest, OnnxBundleManifest, SharedBundleManifest } from "../src/onnx.ts";

export interface LocalAsset { path: string; bytes: ArrayBuffer; }
export interface LocalBundle {
  manifestPath: string;
  manifestBytes: ArrayBuffer;
  manifestHash: string;
  backend: Backend;
  /** Present only for a shared bundle, and persisted with the user's selection. */
  head?: Backend;
  modelId: string;
  assets: LocalAsset[];
}
export const MAX_BUNDLE_BYTES = 128 * 1024 * 1024;

interface RequiredBundleFiles {
  manifest: OnnxBundleManifest;
  paths: string[];
  backend: Backend;
  head?: Backend;
  modelId: string;
}

const SHARED_SCHEMA = "vons.shared-bundle/v1";
const SHARED_HEAD_INPUTS: Record<Backend, string[]> = {
  direct: ["candidate_embeddings", "pooled", "option_mask"],
  diffusion: ["pooled", "option_mask", "initial_noise"],
};

export function safePath(path: string): boolean {
  return path.length > 0 && !path.startsWith("/") && !path.includes("..")
    && !/[\\%?#:\u0000-\u001f]/.test(path)
    && path.split("/").every((part) => part.length > 0);
}
export async function digest(bytes: ArrayBuffer): Promise<string> {
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("");
}

function validateFileRecords(files: unknown, label: string): asserts files is BundleManifest["files"] {
  if (!Array.isArray(files) || files.length === 0 || files.length > 2048) throw new Error(`${label} has an invalid file list.`);
  const paths = new Set<string>();
  for (const file of files) {
    if (!file || typeof file.path !== "string" || !safePath(file.path) || paths.has(file.path)
      || !Number.isSafeInteger(file.bytes) || file.bytes < 0 || file.bytes > MAX_BUNDLE_BYTES
      || !/^[0-9a-f]{64}$/.test(file.sha256) || typeof file.role !== "string") {
      throw new Error("Invalid manifest file record.");
    }
    paths.add(file.path);
  }
}

function exactlyOne(files: BundleManifest["files"], role: string): BundleManifest["files"][number] {
  const matches = files.filter((file) => file.role === role);
  if (matches.length !== 1) throw new Error(`The shared manifest must name exactly one ${role} file.`);
  return matches[0];
}

function tokenizerPaths(files: BundleManifest["files"]): [string, string] {
  const paths = (["tokenizer.json", "tokenizer_config.json"] as const).map((name) => {
    const matches = files.filter((file) => file.role === "tokenizer" && (file.path === name || file.path.endsWith(`/${name}`)));
    if (matches.length !== 1) throw new Error(`The manifest must name exactly one ${name}.`);
    return matches[0].path;
  });
  return paths as [string, string];
}

function requiredFullGraphFiles(manifest: BundleManifest): RequiredBundleFiles {
  if (manifest.schema_version !== "vons.bundle.manifest/v1" || !manifest.metadata
    || (manifest.metadata.backend !== "direct" && manifest.metadata.backend !== "diffusion")) {
    throw new Error("Choose a Vons v1 manifest for a full Direct or Diffusion ONNX graph.");
  }
  validateFileRecords(manifest.files, "Bundle manifest");
  const graphs = manifest.files.filter((file) => file.role === "model_graph");
  const [tokenizerJson, tokenizerConfig] = tokenizerPaths(manifest.files);
  const external = manifest.metadata.graph?.external_data_locations ?? [];
  if (graphs.length !== 1) {
    throw new Error("The manifest must name one full graph and valid external data.");
  }
  if (!Array.isArray(external) || external.some((path) => typeof path !== "string"
    || !safePath(path) || !manifest.files.some((file) => file.path === path && file.role === "model_external_data"))) {
    throw new Error("The full graph manifest contains an invalid external-data reference.");
  }
  return {
    manifest,
    paths: [...new Set([graphs[0].path, tokenizerJson, tokenizerConfig, ...external])],
    backend: manifest.metadata.backend,
    modelId: manifest.metadata.model_id ?? `Vons ${manifest.metadata.backend}`,
  };
}

function requiredSharedFiles(manifest: SharedBundleManifest, selectedHead?: Backend): RequiredBundleFiles {
  if (manifest.schema !== SHARED_SCHEMA || typeof manifest.model_id !== "string" || manifest.model_id.length === 0
    || !manifest.encoder || typeof manifest.encoder.graph !== "string"
    || JSON.stringify(manifest.encoder.outputs) !== JSON.stringify(["candidate_embeddings", "pooled"])
    || !manifest.heads?.direct || !manifest.heads?.diffusion) {
    throw new Error("Choose a valid Vons shared-v1 bundle manifest.");
  }
  if (selectedHead !== "direct" && selectedHead !== "diffusion") {
    throw new Error("Choose Direct or Diffusion for this shared model bundle.");
  }
  validateFileRecords(manifest.files, "Shared bundle manifest");
  const graphContracts = [
    ["encoder_graph", manifest.encoder.graph],
    ["encoder_weights", `${manifest.encoder.graph}.data`],
    ["direct_graph", manifest.heads.direct.graph],
    ["direct_weights", `${manifest.heads.direct.graph}.data`],
    ["diffusion_graph", manifest.heads.diffusion.graph],
    ["diffusion_weights", `${manifest.heads.diffusion.graph}.data`],
  ] as const;
  for (const [role, path] of graphContracts) {
    const record = exactlyOne(manifest.files, role);
    if (!safePath(path) || record.path !== path) throw new Error(`The shared manifest has an invalid ${role} path.`);
  }
  if (JSON.stringify(manifest.heads[selectedHead].inputs) !== JSON.stringify(SHARED_HEAD_INPUTS[selectedHead])) {
    throw new Error(`The shared ${selectedHead} head input contract is unsupported.`);
  }
  const config = exactlyOne(manifest.files, "config");
  const [tokenizerJson, tokenizerConfig] = tokenizerPaths(manifest.files);
  const graph = manifest.heads[selectedHead].graph;
  const required = [manifest.encoder.graph, `${manifest.encoder.graph}.data`, graph, `${graph}.data`, config.path, tokenizerJson, tokenizerConfig];
  return { manifest, paths: [...new Set(required)], backend: selectedHead, head: selectedHead, modelId: manifest.model_id };
}

export function requiredFiles(bytes: ArrayBuffer, selectedHead?: Backend): RequiredBundleFiles {
  if (bytes.byteLength > 2 * 1024 * 1024) throw new Error("Manifest exceeds 2 MiB.");
  const manifest = JSON.parse(new TextDecoder().decode(bytes)) as OnnxBundleManifest;
  if (manifest && typeof manifest === "object" && "schema_version" in manifest) {
    return requiredFullGraphFiles(manifest as BundleManifest);
  }
  if (manifest && typeof manifest === "object" && "schema" in manifest) {
    return requiredSharedFiles(manifest as SharedBundleManifest, selectedHead);
  }
  throw new Error("Choose a supported Vons bundle manifest.");
}

export async function verifyBundle(bundle: LocalBundle): Promise<void> {
  if (!safePath(bundle.manifestPath) || await digest(bundle.manifestBytes) !== bundle.manifestHash) {
    throw new Error("The saved manifest failed its integrity check. Import the model again.");
  }
  const { manifest, paths, backend, head } = requiredFiles(bundle.manifestBytes, bundle.head);
  if (backend !== bundle.backend || head !== bundle.head || bundle.assets.length !== paths.length
    || ("schema_version" in manifest && bundle.head !== undefined)) {
    throw new Error("Model bundle contents or selected head do not match the manifest.");
  }
  let total = bundle.manifestBytes.byteLength;
  for (const path of paths) {
    const matches = bundle.assets.filter((asset) => asset.path === path);
    const record = manifest.files.find((file) => file.path === path)!;
    if (matches.length !== 1 || matches[0].bytes.byteLength !== record.bytes) throw new Error(`Missing or wrong-sized model file: ${path}`);
    total += record.bytes;
    if (total > MAX_BUNDLE_BYTES) throw new Error("The selected runtime assets exceed 128 MiB.");
    if (await digest(matches[0].bytes) !== record.sha256) throw new Error(`Model file failed SHA-256 verification: ${path}`);
  }
}

export async function importBundle(files: File[], selectedHead?: Backend): Promise<LocalBundle> {
  if (files.length === 0 || files.length > 2048) throw new Error("Choose one model folder with at most 2048 files.");
  const manifests = files.filter((file) => file.name === "bundle-manifest-v1.json" || file.name === "manifest.json");
  if (manifests.length !== 1) throw new Error("Choose a folder containing exactly one supported Vons bundle manifest.");
  const file = manifests[0];
  if (file.size > 2 * 1024 * 1024) throw new Error("Manifest exceeds 2 MiB.");
  const root = file.webkitRelativePath.slice(0, -file.name.length);
  const manifestBytes = await file.arrayBuffer();
  const { manifest, paths, backend, head, modelId } = requiredFiles(manifestBytes, selectedHead);
  const assets: LocalAsset[] = [];
  let total = manifestBytes.byteLength;
  for (const path of paths) {
    const selected = files.filter((candidate) => candidate.webkitRelativePath === root + path);
    const expected = manifest.files.find((record) => record.path === path)!;
    if (selected.length !== 1 || selected[0].size !== expected.bytes) throw new Error(`Missing or wrong-sized model file: ${path}`);
    total += selected[0].size;
    if (total > MAX_BUNDLE_BYTES) throw new Error("The selected runtime assets exceed 128 MiB.");
    assets.push({ path, bytes: await selected[0].arrayBuffer() });
  }
  const bundle: LocalBundle = { manifestPath: file.name, manifestBytes,
    manifestHash: await digest(manifestBytes), backend, ...(head ? { head } : {}), modelId, assets };
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

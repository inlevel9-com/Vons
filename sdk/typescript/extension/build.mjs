import { build } from "esbuild";
import { readFile, writeFile, mkdir, copyFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { buildWithStagedOutput, resolveBuildOutput } from "./build-output.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const source = path.join(root, "extension");
const args = process.argv.slice(2);

if (args.length === 1 && args[0] === "--help") {
  console.log("Usage: node extension/build.mjs [--out-dir dist/<new-directory>]");
  process.exit(0);
}

const output = resolveBuildOutput(root, args);

await buildWithStagedOutput(output, async (staging) => {
  await mkdir(path.join(staging, "runtime"), { recursive: true });
  const result = await build({
    absWorkingDir: root,
    entryPoints: ["extension/panel.ts", "extension/worker.ts"],
    bundle: true,
    splitting: true,
    format: "esm",
    platform: "browser",
    target: "chrome114",
    conditions: ["onnxruntime-web-use-extern-wasm"],
    outdir: staging,
    metafile: true,
    legalComments: "eof",
    minify: true,
  });

  for (const name of ["manifest.json", "background.mjs", "panel.html", "panel.css", "privacy.html"]) {
    await copyFile(path.join(source, name), path.join(staging, name));
  }
  await copyFile(path.join(root, "../../docs/assets/inlevel9-signature.png"), path.join(staging, "brand.png"));
  for (const size of [16, 32, 48, 128]) {
    await copyFile(path.join(root, `../../docs/assets/vons-extension-icon-${size}.png`), path.join(staging, `icon-${size}.png`));
  }
  await copyFile(path.join(root, "LICENSE"), path.join(staging, "LICENSE"));
  for (const name of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm",
    "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]) {
    await copyFile(path.join(root, "node_modules/onnxruntime-web/dist", name), path.join(staging, "runtime", name));
  }

  const packages = new Set(["onnxruntime-web"]);
  for (const input of Object.keys(result.metafile.inputs)) {
    const parts = input.split("node_modules/").at(-1).split("/");
    if (input.includes("node_modules/")) packages.add(parts[0].startsWith("@") ? `${parts[0]}/${parts[1]}` : parts[0]);
  }
  const notices = ["# Bundled third-party notices", "", "Vons source uses the accompanying LICENSE. Model weights are not included.", ""];
  for (const name of [...packages].sort()) {
    const packageRoot = path.join(root, "node_modules", name);
    const metadata = JSON.parse(await readFile(path.join(packageRoot, "package.json"), "utf8"));
    const files = (await readdir(packageRoot)).filter((file) => /^(?:licen[sc]e|copying|notice|thirdpartynotices)(?:\.|$)/i.test(file));
    if (!files.length && name.startsWith("onnxruntime-")) {
      if (metadata.version !== "1.30.0") throw new Error("Refresh the pinned ONNX Runtime notices before upgrading.");
      notices.push(`## ${name} ${metadata.version}`, await readFile(path.join(source, "ONNX_RUNTIME_NOTICES.md"), "utf8"));
      continue;
    }
    if (!files.length) throw new Error(`Missing third-party license notice: ${name}`);
    notices.push(`## ${name} ${metadata.version}`, `Declared license: ${metadata.license}`, "");
    for (const file of files) notices.push(`### ${file}`, "", await readFile(path.join(packageRoot, file), "utf8"), "");
  }
  await writeFile(path.join(staging, "THIRD_PARTY_NOTICES.txt"), notices.join("\n"));

  const manifest = JSON.parse(await readFile(path.join(staging, "manifest.json"), "utf8"));
  if (JSON.stringify(manifest.permissions) !== '["sidePanel"]' || manifest.host_permissions || manifest.content_scripts) {
    throw new Error("Unexpected extension access.");
  }
}, { allowedRoot: path.join(root, "dist") });

console.log(`Built ${output}; model weights excluded; package notices included.`);

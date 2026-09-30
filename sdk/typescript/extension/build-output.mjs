import { lstat, mkdir, mkdtemp, realpath, rename, rm } from "node:fs/promises";
import path from "node:path";

function isWithin(parent, candidate, allowEqual = false) {
  const relative = path.relative(parent, candidate);
  return (allowEqual && relative === "") || (relative !== "" && relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
}

export function resolveBuildOutput(root, args) {
  const distRoot = path.resolve(root, "dist");
  if (args.length === 0) return path.join(distRoot, "chrome-extension");
  if (args.length !== 2 || args[0] !== "--out-dir" || args[1].startsWith("--")) {
    throw new TypeError("Use --out-dir followed by a new directory inside sdk/typescript/dist.");
  }

  const output = path.resolve(root, args[1]);
  if (!isWithin(distRoot, output)) {
    throw new RangeError("The extension output directory must be inside sdk/typescript/dist.");
  }
  return output;
}

async function resolveOutputPath(output, allowedRoot) {
  if (!allowedRoot) return output;

  const root = path.resolve(allowedRoot);
  await mkdir(root, { recursive: true });
  const realRoot = await realpath(root);
  const requestedParent = path.dirname(output);
  let existingAncestor = requestedParent;

  while (true) {
    try {
      const realAncestor = await realpath(existingAncestor);
      if (!isWithin(realRoot, realAncestor, true)) {
        throw new RangeError("The extension output parent must resolve inside sdk/typescript/dist.");
      }
      break;
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
      const parent = path.dirname(existingAncestor);
      if (parent === existingAncestor) throw error;
      existingAncestor = parent;
    }
  }

  await mkdir(requestedParent, { recursive: true });
  const realParent = await realpath(requestedParent);
  if (!isWithin(realRoot, realParent, true)) {
    throw new RangeError("The extension output parent must resolve inside sdk/typescript/dist.");
  }
  return path.join(realParent, path.basename(output));
}

async function assertReplaceableDirectory(output) {
  try {
    const info = await lstat(output);
    if (info.isSymbolicLink() || !info.isDirectory()) {
      throw new TypeError(`Refusing to replace a non-directory extension output: ${output}`);
    }
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

function reportCleanupProblem(warn, message) {
  try {
    warn(message);
  } catch {
    // A logging failure must not replace the build or recovery error.
  }
}

export async function buildWithStagedOutput(outputDirectory, build, options = {}) {
  const requestedOutput = path.resolve(outputDirectory);
  const output = await resolveOutputPath(requestedOutput, options.allowedRoot);
  const parent = path.dirname(output);
  await mkdir(parent, { recursive: true });
  await assertReplaceableDirectory(output);

  const staging = await mkdtemp(path.join(parent, `.${path.basename(output)}.stage-`));
  const backup = `${staging}.previous`;
  const renameDirectory = options.renameDirectory ?? rename;
  const removeDirectory = options.removeDirectory ?? rm;
  const warn = options.warn ?? ((message) => console.warn(message));
  let previousMoved = false;

  try {
    await build(staging);

    if (await assertReplaceableDirectory(output)) {
      await renameDirectory(output, backup);
      previousMoved = true;
    }

    await renameDirectory(staging, output);

    if (previousMoved) {
      previousMoved = false;
      try {
        await removeDirectory(backup, { recursive: true, force: true });
      } catch {
        reportCleanupProblem(warn, `Built the new extension package; previous output remains at ${backup}`);
      }
    }
  } catch (error) {
    if (previousMoved) {
      try {
        await renameDirectory(backup, output);
        previousMoved = false;
      } catch (restoreError) {
        throw new AggregateError([error, restoreError], `Build failed; previous output is preserved at ${backup}`);
      }
    }
    throw error;
  } finally {
    try {
      await removeDirectory(staging, { recursive: true, force: true });
    } catch (error) {
      reportCleanupProblem(warn, `Could not remove extension staging output ${staging} (${error?.code ?? "cleanup failure"}).`);
    }
  }
}

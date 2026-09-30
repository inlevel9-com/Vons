import assert from "node:assert/strict";
import { mkdtemp, mkdir, readdir, readFile, rename, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { buildWithStagedOutput, resolveBuildOutput } from "../extension/build-output.mjs";

test("failed extension builds preserve the existing package", async () => {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "vons-extension-build-"));
  const output = path.join(temporaryRoot, "chrome-extension");
  await mkdir(output);
  await writeFile(path.join(output, "previous.txt"), "keep this package");

  try {
    await assert.rejects(
      buildWithStagedOutput(output, async (staging) => {
        await writeFile(path.join(staging, "partial.txt"), "incomplete");
        throw new Error("synthetic build failure");
      }),
      /synthetic build failure/,
    );

    assert.equal(await readFile(path.join(output, "previous.txt"), "utf8"), "keep this package");
    assert.deepEqual(await readdir(output), ["previous.txt"]);
    assert.deepEqual(await readdir(temporaryRoot), ["chrome-extension"]);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("successful extension builds replace the previous package after staging", async () => {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "vons-extension-build-"));
  const output = path.join(temporaryRoot, "chrome-extension");
  await mkdir(output);
  await writeFile(path.join(output, "previous.txt"), "old package");

  try {
    await buildWithStagedOutput(output, async (staging) => {
      await writeFile(path.join(staging, "manifest.json"), '{"manifest_version":3}');
    });

    assert.equal(await readFile(path.join(output, "manifest.json"), "utf8"), '{"manifest_version":3}');
    await assert.rejects(readFile(path.join(output, "previous.txt")), { code: "ENOENT" });
    assert.deepEqual(await readdir(temporaryRoot), ["chrome-extension"]);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("a failed final rename restores the previous package once", async () => {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "vons-extension-build-"));
  const output = path.join(temporaryRoot, "chrome-extension");
  await mkdir(output);
  await writeFile(path.join(output, "previous.txt"), "keep this package");
  let stagingPath;
  let backupPath;
  let injectedFailure = false;
  let restoreCalls = 0;

  const renameDirectory = async (from, to) => {
    if (from === output && to.endsWith(".previous")) backupPath = to;
    if (from === backupPath && to === output) restoreCalls += 1;
    if (!injectedFailure && from === stagingPath && to === output) {
      injectedFailure = true;
      const error = new Error("synthetic final rename failure");
      error.code = "EIO";
      throw error;
    }
    return rename(from, to);
  };

  try {
    await assert.rejects(
      buildWithStagedOutput(output, async (staging) => {
        stagingPath = staging;
        await writeFile(path.join(staging, "partial.txt"), "not installed");
      }, { renameDirectory }),
      /synthetic final rename failure/,
    );

    assert.equal(injectedFailure, true);
    assert.equal(restoreCalls, 1);
    assert.ok(backupPath);
    assert.equal(await readFile(path.join(output, "previous.txt"), "utf8"), "keep this package");
    assert.deepEqual(await readdir(output), ["previous.txt"]);
    assert.deepEqual(await readdir(temporaryRoot), ["chrome-extension"]);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("a failed restore reports both errors and preserves the previous package at its backup", async () => {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "vons-extension-build-"));
  const output = path.join(temporaryRoot, "chrome-extension");
  await mkdir(output);
  await writeFile(path.join(output, "previous.txt"), "recoverable package");
  let stagingPath;
  let backupPath;
  const finalRenameError = new Error("synthetic final rename failure");
  const restoreError = new Error("synthetic restore failure");
  finalRenameError.code = "EIO";
  restoreError.code = "EACCES";

  const renameDirectory = async (from, to) => {
    if (from === output && to.endsWith(".previous")) {
      backupPath = to;
      return rename(from, to);
    }
    if (from === stagingPath && to === output) throw finalRenameError;
    if (from === backupPath && to === output) throw restoreError;
    return rename(from, to);
  };

  try {
    await assert.rejects(
      buildWithStagedOutput(output, async (staging) => {
        stagingPath = staging;
        await writeFile(path.join(staging, "partial.txt"), "not installed");
      }, { renameDirectory }),
      (error) => error instanceof AggregateError
        && error.errors[0] === finalRenameError
        && error.errors[1] === restoreError,
    );

    assert.ok(backupPath);
    assert.equal(await readFile(path.join(backupPath, "previous.txt"), "utf8"), "recoverable package");
    await assert.rejects(readFile(path.join(output, "previous.txt")), { code: "ENOENT" });
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("staging cleanup failures do not mask the original build error", async () => {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "vons-extension-build-"));
  const output = path.join(temporaryRoot, "chrome-extension");
  await mkdir(output);
  await writeFile(path.join(output, "previous.txt"), "keep this package");
  const buildError = new Error("synthetic primary build failure");
  let stagingPath;
  const warnings = [];
  const removeDirectory = async (target, options) => {
    if (target === stagingPath) {
      const error = new Error("synthetic cleanup failure");
      error.code = "EPERM";
      throw error;
    }
    return rm(target, options);
  };

  try {
    await assert.rejects(
      buildWithStagedOutput(output, async (staging) => {
        stagingPath = staging;
        throw buildError;
      }, { removeDirectory, warn: (message) => warnings.push(message) }),
      (error) => error === buildError,
    );

    assert.equal(await readFile(path.join(output, "previous.txt"), "utf8"), "keep this package");
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /EPERM/);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("cleanup and warning failures do not mask a final-rename error after restore", async () => {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "vons-extension-build-"));
  const output = path.join(temporaryRoot, "chrome-extension");
  await mkdir(output);
  await writeFile(path.join(output, "previous.txt"), "keep this package");
  const finalRenameError = new Error("synthetic final rename failure");
  finalRenameError.code = "EIO";
  let stagingPath;
  let backupPath;
  let restoreCalls = 0;
  const warnings = [];
  const renameDirectory = async (from, to) => {
    if (from === output && to.endsWith(".previous")) {
      backupPath = to;
      return rename(from, to);
    }
    if (from === stagingPath && to === output) throw finalRenameError;
    if (from === backupPath && to === output) {
      restoreCalls += 1;
      return rename(from, to);
    }
    return rename(from, to);
  };
  const removeDirectory = async (target, options) => {
    if (target === stagingPath) {
      const error = new Error("synthetic cleanup failure");
      error.code = "EPERM";
      throw error;
    }
    return rm(target, options);
  };

  try {
    await assert.rejects(
      buildWithStagedOutput(output, async (staging) => {
        stagingPath = staging;
        await writeFile(path.join(staging, "partial.txt"), "not installed");
      }, {
        renameDirectory,
        removeDirectory,
        warn: (message) => {
          warnings.push(message);
          throw new Error("synthetic warning failure");
        },
      }),
      (error) => error === finalRenameError,
    );

    assert.equal(restoreCalls, 1);
    assert.ok(backupPath);
    assert.equal(await readFile(path.join(output, "previous.txt"), "utf8"), "keep this package");
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /EPERM/);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("a previous-package cleanup failure warns after the new package is installed", async () => {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "vons-extension-build-"));
  const output = path.join(temporaryRoot, "chrome-extension");
  await mkdir(output);
  await writeFile(path.join(output, "previous.txt"), "previous package");
  let backupPath;
  const warnings = [];
  const renameDirectory = async (from, to) => {
    if (from === output && to.endsWith(".previous")) backupPath = to;
    return rename(from, to);
  };
  const removeDirectory = async (target, options) => {
    if (target === backupPath) {
      const error = new Error("synthetic backup cleanup failure");
      error.code = "EPERM";
      throw error;
    }
    return rm(target, options);
  };

  try {
    await buildWithStagedOutput(output, async (staging) => {
      await writeFile(path.join(staging, "manifest.json"), '{"manifest_version":3}');
    }, { renameDirectory, removeDirectory, warn: (message) => warnings.push(message) });

    assert.equal(await readFile(path.join(output, "manifest.json"), "utf8"), '{"manifest_version":3}');
    assert.ok(backupPath);
    assert.equal(await readFile(path.join(backupPath, "previous.txt"), "utf8"), "previous package");
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /previous output remains/);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("an output parent symlink cannot escape the allowed distribution root", async (context) => {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "vons-extension-build-"));
  const allowedRoot = path.join(temporaryRoot, "dist");
  const outside = path.join(temporaryRoot, "outside");
  const link = path.join(allowedRoot, "link");
  await mkdir(allowedRoot);
  await mkdir(outside);

  try {
    await symlink(outside, link, "dir");
  } catch (error) {
    await rm(temporaryRoot, { recursive: true, force: true });
    if (["EPERM", "EACCES", "ENOTSUP"].includes(error.code)) {
      context.skip(`directory symlinks are unavailable: ${error.code}`);
      return;
    }
    throw error;
  }

  let buildCalled = false;
  try {
    await assert.rejects(
      buildWithStagedOutput(path.join(link, "package"), async () => {
        buildCalled = true;
      }, { allowedRoot }),
      /inside sdk\/typescript\/dist/,
    );
    assert.equal(buildCalled, false);
    assert.deepEqual(await readdir(outside), []);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("extension output paths stay under dist", () => {
  const root = path.resolve(os.tmpdir(), "vons-typescript");

  assert.equal(resolveBuildOutput(root, []), path.join(root, "dist/chrome-extension"));
  assert.equal(resolveBuildOutput(root, ["--out-dir", "dist/review"]), path.join(root, "dist/review"));
  assert.throws(() => resolveBuildOutput(root, ["--out-dir", "../outside"]), /inside sdk\/typescript\/dist/);
  assert.throws(() => resolveBuildOutput(root, ["--out-dir", "dist"]), /inside sdk\/typescript\/dist/);
  assert.throws(() => resolveBuildOutput(root, ["--force"]), /Use --out-dir/);
});

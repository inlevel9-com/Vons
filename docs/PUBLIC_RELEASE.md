# Public source release

The public source set is defined by `configs/public-release.json`. The root
`.gitignore` excludes unreviewed root paths and internal documents; the release
tool independently rejects any Git candidate outside that exact file list.

## Included and excluded material

Included: Python and TypeScript source, tests, source demo files, the SDK
dependency lockfile, pilot configuration, reviewed public documentation and the
owner-supplied Vons and INLEVEL9 logos. Reviewed publications are listed separately in
`docs/publications/README.md`.

Excluded: all local datasets (including synthetic outputs), model weights,
tokenizers, ONNX external-data shards, experiment reports, evidence archives,
private prompts and raw model responses, agent/session records, license drafts,
historical planning documents, generated demo JavaScript, environments,
credentials, service caches and unreviewed binary files. Keep excluded originals
locally; never force-add them to make an upload work.

## Audit and export

Run from the repository root with Python 3.10+ and Git:

```sh
python tools/prepare_public_release.py
python tools/prepare_public_release.py --target github --output releases/github-source
python tools/prepare_public_release.py --target huggingface --output releases/huggingface-source
```

Each export is a new directory containing only allowlisted files and a
`RELEASE_MANIFEST.json` with byte counts and SHA-256 hashes. Existing output
directories are never overwritten. Use a new versioned output path after any
source change. The Hugging Face export substitutes its model card as the root
README. Both exports contain source, the current v1.1 Paper and Tech Report and
their two reviewed historical PDFs, with no pretrained assets.

The audit detects known token/key patterns, personal home paths, private writing
session URLs, unexpected Git files, symlinks and changed reviewed binaries.
Pattern scanning is heuristic: passing it is not a complete secret audit or
proof of redistribution rights. Read the actual export before publishing.

## GitHub

The source checkout uses `main`; review `git status`, the existing commit history
and the export manifest before publishing changes. The owner performs commits,
pushes and visibility changes. This preparation does not perform those actions. Preserve a clean initial tree: `.gitignore` cannot
remove already committed material from history. If existing remote history is
later attached, review that history independently before making it public.

The CI workflow checks the release allowlist, Python tests and lint, and the SDK
tests, three browser builds, and extension type checking/building. Push/pull-request checks run when source,
tests, configuration, packaged extension images or the workflow changes; documentation-only edits do not
trigger the full suite. A manual run remains available. SDK tokenizer tests use
synthetic fixtures and require no private pilot assets. It has read-only repository permissions and
does not publish packages or artifacts. Checks that require unavailable optional
Python dependencies may be skipped; install training/export extras for the full
local suite. The historical pilot-summary regression is also skipped when its
private input files are absent. A separate synthetic regression runs without
those files; do not publish the historical inputs merely to remove that skip.

## Hugging Face

Upload only a reviewed versioned Hugging Face export to `INLEVEL9/Vons`.
The initial verified upload used `releases/huggingface-source-v8`. The reviewed
v1.1 update was published from `releases/huggingface-source-v13` as Hub commit
[`a76a881628c03026a2d0fa0993da5904f13480e6`](https://huggingface.co/INLEVEL9/Vons/commit/a76a881628c03026a2d0fa0993da5904f13480e6); see the
[step-by-step upload guide](HUB_PUBLISHING.md). Do not select the development workspace in the
Hub upload dialog. Use explicit allow/ignore patterns if using `upload_folder`;
its handling of ignore files depends on the client version and upload method.

The current checkout also includes the [Chrome extension source](../sdk/typescript/extension/README.md),
which was not part of that initial 97-file upload but is included in the v1.1
source update. Generated extension packages remain local and
ignored; distribute only their reviewed contents, with dependency notices and
without imported models or prompts. Chrome Web Store version 0.1.0 was submitted
on 2026-09-25, passed review and had its public listing independently read back
the same day: [Vons — Local Decisions](https://chromewebstore.google.com/detail/vons-%E2%80%94-local-decisions/cbmjkokoojinfgafahncmjlfmicinide).
That listing does not establish model quality or production readiness.

This source preview has no weights and no inference widget. A later weight
release needs its own exact file inventory, upstream notices, model/data
provenance and runtime validation. Do not turn the ignored checkpoint directory
into a public upload by adding Git LFS patterns.

Official references: [GitHub ignore guidance](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files)
and [Hub upload guidance](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Terms and publication state

The owner selected noncommercial-free / enterprise-commercial software terms
and CC BY 4.0 articles on 2026-09-24. The prepared root [license](../LICENSE), SDK
license copy, package metadata and Hub card now match that source-available
direction. [Licensing scope](LICENSING.md) explains the boundary and preserves
earlier and third-party rights. Local preparation is complete for the owner to
commit, push and publish; no commercial customer agreement is adopted here.
npm publishing remains disabled with `private: true`.

The Python build backend requires setuptools 77.0.3+ to include the custom SPDX
license reference and exact license file in wheel/sdist metadata. No runtime
dependency was added. Verify built package contents as well as source exports.

The author-approved public contact is Kwangseob Ahn, oswarld@inlevel9.com,
INLEVEL9 / SEJONG UNIV. Logo inclusion does not grant trademark rights or imply
institutional endorsement. Model-assisted writing/review is not peer acceptance.

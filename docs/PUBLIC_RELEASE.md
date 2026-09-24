# Public source release

The public source set is defined by `configs/public-release.json`. The root
`.gitignore` excludes unreviewed root paths and internal documents; the release
tool independently rejects any Git candidate outside that exact file list.

## Included and excluded material

Included: Python and TypeScript source, tests, source demo files, the SDK
dependency lockfile, pilot configuration, reviewed public documentation and the
owner-supplied INLEVEL9 logo. Reviewed publications are listed separately in
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
README. Both exports contain source only, with no pretrained assets.

The audit detects known token/key patterns, personal home paths, private writing
session URLs, unexpected Git files, symlinks and changed reviewed binaries.
Pattern scanning is heuristic: passing it is not a complete secret audit or
proof of redistribution rights. Read the actual export before publishing.

## GitHub

Git is initialized locally on `main`. No initial commit or remote publication
is implied by this preparation. Review `git status` and the export manifest
before the initial commit. Preserve a clean initial tree: `.gitignore` cannot
remove already committed material from history. If existing remote history is
later attached, review that history independently before making it public.

The CI workflow checks the release allowlist, Python tests and lint, and the SDK
tests and three browser builds. It has read-only repository permissions and
does not publish packages or artifacts. Checks that require unavailable optional
Python dependencies may be skipped; install training/export extras for the full
local suite. The historical pilot-summary regression is also skipped when its
private input files are absent. A separate synthetic regression runs without
those files; do not publish the historical inputs merely to remove that skip.

## Hugging Face

Upload only the generated `releases/huggingface-source` directory after reviewing
the prepared terms and choosing the destination. Do not select the development workspace in the
Hub upload dialog. Use explicit allow/ignore patterns if using `upload_folder`;
its handling of ignore files depends on the client version and upload method.

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
earlier and third-party rights. The custom agreement is prepared for owner
review before distribution; no commercial customer agreement is adopted here.
npm publishing remains disabled with `private: true`.

The Python build backend requires setuptools 77.0.3+ to include the custom SPDX
license reference and exact license file in wheel/sdist metadata. No runtime
dependency was added. Verify built package contents as well as source exports.

The author-approved public contact is Kwangseob Ahn, oswarld@inlevel9.com,
INLEVEL9 / SEJONG UNIV. Logo inclusion does not grant trademark rights or imply
institutional endorsement. Model-assisted writing/review is not peer acceptance.

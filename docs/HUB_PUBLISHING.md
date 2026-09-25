# Upload the Vons source preview to Hugging Face

This guide covers the completed initial upload and subsequent reviewed updates.
The current v1.1 snapshot contains 128 reviewed files, including the source,
extension source, Vons logo, current Paper and Tech Report PDFs and their
historical reviewed versions. Drafts, source manuscript bundles, model weights,
restricted data and private records are excluded.

## Destination and scope

Use the existing model repository [INLEVEL9/Vons](https://huggingface.co/INLEVEL9/Vons).
The organization page [INLEVEL9](https://huggingface.co/INLEVEL9) is its parent,
not a folder-upload destination. There is no need to create another repository,
a dataset or a Space for this source preview.

The owner authorized the Hub upload on 2026-09-25. The verified
[initial publication commit](https://huggingface.co/INLEVEL9/Vons/commit/49ad2ef694ffead3be170dd32d43a248edb08c82)
contains all 97 reviewed files. The Hub generated two PDF LFS tracking lines in
`.gitattributes`; the other 96 files matched the reviewed export exactly. The
public model card now uses the Vons Community and Commercial License 1.0.
The article CC BY 4.0 terms are separate. A later documentation update records
the arXiv submission status without adding models or draft material.

The reviewed v1.1 update uses the local snapshot
`releases/huggingface-source-v13` and was published in Hub commit
[`a76a881628c03026a2d0fa0993da5904f13480e6`](https://huggingface.co/INLEVEL9/Vons/commit/a76a881628c03026a2d0fa0993da5904f13480e6).
It adds the 21-page Preprint v1.1, the 29-page
Technical Report v1.1, post-freeze implementation source and tests, Chrome
extension source and the owner-supplied Vons logo. It does not add manuscript
source bundles, claim ledgers, model weights, tokenizer assets or benchmark
data.

A commit to this public repository makes uploaded files publicly readable.
Uploading source does not create hosted inference, pretrained weights or a
working `from_pretrained` package. The card states those limits explicitly.
The main code/contribution destination is
[inlevel9-com/Vons](https://github.com/inlevel9-com/Vons). Its visibility is
controlled separately from Hugging Face.

## 1. Install and sign in

On macOS with Homebrew:

```sh
brew install hf
hf auth login
hf auth whoami
```

If Homebrew already installed `hf`, use `brew upgrade hf` when an update is
needed. Current CLI versions offer browser login: follow the displayed URL and
code, and sign in using your account with write access to `INLEVEL9/Vons`.
`hf auth whoami` confirms the account and organization membership; membership
alone does not guarantee repository write permission.

If your CLI asks for a token instead, create a user access token in
[Hugging Face token settings](https://huggingface.co/settings/tokens), preferably
scoped to write to `INLEVEL9/Vons`, and enter it at the hidden login prompt.
Keep credentials out of source files, shell command arguments and issue reports.
See the official [CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).

## 2. Use the reviewed export

Run the source audit from the Vons checkout:

```sh
python3 tools/prepare_public_release.py
```

The initial uploaded snapshot is `releases/huggingface-source-v8`. The reviewed
v1.1 artifact snapshot is `releases/huggingface-source-v13`; it contains 128 reviewed
files plus a local `RELEASE_MANIFEST.json` receipt with every file's byte count
and SHA-256 digest. Its root README is the Hub model card; the working-tree
README remains the GitHub landing page. Live read-back confirmed a public,
ungated repository at commit `a76a881628c03026a2d0fa0993da5904f13480e6`,
with exact hashes for the two v1.1 PDFs and Vons logo.

This export is a snapshot. If you edit any public source or documentation after
preparation, create a fresh versioned directory and use that path below:

```sh
python3 tools/prepare_public_release.py --target huggingface --output releases/huggingface-source-next
```

The exporter refuses to overwrite an existing directory. Never point the upload
command at the development checkout or an old export containing stale terms.

## 3. Upload

For a subsequent update, create and review a fresh export, then upload it:

```sh
hf upload INLEVEL9/Vons releases/huggingface-source-next . \
  --repo-type model \
  --exclude RELEASE_MANIFEST.json \
  --commit-message "Update reviewed Vons source preview"
```

The final `.` places the files at the repository root while preserving nested
paths. The command creates a Hub commit directly; a separate Git commit/push
is unnecessary. It updates matching remote paths, including README and LICENSE,
and does not request deletion of unrelated remote files. The manifest stays
local as an audit receipt; it is not an additional source file in the allowlist.

A successful command prints a repository or commit URL. For a 401/403 response,
check the account, `INLEVEL9/Vons` write permission and credential scope; do not
create a duplicate personal repository. Official
[upload guidance](https://huggingface.co/docs/huggingface_hub/guides/upload)
describes folder uploads and filtering.

## 4. Verify the result

Open [the model card](https://huggingface.co/INLEVEL9/Vons) and
[Files and versions](https://huggingface.co/INLEVEL9/Vons/tree/main):

- The card says this is a source preview with no pretrained weights or hosted inference.
- License metadata uses `other`, `vons-community-commercial-1.0` and `LICENSE`.
  It must no longer identify the software as Apache-2.0 or CC BY 4.0.
- Root `LICENSE` and `sdk/typescript/LICENSE` contain the same custom terms.
- Source paths such as `vons/`, `tools/`, `tests/` and `sdk/typescript/` are present.
- The Paper and Tech Report links open the 21-page v1.1 paper and 29-page v1.1 report.
  Their digests match the publication index. No manuscript source bundle or
  private review material is uploaded.
- The Vons and INLEVEL9 logo links resolve, and a signed-out visitor can read the card.

Save the resulting Hub commit URL with the local release manifest. GitHub
publication, manuscript publication and a hosted demo are separate actions.
For subsequent source changes, create a new reviewed export and repeat the same
upload command with its new path. Add research files only after their separate
review and an explicit public-allowlist update.

The initial Hub upload and publication-document update were owner-authorized.
GitHub commits, pushes and visibility changes remain owner-operated. The v1.1
Hub source update includes reviewed extension source, but generated extension
packages and imported local models remain excluded.
Official CLI/upload guidance checked 2026-09-25.

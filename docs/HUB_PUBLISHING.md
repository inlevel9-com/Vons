# Publishing on the Hugging Face Hub

The owner-provided organization is [INLEVEL9](https://huggingface.co/INLEVEL9).
Its profile exists, but no Vons model repository or demo is announced here.
Keep the main code and contribution workflow at
[inlevel9-com/Jons](https://github.com/inlevel9-com/Jons).

## Choose the surface

| Surface | Purpose for Vons | Ready now? |
| --- | --- | --- |
| Organization card | Introduce INLEVEL9 and link research and source | Copy draft below; not published |
| Model repository | Model card and versioned downloadable assets | Source-only export prepared; no weights included |
| Space | A browser-accessible demonstration | Requires a separate app implementation and verification |
| Dataset repository | Permitted, documented datasets | No dataset release prepared |

Uploading source files does not create a working inference service. If a model
repository initially contains only this source preview, its card must retain
the explicit no-weights/no-hosted-inference notice. Do not suggest
`from_pretrained` works until a compatible model package is actually released.

## Upload a reviewed source preview

1. Open [New model](https://huggingface.co/new). Choose owner `INLEVEL9`, a
   repository name such as `vons-source-preview`, and **Private** for review.
   The example name is a proposal, not an existing repository.
2. In **Files**, select **Add file**, then **Upload files**. Upload only a newly
   audited export from `tools/prepare_public_release.py --target huggingface`.
   Preserve directory paths and put its model-card `README.md` at the root.
3. Retain the prepared custom software license and model-card metadata. Do not
   select Apache-2.0 or CC BY 4.0 as the software license. Article rights are
   separate. Keep the UI commit message descriptive.
4. Read back the uploaded files, paths, rendered card and license before release.
   For a directory with many files, the official CLI can preserve the layout:

```sh
hf auth login
hf upload INLEVEL9/vons-source-preview releases/huggingface-source-v5 . --repo-type model
```

The command is an upload example, not something this preparation executes.
Use it only after confirming the actual repository name and export directory.
Authenticate locally; never put a token in a committed file or paste it into
an issue. See [repository and upload guidance](https://huggingface.co/docs/hub/repositories-getting-started).

## Make the repository public

An owner can change visibility in the repository's **Settings** tab. Review the
complete contents and history, then choose **Public** and complete the displayed
confirmation. Private repositories return a not-found response to unauthorized
visitors. After release, check the page and files while signed out. For a public
Space, both the app and its source are visible. These steps are described in
[repository settings](https://huggingface.co/docs/hub/repositories-settings).

## Add the organization introduction

The organization's **Create a Card** action creates a static Space named
`README`. Its `README.md` becomes the organization card when the Space is Public.
See [organization cards](https://huggingface.co/docs/hub/organizations-cards).
Review the following copy against current availability before publishing:

```markdown
# INLEVEL9

Research, practical tools and a place to learn from what works and what fails.

## Vons

Vons explores compact local decision models for agent workflows. A planner
provides state and candidate choices; the host retains execution and consent.

We are preparing a source preview and an English research paper for arXiv.
Pretrained weights and a hosted inference demo are not yet available.
Questions, experience reports, reproductions and contributions are welcome
through the project repository once it is public.

- Website: https://inlevel9.com/
- Source destination: https://github.com/inlevel9-com/Jons
- Contact: oswarld@inlevel9.com

Software and article terms are separate. Qualifying noncommercial software use
is free; enterprise/commercial software use requires a written agreement.
The paper is being prepared under CC BY 4.0.
```

## Add an interactive Space later

Create a Space under `INLEVEL9` and choose a suitable SDK. Static HTML suits a
browser-only contract explorer or an independently verified local ONNX demo;
Gradio or Docker suits a server-backed application. Current official guidance
lists static Spaces as free and describes plan requirements for compute-backed
Gradio/Docker Spaces. Confirm current pricing before choosing paid compute.
See [Spaces overview](https://huggingface.co/docs/hub/spaces-overview).

A Space needs actual app files and a working build. A no-weight contract demo
must say it is deterministic. A real model demo needs permitted assets,
tokenizer/runtime validation, provider and failure reporting, and accurate
download-size information. Reading and exploration should not require a new
Vons account. Use the existing report forms for feedback before introducing a
separate account system or feedback database.

No Hub upload, repository creation, visibility change or paid resource is
performed by these instructions. Official documentation checked 2026-09-24.

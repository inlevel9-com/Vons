---
license: other
license_name: vons-community-commercial-1.0
license_link: LICENSE
language:
  - en
tags:
  - vons
  - research
  - candidate-selection
  - onnx
---

![Vons](../docs/assets/vons-logo.png)

![INLEVEL9](../docs/assets/inlevel9-signature.png)

# Vons

**Plan globally. Decide locally. Keep the host in control.**

Compact local decision models for frontier-agent workflows.

**Kwangseob Ahn** · **INLEVEL9 / SEJONG UNIV.** · **oswarld@inlevel9.com**

## Source preview

This repository contains research source and documentation. It does not contain
pretrained model weights, tokenizer assets or datasets. Hosted inference and
`from_pretrained` loading are not available from this source preview.

Vons studies Direct and Diffusion candidate scoring with abstention. The host
owns execution, permission and consent. Model output never authorizes an action.
English-first operation and a model asset budget under 64 MiB are design targets;
they are not general quality or complete browser-package guarantees.

See [the model card](../docs/MODEL_CARD.md),
[the TypeScript SDK](../sdk/typescript/README.md),
[the paper and Tech Report](../docs/publications/README.md), and
[release preparation](../docs/PUBLIC_RELEASE.md).

## Read, try and contribute

This source preview is distributed through [INLEVEL9/Vons](https://huggingface.co/INLEVEL9/Vons).
Source and contribution reports belong in
[inlevel9-com/Vons](https://github.com/inlevel9-com/Vons). The reviewed paper and Tech Report are included as PDF downloads below. The software remains a 0.1.0
research preview. No pretrained weights or working Space are announced.

Follow the [community guide](../docs/COMMUNITY.md) for the no-weight walkthrough,
experience and reproduction reports, and the
[contribution guide](../CONTRIBUTING.md) for review expectations. The paper's
[arXiv submission record](../docs/publications/ARXIV.md) is available. User reports are not
independent research validation.

Maintainers can use the [Hub upload guide](../docs/HUB_PUBLISHING.md) to prepare
an organization card, source preview and later a separately validated Space.

## Paper and Tech Report

| Publication | Download |
| --- | --- |
| Vons: A Compact, Host-Controlled Decision Component for Agent Workflows — Preprint v1.1 | [PDF · 21 pages](../docs/publications/vons-paper-v1.1.pdf) |
| Vons: Compact Decision Models for Frontier-Agent Workflows — Technical Report v1.1 | [PDF · 29 pages](../docs/publications/vons-tech-report-v1.1.pdf) |

Articles and original figures use **CC BY 4.0**. Their version, SHA-256 digests
and verification scope are in the [publication index](../docs/publications/README.md).
These preprints preserve negative results, missing measurements and the boundary
between frozen v1 evidence and the separately labelled post-freeze v2 handoff.
The paper uses arXiv **submission number 8127259**. The v1.1 PDF and metadata
were processed successfully on **2026-09-25** and submitted on **2026-09-26**.
The arXiv account currently reports **on hold** for moderation. This tracking
number is not a public arXiv article identifier. No public announcement or
peer-review acceptance is claimed.

Chrome Web Store version **0.1.0** passed review and is publicly available as
[Vons — Local Decisions](https://chromewebstore.google.com/detail/vons-%E2%80%94-local-decisions/cbmjkokoojinfgafahncmjlfmicinide),
verified on 2026-09-25. Store availability is not evidence of model quality,
scientific validity or production readiness.

## Local use

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m vons.cli generate-smoke --output data/generated/smoke.jsonl
python -m vons.cli validate-data --input data/generated/smoke.jsonl
```

The source export includes the research tools and pinned pilot configurations.
Obtain and review upstream assets separately before running model experiments.
Raw restricted benchmark data and private evidence are not redistributed here.

## Terms

Source is distributed under [Vons Community and Commercial License 1.0](https://huggingface.co/INLEVEL9/Vons/blob/main/LICENSE):
qualifying noncommercial use is free; enterprise and other commercial use require
a separate written agreement. Contact oswarld@inlevel9.com for commercial terms.
This is source-available software, not OSI-approved open source.

The paper and Tech Report are distributed under [CC BY 4.0](../docs/publications/LICENSE.md).
This permits commercial article reuse with attribution; software rights are
separate. Third-party assets retain their own terms. See
[licensing scope](../docs/LICENSING.md). This source release does not establish research validation or a commercial
customer agreement.

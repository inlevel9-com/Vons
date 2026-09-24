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

# Vons

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
[inlevel9-com/Vons](https://github.com/inlevel9-com/Vons). The paper and Tech Report are coming soon and intentionally excluded while
v1.0 manuscripts are prepared and reviewed. The software remains a 0.1.0
research preview. No pretrained weights or working Space are announced.

Follow the [community guide](../docs/COMMUNITY.md) for the no-weight walkthrough,
experience and reproduction reports, and the
[contribution guide](../CONTRIBUTING.md) for review expectations. The paper is
being prepared for [arXiv](../docs/publications/ARXIV.md). User reports are not
independent research validation.

Maintainers can use the [Hub upload guide](../docs/HUB_PUBLISHING.md) to prepare
an organization card, source preview and later a separately validated Space.

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

Source is distributed under [Vons Community and Commercial License 1.0](../LICENSE):
qualifying noncommercial use is free; enterprise and other commercial use require
a separate written agreement. Contact oswarld@inlevel9.com for commercial terms.
This is source-available software, not OSI-approved open source.

The paper and Tech Report are prepared under [CC BY 4.0](../docs/publications/LICENSE.md).
This permits commercial article reuse with attribution; software rights are
separate. Third-party assets retain their own terms. See
[licensing scope](../docs/LICENSING.md). This source release does not establish research validation or a commercial
customer agreement.

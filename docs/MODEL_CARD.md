# Vons model card

Status: research source preview. This package contains no pretrained weights,
tokenizer files or benchmark datasets.

## Intended use

Vons studies bounded candidate selection and abstention in agent workflows.
A host supplies the state and candidates. The host alone enforces permission,
consent and execution policy. KASI is a separate compatibility adapter.

## Architecture and target

The research implementation offers a Direct scorer and a conditional Diffusion
scorer. Pilot configurations pin `google/bert_uncased_L-4_H-256_A-4` at revision
`387825ce42dbb39b87911cdf8e383ee3b25184f8`. This identifies a research dependency;
its assets are not included or relicensed here.

English-first input and a model asset budget below 64 MiB are design targets.
Runtime libraries, complete download size and memory use are distinct quantities.

## Evaluation and limitations

Historical local experiments include synthetic pilots and an evaluation-only
external benchmark diagnostic. They do not establish general task quality,
production readiness, calibrated safety or a Direct/Diffusion superiority claim.
Reviewed numerical claims belong to the versioned
[paper and Tech Report](publications/README.md), with exact evidence references.
Unreviewed historical tables are not distributed as current results.

The ONNX candidate head does not support general score questions. Python and
TypeScript token-budget checks are not fully identical. Browser smoke, repeated
latency, full numerical parity and memory evidence remain separate gates.

Do not use model confidence as a permission grant or assume it is calibrated.
Vons cannot execute tools or approve high-risk actions.

## Data and provenance

Synthetic data can be generated from the source. Restricted data, model-generated
raw responses, private prompts, model checkpoints and local manifests are
excluded from this distribution. Obtain any external dataset under its own
terms. The source release manifest records hashes of the files actually exported.

## Author and terms

Kwangseob Ahn — INLEVEL9 / SEJONG UNIV. — oswarld@inlevel9.com.

The Vons-owned source is prepared under [Vons Community and Commercial License
1.0](../LICENSE): noncommercial community use is free; enterprise/commercial use
requires a separate written agreement. The articles and their original figures
use [CC BY 4.0](publications/LICENSE.md). Third-party assets retain their own
terms, and no model weights are licensed or included by this source-only card.
See [licensing scope](LICENSING.md) and [release preparation](PUBLIC_RELEASE.md).

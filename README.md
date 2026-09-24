<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/inlevel9-signature-dark.png">
    <img src="docs/assets/inlevel9-signature.png" alt="INLEVEL9" width="320">
  </picture>
</p>
<h1 align="center">Vons</h1>
<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0--preview-c5ff7a?style=flat-square&amp;labelColor=252525" alt="Version 0.1.0 preview">
  <img src="https://img.shields.io/badge/status-research_preview-8e9aaf?style=flat-square&amp;labelColor=252525" alt="Research preview">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-community%20%2F%20commercial-c5ff7a?style=flat-square&amp;labelColor=252525" alt="Noncommercial community use; commercial agreement required"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-3776ab?style=flat-square&amp;labelColor=252525" alt="Python 3.10 or later">
  <img src="https://img.shields.io/badge/node-%E2%89%A522-68a063?style=flat-square&amp;labelColor=252525" alt="Node 22 or later">
</p>
<h3 align="center">Plan globally. Decide locally. Keep the host in control.</h3>
<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="sdk/typescript/README.md">TypeScript SDK</a> ·
  <a href="docs/MODEL_CARD.md">Model card</a> ·
  <a href="docs/publications/README.md">Paper &amp; Tech Report</a> ·
  <a href="docs/COMMUNITY.md">Community</a> ·
  <a href="CONTRIBUTING.md">Contribute</a> ·
  <a href="docs/PUBLIC_RELEASE.md">Release guide</a>
</p>

---

Vons explores compact local decision models for frontier-agent workflows.
A planner supplies a state and a bounded set of candidates; Vons returns a
structured choice or abstention. The host owns execution, policy and consent.

The research code includes a one-pass **Direct** scorer and a conditional
**Diffusion** scorer with deterministic DDIM sampling. The English-first design
target is a model asset bundle under **64 MiB**, with CPU/WASM and optional
WebGPU execution. This is a target, not a complete browser-package guarantee.

This source preview includes the Python contract, training/evaluation/export
tools, a TypeScript SDK and tests. Pretrained weights, tokenizer assets, raw
benchmark data, private prompts and experiment logs are not distributed here.
There is no published npm package or hosted inference service implied by the
version badge.

## Read, try and participate

Start with the [community guide](docs/COMMUNITY.md): read the documentation, try a
deterministic example without weights, share an experience or reproduction
report, and contribute improvements. Failures and critical feedback are welcome.
The [contribution guide](CONTRIBUTING.md) explains review, privacy and attribution.

Destinations: [GitHub: inlevel9-com/Vons](https://github.com/inlevel9-com/Vons) ·
[Hugging Face: INLEVEL9/Vons](https://huggingface.co/INLEVEL9/Vons).
This release contains source, documentation and participation forms. The paper
and Tech Report are **coming soon**: both are being improved toward v1.0 and
are intentionally absent from this upload. See the
[publication status](docs/publications/README.md) and
[Hub release guide](docs/HUB_PUBLISHING.md). The software remains a 0.1.0 research
preview; the manuscript v1.0 goal is not a software release or a completed result.
No arXiv submission or hosted inference service is announced.

## Quickstart

Python 3.10+ is sufficient for the contract and synthetic data commands:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m vons.cli generate-smoke --output data/generated/smoke.jsonl
python -m vons.cli validate-data --input data/generated/smoke.jsonl
```

The smoke generator creates local synthetic examples; it does not download a
benchmark or invoke a model. Generated files are ignored by Git.

A deterministic host-policy example:

```python
from vons import KASIAdapter, KASIProposal

adapter = KASIAdapter(tool_policy={"read_weather": "low"})
decision = adapter.decide(
    KASIProposal.from_mapping({
        "calls": [{"name": "read_weather"}],
        "confidence": 0.9,
        "risk": "low",
    })
)
```

This adapter returns a decision. It never runs the proposed tool, and an input
confidence value is not a calibrated probability or an authorization grant.
Unregistered tools are refused. See the [KASI contract](docs/KASI_INTEGRATION.md).

## TypeScript SDK

```sh
cd sdk/typescript
npm ci
npm test
npm run build:browser
npm run build:benchmark
npm run build:parity
```

The contract entry point is `src/index.ts`; the optional `src/onnx.ts` runtime
verifies model manifests and runs a supplied ONNX bundle with an explicit WASM
or WebGPU provider. The demo needs locally exported models and tokenizers.
A source checkout alone cannot run pretrained inference.

Read the [SDK guide](sdk/typescript/README.md) for local serving and runtime
requirements.

## Research workflow

Install the optional dependencies when running your own experiments:

```sh
python -m pip install -e '.[train,export,ollama,dev]'
python -m vons.cli generate-synthetic --count 2000 --output data/generated/pilot.jsonl
python -m vons.cli --help
```

The pinned encoder and experiment settings are in
[`configs/pilot.json`](configs/pilot.json) and
[`configs/pilot-diffusion.json`](configs/pilot-diffusion.json).
Review upstream terms before downloading or redistributing assets. Reports
must preserve seeds, source revisions, failures, raw responses and measurement
scope. Never interpret a model's self-reported confidence as calibrated probability.

## Scope and limitations

- Synthetic pilot results do not establish external-task generalization.
- Direct and Diffusion results depend on training, candidate shape and padding;
  this preview does not establish a general method ranking.
- Browser smoke, repeated latency, numerical parity and memory measurement are
  separate checks. Missing measurements are unavailable, not zero.
- Python uses an approximate input-token estimate; the TypeScript ONNX path
  checks the exported tokenizer's aggregate budget.
- The general contract supports score questions, while the current candidate
  ONNX head reports them as unsupported.
- KASI is a host compatibility adapter, kept separate from the general decision
  contract. Model output cannot grant consent or authorize high-risk actions.

The [model card](docs/MODEL_CARD.md) describes intended use and release scope.
The [publication index](docs/publications/README.md) records manuscript availability.

## Development checks

```sh
python -m pytest -q
ruff check .
python tools/prepare_public_release.py
```

Some model/export tests require the optional training and export dependencies;
pytest reports those skips explicitly. The public-release audit checks an exact
file allowlist, known sensitive patterns, symlinks and reviewed binary digests.
It does not replace a complete secret or redistribution-rights review.

## Author

**Kwangseob Ahn**  
INLEVEL9 / SEJONG UNIV.  
[oswarld@inlevel9.com](mailto:oswarld@inlevel9.com)

## License and distribution

The [Vons Community and Commercial License 1.0](LICENSE) covers this
source release: qualifying **noncommercial use is free**; **enterprise and other
commercial use require a separate written agreement**. Contact
[oswarld@inlevel9.com](mailto:oswarld@inlevel9.com) for commercial terms.
This is source-available software with use restrictions.

The paper and Tech Report are separately prepared under
[CC BY 4.0](docs/publications/LICENSE.md), which permits commercial article reuse
with attribution. Their license does not grant commercial software rights.
Third-party components retain their original terms. See
[licensing scope](docs/LICENSING.md) and the [release guide](docs/PUBLIC_RELEASE.md).

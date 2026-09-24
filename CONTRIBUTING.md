# Contributing to Vons

Help make bounded local decisions easier to understand, reproduce and improve.
Questions, unsuccessful trials, documentation fixes and small code changes are
welcome. Start with the [community guide](docs/COMMUNITY.md).

## Choose a contribution

- Share an experience using the **Experience report** issue form. Describe what
  happened, including failures; a positive result is not required.
- Submit a **Reproduction report** with exact commands, versions, denominators
  and the scope of the claim you checked.
- Use **Bug report** for a minimal, preferably synthetic, reproduction.
- Propose a focused change through a pull request. Discuss changes to the public
  contract, data handling, model architecture or dependencies before implementing
  them. Small documentation corrections can go directly to a pull request.

The intended repository is [inlevel9-com/Jons](https://github.com/inlevel9-com/Jons).
Its URL spelling differs from the Vons project name. It was private and empty
when these instructions were prepared; public participation starts after the
maintainer publishes the reviewed source and verifies the issue forms.

## Work locally

Use Python 3.10 or later. From a source checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
python tools/prepare_public_release.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Tests for
optional training/export dependencies may skip explicitly. Do not download
restricted data or model weights simply to remove a skip. For TypeScript work,
use the [SDK checks](sdk/typescript/README.md).

Keep one purpose per change. Add behavior-focused regression coverage for fixes.
Report checks you ran, checks you could not run, and the reason. English is used
for code, comments and repository documentation; community questions and
experience reports may be in English or Korean.

## Preserve the boundaries

The general decision contract and the KASI compatibility adapter are separate.
Model output never executes a tool, grants consent or authorizes a high-risk
action. Self-reported confidence is not a calibrated probability. Missing
measurements stay unavailable; design targets are not achieved results.

Use synthetic or redacted examples in public reports. Do not attach credentials,
private prompts, customer traces, personal data, restricted benchmark rows,
model weights or private evidence archives. Keep your own raw records locally;
share only material you are entitled to disclose. Contact
[oswarld@inlevel9.com](mailto:oswarld@inlevel9.com) before disclosing a security
issue; the first message should describe the issue without credentials or
sensitive attachments.

New public files must be reviewed and added to `configs/public-release.json`
and, where necessary, `.gitignore`. Never force-add ignored research material.

## Review and attribution

Maintainers review scope, correctness, evidence and redistribution rights before
merging. An issue, testimonial or model-assisted review is not acceptance,
independent replication or peer review. Review may request changes or decline a
proposal; no response-time or merge guarantee is implied.

Be specific and respectful. Critique claims and code rather than people. Spam,
harassment, fabricated evidence and exposed private information may be removed;
technical criticism and negative results are welcome. Corrections should keep
the original claim and explain what changed whenever it is safe to do so.

The software uses the [Vons Community and Commercial License](LICENSE), with
separate [article terms](docs/publications/LICENSE.md). Submitting an issue does
not grant software-use rights or permission to use it as a promotional quote.
Code contributions are reviewed for compatibility with the repository license;
identify third-party material and obtain any required employer permission.
No copyright assignment or separate contributor agreement is created here.

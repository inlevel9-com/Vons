# Read, try, share and contribute

Vons explores compact local decisions inside agent workflows. A planner supplies
state and bounded candidates; the host retains execution, policy and consent.
The community should help people understand this boundary, try the software and
report what works or fails.

## Start here

| Your goal | Entry point | Current availability |
| --- | --- | --- |
| Understand Vons | [README](../README.md) and [model card](MODEL_CARD.md) | Included in this source preview |
| Read the research | [Paper and Tech Report](publications/README.md) | Coming soon; v1.0 manuscripts are in preparation and excluded from this release |
| Try without weights | The walkthrough below | Deterministic contract and synthetic data only |
| Run model inference | [TypeScript SDK](../sdk/typescript/README.md) | Requires separately obtained, reviewed model/tokenizer assets |
| Share experience | [Experience report](https://github.com/inlevel9-com/Vons/issues/new?template=experience.yml) | GitHub account required to submit |
| Reproduce a result | [Reproduction report](https://github.com/inlevel9-com/Vons/issues/new?template=reproduction.yml) | Form included; evidence reviewed separately |
| Improve Vons | [Contributing](../CONTRIBUTING.md) | Documentation, tests and scoped code changes welcome |

The destinations are [GitHub: inlevel9-com/Vons](https://github.com/inlevel9-com/Vons)
and [Hugging Face: INLEVEL9/Vons](https://huggingface.co/INLEVEL9/Vons).
The source preview includes documentation and GitHub issue forms for feedback.
The paper and Tech Report will be added after v1.0 review; this release contains
neither manuscript. A source upload does not provide hosted model inference.
The current software terms require a separate agreement for
enterprise/commercial use, including enterprise evaluation. Reading the paper
under its article terms is separate from software use.

## A five-minute first exploration

Five minutes is a usability target, not a measured onboarding result. Start in
the root of a source checkout with Python 3.10+. This path needs no model assets,
GPU, API token or hosted inference account. Python packaging may need internet
access to install its build tools.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m vons.cli generate-smoke --output data/generated/community-smoke.jsonl
python -m vons.cli validate-data --input data/generated/community-smoke.jsonl
```

On Windows PowerShell use `.venv\Scripts\Activate.ps1` for activation. The
generator reports four synthetic rows. Validation prints their data manifest;
this checks the data contract, not model quality.

Run this Python example in the same environment:

```python
from vons import KASIAdapter, KASIProposal

adapter = KASIAdapter(tool_policy={"read_weather": "low"})
for tool in ("read_weather", "unknown_tool"):
    proposal = KASIProposal.from_mapping({
        "calls": [{"name": tool}],
        "confidence": 0.9,
        "risk": "low",
    })
    decision = adapter.decide(proposal)
    print(tool, decision.action.value, decision.reason)
```

Expected output:

```text
read_weather call approved_by_host_policy
unknown_tool refuse unregistered_or_no_proposed_call
```

This is the deterministic KASI adapter, not a learned Vons prediction. The word
`call` is a returned decision; the example never fetches weather or executes a
tool. The host supplies the tool policy. The literal confidence is sample input,
not a calibrated probability. See the [general scope](MODEL_CARD.md) and
[KASI integration](KASI_INTEGRATION.md) before building an integration.

## Share an honest experience

Use a report form with the source revision or release-manifest digest, scenario,
OS/device, runtime/provider, exact command, expected behavior and observed
behavior. Mark whether you tried the contract, a synthetic example or real model
inference. Include attempted, successful, abstained and failed counts with
definitions; explain any overlapping categories. Report unavailable quantities
as unavailable and keep errors in the denominator.

A short negative report is useful. Personal experience and measured benchmarks
serve different purposes: user ratings, stars and downloads do not establish
accuracy, safety or independent replication. Public reports are not automatically
training data or promotional testimonials. Link only safe evidence you have
reviewed and have permission to share. No name, employer or private workflow is
required beyond the hosting service's account requirements.

## A practical platform roadmap

These are proposed acceptance criteria, not completed or measured milestones.

1. **Reading and participation:** publish a reviewed source preview, readable
   research summary and accurate availability notices. A signed-out visitor can
   read the documentation; a signed-in contributor can submit each issue form
   and find the contribution rules. Check keyboard navigation and narrow screens.
2. **First use:** test the no-weight walkthrough from a clean checkout. Invite
   voluntary reports of completion, failure and time spent. Keep counts and
   denominators; do not infer actual use from page views or stars.
3. **Real inference:** offer a versioned demo only after permitted model assets,
   tokenizer, notices, provider evidence and numerical parity are reviewed.
   Clearly label unsupported requests and known failures. Host policy still
   controls actions. Do not present static or simulated results as inference.
4. **Community learning:** triage reports, publish reproducible improvements,
   credit contributions and keep corrections visible. Record submitted reports,
   independently reproduced reports and accepted contributions separately.

Start with the existing GitHub and Hugging Face channels. A dedicated website,
review database, accounts and telemetry need a separate implementation decision;
none is required for the first source preview. Reading should remain available
without a Vons account. Follow the [contribution and moderation rules](../CONTRIBUTING.md).

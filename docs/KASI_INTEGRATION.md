# KASI integration in Vons

KASI is the first Vons domain adapter, not a separate model product. Vons owns
the general candidate judgment contract; the adapter preserves KASI's public
five-action contract for tool-calling hosts:

`call`, `clarify`, `confirm`, `refuse`, `respond`.

The model may propose ordinary calls, clarification, refusal, or a response. It
cannot grant consent. The host wrapper requires a host-owned registry of tool
names and risk levels, validates tools and arguments, applies confidence
thresholds, and derives `confirm` from that policy. The proposal's `risk` is
only advisory: it can never lower the registered host risk, and missing or
unknown risk is fail-closed. The host remains the only execution boundary.

Consent is bound to the complete call, including its arguments. Pass the
`KASICall.fingerprint()` returned for the exact call in `confirmed_calls`; a
tool name in `confirmed_tools` is retained only for compatibility and cannot
authorize a high-risk call.

The Python adapter stores a SHA-256 fingerprint. The TypeScript SDK exposes the
same canonical `{name, arguments}` consent key as a portable string because
browser runtimes do not provide synchronous SHA-256. A host that shares consent
records between the Python and TypeScript adapters must hash the TypeScript key
with SHA-256 before persistence and comparison.

Unregistered tools are refused. A minimal host setup is:

```python
from vons import KASIAdapter, KASICall, KASIProposal

adapter = KASIAdapter(tool_policy={"read_weather": "low", "unlock": "high"})
proposal = KASIProposal(
    calls=(KASICall("read_weather", {"city": "Seoul"}),),
    confidence=0.9,
    risk="low",
)
decision = adapter.decide(proposal)
```

The mapping is intentionally one-way: KASI calls can be represented as Vons
choices, but arbitrary Vons choices do not become tool calls without a host
adapter. Existing KASI datasets, safety policy, and C99 adapter can be imported
behind a manifest when their provenance and licensing are recorded.

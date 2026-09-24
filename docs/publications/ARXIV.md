# arXiv preparation

The paper is being improved toward v1.0 for a future arXiv submission. No
manuscript, submission bundle, identifier, acceptance or public paper is
included in the current source release. Community-platform goals are future
work, not measured adoption or implemented features. arXiv performs content
moderation, not journal peer review; see its
[moderation policy](https://info.arxiv.org/help/moderation/index.html).

## Release target

Prepare the paper and Tech Report as separately reviewed v1.0 research
artifacts. Verify exact final bytes, compile the submission sources in a clean
environment, and inspect the resulting PDF before adding public download links.
A preview PDF does not establish that a TeX source package compiles. Maintain
historical drafts privately and preserve measured values and evidence provenance.

## Research scope

Present the bounded decision contract, engineering evidence and informative
negative results. State the synthetic pilot's label imbalance and baseline,
external-evaluation failures, token budget and browser-verification limitations
alongside the relevant results. A narrower research preprint can describe open
engineering gates without claiming that those gates have passed or that a model
is ready for deployment. Preserve a separate private claim-to-source audit.

The publication language is English. Author metadata is **Kwangseob Ahn**,
**INLEVEL9 / SEJONG UNIV.**, **oswarld@inlevel9.com**. Article preparation uses
[CC BY 4.0](LICENSE.md); software terms are separate. The author remains
responsible for scientific claims, references and model-assistance disclosure.

## Files and checks

Prepare a readable PDF, editable manuscript source, included figures and
bibliography, together with a reproducible build instruction. Verify every
source-bundle entry and record exact hashes. Resolve missing references,
orphan captions, table breaks and unavailable figures; inspect all PDF pages.
Keep private claim ledgers, prompts, review conversations, restricted benchmark
rows and unrelated repository files outside the submission bundle.

arXiv prefers TeX/LaTeX and requires source when the PDF was made from TeX/LaTeX.
Include figure files instead of external figure links. See the official
[submission overview](https://info.arxiv.org/help/submit/index.html).
For a TeX package, include the necessary bibliography source or compatible
generated bibliography and confirm the top-level file builds from the bundle
root. Inspect arXiv's processed PDF as well as the local build. See
[TeX submission guidance](https://info.arxiv.org/help/submit_tex.html).

## Author submission handoff

1. Review the exact final manuscript, source bundle, hashes and remaining limits.
2. Confirm title, abstract, author affiliation, article rights and a subject
   category based on the paper's actual contribution. `cs.AI` and `cs.LG` are
   candidates to assess, not confirmed classifications.
3. Check the account and any applicable [endorsement requirement](https://info.arxiv.org/help/endorsement.html).
4. Prepare the upload and metadata; verify the processed PDF and the selected
   [arXiv license](https://info.arxiv.org/help/license/index.html). CC BY 4.0 is
   an available option consistent with the selected article terms.
5. The author approves the concrete submission and completes the submission
   agreement. Add the real arXiv URL to the publication index only after it exists.

Official guidance was checked on 2026-09-25; recheck it at submission time.
The GitHub destination is [inlevel9-com/Vons](https://github.com/inlevel9-com/Vons)
and the Hugging Face repository is [INLEVEL9/Vons](https://huggingface.co/INLEVEL9/Vons).
Neither link establishes public artifact availability, replication or acceptance.

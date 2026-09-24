"""Audit saved local-model suitability records without making new model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def audit(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    report = json.loads(raw)
    records = report["records"]
    groups: Counter[str] = Counter()
    evidence = []
    correct = 0
    strict_shape = 0
    null_on_abstain = 0
    prompt_set = set()
    for record in records:
        prompt = json.loads(record["prompt"])
        parsed = json.loads(record["response"]) if record["response"] else None
        valid = (isinstance(parsed, dict) and set(parsed) == {"choice", "abstain"}
                 and isinstance(parsed["abstain"], bool)
                 and (parsed["choice"] is None or parsed["choice"] in prompt["candidates"]))
        strict_shape += valid
        if valid:
            correct += ((record["expected_answerable"] and not parsed["abstain"]
                         and parsed["choice"] == record["expected_label"])
                        or (not record["expected_answerable"] and parsed["abstain"]))
            null_on_abstain += not parsed["abstain"] or parsed["choice"] is None
        key = json.dumps({"expected_label": record["expected_label"],
                          "answerable": record["expected_answerable"], "parsed": parsed},
                         sort_keys=True)
        groups[key] += 1
        prompt_hash = sha256(record["prompt"].encode())
        prompt_set.add(prompt_hash)
        evidence.append({"example_id": record["example_id"], "prompt_sha256": prompt_hash,
                         "response_sha256": sha256((record["response"] or "").encode()),
                         "accepted_for_training": False, "reason": "prompt_label_alignment_unresolved"})
    return {"model": report["model"], "role_metadata": report["role"],
            "raw_path": str(path), "raw_sha256": sha256(raw), "rows": len(records),
            "known_label_accuracy": correct / len(records) if records else None,
            "strict_json_shape_and_candidate_rows": strict_shape,
            "null_on_abstention_contract_rows": null_on_abstain,
            "unique_prompt_count": len(prompt_set), "prompt_hashes": sorted(prompt_set),
            "response_groups": [{**json.loads(key), "rows": count} for key, count in sorted(groups.items())],
            "record_hashes": evidence,
            "digest": records[0]["metadata"].get("digest") if records else None,
            "upstream_revision": None,
            "upstream_revision_reason": "not established by saved local metadata"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite historical audit")
    models = [audit(path) for path in args.input]
    identical = all(row["prompt_hashes"] == models[0]["prompt_hashes"] for row in models)
    result = {"schema": "vons.local-model-alignment/v1", "models": models,
              "identical_prompt_sets_across_roles": identical,
              "source_sha256": sha256(Path(__file__).read_bytes()),
              "new_model_calls": 0, "teacher_outputs_used_for_training": False,
              "interpretation": [
                  "Every answerable pilot label is call_tool, while all three models select respond_directly.",
                  "The saved state does not identify a required tool or a rule distinguishing those two actions. This is a prompt/label specification limitation, not a proven model error rate on real workflows.",
                  "Clarify is an available action for underspecified inputs, but the label instead requires abstention; the prompt does not state that mapping.",
                  "Role names are metadata; the benchmark uses identical task prompts and one generic system instruction, not distinct writer/judge/analyst evaluations.",
                  "Muse's 0.09 comes from its 18 abstention booleans. Its nonnull choice on those rows would violate Vons's public abstention contract despite passing the benchmark JSON schema.",
                  "Historical scores are preserved. No general model ranking, swapped-role validation, development-task benchmark or teacher-training admission follows from this audit.",
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"models": len(models), "identical_prompt_sets_across_roles": identical,
                      "rows": sum(row["rows"] for row in models)}))


if __name__ == "__main__":
    main()

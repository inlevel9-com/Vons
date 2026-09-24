"""Auditable Ollama adapter for local data generation and cross-review."""

from __future__ import annotations

import json
import platform
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import Example


@dataclass(frozen=True)
class OllamaModelInfo:
    tag: str
    digest: str | None
    details: Mapping[str, Any]
    capabilities: tuple[str, ...]
    license_present: bool


@dataclass(frozen=True)
class OllamaRecord:
    model: str
    prompt_kind: str
    prompt: str
    response: str | None
    created_at: str
    elapsed_ms: float
    metadata: Mapping[str, Any]
    error: str | None = None


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: float = 120.0) -> None:
        self.host, self.timeout = host.rstrip("/"), timeout

    def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.host + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def version(self) -> str:
        return str(self._request("/api/version")["version"])

    def tags(self) -> list[Mapping[str, Any]]:
        value = self._request("/api/tags")
        models = value.get("models", []) if isinstance(value, Mapping) else []
        return [item for item in models if isinstance(item, Mapping)]

    def show(self, model: str) -> OllamaModelInfo:
        value = self._request("/api/show", {"model": model})
        digest = value.get("digest")
        if not digest:
            digest = next((item.get("digest") for item in self.tags() if item.get("name") == model or item.get("model") == model), None)
        return OllamaModelInfo(model, digest, dict(value.get("details", {})), tuple(value.get("capabilities", ())), bool(value.get("license")))

    def chat_json(self, model: str, *, system: str, prompt: str, schema: Mapping[str, Any], seed: int = 7) -> OllamaRecord:
        started = time.perf_counter()
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        body: Mapping[str, Any] = {}
        try:
            body = self._request("/api/chat", {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "stream": False, "format": schema, "options": {"temperature": 0, "seed": seed}, "think": False})
            response = body.get("message", {}).get("content", "")
            info = self.show(model)
            error = None
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
            response, info, error = None, None, f"{type(exc).__name__}: {exc}"
        return OllamaRecord(model, "chat_json", prompt, response, created_at, (time.perf_counter() - started) * 1000, {"host": self.host, "platform": platform.platform(), "ollama_version": self.version() if error is None else None, "digest": info.digest if info else None, "details": dict(info.details) if info else {}, "capabilities": list(info.capabilities) if info else [], "license_present": info.license_present if info else None, "raw_fields": sorted(body), "raw_response": body}, error)


def record_to_mapping(record: OllamaRecord) -> dict[str, Any]:
    return asdict(record)


def benchmark_ollama(client: OllamaClient, model: str, examples: Sequence[Example], *, role: str = "judge", limit: int = 200, seed: int = 7) -> dict[str, Any]:
    """Measure JSON compliance and judgment on a bounded, known-label set.

    The model's self-reported confidence is retained only as raw output. It is
    never used as a probability or as a substitute for the dataset label.
    """
    schema = {
        "type": "object",
        "properties": {"choice": {"type": ["string", "null"]}, "abstain": {"type": "boolean"}},
        "required": ["choice", "abstain"],
        "additionalProperties": False,
    }
    system = "Return only JSON with a choice from the candidates or null and an abstain boolean. Do not call tools."
    records: list[dict[str, Any]] = []
    for row in list(examples)[:limit]:
        prompt = json.dumps({"state": row.state, "question": row.question, "candidates": list(row.options)}, ensure_ascii=False)
        record = client.chat_json(model, system=system, prompt=prompt, schema=schema, seed=seed)
        item = record_to_mapping(record)
        item.update({"example_id": row.id, "role": role, "expected_label": row.label, "expected_answerable": row.answerable})
        parsed: Mapping[str, Any] | None = None
        valid_json = False
        if record.response:
            try:
                candidate = json.loads(record.response)
                if isinstance(candidate, Mapping) and isinstance(candidate.get("abstain"), bool):
                    choice = candidate.get("choice")
                    valid_json = choice is None or isinstance(choice, str)
                    parsed = candidate if valid_json else None
            except json.JSONDecodeError:
                parsed = None
        item["parsed"] = dict(parsed) if parsed is not None else None
        item["valid_json"] = valid_json
        item["correct"] = bool(valid_json and ((row.answerable and parsed and parsed.get("choice") == row.label and not parsed.get("abstain")) or (not row.answerable and parsed and parsed.get("abstain"))))
        records.append(item)
    valid = [item for item in records if item["valid_json"]]
    correct = [item for item in records if item["correct"]]
    latencies = sorted(float(item["elapsed_ms"]) for item in records)
    return {
        "model": model,
        "role": role,
        "seed": seed,
        "rows": len(records),
        "json_valid_rate": len(valid) / len(records) if records else 0.0,
        "judgment_accuracy": len(correct) / len(records) if records else 0.0,
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else None,
        "records": records,
    }


def write_benchmark(path: str | Path, result: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

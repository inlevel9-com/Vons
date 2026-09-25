"""Command-line entry points for reproducible Vons experiments."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from .bundle import build_bundle_manifest, verify_bundle_manifest
from .calibration import calibrate_report
from .data import dataset_manifest, read_jsonl, smoke_examples, synthetic_examples, write_jsonl
from .ollama import OllamaClient, benchmark_ollama, write_benchmark
from .research import build_research_packet
from .training import evaluate, export_diffusion_bundle, export_direct_bundle, export_head, train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vons")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("generate-smoke"); command.add_argument("--output", required=True)
    command = sub.add_parser("generate-synthetic"); command.add_argument("--count", type=int, default=2000); command.add_argument("--seed", type=int, default=7); command.add_argument("--output", required=True)
    command = sub.add_parser("validate-data"); command.add_argument("--input", required=True); command.add_argument("--manifest")
    command = sub.add_parser("split-data"); command.add_argument("--input", required=True); command.add_argument("--split", choices=["train", "development", "calibration", "test", "smoke"], required=True); command.add_argument("--output", required=True)
    command = sub.add_parser("ollama-check"); command.add_argument("--models", nargs="+", required=True); command.add_argument("--host", default="http://127.0.0.1:11434"); command.add_argument("--output")
    command = sub.add_parser("ollama-benchmark"); command.add_argument("--model", required=True); command.add_argument("--input", required=True); command.add_argument("--role", default="judge"); command.add_argument("--limit", type=int, default=200); command.add_argument("--seed", type=int, default=7); command.add_argument("--host", default="http://127.0.0.1:11434"); command.add_argument("--output", required=True)
    command = sub.add_parser("build-research-packet"); command.add_argument("--results", required=True); command.add_argument("--output", required=True)
    command = sub.add_parser("build-bundle-manifest"); command.add_argument("--onnx", required=True); command.add_argument("--tokenizer", required=True); command.add_argument("--output", required=True); command.add_argument("--checkpoint"); command.add_argument("--model-manifest"); command.add_argument("--calibration", action="append", default=[]); command.add_argument("--runtime-asset", action="append", default=[]); command.add_argument("--license", action="append", default=[]); command.add_argument("--config", action="append", default=[]); command.add_argument("--source-root"); command.add_argument("--source-file", action="append", default=[])
    command = sub.add_parser("verify-bundle-manifest"); command.add_argument("--manifest", required=True)
    command = sub.add_parser("calibrate"); command.add_argument("--calibration-input", required=True); command.add_argument("--calibration-report", required=True); command.add_argument("--test-input", required=True); command.add_argument("--test-report", required=True); command.add_argument("--output", required=True); command.add_argument("--target-brier", type=float); command.add_argument("--target-risk", type=float)
    for name in ("train", "evaluate", "export", "export-bundle", "export-diffusion-bundle"):
        command = sub.add_parser(name); command.add_argument("--config"); command.add_argument("--model"); command.add_argument("--input"); command.add_argument("--checkpoint"); command.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate-smoke":
        write_jsonl(args.output, smoke_examples()); print(json.dumps({"output": args.output, "rows": 4}, indent=2)); return 0
    if args.command == "generate-synthetic":
        rows = synthetic_examples(args.count, seed=args.seed); write_jsonl(args.output, rows); print(json.dumps({"output": args.output, "rows": len(rows), "seed": args.seed}, indent=2)); return 0
    if args.command == "validate-data":
        rows = read_jsonl(args.input); manifest = dataset_manifest(args.input, rows)
        if args.manifest:
            Path(args.manifest).parent.mkdir(parents=True, exist_ok=True); Path(args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True)); return 0
    if args.command == "split-data":
        rows = [row for row in read_jsonl(args.input) if row.split == args.split]
        if not rows:
            raise SystemExit(f"input has no {args.split!r} rows")
        write_jsonl(args.output, rows)
        print(json.dumps({"output": args.output, "rows": len(rows), "split": args.split}, indent=2)); return 0
    if args.command == "ollama-check":
        client, payload = OllamaClient(args.host), {"host": args.host, "platform": platform.platform(), "models": []}
        for model in args.models:
            try:
                info = client.show(model); payload["models"].append({"tag": info.tag, "digest": info.digest, "details": info.details, "capabilities": info.capabilities, "license_present": info.license_present})
            except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
                payload["models"].append({"tag": model, "error": f"{type(exc).__name__}: {exc}"})
        serialized = json.dumps(payload, indent=2, sort_keys=True, default=list)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(serialized + "\n", encoding="utf-8")
        print(serialized); return 0
    if args.command == "ollama-benchmark":
        rows = read_jsonl(args.input)
        result = benchmark_ollama(OllamaClient(args.host), args.model, rows, role=args.role, limit=args.limit, seed=args.seed)
        write_benchmark(args.output, result)
        print(json.dumps({key: result[key] for key in ("model", "role", "rows", "json_valid_rate", "judgment_accuracy", "latency_p50_ms")}, indent=2))
        return 0
    if args.command == "build-research-packet":
        output = build_research_packet(results_dir=args.results, output=args.output); print(json.dumps({"output": str(output)}, indent=2)); return 0
    if args.command == "build-bundle-manifest":
        output = build_bundle_manifest(
            args.onnx, args.tokenizer, args.output,
            checkpoint_path=args.checkpoint,
            model_manifest_path=args.model_manifest,
            calibration_paths=args.calibration,
            runtime_asset_paths=args.runtime_asset,
            license_paths=args.license,
            config_paths=args.config,
            source_root=args.source_root,
            source_paths=args.source_file,
        )
        print(json.dumps({"output": str(output), "verification": verify_bundle_manifest(output)}, indent=2)); return 0
    if args.command == "verify-bundle-manifest":
        print(json.dumps(verify_bundle_manifest(args.manifest), indent=2)); return 0
    if args.command == "calibrate":
        output = calibrate_report(
            calibration_input=args.calibration_input,
            calibration_report=args.calibration_report,
            test_input=args.test_input,
            test_report=args.test_report,
            output=args.output,
            target_risk=args.target_risk,
            target_brier=args.target_brier,
        )
        print(json.dumps({"report": str(output)}, indent=2)); return 0
    if args.command == "train":
        if not args.config:
            raise SystemExit("train requires --config")
        print(json.dumps({"checkpoint": str(train(args.config))}, indent=2)); return 0
    if args.command == "evaluate":
        if not args.checkpoint or not args.input or not args.output:
            raise SystemExit("evaluate requires --checkpoint, --input, and --output")
        print(json.dumps({"report": str(evaluate(args.checkpoint, args.input, args.output))}, indent=2)); return 0
    if args.command == "export":
        if not args.checkpoint or not args.output:
            raise SystemExit("export requires --checkpoint and --output")
        print(json.dumps({"onnx": str(export_head(args.checkpoint, args.output))}, indent=2)); return 0
    if args.command == "export-bundle":
        if not args.checkpoint or not args.output:
            raise SystemExit("export-bundle requires --checkpoint and --output")
        print(json.dumps({"onnx": str(export_direct_bundle(args.checkpoint, args.output))}, indent=2)); return 0
    if args.command == "export-diffusion-bundle":
        if not args.checkpoint or not args.output:
            raise SystemExit("export-diffusion-bundle requires --checkpoint and --output")
        print(json.dumps({"onnx": str(export_diffusion_bundle(args.checkpoint, args.output))}, indent=2)); return 0
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

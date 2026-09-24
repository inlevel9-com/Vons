"""Archive exact manifest bytes for a browser measurement provenance record."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def archive(source: Path, output: Path, expected_sha256: str) -> dict[str, object]:
    data = source.read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"source manifest hash {actual_sha256} != expected {expected_sha256}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    if output.read_bytes() != data:
        raise OSError(f"archived manifest bytes do not match source: {output}")
    return {"source": str(source), "output": str(output), "bytes": len(data), "sha256": actual_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    result = archive(args.source, args.output, args.expected_sha256)
    print(result)


if __name__ == "__main__":
    main()

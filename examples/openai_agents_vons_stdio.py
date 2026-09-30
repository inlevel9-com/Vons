"""Run an OpenAI Agents SDK planner with the local Vons stdio MCP server."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from agents import Agent, RunConfig, Runner
from agents.mcp import MCPServerStdio, create_static_tool_filter

INSTRUCTIONS = """Use vons_decide only for a bounded choice among explicit options.
Treat its response as a local model proposal with uncalibrated scores. Report
abstentions accurately. Never execute an action, claim consent, or treat the
decision as permission to act."""


def _validate_node_runtime(node: Path, child_env: dict[str, str]) -> None:
    if not os.access(node, os.X_OK):
        raise ValueError("VONS_NODE must name an executable Node.js binary.")

    try:
        result = subprocess.run(
            [str(node), "--version"],
            check=True,
            capture_output=True,
            env=child_env,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            "Unable to verify the Node.js version; install Node.js 22.6+ or "
            "set VONS_NODE to its executable."
        ) from error

    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", result.stdout.strip())
    if not match:
        raise ValueError("VONS_NODE must report a stable semantic version with --version.")

    version = tuple(int(part) for part in match.groups())
    if version < (22, 6, 0):
        reported = ".".join(str(part) for part in version)
        raise ValueError(f"Node.js 22.6+ is required; VONS_NODE reports {reported}.")


def _local_server_settings() -> tuple[str, list[str], dict[str, str]]:
    repository = (
        Path(os.environ.get("VONS_REPO", Path(__file__).resolve().parents[1]))
        .expanduser()
        .resolve(strict=True)
    )
    bundle_value = os.environ.get("VONS_BUNDLE")
    if not bundle_value:
        raise ValueError("Set VONS_BUNDLE to a reviewed local Vons model bundle.")
    bundle = Path(bundle_value).expanduser().resolve(strict=True)
    if not bundle.is_dir():
        raise ValueError("VONS_BUNDLE must name a local directory.")

    node_value = os.environ.get("VONS_NODE") or shutil.which("node")
    if not node_value:
        raise ValueError("Node.js 22.6+ must be on PATH or set in VONS_NODE.")
    node = Path(node_value).expanduser().resolve(strict=True)
    if not node.is_file():
        raise ValueError("VONS_NODE must name a Node.js executable.")
    # The version probe runs the configured executable too; keep parent secrets
    # out of that subprocess as well as the later MCP server process.
    inherited_keys = ("HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "WINDIR")
    child_env = {key: os.environ[key] for key in inherited_keys if key in os.environ}
    _validate_node_runtime(node, child_env)

    server_script = repository / "sdk/typescript/src/mcp-cli.ts"
    if not server_script.is_file():
        raise ValueError("VONS_REPO must point to the Vons source checkout.")

    args = [
        "--experimental-strip-types",
        str(server_script),
        "--bundle",
        str(bundle),
        "--provider",
        "wasm",
    ]
    manifest_hash = os.environ.get("VONS_MANIFEST_SHA256")
    if manifest_hash:
        args.extend(["--manifest-sha256", manifest_hash])

    # Keep the OpenAI API key in the Python parent; the local MCP child does
    # not need it. Node is addressed by absolute path, so no PATH is required.
    return str(node), args, child_env


async def _run(prompt: str) -> str:
    command, args, child_env = _local_server_settings()
    server = MCPServerStdio(
        name="Vons local decision",
        params={"command": command, "args": args, "env": child_env},
        cache_tools_list=True,
        tool_filter=create_static_tool_filter(allowed_tool_names=["vons_decide"]),
        # vons_decide returns a proposal only; it has no side effects or action authority.
        require_approval="never",
    )
    async with server:
        agent = Agent(
            name="Vons-assisted agent",
            instructions=INSTRUCTIONS,
            mcp_servers=[server],
            model=os.environ.get("OPENAI_MODEL") or None,
        )
        result = await Runner.run(
            agent,
            prompt,
            run_config=RunConfig(
                tracing_disabled=True,
                trace_include_sensitive_data=False,
            ),
        )
    if not isinstance(result.final_output, str):
        raise TypeError("The agent returned non-text output; expected a string.")
    return result.final_output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an OpenAI Agents SDK agent with local Vons decision inference."
    )
    parser.add_argument("prompt", nargs="+", help="Task prompt for the OpenAI agent.")
    arguments = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in the environment before running.")
    try:
        output = asyncio.run(_run(" ".join(arguments.prompt)))
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

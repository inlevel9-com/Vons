from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "examples/openai_agents_vons_stdio.py"


def _load_example(monkeypatch: pytest.MonkeyPatch):
    agents = types.ModuleType("agents")
    agents.__path__ = []  # type: ignore[attr-defined]
    agents.Agent = object  # type: ignore[attr-defined]
    agents.RunConfig = object  # type: ignore[attr-defined]
    agents.Runner = object  # type: ignore[attr-defined]

    mcp = types.ModuleType("agents.mcp")
    mcp.MCPServerStdio = object  # type: ignore[attr-defined]
    mcp.create_static_tool_filter = lambda **_: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agents", agents)
    monkeypatch.setitem(sys.modules, "agents.mcp", mcp)

    spec = importlib.util.spec_from_file_location("openai_agents_vons_stdio_test_subject", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _configure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module):
    repository = tmp_path / "vons"
    server_script = repository / "sdk/typescript/src/mcp-cli.ts"
    server_script.parent.mkdir(parents=True)
    server_script.touch()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    node = tmp_path / "node"
    node.touch()
    node.chmod(0o755)

    monkeypatch.setenv("VONS_REPO", str(repository))
    monkeypatch.setenv("VONS_BUNDLE", str(bundle))
    monkeypatch.setenv("VONS_NODE", str(node))
    monkeypatch.setenv("OPENAI_API_KEY", "parent-only-test-key")
    monkeypatch.setenv("VONS_TEST_PARENT_SECRET", "parent-only-test-secret")
    return node, bundle, server_script


@pytest.mark.parametrize("version", ["v22.6.0", "v22.6.1", "v23.0.0"])
def test_node_minimum_and_newer_versions_are_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: str
) -> None:
    module = _load_example(monkeypatch)
    node, bundle, server_script = _configure_paths(monkeypatch, tmp_path, module)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=version, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    command, args, child_env = module._local_server_settings()

    assert command == str(node)
    assert args[1] == str(server_script)
    assert args[3] == str(bundle)
    assert child_env.get("OPENAI_API_KEY") is None
    assert child_env.get("VONS_TEST_PARENT_SECRET") is None
    assert calls == [
        (
            [str(node), "--version"],
            {
                "check": True,
                "capture_output": True,
                "env": child_env,
                "text": True,
                "timeout": 5,
            },
        )
    ]


@pytest.mark.parametrize("version", ["v22.5.9", "v21.99.99"])
def test_older_node_versions_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: str
) -> None:
    module = _load_example(monkeypatch)
    _configure_paths(monkeypatch, tmp_path, module)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, stdout=version, stderr=""),
    )

    with pytest.raises(ValueError, match=r"Node.js 22\.6\+ is required"):
        module._local_server_settings()


@pytest.mark.parametrize("version", ["", "node v22.6.0", "v22.6.0-pre"])
def test_malformed_or_prerelease_node_versions_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: str
) -> None:
    module = _load_example(monkeypatch)
    _configure_paths(monkeypatch, tmp_path, module)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, stdout=version, stderr=""),
    )

    with pytest.raises(ValueError, match="stable semantic version"):
        module._local_server_settings()


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "node --version", stderr="private diagnostic"),
        subprocess.TimeoutExpired("node --version", timeout=5, stderr="private diagnostic"),
        OSError("private diagnostic"),
    ],
)
def test_node_version_command_failures_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: Exception,
) -> None:
    module = _load_example(monkeypatch)
    _configure_paths(monkeypatch, tmp_path, module)

    def fail_run(*_: object, **__: object) -> None:
        raise failure

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    with pytest.raises(ValueError, match="Unable to verify the Node.js version") as error:
        module._local_server_settings()
    assert "private diagnostic" not in str(error.value)


def test_non_executable_node_is_rejected_before_running_version_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_example(monkeypatch)
    node, _, _ = _configure_paths(monkeypatch, tmp_path, module)
    original_access: Callable[[str | Path, int], bool] = module.os.access
    monkeypatch.setattr(
        module.os,
        "access",
        lambda path, mode: False if Path(path) == node else original_access(path, mode),
    )
    run_calls: list[object] = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: run_calls.append(args))

    with pytest.raises(ValueError, match="must name an executable Node.js binary"):
        module._local_server_settings()
    assert run_calls == []


def test_non_text_agent_output_is_reported_without_traceback_or_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_example(monkeypatch)
    server_calls: list[dict[str, object]] = []

    class StubServer:
        def __init__(self, **kwargs: object) -> None:
            server_calls.append(kwargs)

        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *_: object) -> bool:
            return False

    class StubRunConfig:
        def __init__(self, **_: object) -> None:
            pass

    class StubRunner:
        @staticmethod
        async def run(*_: object, **__: object) -> types.SimpleNamespace:
            return types.SimpleNamespace(final_output={"private": "agent result"})

    child_env = {"HOME": "/synthetic/home", "TMPDIR": "/synthetic/tmp"}
    monkeypatch.setattr(module, "_local_server_settings", lambda: ("node", [], child_env))
    monkeypatch.setattr(module, "MCPServerStdio", StubServer)
    monkeypatch.setattr(module, "create_static_tool_filter", lambda **_: None)
    monkeypatch.setattr(module, "Agent", lambda **_: object())
    monkeypatch.setattr(module, "RunConfig", StubRunConfig)
    monkeypatch.setattr(module, "Runner", StubRunner)
    monkeypatch.setenv("OPENAI_API_KEY", "parent-only-test-key")
    monkeypatch.setattr(sys, "argv", ["openai_agents_vons_stdio.py", "synthetic prompt"])

    with pytest.raises(SystemExit) as error:
        module.main()

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "The agent returned non-text output; expected a string." in captured.err
    assert "Traceback" not in captured.err
    assert "agent result" not in captured.err
    assert len(server_calls) == 1
    params = server_calls[0]["params"]
    assert isinstance(params, dict)
    assert params["env"] is child_env
    assert "OPENAI_API_KEY" not in params["env"]
    assert "VONS_TEST_PARENT_SECRET" not in params["env"]

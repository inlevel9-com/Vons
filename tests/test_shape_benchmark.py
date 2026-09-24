from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from vons.data import Example

_SPEC = importlib.util.spec_from_file_location("benchmark_onnx_shapes", Path(__file__).parents[1] / "tools/benchmark_onnx_shapes.py")
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_options_for = _MODULE._options_for
shape_cells = _MODULE.shape_cells


def _row(options: tuple[str, ...]) -> Example:
    return Example(
        id="fixture",
        task_group="shape",
        state="ready",
        question="Choose",
        options=options,
        label=options[0],
        answerable=True,
        split="smoke",
        provenance={},
        metadata={},
    )


def test_shape_cells_keep_the_four_padding_comparisons_explicit() -> None:
    cells = shape_cells(live_candidates=4, short_sequence=64, full_sequence=512)
    assert [cell.cell_id for cell in cells] == [
        "direct_live_short",
        "direct_padded_short",
        "direct_padded_full",
        "diffusion_padded_full",
    ]
    assert [(cell.backend, cell.slots, cell.sequence_length) for cell in cells] == [
        ("direct", 4, 64),
        ("direct", 32, 64),
        ("direct", 32, 512),
        ("diffusion", 32, 512),
    ]


def test_shape_fixture_expands_candidates_without_changing_live_prefix() -> None:
    options = _options_for(_row(("a", "b")), 4)
    assert options[:2] == ("a", "b")
    assert len(options) == 4
    assert len(set(options)) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"live_candidates": 1, "short_sequence": 64, "full_sequence": 512},
        {"live_candidates": 33, "short_sequence": 64, "full_sequence": 512},
        {"live_candidates": 4, "short_sequence": 512, "full_sequence": 64},
    ],
)
def test_shape_cells_reject_invalid_dimensions(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        shape_cells(**kwargs)

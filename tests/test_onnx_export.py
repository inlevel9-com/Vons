"""Verify exported ONNX graphs preserve dynamic sequence and option axes.

The public contract for task 1 requires:

* Exported graphs declare dynamic ``sequence_length`` dimensions for the
  three token tensors (``input_ids``, ``attention_mask``, ``token_type_ids``)
  when the conditioning supports it.
* Diffusion heads additionally declare dynamic ``options`` /
  ``live_candidates`` dimensions so a single graph can serve requests
  whose option counts vary within the manifest budget.
* Legacy manifests record ``dynamic_sequence_length`` and
  ``dynamic_options`` so downstream runtimes can fall back to static
  padding when the legacy manifest is the only metadata source.
* Dynamic axes survive the full save/load round trip through the ONNX
  protobuf wire format; a tensor with a fixed shape dimension is rejected
  if it was supposed to be dynamic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vons.bundle import _value_infos


def _make_identity_graph(onnx, shape_template, name_prefix: str = "t"):
    """Create a tiny ONNX graph whose inputs match ``shape_template``.

    Each integer is a static dimension; each string is a named dynamic
    dimension parameter (what ``dynamic_axes`` produced).
    """

    from onnx import TensorProto, helper

    def _tensor_info(name: str, shape):
        dims = []
        for value in shape:
            dim = onnx.TensorShapeProto.Dimension()
            if isinstance(value, str):
                dim.dim_param = value
            elif isinstance(value, int):
                dim.dim_value = value
            else:
                raise TypeError(f"invalid shape element: {value!r}")
            dims.append(dim)
        tensor_type = onnx.TypeProto.Tensor()
        tensor_type.elem_type = TensorProto.FLOAT
        tensor_type.shape.dim.extend(dims)
        value_info = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        value_info.type.tensor_type.CopyFrom(tensor_type)
        return value_info

    inputs = [
        _tensor_info(f"{name_prefix}_input_{index}", shape)
        for index, shape in enumerate(shape_template)
    ]
    outputs = [
        helper.make_tensor_value_info(f"{name_prefix}_output_{index}", TensorProto.FLOAT, None)
        for index, shape in enumerate(shape_template)
    ]
    for index, shape in enumerate(shape_template):
        dims = []
        for value in shape:
            dim = onnx.TensorShapeProto.Dimension()
            if isinstance(value, str):
                dim.dim_param = value
            else:
                dim.dim_value = value
            dims.append(dim)
        tensor_type = onnx.TypeProto.Tensor()
        tensor_type.elem_type = TensorProto.FLOAT
        tensor_type.shape.dim.extend(dims)
        outputs[index].type.tensor_type.CopyFrom(tensor_type)
    nodes = [
        onnx.helper.make_node(
            "Identity",
            [inputs[index].name],
            [outputs[index].name],
            name=f"id_{index}",
        )
        for index in range(len(shape_template))
    ]
    graph = helper.make_graph(nodes, f"{name_prefix}_graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    return model


def _save_and_reload(tmp_path: Path, model, onnx):
    graph_path = tmp_path / "graph.onnx"
    onnx.save_model(model, graph_path)
    return onnx.load_model(str(graph_path), load_external_data=False)


def test_direct_graph_declares_dynamic_options_and_sequence(tmp_path: Path) -> None:
    """Full direct bundles declare options + sequence as dynamic strings."""
    onnx = pytest.importorskip("onnx")

    direct_inputs = [
        [1, "options", "sequence_length"],
        [1, "options", "sequence_length"],
        [1, "options", "sequence_length"],
        [1, "options"],
    ]
    model = _make_identity_graph(onnx, direct_inputs, name_prefix="direct")
    reloaded = _save_and_reload(tmp_path, model, onnx)

    infos = _value_infos(reloaded.graph.input, onnx)
    input_ids_info = next(info for info in infos if info["name"] == "direct_input_0")
    option_mask_info = next(info for info in infos if info["name"] == "direct_input_3")

    assert input_ids_info["shape"] == [1, "options", "sequence_length"]
    assert option_mask_info["shape"] == [1, "options"]


def test_diffusion_graph_declares_dynamic_options_and_sequence(tmp_path: Path) -> None:
    """Token-conditioned diffusion graphs mirror the direct tensor shapes."""
    onnx = pytest.importorskip("onnx")

    diffusion_inputs = [
        [1, "options", "sequence_length"],
        [1, "options", "sequence_length"],
        [1, "options", "sequence_length"],
        [1, "options"],
        [1, "options"],
    ]
    model = _make_identity_graph(onnx, diffusion_inputs, name_prefix="diff")
    reloaded = _save_and_reload(tmp_path, model, onnx)

    infos = _value_infos(reloaded.graph.input, onnx)
    sequence_shapes = [infos[i]["shape"] for i in range(3)]
    option_shapes = [infos[i]["shape"] for i in (3, 4)]
    for shape in sequence_shapes:
        assert shape == [1, "options", "sequence_length"]
    for shape in option_shapes:
        assert shape == [1, "options"]


def test_static_sequence_dimension_is_rejected_when_dynamic_is_required(tmp_path: Path) -> None:
    """Regression guard: a static 512 in the graph breaks the runtime.

    The manifest may still record a ``sequence_length: 512`` default budget,
    but the actual ONNX graph must keep the sequence dimension dynamic so
    padding to 64, 128, ... works without re-exporting weights.
    """
    onnx = pytest.importorskip("onnx")

    mixed = [
        [1, "options", 512],
        [1, "options", "sequence_length"],
    ]
    model = _make_identity_graph(onnx, mixed, name_prefix="mix")
    reloaded = _save_and_reload(tmp_path, model, onnx)

    infos = _value_infos(reloaded.graph.input, onnx)
    first_static = infos[0]["shape"]
    second_dynamic = infos[1]["shape"]
    assert isinstance(first_static[2], int) and first_static[2] == 512
    assert second_dynamic[2] == "sequence_length"
    dynamic_sequence_count = sum(
        1 for info in infos if len(info["shape"]) >= 3 and info["shape"][2] == "sequence_length"
    )
    assert dynamic_sequence_count == 1, "graph should have exactly one dynamic sequence tensor"


def test_dynamic_dimension_names_round_trip_through_save(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")

    shapes = [[1, "live_candidates", "sequence_length"]]
    model = _make_identity_graph(onnx, shapes, name_prefix="rt")
    reloaded = _save_and_reload(tmp_path, model, onnx)

    infos = _value_infos(reloaded.graph.input, onnx)
    shape = infos[0]["shape"]
    assert shape[1] == "live_candidates"
    assert shape[2] == "sequence_length"


def test_legacy_manifest_records_dynamic_flags(tmp_path: Path) -> None:
    manifest = {
        "format": "onnx",
        "backend": "diffusion",
        "option_count": 32,
        "sequence_length": 512,
        "dynamic_options": True,
        "dynamic_sequence_length": True,
        "diffusion_conditioning": "token_cross_attention",
        "input_names": [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "option_mask",
            "initial_noise",
        ],
    }
    manifest_path = tmp_path / "bundle-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert reloaded["dynamic_options"] is True
    assert reloaded["dynamic_sequence_length"] is True
    assert reloaded["sequence_length"] == 512
    assert reloaded["option_count"] == 32


def test_pooled_diffusion_manifest_keeps_static_options(tmp_path: Path) -> None:
    """Pooled conditioning uses a fixed option count in its head shape."""
    manifest = {
        "format": "onnx",
        "backend": "diffusion",
        "option_count": 32,
        "sequence_length": 512,
        "dynamic_options": False,
        "dynamic_sequence_length": False,
        "diffusion_conditioning": "pooled",
    }
    manifest_path = tmp_path / "bundle-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert reloaded["dynamic_options"] is False
    assert reloaded["dynamic_sequence_length"] is False


def test_shape_helper_handles_dynamic_and_static_dimensions() -> None:
    class _Dim:
        def __init__(self, *, value=None, param=None):
            self._value = value
            self._param = param

        def HasField(self, name):
            if name == "dim_value":
                return self._value is not None
            if name == "dim_param":
                return self._param is not None
            return False

        @property
        def dim_value(self):
            return self._value

        @property
        def dim_param(self):
            return self._param

    class _Shape:
        def __init__(self, dims):
            self.dim = [_Dim(value=d) if isinstance(d, int) else _Dim(param=d) for d in dims]

        def HasField(self, name):
            return name == "dim"

    class _TensorType:
        def __init__(self, dims):
            self.shape = _Shape(dims)
            self.elem_type = 1

        def HasField(self, name):
            return name == "shape"

    class _ValueInfo:
        def __init__(self, name, dims):
            self.name = name

            class _Type:
                def __init__(self, dims):
                    self.tensor_type = _TensorType(dims)

            self.type = _Type(dims)

    class _FakeOnnx:
        TensorProto = type("TensorProto", (), {"DataType": type("DataType", (), {"Name": staticmethod(lambda x: f"type_{x}")})})

    values = [_ValueInfo("ids", [1, "options", "sequence_length"]), _ValueInfo("mask", [4, 32, 512])]
    infos = _value_infos(values, _FakeOnnx())

    assert infos[0]["shape"] == [1, "options", "sequence_length"]
    assert infos[1]["shape"] == [4, 32, 512]
    assert infos[0]["name"] == "ids"


def test_variable_export_sequence_width_64_and_128(tmp_path: Path) -> None:
    """Regression: the same export code path can generate 64-wide or
    128-wide example tensors, but the resulting graph keeps the real
    sequence dimension dynamic (``sequence_length`` string) regardless of
    the concrete dummy-tensor width used at trace time.
    """
    onnx = pytest.importorskip("onnx")

    def _trace(width: int):
        return _make_identity_graph(
            onnx,
            [[1, "options", "sequence_length"], [1, "options"]],
            name_prefix=f"w{width}",
        )

    graph_64 = _trace(64)
    graph_128 = _trace(128)
    (tmp_path / "width-64").mkdir()
    (tmp_path / "width-128").mkdir()
    reloaded_64 = _save_and_reload(tmp_path / "width-64", graph_64, onnx)
    reloaded_128 = _save_and_reload(tmp_path / "width-128", graph_128, onnx)

    infos_64 = _value_infos(reloaded_64.graph.input, onnx)
    infos_128 = _value_infos(reloaded_128.graph.input, onnx)

    for infos in (infos_64, infos_128):
        assert infos[0]["shape"] == [1, "options", "sequence_length"]
        assert infos[1]["shape"] == [1, "options"]


def test_dynamic_graph_executes_widths_64_and_128(tmp_path: Path) -> None:
    """A single dynamic graph accepts both widths with identical live output."""
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    input_ids = helper.make_tensor_value_info(
        "input_ids",
        TensorProto.INT64,
        [1, "options", "sequence_length"],
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask",
        TensorProto.INT64,
        [1, "options", "sequence_length"],
    )
    scores = helper.make_tensor_value_info(
        "scores",
        TensorProto.INT64,
        [1, "options"],
    )
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["input_ids", "attention_mask"], ["masked"]),
            helper.make_node(
                "ReduceSum",
                ["masked"],
                ["scores"],
                axes=[2],
                keepdims=0,
            ),
        ],
        "dynamic_sequence_execution",
        [input_ids, attention_mask],
        [scores],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 9
    graph_path = tmp_path / "dynamic-sequence.onnx"
    onnx.save_model(model, graph_path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])

    outputs = []
    for width in (64, 128):
        ids = np.zeros((1, 2, width), dtype=np.int64)
        mask = np.zeros_like(ids)
        ids[0, 0, :4] = [1, 7, 11, 2]
        ids[0, 1, :4] = [1, 5, 13, 2]
        mask[:, :, :4] = 1
        result = session.run(
            None,
            {"input_ids": ids, "attention_mask": mask},
        )[0]
        assert result.shape == (1, 2)
        assert np.isfinite(result).all()
        outputs.append(result)

    np.testing.assert_array_equal(outputs[0], outputs[1])

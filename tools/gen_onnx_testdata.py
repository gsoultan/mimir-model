#!/usr/bin/env python3
"""Generate the ONNX fixture in testdata/onnx.

Run from the repository root with `python3 -P tools/gen_onnx_testdata.py`.
Needs NumPy, `onnx` and `onnxruntime`.

tiny-cnn.onnx is a small image classifier with random weights that uses
the operators the importer supports in a convolutional network: Conv
(padded, strided, grouped with auto_pad, dilated, and 1x1),
BatchNormalization, Relu, Clip, MaxPool, AveragePool (with and without the
padding in the count), GlobalAveragePool, Sigmoid, Tanh, Concat, Add, Sub,
Mul, Div, Flatten, Identity, Dropout, Gemm (both weight layouts, with alpha
and beta), MatMul, Reshape, Transpose, Constant and Softmax. The batch size
is the symbolic dimension N.

expected.json holds two input batches (N = 1 and N = 3) and onnxruntime's
outputs for them, as base64 of little-endian float32. The Go test imports
the model, checks that the result is byte for byte tiny-cnn.mim, and runs
it; the Python test runs tiny-cnn.mim. Both compare with onnxruntime.
"""

import base64
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT = os.path.join("testdata", "onnx")
rng = np.random.default_rng(7)


def arr(name, a):
    return numpy_helper.from_array(np.asarray(a), name)


def w(name, *shape, scale=0.3):
    return arr(name, (rng.standard_normal(shape) * scale).astype(np.float32))


def build():
    inits = [
        w("conv1.weight", 8, 3, 3, 3), w("conv1.bias", 8, scale=0.1),
        arr("bn1.scale", (1 + 0.2 * rng.standard_normal(8)).astype(np.float32)),
        w("bn1.bias", 8, scale=0.1), w("bn1.mean", 8, scale=0.1),
        arr("bn1.var", (0.5 + rng.random(8)).astype(np.float32)),
        w("conv2.weight", 8, 2, 3, 3), w("conv2.bias", 8, scale=0.1),
        w("skip.weight", 8, 8, 1, 1),
        w("channel_scale", 1, 8, 1, 1, scale=1.0),
        arr("clip.min", np.float32(0)), arr("clip.max", np.float32(6)),
        w("conv3.weight", 4, 16, 3, 3, scale=0.2),
        w("fc.weight", 10, 20), w("fc.bias", 10, scale=0.1),
        w("proj", 10, 10),
        w("head.weight", 10, 10), w("head.bias", 10, scale=0.1),
        arr("shape.split", np.array([-1, 2, 5], np.int64)),
        arr("shape.flat", np.array([0, -1], np.int64)),
    ]
    n = helper.make_node
    half = helper.make_tensor("half", TensorProto.FLOAT, [], [2.0])  # float_data, not raw
    nodes = [
        n("Conv", ["x", "conv1.weight", "conv1.bias"], ["c1"], name="conv1", pads=[1, 1, 1, 1]),
        n("BatchNormalization", ["c1", "bn1.scale", "bn1.bias", "bn1.mean", "bn1.var"], ["b1"], name="bn1", epsilon=1e-3),
        n("Relu", ["b1"], ["r1"], name="relu1"),
        n("MaxPool", ["r1"], ["p1"], name="pool1", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]),
        n("Conv", ["p1", "conv2.weight", "conv2.bias"], ["c2"], name="conv2", group=4, strides=[2, 2],
          auto_pad="SAME_UPPER", kernel_shape=[3, 3]),
        n("Conv", ["p1", "skip.weight"], ["sk"], name="skip", strides=[2, 2]),
        n("Add", ["c2", "sk"], ["res"], name="residual"),
        n("Mul", ["res", "channel_scale"], ["scaled"], name="scale"),
        n("Clip", ["scaled", "clip.min", "clip.max"], ["r6"], name="relu6"),
        n("Sigmoid", ["r6"], ["sg"], name="sigmoid"),
        n("Sub", ["r6", "sg"], ["d"], name="sub"),
        n("Concat", ["r6", "d"], ["cat"], name="concat", axis=1),
        n("Conv", ["cat", "conv3.weight"], ["c3"], name="conv3", dilations=[2, 2], pads=[2, 2, 2, 2]),
        n("AveragePool", ["c3"], ["a1"], name="avgpool1", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        n("AveragePool", ["a1"], ["a2"], name="avgpool2", kernel_shape=[2, 2], strides=[2, 2],
          pads=[0, 0, 1, 1], count_include_pad=1),
        n("Tanh", ["c3"], ["t3"], name="tanh"),
        n("GlobalAveragePool", ["t3"], ["g"], name="gap"),
        n("Flatten", ["a2"], ["f1"], name="flatten1"),
        n("Flatten", ["g"], ["f2"], name="flatten2"),
        n("Concat", ["f1", "f2"], ["feat"], name="features", axis=-1),
        n("Identity", ["feat"], ["feat2"], name="identity"),
        n("Gemm", ["feat2", "fc.weight", "fc.bias"], ["h1"], name="fc", transB=1),
        n("Dropout", ["h1"], ["h2"], name="dropout"),
        n("MatMul", ["h2", "proj"], ["h3"], name="proj"),
        n("Constant", [], ["half"], name="half", value=half),
        n("Div", ["h3", "half"], ["h4"], name="div"),
        n("Reshape", ["h4", "shape.split"], ["h5"], name="split"),
        n("Transpose", ["h5"], ["h6"], name="transpose", perm=[0, 2, 1]),
        n("Reshape", ["h6", "shape.flat"], ["h7"], name="flat"),
        n("Gemm", ["h7", "head.weight", "head.bias"], ["logits"], name="head", alpha=0.5, beta=2.0),
        n("Softmax", ["logits"], ["probs"], name="softmax", axis=-1),
    ]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 16, 16])
    y = helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["N", 10])
    g = helper.make_graph(nodes, "tiny-cnn", [x], [y], inits)
    m = helper.make_model(g, producer_name="mim-testdata", opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m, full_check=True)
    return m


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def main():
    m = build()
    os.makedirs(OUT, exist_ok=True)
    data = m.SerializeToString()
    with open(os.path.join(OUT, "tiny-cnn.onnx"), "wb") as fh:
        fh.write(data)
    sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    runs = []
    for batch in (1, 3):
        x = rng.standard_normal((batch, 3, 16, 16)).astype(np.float32)
        (probs,) = sess.run(["probs"], {"x": x})
        runs.append({
            "inputs": {"x": {"shape": list(x.shape), "data": b64(x)}},
            "outputs": {"probs": {"shape": list(probs.shape), "data": b64(probs)}},
        })
    exp = {
        "model": "tiny-cnn.onnx",
        "mim": "tiny-cnn.mim",
        "producer": "onnxruntime " + ort.__version__,
        "tolerance": 1e-5,
        "runs": runs,
    }
    with open(os.path.join(OUT, "expected.json"), "w", encoding="utf-8") as fh:
        json.dump(exp, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()

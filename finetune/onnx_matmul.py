"""
Rewrite the Einsum nodes of a torch-exported CCT OCR model as MatMul (+ Reshape / Transpose).

Keras on the torch backend implements Dense/EinsumDense and multi-head attention with einsum, and
torch.onnx keeps them as ONNX `Einsum` nodes. ONNX Runtime runs those far slower than `MatMul`:
for india-v1, 54 Einsum nodes made OCR 33 ms/crop on CPU vs 5.3 ms after this rewrite (4.8 ms vs
~5-11 ms on an RTX 4060), with identical outputs (max |diff| 1e-6 on 500 real crops).

Handles the four equations the CCT attention blocks produce, and folds weight x weight einsums
into constants. Anything else raises, so a new architecture fails loudly instead of silently.

Usage:
    python finetune/onnx_matmul.py in.onnx out.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def einsum_to_matmul(src: str | Path, dst: str | Path) -> int:
    """Convert ``src`` to ``dst``; returns the number of Einsum nodes rewritten."""
    model = onnx.load(str(src))
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    nodes: list[onnx.NodeProto] = []
    extra_inits: list[onnx.TensorProto] = []
    counter = [0]

    def fresh(base: str) -> str:
        counter[0] += 1
        return f"{base}__mm{counter[0]}"

    def const_i64(values: list[int]) -> str:
        name = fresh("const")
        extra_inits.append(numpy_helper.from_array(np.asarray(values, np.int64), name))
        return name

    def leading_dims(tensor: str, n: int) -> str:
        """Runtime shape[:n] of ``tensor`` (keeps the batch dimension dynamic)."""
        shape, out = fresh("shape"), fresh("dims")
        nodes.append(helper.make_node("Shape", [tensor], [shape]))
        nodes.append(helper.make_node("Slice", [shape, const_i64([0]), const_i64([n])], [out]))
        return out

    def concat(parts: list[str]) -> str:
        out = fresh("cat")
        nodes.append(helper.make_node("Concat", parts, [out], axis=0))
        return out

    converted = 0
    for node in graph.node:
        if node.op_type != "Einsum":
            nodes.append(node)
            continue
        eq = next(a.s.decode() for a in node.attribute if a.name == "equation").replace(" ", "")
        x, w = node.input
        y = node.output[0]
        converted += 1

        if x in inits and w in inits:  # constant expression: fold it
            val = np.einsum(eq, numpy_helper.to_array(inits[x]), numpy_helper.to_array(inits[w]))
            extra_inits.append(numpy_helper.from_array(val.astype(np.float32), y))
        elif eq == "abc,cde->abde" and w in inits:  # Q/K/V projection: x[a,b,c] @ W[c,(d,e)]
            wv = numpy_helper.to_array(inits[w])
            c, d, e = wv.shape
            w2, mm = fresh("w2d"), fresh("mm")
            extra_inits.append(numpy_helper.from_array(wv.reshape(c, d * e), w2))
            nodes.append(helper.make_node("MatMul", [x, w2], [mm]))
            shape = concat([leading_dims(x, 2), const_i64([d, e])])
            nodes.append(helper.make_node("Reshape", [mm, shape], [y]))
        elif eq == "abcd,cde->abe" and w in inits:  # attention output: x[a,b,(c,d)] @ W[(c,d),e]
            wv = numpy_helper.to_array(inits[w])
            c, d, e = wv.shape
            w2, xr = fresh("w2d"), fresh("xr")
            extra_inits.append(numpy_helper.from_array(wv.reshape(c * d, e), w2))
            shape = concat([leading_dims(x, 2), const_i64([c * d])])
            nodes.append(helper.make_node("Reshape", [x, shape], [xr]))
            nodes.append(helper.make_node("MatMul", [xr, w2], [y]))
        elif eq == "aecd,abcd->acbe":  # scores = Q.K^T: key[a,e,c,d], query[a,b,c,d] -> [a,c,b,e]
            kt, qt = fresh("kT"), fresh("qT")
            nodes.append(helper.make_node("Transpose", [x], [kt], perm=[0, 2, 3, 1]))  # a,c,d,e
            nodes.append(helper.make_node("Transpose", [w], [qt], perm=[0, 2, 1, 3]))  # a,c,b,d
            nodes.append(helper.make_node("MatMul", [qt, kt], [y]))
        elif eq == "acbe,aecd->abcd":  # attn[a,c,b,e] @ value[a,e,c,d] -> [a,b,c,d]
            vt, mm = fresh("vT"), fresh("mm")
            nodes.append(helper.make_node("Transpose", [w], [vt], perm=[0, 2, 1, 3]))  # a,c,e,d
            nodes.append(helper.make_node("MatMul", [x, vt], [mm]))  # a,c,b,d
            nodes.append(helper.make_node("Transpose", [mm], [y], perm=[0, 2, 1, 3]))
        else:
            raise ValueError(f"Unsupported Einsum {eq!r} (inputs {list(node.input)})")

    del graph.node[:]
    graph.node.extend(nodes)
    graph.initializer.extend(extra_inits)
    used = {i for n in graph.node for i in n.input} | {o.name for o in graph.output}
    kept = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return converted


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    n = einsum_to_matmul(sys.argv[1], sys.argv[2])
    print(f"rewrote {n} Einsum nodes -> {sys.argv[2]}")

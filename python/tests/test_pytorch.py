"""The PyTorch exporter: models exported with mim.pytorch run in the
Python runtime with PyTorch's results. Skipped without PyTorch."""

import os
import tempfile
import unittest

import numpy as np

import _paths  # noqa: F401
from mim import dtypes as D
from mim.container import File
from mim.executor import Program
from mim.tensor import Tensor

try:
    import torch
    import torch.nn as nn
    from torch.export import Dim

    import torch_models as TM
    from mim.pytorch import ExportError, export
except ImportError:  # pragma: no cover
    torch = None

TOL = 2e-6  # relative to the largest output


def to_mim(t):
    from mim.pytorch import _DTYPES

    a = t.detach().contiguous()
    if a.dtype == torch.bfloat16:
        return Tensor(D.BF16, a.shape, a.view(torch.int16).numpy().view(np.uint16))
    return Tensor(_DTYPES[a.dtype], a.shape, a.numpy())


def values(t):
    return D.to_f64(t.arr, t.dtype) if t.dtype.is_float else t.arr


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TestExport(unittest.TestCase):
    def run_model(self, m, args, dyn, more=(), **kw):
        f = File(export(m, args, dynamic_shapes=dyn, **kw))
        f.verify_hashes()
        s = Program.load(f).session()
        names = [f.graph().values[v].name for v in f.graph().main.inputs]
        outs = [f.graph().values[v].name for v in f.graph().main.outputs]
        for ins in (args, *more):
            got = s.run({n: to_mim(t) for n, t in zip(names, ins)})
            with torch.no_grad():
                ref = m(*ins)
            ref = ref if isinstance(ref, tuple) else (ref,)
            for name, r in zip(outs, ref):
                g = got[name]
                self.assertEqual(tuple(g.shape), tuple(r.shape), name)
                if r.dtype.is_floating_point:
                    want = r.double().numpy()
                    d = np.abs(values(g) - want).max() if want.size else 0
                    self.assertLessEqual(d, TOL * max(np.abs(want).max(), 1), f"{name}: {d:g}")
                else:
                    np.testing.assert_array_equal(g.arr, r.numpy(), name)
        return f

    def test_models(self):
        more = {
            "tiny-cnn": [(torch.randn(5, 3, 16, 16),)],
            "audio": [(torch.randn(3, 4, 40),)],
            "tiny-gpt": [(torch.randint(0, 64, (3, 1)),), (torch.randint(0, 64, (1, 32)),)],
            "tiny-llama": [(torch.randint(0, 64, (1, 2)),), (torch.randint(0, 64, (1, 33)),)],
        }
        for name, cls in TM.MODELS.items():
            with self.subTest(model=name):
                args, dyn = cls.example()
                self.run_model(TM.build(name), args, dyn, more[name])

    def test_ops(self):
        args, _ = TM.Ops.example()
        self.run_model(TM.Ops().eval(), args, None)

    def test_names_and_dimensions(self):
        m = TM.build("tiny-gpt")
        args, dyn = TM.TinyGPT.example()
        f = File(export(m, args, dynamic_shapes=dyn))
        self.assertEqual([(d.name, d.bound) for d in f.graph().symdims], [("batch", 8), ("seq", 32)])
        sd = m.state_dict()
        for name in ("wte.weight", "blocks.0.qkv.weight", "blocks.1.ln2.bias"):
            t = f.tensor(name)
            self.assertEqual(bytes(f.tensor_data(t)), sd[name].numpy().tobytes(), name)
        # The tied head reads the embedding table; it is stored once.
        self.assertEqual(sum(t.name.startswith("wte") for t in f.tensors()), 1)
        kinds = {f.graph().values[i].name: f.graph().values[i].kind for i in range(len(f.graph().values))}
        self.assertEqual(kinds["wte.weight"], 3)  # weight
        self.assertEqual(kinds["blocks.1.mask"], 5)  # a buffer is a const
        self.assertNotIn("blocks.0.mask", kinds)  # unused: block 0 calls attention
        md = f.metadata()
        self.assertEqual(md["general.name"], ("string", "TinyGPT"))
        self.assertEqual(md["export.source_format"], ("string", "pytorch"))

    def test_batch_norm_is_folded(self):
        f = File(export(TM.build("tiny-cnn"), *TM.TinyCNN.example()[:1], dynamic_shapes=TM.TinyCNN.example()[1]))
        names = {t.name for t in f.tensors()}
        self.assertIn("bn1.folded_scale", names)
        self.assertNotIn("bn1.running_mean", names)

    def test_low_precision(self):
        args, dyn = TM.TinyLlama.example()
        with torch.no_grad():
            r32 = TM.build("tiny-llama")(*args).double().numpy()
        for td, dt in ((torch.bfloat16, D.BF16), (torch.float16, D.F16)):
            m = TM.build("tiny-llama").to(td)
            f = File(export(m, args, dynamic_shapes=dyn))
            out = Program.load(f).session().run({"ids": to_mim(args[0])})["output"]
            self.assertIs(out.dtype, dt)
            with torch.no_grad():
                rt = m(*args).double().numpy()
            # As close to the float32 model as PyTorch's own result.
            self.assertLessEqual(np.abs(values(out) - r32).max(), 2 * np.abs(rt - r32).max() + 1e-3)

    def test_write_to_path(self):
        args, dyn = TM.AudioNet.example()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "audio.mim")
            data = export(TM.build("audio"), args, p, dynamic_shapes=dyn, metadata={"general.license": "MIT"})
            with open(p, "rb") as fh:
                self.assertEqual(fh.read(), data)
            self.assertEqual(File(data).metadata()["general.license"], ("string", "MIT"))

    def test_kv_cache(self):
        m = TM.build_kv()
        args, dyn = TM.TinyLlamaKV.example()
        f = File(export(m, args, dynamic_shapes=dyn))
        g = f.graph()
        caches = [v for v in g.values if v.name.endswith("_cache")]
        self.assertEqual(len(caches), 4)
        for v in caches:
            self.assertEqual(v.kind, 4)  # variable
            self.assertTrue(v.zero)  # an empty cache takes no file space
            self.assertEqual(list(v.shape), [1, 2, TM.TinyLlamaKV.MAX_LEN, 8])
        self.assertFalse([t.name for t in f.tensors() if "cache" in t.name])
        self.assertEqual(sorted(g.values[u.variable].name for u in g.updates), sorted(v.name for v in caches))
        # Each update is a scatter into its own variable, so a runtime can
        # write the new rows in place.
        producer = {o: op for op in g.main.ops for o in op.outputs}
        for u in g.updates:
            op = producer[u.value]
            self.assertEqual(op.name, "scatter")
            self.assertEqual(op.inputs[0], u.variable)

        full = TM.build("tiny-llama")
        torch.manual_seed(3)
        for prompt, steps in ((5, 4), (1, 3), (9, 0)):
            ids = torch.randint(0, 64, (1, prompt + steps))
            s = Program.load(f).session()
            m.reset()
            with torch.no_grad():
                want = full(ids).double().numpy()
            t = 0
            for n in [prompt] + [1] * steps:
                x, pos = ids[:, t:t + n], torch.arange(t, t + n)
                got = values(s.run({"ids": to_mim(x), "pos": to_mim(pos)})["output"])
                with torch.no_grad():
                    ref = m(x, pos).double().numpy()
                scale = max(np.abs(ref).max(), 1)
                self.assertLessEqual(np.abs(got - ref).max(), TOL * scale, f"step at {t}")
                # A cached decode gives the logits of the full forward pass.
                self.assertLessEqual(np.abs(got - want[:, t:t + n]).max(), TOL * scale, f"full pass at {t}")
                t += n
            k = s.variable("layers.0.k_cache")
            self.assertTrue(np.all(D.to_f64(k.arr, k.dtype)[:, :, t:] == 0))

    def test_rejections(self):
        class Sort(nn.Module):
            def forward(self, x):
                return torch.sort(x).values

        class Flat(nn.Module):
            def forward(self, x):
                return x.reshape(-1)

        class Mutate(nn.Module):
            def forward(self, x):
                x.add_(1)
                return x * 2

        class Accumulate(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("hist", torch.zeros(4))

            def forward(self, i):
                self.hist.index_put_((i,), torch.ones(i.shape), accumulate=True)
                return self.hist * 1

        with self.assertRaisesRegex(ExportError, "mutates an input"):
            export(Mutate().eval(), (torch.randn(3),))
        with self.assertRaisesRegex(ExportError, "accumulate"):
            export(Accumulate().eval(), (torch.tensor([1, 2]),))
        with self.assertRaisesRegex(ExportError, "aten.sort"):
            export(Sort().eval(), (torch.randn(4),))
        b, t = Dim("b", max=8), Dim("t", max=8)
        with self.assertRaisesRegex(ExportError, "expression of symbolic sizes"):
            export(Flat().eval(), (torch.randn(2, 3),), dynamic_shapes={"x": {0: b, 1: t}})
        with self.assertRaisesRegex(ExportError, "training mode"):
            export(TM.TinyCNN(), (torch.randn(1, 3, 16, 16),))


if __name__ == "__main__":
    unittest.main()

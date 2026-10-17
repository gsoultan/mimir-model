#!/usr/bin/env python3
"""Measure text generation speed and peak memory, for docs/benchmarks.md.

    go build -o /tmp/mmf-run ./cmd/mmf-run
    python3 -I tools/bench_generate.py --mmf-run /tmp/mmf-run model.gguf model.mmf ...

An .mmf file is run with mmf-run; a .gguf file with llama.cpp through
llama-cpp-python, as the reference. Every run is a fresh process: the table
reports the median over the runs of the load time, the prompt and
generation speed (greedy, one token per step) and the peak resident memory
of the process, which counts the mapped weights the run touched. With
--repeat N the prompt is the sentence below N times over, for long prompts.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

PROMPT = "Once upon a time there was a little girl who lived in a village near the forest."

LLAMA_CPP = r"""
import json, sys, time
import llama_cpp
path, n, threads, prompt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
t = time.perf_counter()
llm = llama_cpp.Llama(model_path=path, n_ctx=2048, n_threads=threads, n_batch=512, verbose=False, seed=0)
load = time.perf_counter() - t
ids = llm.tokenize(prompt.encode(), add_bos=True)
t = time.perf_counter()
llm.eval(ids)
pt = time.perf_counter() - t
t = time.perf_counter()
for _ in range(n):
    llm.eval([llm.sample(top_k=1)])
gt = time.perf_counter() - t
print(json.dumps({"load": load, "prompt_tokens": len(ids), "prompt_s": pt, "tokens": n, "gen_s": gt}))
"""

MMF_RUN = re.compile(
    r"load (\S+); prompt (\d+) tokens in (\S+) \(.*\); generated (\d+) tokens in (\S+) \("
)


def seconds(s):
    for unit, f in (("ms", 1e-3), ("µs", 1e-6), ("us", 1e-6), ("s", 1.0)):
        if s.endswith(unit):
            return float(s[: -len(unit)]) * f
    raise ValueError(s)


def run(cmd):
    """Runs cmd and returns its stdout, stderr and peak RSS in bytes."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        p = subprocess.Popen(cmd, stdout=out, stderr=err)
        _, status, usage = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        o, e = out.read().decode(errors="replace"), err.read().decode(errors="replace")
    if p.returncode != 0:
        sys.exit(f"{cmd[0]} failed: {e}")
    # ru_maxrss is in KiB on Linux and in bytes on macOS.
    return o, e, usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def measure(path, args):
    prompt = " ".join([PROMPT] * args.repeat)
    if path.endswith(".gguf"):
        out, _, rss = run([sys.executable, "-c", LLAMA_CPP, path, str(args.tokens), str(args.threads), prompt])
        r = json.loads(out)
        return "llama.cpp", r["load"], r["prompt_tokens"] / r["prompt_s"], r["tokens"] / r["gen_s"], rss, r["prompt_tokens"]
    _, err, rss = run([args.mmf_run, "-m", path, "-p", prompt, "-n", str(args.tokens)])
    m = MMF_RUN.search(err)
    if not m:
        sys.exit(f"cannot parse mmf-run output: {err}")
    load, np_, pt, ng, gt = seconds(m[1]), int(m[2]), seconds(m[3]), int(m[4]), seconds(m[5])
    return "mmf-run", load, np_ / pt, max(ng - 1, 0) / gt, rss, np_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmf-run", default="mmf-run")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    print(f"median of {args.runs} runs, {args.tokens} generated tokens, {args.threads} threads\n")
    print("| file | runtime | load | prompt | generation | peak RSS | prompt tokens |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for f in args.files:
        rs = [measure(f, args) for _ in range(args.runs)]
        med = [statistics.median(r[i] for r in rs) for i in range(1, 5)]
        print(f"| {os.path.basename(f)} | {rs[0][0]} | {med[0]*1000:.0f} ms | {med[1]:.1f} tok/s "
              f"| {med[2]:.1f} tok/s | {med[3]/2**20:.0f} MiB | {rs[0][5]} |")


if __name__ == "__main__":
    main()

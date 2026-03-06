#!/usr/bin/env python3
import argparse, csv, os, time, json
from dataclasses import asdict, dataclass
from typing import Optional, Dict, Any, List

import torch

# ---------- Prompt helpers ----------

def make_synth_prompt(target_tokens: int) -> str:
    """
    Deterministic filler prompt. We don't try to exactly hit token count here;
    we approximate by characters. For Phase 3 synthetic sweeps that's enough.
    For true apples-to-apples later, use --prompt-file.
    """
    base = (
        "SYSTEM: You are a helpful assistant.\n"
        "USER: Please answer clearly.\n"
        "CONTEXT:\n"
    )
    filler = ("lorem ipsum " * 50000)
    text = base + filler + "\nASSISTANT:"
    # crude char->token approx; safe and deterministic
    # 1 token ~ 4 chars is a common rough estimate
    approx_chars = max(200, target_tokens * 4)
    return text[:approx_chars]

def read_prompt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------- Metrics row ----------

@dataclass
class BenchRow:
    timestamp: str
    run_id: str
    experiment_name: str
    backend: str                 # hf | vllm
    model: str
    dtype: str                   # fp16 | bf16
    prompt_len: int
    gen_len: int
    batch: int
    use_cache: int               # 1/0
    attn: str                    # auto/sdpa/flash2 (hf only)
    ttft_ms: str                 # empty if not measured
    total_ms: float
    tokens_per_sec: float
    peak_vram_mb: float
    status: str                  # ok | oom | error:<Type>

def append_rows_csv(path: str, rows: List[BenchRow]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

# ---------- HF runner ----------

def bench_hf(
    model_path: str,
    dtype: str,
    attn: str,
    use_cache: bool,
    prompt: str,
    gen_len: int,
    runs: int,
    warmup: int,
) -> List[Dict[str, Any]]:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cuda",
    )
    # attention impl (HF)
    if attn != "auto":
        # transformers uses torch backend settings; easiest: set attribute if exists
        # This is best-effort; if not supported it will just behave as default.
        try:
            model.config.attn_implementation = attn
        except Exception:
            pass

    inputs = tok(prompt, return_tensors="pt").to("cuda")

    def one():
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=gen_len,
                do_sample=False,
                use_cache=use_cache,
            )
        t1 = time.perf_counter()
        peak = torch.cuda.max_memory_allocated() / (1024**2)

        # generated tokens only (approx by subtracting prompt tokens)
        out_ids = out[0]
        gen_tokens = max(0, out_ids.shape[0] - inputs["input_ids"].shape[1])
        total_s = (t1 - t0)
        tokps = (gen_tokens / total_s) if total_s > 0 else 0.0
        return {
            "ttft_ms": "",  # not streaming here
            "total_ms": total_s * 1000.0,
            "tokens_per_sec": tokps,
            "peak_vram_mb": peak,
        }

    # warmup
    for _ in range(warmup):
        try:
            one()
        except Exception:
            pass

    out = []
    for _ in range(runs):
        out.append(one())
    return out

# ---------- vLLM runner ----------

def bench_vllm(
    model_path: str,
    dtype: str,
    prompt: str,
    gen_len: int,
    runs: int,
    warmup: int,
) -> List[Dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    vllm_dtype = "float16" if dtype == "fp16" else "bfloat16"
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    llm = LLM(
        model=model_path,
        dtype=vllm_dtype,
        gpu_memory_utilization = 0.80,
        max_model_len = 4096,
        enforce_eager = True,
    )
    sp = SamplingParams(max_tokens=gen_len, temperature=0.0)

    def one():
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)  # batch=1 list
        t1 = time.perf_counter()
        peak = torch.cuda.max_memory_allocated() / (1024**2)

        gen_text = out[0].outputs[0].text
        gen_tokens = len(tok(gen_text, add_special_tokens=False).input_ids)
        total_s = (t1 - t0)
        tokps = (gen_tokens / total_s) if total_s > 0 else 0.0
        return {
            "ttft_ms": "",  # needs streaming server for real TTFT
            "total_ms": total_s * 1000.0,
            "tokens_per_sec": tokps,
            "peak_vram_mb": peak,
        }

    for _ in range(warmup):
        try:
            one()
        except Exception:
            pass

    out = []
    for _ in range(runs):
        out.append(one())
    return out

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["hf", "vllm"])
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--prompt-len", type=int, default=128, help="Synthetic prompt length (approx). Ignored if --prompt-file set.")
    ap.add_argument("--prompt-file", type=str, default=None, help="Path to a prompt .txt file (overrides synthetic).")
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1, help="Currently only batch=1 supported in this script.")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--attn", choices=["auto", "sdpa", "flash2"], default="auto", help="HF only.")
    ap.add_argument("--use-cache", action="store_true", default=True)
    ap.add_argument("--no-use-cache", dest="use_cache", action="store_false")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--experiment-name", type=str, default="phase3_bench")
    ap.add_argument("--out", type=str, default="results/llm_bench.csv")
    args = ap.parse_args()

    if args.batch != 1:
        raise SystemExit("This script currently supports --batch 1 only (keeps Phase 3 minimal).")

    prompt = read_prompt_file(args.prompt_file) if args.prompt_file else make_synth_prompt(args.prompt_len)

    run_id = args.run_id or f"{args.backend}_{time.strftime('%Y%m%d_%H%M%S')}"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        if args.backend == "hf":
            metrics = bench_hf(
                model_path=args.model,
                dtype=args.dtype,
                attn=args.attn,
                use_cache=args.use_cache,
                prompt=prompt,
                gen_len=args.gen_len,
                runs=args.runs,
                warmup=args.warmup,
            )
        else:
            metrics = bench_vllm(
                model_path=args.model,
                dtype=args.dtype,
                prompt=prompt,
                gen_len=args.gen_len,
                runs=args.runs,
                warmup=args.warmup,
            )

        rows = []
        for m in metrics:
            rows.append(BenchRow(
                timestamp=ts,
                run_id=run_id,
                experiment_name=args.experiment_name,
                backend=args.backend,
                model=args.model,
                dtype=args.dtype,
                prompt_len=args.prompt_len,
                gen_len=args.gen_len,
                batch=args.batch,
                use_cache=1 if args.use_cache else 0,
                attn=args.attn if args.backend == "hf" else "",
                ttft_ms=m["ttft_ms"],
                total_ms=float(m["total_ms"]),
                tokens_per_sec=float(m["tokens_per_sec"]),
                peak_vram_mb=float(m["peak_vram_mb"]),
                status="ok",
            ))

        append_rows_csv(args.out, rows)
        print(f"Appended {len(rows)} rows -> {args.out}")
        print("run_id =", run_id)

    except torch.cuda.OutOfMemoryError:
        row = BenchRow(
            timestamp=ts, run_id=run_id, experiment_name=args.experiment_name,
            backend=args.backend, model=args.model, dtype=args.dtype,
            prompt_len=args.prompt_len, gen_len=args.gen_len, batch=args.batch,
            use_cache=1 if args.use_cache else 0, attn=args.attn if args.backend=="hf" else "",
            ttft_ms="", total_ms=0.0, tokens_per_sec=0.0, peak_vram_mb=0.0, status="oom"
        )
        append_rows_csv(args.out, [row])
        print("OOM -> row appended")
    except Exception as e:
        row = BenchRow(
            timestamp=ts, run_id=run_id, experiment_name=args.experiment_name,
            backend=args.backend, model=args.model, dtype=args.dtype,
            prompt_len=args.prompt_len, gen_len=args.gen_len, batch=args.batch,
            use_cache=1 if args.use_cache else 0, attn=args.attn if args.backend=="hf" else "",
            ttft_ms="", total_ms=0.0, tokens_per_sec=0.0, peak_vram_mb=0.0, status=f"error:{type(e).__name__}"
        )
        append_rows_csv(args.out, [row])
        raise

if __name__ == "__main__":
    main()
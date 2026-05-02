#!/usr/bin/env python3
"""
Phase 0.5 — Hardware Characterization Microbenchmarks
GB10 Grace Blackwell: memory BW, TFLOPS, NVLink-C2C, thermal, roofline
Results → /home/student/Desktop/Test/results/hardware/
"""
import os, sys, time, json, subprocess, platform
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/home/student/Desktop/Test/results/hardware")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG = RESULTS_DIR / f"hw_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

results = {
    "timestamp": datetime.now().isoformat(),
    "device": "NVIDIA GB10 Grace Blackwell",
    "theoretical": {
        "memory_bw_gbs": 273,
        "bf16_tflops": 67,
        "fp8_tflops": 134,
        "int8_tops": 134,
        "nvlink_c2c_gbs": 900,
        "roofline_ridge_flops_per_byte": round(67e12 / 273e9, 1),  # ~245
    },
    "hw1_memory_bw": {},
    "hw2_compute": {},
    "hw3_nvlink": {},
    "hw4_thermal": {},
    "hw5_roofline": {},
    "hw5b_attention": {},
    "hw5c_nvme": {},
    "hw5d_memory_pressure": {},
    "hw6_model_bw_efficiency": {},
}

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def save():
    with open(LOG, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {LOG}")

# ─────────────────────────────────────────────────────────────
def check_cuda():
    try:
        import torch
        if not torch.cuda.is_available():
            print("[ERROR] CUDA not available — run phase0_install.sh first")
            sys.exit(1)
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device : {name}")
        print(f"  SM cap : {cap[0]}.{cap[1]}  (expected 12.1 for GB10)")
        print(f"  Memory : {total_gb:.1f} GB")
        results["device_info"] = {
            "name": name, "compute_cap": f"{cap[0]}.{cap[1]}", "total_memory_gb": round(total_gb, 1)
        }
        return torch
    except ImportError:
        print("[ERROR] PyTorch not installed")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────
def hw1_memory_bandwidth(torch):
    print_section("HW-1: Memory Bandwidth")
    PEAK = 273  # GB/s theoretical
    data = {}

    sizes_gb = [1, 4, 16, 32, 64]
    for s in sizes_gb:
        try:
            n = int(s * 1024**3 / 4)  # float32 elements
            x = torch.randn(n, device='cuda', dtype=torch.float32)
            torch.cuda.synchronize()

            # warmup
            for _ in range(3):
                _ = x * 2.0
            torch.cuda.synchronize()

            # timed
            t0 = time.perf_counter()
            for _ in range(5):
                y = x * 2.0  # read + write = 2× size
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 5

            bw = 2 * s / dt
            util_pct = bw / PEAK * 100
            data[f"{s}GB"] = {"bw_gbs": round(bw, 1), "util_pct": round(util_pct, 1)}
            print(f"  {s:4d} GB tensor: {bw:6.1f} GB/s  ({util_pct:.1f}% of {PEAK} GB/s peak)")
            del x, y
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  {s:4d} GB tensor: OOM (skipped)")
            data[f"{s}GB"] = {"error": "OOM"}

    results["hw1_memory_bw"] = data
    save()

# ─────────────────────────────────────────────────────────────
def hw2_compute_tflops(torch):
    print_section("HW-2: Compute Throughput (GEMM)")
    PEAK_BF16 = 67    # TFLOPS theoretical
    PEAK_FP8  = 134
    data = {}

    # Enable TF32 to allow Tensor Core usage for BF16/FP32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sizes = [(4096, 4096, 4096), (8192, 8192, 8192), (16384, 4096, 4096)]
    WARMUP, ITERS = 15, 50

    for dtype_name, dtype, peak in [("bf16", torch.bfloat16, PEAK_BF16),
                                      ("fp32", torch.float32, PEAK_BF16 / 2)]:
        flop_results = []
        for M, N, K in sizes:
            try:
                A = torch.randn(M, K, device='cuda', dtype=dtype)
                B = torch.randn(K, N, device='cuda', dtype=dtype)
                torch.cuda.synchronize()

                for _ in range(WARMUP):
                    C = torch.mm(A, B)
                torch.cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(ITERS):
                    C = torch.mm(A, B)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / ITERS

                flops = 2 * M * N * K
                tflops = flops / dt / 1e12
                flop_results.append(tflops)
                print(f"  {dtype_name} {M}x{N}x{K}: {tflops:.1f} TFLOPS  ({tflops/PEAK_BF16*100:.1f}% of {PEAK_BF16}T peak)")
                del A, B, C
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  {dtype_name} {M}x{N}x{K}: OOM")

        if flop_results:
            data[dtype_name] = {
                "peak_tflops": round(max(flop_results), 1),
                "util_pct": round(max(flop_results) / PEAK_BF16 * 100, 1),
            }

    # FP8 via _scaled_mm — torch.randn doesn't support fp8 directly, cast instead
    try:
        M, N, K = 8192, 8192, 8192
        A = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
        B = torch.randn(K, N, device='cuda').to(torch.float8_e4m3fn)
        A_s = torch.ones(1, device='cuda')
        B_s = torch.ones(1, device='cuda')
        torch.cuda.synchronize()

        for _ in range(WARMUP):
            C = torch._scaled_mm(A, B.T, scale_a=A_s, scale_b=B_s, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            C = torch._scaled_mm(A, B.T, scale_a=A_s, scale_b=B_s, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / ITERS
        flops = 2 * M * N * K
        tflops = flops / dt / 1e12
        data["fp8"] = {"peak_tflops": round(tflops, 1), "util_pct": round(tflops / PEAK_FP8 * 100, 1)}
        print(f"  fp8  {M}x{N}x{K}: {tflops:.1f} TFLOPS  ({tflops/PEAK_FP8*100:.1f}% of {PEAK_FP8}T peak)")
        bf16_tps = data.get("bf16", {}).get("peak_tflops", 0)
        if bf16_tps:
            print(f"  FP8 vs BF16 speedup: {tflops/bf16_tps:.1f}x  (theoretical 2x)")
        del A, B, C, A_s, B_s
        torch.cuda.empty_cache()
    except Exception as e:
        data["fp8"] = {"error": str(e)}
        print(f"  fp8: {e}")

    results["hw2_compute"] = data
    save()

# ─────────────────────────────────────────────────────────────
def hw3_nvlink_bandwidth(torch):
    print_section("HW-3: NVLink-C2C CPU↔GPU Bandwidth")
    data = {}
    sizes_gb = [0.1, 0.5, 1, 4, 8]  # cap at 8 GB to avoid pinned-memory exhaustion
    ITERS = 3

    for s in sizes_gb:
        try:
            n = int(s * 1024**3 / 4)
            cpu_t = torch.zeros(n, dtype=torch.float32).pin_memory()
            gpu_t = torch.zeros(n, dtype=torch.float32, device='cuda')
            torch.cuda.synchronize()

            # H2D: pinned CPU → GPU (timed with synchronize inside)
            h2d_times = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                tmp = cpu_t.to('cuda', non_blocking=False)
                torch.cuda.synchronize()
                h2d_times.append(time.perf_counter() - t0)
                del tmp
            h2d = s / (sum(h2d_times) / len(h2d_times))

            # D2H: GPU → CPU (blocking .cpu() call)
            d2h_times = []
            for _ in range(ITERS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                tmp = gpu_t.cpu()
                d2h_times.append(time.perf_counter() - t0)
                del tmp
            d2h = s / (sum(d2h_times) / len(d2h_times))

            data[f"{s}GB"] = {"h2d_gbs": round(h2d, 1), "d2h_gbs": round(d2h, 1)}
            print(f"  {s:.1f} GB  H2D: {h2d:6.1f} GB/s   D2H: {d2h:6.1f} GB/s")
            del cpu_t, gpu_t
            torch.cuda.empty_cache()
        except Exception as e:
            data[f"{s}GB"] = {"error": str(e)}
            print(f"  {s:.1f} GB  ERROR: {e}")
            torch.cuda.empty_cache()

    data["note"] = "H2D via pinned→cuda DMA; D2H via blocking .cpu(). NVLink-C2C unified mem."
    results["hw3_nvlink"] = data
    save()

# ─────────────────────────────────────────────────────────────
def hw4_thermal_power(torch):
    print_section("HW-4: Thermal & Power (idle + sustained load)")
    THROTTLE_C = 53
    data = {}

    def snap():
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=power.draw,temperature.gpu,utilization.gpu,utilization.memory,clocks.sm",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True)
            vals = [v.strip() for v in r.stdout.strip().split(",")]
            return {
                "power_draw_w":  float(vals[0]),
                "temperature_c": int(vals[1]),
                "gpu_util_pct":  int(vals[2]),
                "mem_util_pct":  int(vals[3]),
                "sm_clock_mhz":  int(vals[4]),
            }
        except Exception as e:
            return {"error": str(e)}

    # Idle snapshot
    idle = snap()
    data["idle"] = idle
    print(f"  Idle : {idle.get('power_draw_w')}W  {idle.get('temperature_c')}°C  "
          f"SM={idle.get('sm_clock_mhz')}MHz")

    # Sustained load: run BF16 matmul for 15s and sample every 3s
    print("  Running sustained BF16 load (15s)...")
    M = 8192
    A = torch.randn(M, M, device='cuda', dtype=torch.bfloat16)
    B = torch.randn(M, M, device='cuda', dtype=torch.bfloat16)
    torch.cuda.synchronize()

    load_snaps = []
    t_end = time.perf_counter() + 15
    while time.perf_counter() < t_end:
        for _ in range(50):
            C = torch.mm(A, B)
        torch.cuda.synchronize()
        s = snap()
        load_snaps.append(s)
        print(f"    {s.get('power_draw_w')}W  {s.get('temperature_c')}°C  "
              f"util={s.get('gpu_util_pct')}%")

    del A, B, C
    torch.cuda.empty_cache()

    peak_power = max((s.get("power_draw_w", 0) for s in load_snaps), default=0)
    peak_temp  = max((s.get("temperature_c", 0) for s in load_snaps), default=0)
    throttled  = peak_temp >= THROTTLE_C

    data["sustained_load"] = {
        "duration_sec": 15,
        "peak_power_w": peak_power,
        "peak_temp_c": peak_temp,
        "throttle_threshold_c": THROTTLE_C,
        "throttle_occurred": throttled,
        "snapshots": load_snaps,
    }
    print(f"  Peak : {peak_power}W  {peak_temp}°C  throttle={throttled}")
    results["hw4_thermal"] = data
    save()

# ─────────────────────────────────────────────────────────────
def hw5_roofline(torch):
    print_section("HW-5: Roofline Analysis")
    BW  = 273e9    # bytes/s
    BF16 = 67e12   # FLOP/s
    FP8  = 134e12

    ridge_bf16 = BF16 / BW
    ridge_fp8  = FP8  / BW

    workloads = {
        "LLM decode batch=1  (BF16)":  1.0,
        "LLM decode batch=32 (BF16)":  32.0,
        "LLM prefill seq=2048":        1000.0,
        "Training fwd+bwd":             2000.0,
        "GEMM BF16 (roofline)":         ridge_bf16,
    }

    print(f"\n  Ridge BF16 = {BF16/1e12:.0f} TFLOPS / {BW/1e9:.0f} GB/s = {ridge_bf16:.1f} FLOPs/byte")
    print(f"  Ridge FP8  = {FP8/1e12:.0f} TFLOPS / {BW/1e9:.0f} GB/s = {ridge_fp8:.1f} FLOPs/byte")
    print(f"\n  {'Workload':<40} {'AI (FLOPs/B)':>14} {'Bound':>12}")
    print(f"  {'-'*70}")

    roofline_data = {"ridge_bf16_flops_per_byte": round(ridge_bf16, 1),
                     "ridge_fp8_flops_per_byte": round(ridge_fp8, 1),
                     "workloads": {}}

    for name, ai in workloads.items():
        bound = "MEMORY" if ai < ridge_bf16 else "COMPUTE"
        print(f"  {name:<40} {ai:>14.1f}   {bound:>12}")
        roofline_data["workloads"][name] = {"arithmetic_intensity": ai, "bottleneck": bound}

    results["hw5_roofline"] = roofline_data
    save()

# ─────────────────────────────────────────────────────────────
def hw5b_attention_kernels(torch):
    print_section("HW-5b: Attention Kernel Comparison")
    data = {}
    seq_lens = [512, 1024, 2048, 4096, 8192]
    B, H, D = 1, 32, 128  # batch, heads, head_dim

    for sl in seq_lens:
        row = {}
        Q = torch.randn(B, H, sl, D, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(B, H, sl, D, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(B, H, sl, D, device='cuda', dtype=torch.bfloat16)

        # torch SDPA (flash-attention via cuDNN if available)
        try:
            torch.cuda.synchronize()
            for _ in range(3): _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()
            ITERS = 20
            t0 = time.perf_counter()
            for _ in range(ITERS):
                _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()
            dt_sdpa = (time.perf_counter() - t0) / ITERS * 1000
            row["sdpa_ms"] = round(dt_sdpa, 3)
        except Exception as e:
            row["sdpa_ms"] = f"error: {e}"

        # Flash Attention 3 (if installed)
        try:
            from flash_attn import flash_attn_qkvpacked_func
            qkv = torch.randn(B * sl, 3, H, D, device='cuda', dtype=torch.bfloat16)
            torch.cuda.synchronize()
            for _ in range(3): _ = flash_attn_qkvpacked_func(qkv)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                _ = flash_attn_qkvpacked_func(qkv)
            torch.cuda.synchronize()
            dt_fa3 = (time.perf_counter() - t0) / ITERS * 1000
            row["fa3_ms"] = round(dt_fa3, 3)
            row["fa3_speedup_vs_sdpa"] = round(dt_sdpa / dt_fa3, 2)
        except ImportError:
            row["fa3_ms"] = "not_installed"
        except Exception as e:
            row["fa3_ms"] = f"error: {e}"

        print(f"  seq={sl:5d}  SDPA: {row.get('sdpa_ms','?')} ms   FA3: {row.get('fa3_ms','?')} ms")
        data[f"seq_{sl}"] = row
        del Q, K, V
        torch.cuda.empty_cache()

    results["hw5b_attention"] = data
    save()

# ─────────────────────────────────────────────────────────────
def hw5c_nvme_io():
    print_section("HW-5c: NVMe Storage I/O")
    data = {}

    test_file = "/tmp/nvme_test_4gb.bin"
    SIZE_GB = 4

    try:
        # Write test
        t0 = time.perf_counter()
        subprocess.run(
            f"dd if=/dev/zero of={test_file} bs=1M count={SIZE_GB*1024} conv=fdatasync 2>/dev/null",
            shell=True, check=True)
        dt_w = time.perf_counter() - t0
        write_gbs = SIZE_GB / dt_w
        data["write_gbs"] = round(write_gbs, 2)
        print(f"  Sequential write: {write_gbs:.2f} GB/s")

        # Read test (drop cache first)
        subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null", shell=True)
        t0 = time.perf_counter()
        subprocess.run(f"dd if={test_file} of=/dev/null bs=1M iflag=direct 2>/dev/null", shell=True, check=True)
        dt_r = time.perf_counter() - t0
        read_gbs = SIZE_GB / dt_r
        data["read_gbs"] = round(read_gbs, 2)
        print(f"  Sequential read : {read_gbs:.2f} GB/s")

        # Estimate model load times
        for model_size_gb, name in [(14, "7B BF16"), (64, "32B BF16"), (144, "72B BF16"), (118, "235B NF4")]:
            t_load = model_size_gb / read_gbs
            data[f"load_time_{name.replace(' ', '_')}"] = round(t_load, 1)
            print(f"  Est. load {name:12s}: {t_load:.1f}s")

        os.remove(test_file)
    except Exception as e:
        data["error"] = str(e)
        print(f"  [ERROR] {e}")

    results["hw5c_nvme"] = data
    save()

# ─────────────────────────────────────────────────────────────
def hw5d_memory_pressure(torch):
    print_section("HW-5d: Memory Pressure Behavior")
    # Test latency at increasing memory occupancy levels.
    # Each level: allocate ONE tensor of that absolute size, measure, then FREE IT
    # before moving to the next. We never accumulate — previous tensors are released.
    # This avoids the OOM that happens when holding all levels simultaneously.
    data = {}
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Test absolute sizes in GB (not cumulative); cap well below total to stay safe
    test_sizes_gb = [10, 20, 40, 60, 80, 96, 108, 112]
    safe_max_gb   = total_gb * 0.90  # never exceed 90%

    print(f"  Total unified memory: {total_gb:.1f} GB  (safe max: {safe_max_gb:.0f} GB)")

    for target_gb in test_sizes_gb:
        if target_gb > safe_max_gb:
            print(f"  {target_gb:.0f} GB  — exceeds safe 90% limit, stopping")
            break
        try:
            torch.cuda.empty_cache()
            n = int(target_gb * 1e9 / 2)  # bfloat16 = 2 bytes/elem

            t0 = time.perf_counter()
            ballast = torch.zeros(n, dtype=torch.bfloat16, device='cuda')
            torch.cuda.synchronize()
            dt_alloc = (time.perf_counter() - t0) * 1000

            alloc_now = torch.cuda.memory_allocated() / 1e9
            pct_used  = alloc_now / total_gb * 100

            # Measure small-matmul latency while memory is under pressure
            x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                y = x @ x
            torch.cuda.synchronize()
            op_ms = (time.perf_counter() - t0) / 200 * 1000

            print(f"  {target_gb:5.0f} GB ({pct_used:.0f}%)  "
                  f"alloc={dt_alloc:.0f}ms  op_latency={op_ms:.3f}ms")
            data[f"{target_gb:.0f}GB"] = {
                "target_gb":     target_gb,
                "allocated_gb":  round(alloc_now, 2),
                "pct_used":      round(pct_used, 1),
                "alloc_time_ms": round(dt_alloc, 1),
                "op_latency_ms": round(op_ms, 4),
            }
            # ALWAYS free before next iteration
            del ballast, x, y
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            oom_gb = torch.cuda.memory_allocated() / 1e9
            data["oom_at_gb"]  = round(oom_gb, 2)
            data["oom_at_pct"] = round(oom_gb / total_gb * 100, 1)
            print(f"  OOM at target={target_gb} GB  (allocated={oom_gb:.1f} GB)")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            data[f"{target_gb:.0f}GB"] = {"error": str(e)}
            print(f"  {target_gb} GB  ERROR: {e}")
            torch.cuda.empty_cache()

    results["hw5d_memory_pressure"] = data
    save()

# ─────────────────────────────────────────────────────────────
def hw6_model_bw_efficiency(torch):
    print_section("HW-6: Model Bandwidth Efficiency (decode simulation)")
    # LLM decode is memory-bandwidth-bound (arithmetic intensity ~1 FLOPs/byte).
    # Proxy: read a large 1D weight tensor and compute a dot product (batch=1).
    # Achieved BW = bytes_read / time_per_token
    PEAK_BW = 273  # GB/s
    data = {}
    WARMUP, ITERS = 10, 50

    configs = [
        ("7B_bf16",  7e9,  2),    # 7B params × 2 bytes/param  → 14 GB
        ("32B_bf16", 32e9, 2),    # 32B params × 2 bytes/param → 64 GB
        ("70B_nf4",  70e9, 0.5),  # 70B params × 0.5 bytes     → 35 GB
    ]

    for name, params, bytes_per_param in configs:
        model_bytes = params * bytes_per_param
        model_gb    = model_bytes / 1e9

        # Simulate ONE layer read: use ~model_bytes/32 but cap at 4 GB
        # so we never risk OOM — both W and x must fit simultaneously
        layer_bytes = min(model_bytes / 32, 4e9)
        n = int(layer_bytes / 2)  # number of bfloat16 elements

        try:
            torch.cuda.empty_cache()
            W = torch.randn(n, device='cuda', dtype=torch.bfloat16)
            x = torch.ones(1, n, device='cuda', dtype=torch.bfloat16)
            torch.cuda.synchronize()

            for _ in range(WARMUP):
                r = (x * W).sum()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(ITERS):
                r = (x * W).sum()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / ITERS

            # bytes accessed: read W (n×2) + read x (n×2)
            bytes_accessed = n * 2 * 2
            achieved_bw    = bytes_accessed / dt / 1e9
            util_pct       = achieved_bw / PEAK_BW * 100
            tps_estimate   = achieved_bw * 1e9 / model_bytes  # if full model read/token

            data[name] = {
                "model_size_gb":   round(model_gb, 1),
                "layer_size_gb":   round(layer_bytes / 1e9, 2),
                "achieved_bw_gbs": round(achieved_bw, 1),
                "bw_util_pct":     round(util_pct, 1),
                "estimated_tps":   round(tps_estimate, 1),
            }
            print(f"  {name:15s}  BW={achieved_bw:.1f} GB/s ({util_pct:.1f}%)  "
                  f"~{tps_estimate:.0f} tok/s (decode est.)")
            del W, x, r
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            data[name] = {"error": "OOM"}
            print(f"  {name:15s}  OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            data[name] = {"error": str(e)}
            print(f"  {name:15s}  ERROR: {e}")
            torch.cuda.empty_cache()

    results["hw6_model_bw_efficiency"] = data
    save()

# ─────────────────────────────────────────────────────────────
def print_summary():
    print_section("SUMMARY")
    r = results
    peak_bw = max((v["bw_gbs"] for v in r["hw1_memory_bw"].values() if "bw_gbs" in v), default=0)
    peak_bf16 = r["hw2_compute"].get("bf16", {}).get("peak_tflops", "?")
    peak_fp8  = r["hw2_compute"].get("fp8", {}).get("peak_tflops", "?")
    h2d = max((v["h2d_gbs"] for v in r["hw3_nvlink"].values() if "h2d_gbs" in v), default=0)
    ridge = r["hw5_roofline"].get("ridge_bf16_flops_per_byte", "?")

    print(f"  Peak memory bandwidth : {peak_bw:.1f} / 273 GB/s ({peak_bw/273*100:.0f}%)")
    print(f"  Peak BF16 TFLOPS      : {peak_bf16} / 67 TFLOPS")
    print(f"  Peak FP8 TFLOPS       : {peak_fp8} / 134 TFLOPS")
    print(f"  NVLink-C2C H2D        : {h2d:.1f} GB/s")
    print(f"  Roofline ridge point  : {ridge} FLOPs/byte")
    print(f"\n  Full results saved to: {LOG}")

# ─────────────────────────────────────────────────────────────
def run_section(name, fn, *args):
    try:
        fn(*args)
    except Exception as e:
        print(f"\n  [SKIP] {name} failed: {e}")
        results.setdefault(name, {"error": str(e)})
        save()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+",
                        choices=["hw1","hw2","hw3","hw4","hw5","hw5b","hw5c","hw5d","hw6"],
                        help="Run only these sections (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Load latest result file and skip already-completed sections")
    cli = parser.parse_args()

    print(f"\nGB10 Hardware Characterization — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    torch = check_cuda()

    # --resume: load the most recently written result file so completed sections aren't re-run
    if cli.resume:
        import glob as _glob
        existing = sorted(_glob.glob(str(RESULTS_DIR / "hw_bench_*.json")))
        if existing:
            with open(existing[-1]) as f:
                loaded = json.load(f)
            for k, v in loaded.items():
                if k in results:
                    results[k] = v
            print(f"  Resumed from: {existing[-1]}")

    def should_run(name):
        if cli.only:
            return name in cli.only
        if cli.resume and results.get(name):
            print(f"  [skip] {name} already has data")
            return False
        return True

    ALL = [
        ("hw1",  hw1_memory_bandwidth,    torch),
        ("hw2",  hw2_compute_tflops,      torch),
        ("hw3",  hw3_nvlink_bandwidth,    torch),
        ("hw4",  hw4_thermal_power,       torch),
        ("hw5",  hw5_roofline,            torch),
        ("hw5b", hw5b_attention_kernels,  torch),
        ("hw5c", hw5c_nvme_io),
        ("hw5d", hw5d_memory_pressure,    torch),
        ("hw6",  hw6_model_bw_efficiency, torch),
    ]
    for name, fn, *args in ALL:
        if should_run(name):
            run_section(name, fn, *args)

    print_summary()

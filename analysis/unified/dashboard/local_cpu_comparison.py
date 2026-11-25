"""Read-only, controlled local Pi CPU vs GPU+H2D comparisons for MobileNetV2.

The CPU data provide CON=1 only. CON > 1 is a *sequential-model estimate*,
not an observation. GPU data are measured group makespans and already include
H2D because the matching GPU records use transfer_mode='h2d'.
Full network-to-GPU E2E is deliberately not inferred.
"""
from pathlib import Path
import numpy as np
import pandas as pd

CONS = (1, 2, 4, 8, 16, 32)
MATCHES = {
    "flatten": ("flatten", "flatten", 5120),
    "dropout": ("dropout", "classifier_0", 5120),
    "linear": ("linear", "classifier_1", 5120),
}


def load_pi_cpu_baselines(root: Path) -> pd.DataFrame:
    """Load existing Raspberry Pi CPU BS=1 CSVs; no benchmark is rerun."""
    root = Path(root)
    columns = ["op", "layer", "input_bytes", "cpu_p50_ms", "cpu_samples", "cpu_csv"]
    if not root.is_dir():
        return pd.DataFrame(columns=columns)
    rows = []
    for op, (expected_op, layer, size) in MATCHES.items():
        matches = sorted(root.glob(f"*_{op}/bs_0001/cpu.csv"))
        valid = []
        for path in matches:
            try:
                d = pd.read_csv(path)
                required = {"device", "layer", "op", "input_bytes", "con", "global_latency_ms"}
                if not required.issubset(d.columns):
                    continue
                d = d[
                    d["device"].eq("pi_cpu") &
                    d["layer"].eq(layer) &
                    d["op"].eq(expected_op) &
                    pd.to_numeric(d["input_bytes"], errors="coerce").eq(size) &
                    pd.to_numeric(d["con"], errors="coerce").eq(1)
                ]
                times = pd.to_numeric(d["global_latency_ms"], errors="coerce")
                times = times[np.isfinite(times) & times.gt(0)]
                if len(times):
                    valid.append((path, times))
            except (OSError, pd.errors.ParserError, ValueError):
                continue
        # Never silently average distinct runs if multiple candidate baselines exist.
        if len(valid) != 1:
            continue
        path, samples = valid[0]
        rows.append({"op": op, "layer": layer, "input_bytes": size,
                     "cpu_p50_ms": float(samples.median()), "cpu_samples": int(len(samples)),
                     "cpu_csv": str(path)})
    return pd.DataFrame(rows, columns=columns)


def build_local_vs_gpu_group(gpu_sources: pd.DataFrame,
                             cpu_baselines: pd.DataFrame,
                             operator: str) -> pd.DataFrame:
    """Measured GPU+H2D group vs CON=1 CPU and labeled sequential extrapolation.

    The matching GPU source must be unique, to avoid combining different layers.
    Missing GPU CON conditions remain NaN; no extrapolation is applied to GPU.
    """
    if operator not in MATCHES:
        raise ValueError("Unsupported operator; choose flatten, dropout or linear")
    op, layer, size = MATCHES[operator]
    base = cpu_baselines[
        cpu_baselines["op"].eq(op) & cpu_baselines["layer"].eq(layer) &
        cpu_baselines["input_bytes"].eq(size)
    ]
    if len(base) != 1:
        raise ValueError(f"No unique BS=1 Raspberry Pi CPU baseline for {operator} / {size} bytes")
    cpu = float(base.iloc[0]["cpu_p50_ms"])
    if not np.isfinite(cpu) or cpu <= 0:
        raise ValueError("CPU baseline is not a positive measured time")
    g = gpu_sources[
        gpu_sources["op"].eq(op) & gpu_sources["layer"].eq(layer) &
        gpu_sources["input_bytes"].eq(size) &
        gpu_sources["con"].isin(CONS) &
        gpu_sources["transfer_mode"].astype(str).str.lower().eq("h2d")
    ].copy()
    if g["source_file"].nunique() > 1:
        raise ValueError("Multiple GPU sources matched; refusing to average different workloads")
    if g.duplicated(["con"]).any():
        raise ValueError("Duplicate CON condition for the selected GPU source")
    out = pd.DataFrame({"CON": list(CONS)})
    out = out.merge(
        g[["con", "group_p50_ms", "group_p95_ms", "complete_repeats", "source_file"]]
        .rename(columns={"con": "CON", "group_p50_ms": "gpu_plus_h2d_group_measured_p50_ms",
                         "group_p95_ms": "gpu_plus_h2d_group_measured_p95_ms",
                         "complete_repeats": "gpu_complete_repeats", "source_file": "gpu_source_file"}),
        on="CON", how="left", validate="one_to_one")
    out["node_a_cpu_group_measured_p50_ms"] = np.where(out.CON.eq(1), cpu, np.nan)
    out["node_a_cpu_sequential_model_ms"] = cpu * out.CON
    out["cpu_group_evidence"] = np.where(out.CON.eq(1), "MEASURED CON=1", "ESTIMATED: CON × CPU P50")
    out["cpu_over_gpu_ratio"] = out.node_a_cpu_sequential_model_ms / out.gpu_plus_h2d_group_measured_p50_ms
    out["ratio_evidence"] = np.where(out.CON.eq(1), "CPU and GPU measured independently",
                                      "CPU sequential estimate / GPU measured")
    return out

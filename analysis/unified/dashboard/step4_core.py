"""GPU Con Bench: read-only analysis of three independently measured stages.

Stage 1: Physical network request latency and ACK-round goodput.
Stage 2: Dedicated H2D batch time and derived effective bandwidth.
Stage 3: Nsight request completion (global_latency_ms), group completion
(group_makespan_ms), with kernel-only timing as optional diagnostic.

Never substitute missing/OOM measurements with zero. Cross-stage sums are
*illustrative service-demand indicators*, not observed end-to-end latency.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

CONS = (1, 2, 4, 8, 16, 32)
MIB = 1024.0 ** 2
NETWORK_METRICS = ("payload_recv_ms", "request_rtt_ms", "host_ready_ms", "validation_ms")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path, low_memory=False)


def require(df: pd.DataFrame, columns, label: str):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label}: missing columns: {', '.join(missing)}")


def numeric(df: pd.DataFrame, *columns):
    d = df.copy()
    for col in columns:
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def payload_label(n: int | float) -> str:
    n = int(n)
    if n >= MIB:
        return f"{n / MIB:g} MiB"
    if n >= 1024:
        return f"{n / 1024:g} KiB"
    return f"{n:,} B"


def network_summary(latency: pd.DataFrame, rounds: pd.DataFrame | None, date: str = "ALL"):
    """Return file-median request latency, and per-file pooled-goodput median.

    Physical unified_latency is file-level summary (not raw request latencies).
    Aggregating medians across files is deliberate; it is *not* a pooled P50.
    """
    require(latency, ["dataset", "metric", "payload_bytes", "concurrency", "p50_ms", "source_file"], "unified_latency.csv")
    l = numeric(latency, "payload_bytes", "concurrency", "p50_ms", "p95_ms")
    l = l[l.dataset.astype(str).eq("physical") & l.metric.astype(str).isin(NETWORK_METRICS)]
    if date != "ALL":
        l = l[l.source_file.astype(str).str.startswith(date + "_")]
    l = l[l.payload_bytes.gt(0) & l.concurrency.isin(CONS)].copy()
    agg = {"p50_ms": ("p50_ms", "median"), "file_rows": ("p50_ms", "count")}
    if "p95_ms" in l.columns:
        agg["p95_ms"] = ("p95_ms", "median")
    summary = l.groupby(["payload_bytes", "concurrency", "metric"], as_index=False).agg(**agg)
    if "p95_ms" not in summary:
        summary["p95_ms"] = np.nan

    base_columns = ["payload_bytes", "concurrency", "goodput_mib_s", "files"]
    if rounds is None or rounds.empty:
        return summary, pd.DataFrame(columns=base_columns)
    require(rounds, ["source_file", "payload_bytes", "network_con", "total_requests", "batch_elapsed_ms", "valid"], "physical_rounds.csv")
    r = numeric(rounds, "payload_bytes", "network_con", "total_requests", "batch_elapsed_ms")
    r = r[r.valid.astype(str).str.lower().eq("true")]
    if date != "ALL":
        r = r[r.source_file.astype(str).str.startswith(date + "_")]
    r = r[r.payload_bytes.gt(0) & r.total_requests.gt(0) & r.batch_elapsed_ms.gt(0) & r.network_con.isin(CONS)].copy()
    if r.empty:
        return summary, pd.DataFrame(columns=base_columns)
    r["transferred_bytes"] = r["payload_bytes"] * r["total_requests"]
    per_file = r.groupby(["payload_bytes", "network_con", "source_file"], as_index=False).agg(
        transferred_bytes=("transferred_bytes", "sum"), elapsed_ms=("batch_elapsed_ms", "sum"))
    per_file["goodput_mib_s"] = per_file.transferred_bytes / MIB / (per_file.elapsed_ms / 1000)
    goodput = per_file.groupby(["payload_bytes", "network_con"], as_index=False).agg(
        goodput_mib_s=("goodput_mib_s", "median"), files=("source_file", "nunique"))
    return summary, goodput.rename(columns={"network_con": "concurrency"})


def h2d_summary(summary: pd.DataFrame | None, coverage: pd.DataFrame | None) -> pd.DataFrame:
    """One audited record per exact (payload bytes, CON); no fallback matching."""
    cols = ["payload_bytes", "concurrency", "gpu_p50_ms", "gpu_p95_ms", "host_p50_ms",
            "host_p95_ms", "amortized_ms_per_request", "goodput_mib_s", "requests_per_s",
            "n_rounds", "status"]
    if summary is None or summary.empty:
        return pd.DataFrame(columns=cols)
    require(summary, ["payload_bytes", "con", "gpu_p50_ms", "gpu_p95_ms",
                      "host_p50_ms", "effective_gib_s_from_gpu_p50", "n_rounds"],
            "condition_summary.csv")
    s = numeric(summary, "payload_bytes", "con", "gpu_p50_ms", "gpu_p95_ms", "host_p50_ms",
                "host_p95_ms", "effective_gib_s_from_gpu_p50", "n_rounds")
    s = s[s.payload_bytes.gt(0) & s.con.isin(CONS)].copy()
    cov = None
    if coverage is not None and not coverage.empty:
        require(coverage, ["payload_bytes", "con", "status"], "coverage_matrix.csv")
        cov = numeric(coverage, "payload_bytes", "con")
        cov = cov[cov.payload_bytes.gt(0) & cov.con.isin(CONS)]

    rows = []
    keys = set(zip(s.payload_bytes.astype('int64'), s.con.astype('int64')))
    if cov is not None:
        keys.update(zip(cov.payload_bytes.astype('int64'), cov.con.astype('int64')))
    for payload, con in sorted(keys):
        candidates = s[s.payload_bytes.eq(payload) & s.con.eq(con)]
        status = "MEASURED" if len(candidates) == 1 else ("MISSING" if len(candidates) == 0 else "DUPLICATE")
        if cov is not None:
            cv = cov[cov.payload_bytes.eq(payload) & cov.con.eq(con)]
            if len(cv) != 1:
                status = "COVERAGE_AMBIGUOUS"
            elif str(cv.iloc[0].status).strip().lower() != "measured":
                status = str(cv.iloc[0].status).upper()
        measured = candidates.iloc[0] if status == "MEASURED" else None
        def val(name):
            return float(measured[name]) if measured is not None and name in measured and pd.notna(measured[name]) else np.nan
        batch = val("gpu_p50_ms")
        goodput = val("effective_gib_s_from_gpu_p50") * 1024
        rows.append(dict(
            payload_bytes=int(payload), concurrency=int(con), gpu_p50_ms=batch,
            gpu_p95_ms=val("gpu_p95_ms"), host_p50_ms=val("host_p50_ms"),
            host_p95_ms=val("host_p95_ms"),
            amortized_ms_per_request=(batch / con if batch > 0 else np.nan),
            goodput_mib_s=(goodput if goodput > 0 else np.nan),
            requests_per_s=(1000 * con / batch if batch > 0 else np.nan),
            n_rounds=int(val("n_rounds")) if pd.notna(val("n_rounds")) else 0,
            status=status))
    return pd.DataFrame(rows, columns=cols)


def network_batch_summary(rounds: pd.DataFrame | None, date: str = "ALL") -> pd.DataFrame:
    """Batch completion for exactly CON physical-network requests in a round.

    Each file contributes its own median. Mismatched/incomplete rounds are
    excluded rather than silently compared to CON-sized H2D/GPU groups.
    """
    columns = ["payload_bytes", "concurrency", "batch_p50_ms", "batch_p95_ms", "files", "rounds"]
    if rounds is None or rounds.empty:
        return pd.DataFrame(columns=columns)
    require(rounds, ["source_file", "payload_bytes", "network_con", "total_requests",
                     "batch_elapsed_ms", "valid"], "physical_rounds.csv")
    r = numeric(rounds, "payload_bytes", "network_con", "total_requests", "batch_elapsed_ms")
    r = r[r.valid.astype(str).str.lower().eq("true") &
          r.network_con.isin(CONS) & r.payload_bytes.gt(0) &
          r.total_requests.eq(r.network_con) & r.batch_elapsed_ms.gt(0)].copy()
    if date != "ALL":
        r = r[r.source_file.astype(str).str.startswith(date + "_")]
    if r.empty:
        return pd.DataFrame(columns=columns)
    by_file = r.groupby(["payload_bytes", "network_con", "source_file"], as_index=False).agg(
        batch_p50_ms=("batch_elapsed_ms", "median"),
        batch_p95_ms=("batch_elapsed_ms", lambda x: x.quantile(.95)),
        rounds=("batch_elapsed_ms", "size"))
    agg = by_file.groupby(["payload_bytes", "network_con"], as_index=False).agg(
        batch_p50_ms=("batch_p50_ms", "median"),
        batch_p95_ms=("batch_p95_ms", "median"),
        files=("source_file", "nunique"), rounds=("rounds", "sum"))
    return agg.rename(columns={"network_con": "concurrency"})[columns]


def gpu_summary(requests: pd.DataFrame):
    """Summarize *full request and group completion*, not just GPU kernels.

    global_latency_ms starts at the shared CON-group origin and ends when the
    corresponding request completes. group_makespan_ms ends when the final
    request in that CON group finishes. We count each (source,CON,repeat)
    only once for batch statistics, not once per request.
    """
    required = ["source_file", "op", "layer", "input_bytes", "con", "repeat",
                "request_id", "global_latency_ms", "group_makespan_ms"]
    require(requests, required, "nsys_gpu_requests.csv")
    g = numeric(requests, "input_bytes", "con", "repeat", "request_id",
                "global_latency_ms", "group_makespan_ms", "kernel_count",
                "kernel_exec_ms", "throughput_group_per_s")
    g = g[g.con.isin(CONS) & g.input_bytes.gt(0) &
          g.global_latency_ms.notna() & g.global_latency_ms.gt(0)].copy()
    source_key = ["source_file", "op", "layer", "input_bytes", "con"]
    group_key = source_key + ["repeat"]
    cols = source_key + ["request_p50_ms", "request_p95_ms", "group_p50_ms",
                         "group_p95_ms", "kernel_p50_ms", "kernel_p95_ms",
                         "gpu_group_requests_s", "requests", "complete_repeats", "transfer_mode"]
    slow_cols = ["op", "concurrency", "median_request_slowdown_x",
                 "median_group_slowdown_x", "paired_sources"]
    if g.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=slow_cols)

    # Only CON complete unique requests contribute to the group completion series.
    grouped = g.groupby(group_key, dropna=False, as_index=False).agg(
        n_rows=("request_id", "size"), n_unique=("request_id", "nunique"),
        group_ms=("group_makespan_ms", "median"),
        group_min_ms=("group_makespan_ms", "min"),
        group_max_ms=("group_makespan_ms", "max"))
    grouped = grouped[grouped.n_rows.eq(grouped.con) &
                      grouped.n_unique.eq(grouped.con) &
                      grouped.group_ms.gt(0) &
                      (grouped.group_max_ms - grouped.group_min_ms).abs().le(1e-5)].copy()
    # Match whole repeat to all of its request rows; reject partial repeats.
    g = g.merge(grouped[group_key], on=group_key, how="inner", validate="many_to_one")
    if g.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=slow_cols)

    # Per-request global completion (includes time before CUDA activity starts).
    req = g.groupby(source_key, dropna=False, as_index=False).agg(
        request_p50_ms=("global_latency_ms", "median"),
        request_p95_ms=("global_latency_ms", lambda x: x.quantile(.95)),
        requests=("global_latency_ms", "size"))
    # Per-repeat group finish time, counted once per repeat.
    grp = grouped.groupby(source_key, dropna=False, as_index=False).agg(
        group_p50_ms=("group_ms", "median"),
        group_p95_ms=("group_ms", lambda x: x.quantile(.95)),
        complete_repeats=("group_ms", "size"))
    grp["gpu_group_requests_s"] = np.where(
        grp.group_p50_ms.gt(0), grp.con * 1000.0 / grp.group_p50_ms, np.nan)
    out = req.merge(grp, on=source_key, validate="one_to_one")
    if "transfer_mode" in g.columns:
        modes = g.groupby(source_key, dropna=False, as_index=False).agg(
            transfer_mode=("transfer_mode", lambda x: "|".join(sorted({str(v).strip().lower() for v in x if pd.notna(v)}))))
        out = out.merge(modes, on=source_key, how="left", validate="one_to_one")
    else:
        out["transfer_mode"] = "unknown"
    out["transfer_mode"] = out["transfer_mode"].replace("", "unknown")
    if {"kernel_exec_ms", "kernel_count"}.issubset(g.columns):
        k = g[g.kernel_count.gt(0) & g.kernel_exec_ms.notna()]
        if not k.empty:
            kd = k.groupby(source_key, dropna=False, as_index=False).agg(
                kernel_p50_ms=("kernel_exec_ms", "median"),
                kernel_p95_ms=("kernel_exec_ms", lambda x: x.quantile(.95)))
            out = out.merge(kd, on=source_key, how="left", validate="one_to_one")
    for name in ["kernel_p50_ms", "kernel_p95_ms"]:
        if name not in out:
            out[name] = np.nan
    out = out[cols]

    # Pair each source with its own CON=1 baseline to avoid mixing workloads.
    keys = ["source_file", "op", "layer", "input_bytes"]
    base = out[out.con.eq(1)][keys + ["request_p50_ms", "group_p50_ms"]].rename(
        columns={"request_p50_ms": "req_base", "group_p50_ms": "group_base"})
    paired = out.merge(base, on=keys, how="left", validate="many_to_one")
    paired["req_slowdown"] = np.where(paired.req_base.gt(0),
                                       paired.request_p50_ms / paired.req_base, np.nan)
    paired["grp_slowdown"] = np.where(paired.group_base.gt(0),
                                       paired.group_p50_ms / paired.group_base, np.nan)
    slow = paired.groupby(["op", "con"], as_index=False).agg(
        median_request_slowdown_x=("req_slowdown", "median"),
        median_group_slowdown_x=("grp_slowdown", "median"),
        paired_sources=("req_slowdown", "count")).rename(columns={"con": "concurrency"})
    return out, slow


def gpu_size_summary(sources: pd.DataFrame, operator: str = "ALL") -> pd.DataFrame:
    """Input bytes × CON; each source contributes exactly one source-level summary.

    The plotted P50 at each size is median(source-level P50), NOT pooled
    request quantile. Operators with the same bytes may have different work.
    """
    cols = ["input_bytes", "con", "request_p50_ms", "request_p95_ms",
            "group_p50_ms", "group_p95_ms", "kernel_p50_ms", "kernel_p95_ms",
            "gpu_group_requests_s", "source_files", "operators", "requests",
            "complete_repeats", "paired_sources", "source_paired_request_slowdown_x",
            "source_paired_group_slowdown_x", "transfer_modes"]
    if sources is None or sources.empty:
        return pd.DataFrame(columns=cols)
    d = sources.copy()
    if operator != "ALL":
        d = d[d.op.astype(str).eq(operator)].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = numeric(d, "input_bytes", "con", "request_p50_ms", "group_p50_ms", "kernel_p50_ms")
    keys = ["source_file", "op", "layer", "input_bytes"]
    base = d[d.con.eq(1)][keys + ["request_p50_ms", "group_p50_ms"]].rename(
        columns={"request_p50_ms": "base_request_p50_ms", "group_p50_ms": "base_group_p50_ms"})
    d = d.merge(base, on=keys, how="left", validate="many_to_one")
    d["req_slow"] = np.where(d.base_request_p50_ms.gt(0),
                              d.request_p50_ms / d.base_request_p50_ms, np.nan)
    d["group_slow"] = np.where(d.base_group_p50_ms.gt(0),
                                d.group_p50_ms / d.base_group_p50_ms, np.nan)
    result = d.groupby(["input_bytes", "con"], as_index=False).agg(
        request_p50_ms=("request_p50_ms", "median"),
        request_p95_ms=("request_p95_ms", "median"),
        group_p50_ms=("group_p50_ms", "median"),
        group_p95_ms=("group_p95_ms", "median"),
        kernel_p50_ms=("kernel_p50_ms", "median"),
        kernel_p95_ms=("kernel_p95_ms", "median"),
        gpu_group_requests_s=("gpu_group_requests_s", "median"),
        source_files=("source_file", "nunique"),
        operators=("op", "nunique"), requests=("requests", "sum"),
        complete_repeats=("complete_repeats", "sum"),
        paired_sources=("req_slow", "count"),
        source_paired_request_slowdown_x=("req_slow", "median"),
        source_paired_group_slowdown_x=("group_slow", "median"),
        transfer_modes=("transfer_mode", lambda x: "|".join(sorted({str(v) for v in x if pd.notna(v)}))))
    return result.sort_values(["input_bytes", "con"])[cols].reset_index(drop=True)


def combined_sweep(net: pd.DataFrame, network_rates: pd.DataFrame, h2d: pd.DataFrame,
                   gpu_sizes: pd.DataFrame, *, payload: int, gpu_input_bytes: int,
                   network_batches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compare independently measured group completions across the same CON sweep.

    The sum is a *hypothetical sum of standalone stage medians* and MUST NOT
    be interpreted as measured pipeline end-to-end latency or P50.
    """
    net_receive = net[net.payload_bytes.eq(payload) & net.metric.eq("payload_recv_ms")]
    net_rtt = net[net.payload_bytes.eq(payload) & net.metric.eq("request_rtt_ms")]
    nr = network_rates[network_rates.payload_bytes.eq(payload)] if not network_rates.empty else network_rates
    h = h2d[h2d.payload_bytes.eq(payload)] if not h2d.empty else h2d
    g = gpu_sizes[gpu_sizes.input_bytes.eq(gpu_input_bytes)]
    nb = network_batches[network_batches.payload_bytes.eq(payload)] if network_batches is not None and not network_batches.empty else None

    def single(frame, col, c):
        if frame is None or frame.empty or col not in frame.columns or c not in frame.columns:
            return np.nan
        sub = frame.loc[frame[c].eq(con), col]
        return float(sub.iloc[0]) if len(sub) == 1 and pd.notna(sub.iloc[0]) else np.nan

    rows = []
    for con in CONS:
        net_recv = single(net_receive, "p50_ms", "concurrency")
        net_req = single(net_rtt, "p50_ms", "concurrency")
        net_group = single(nb, "batch_p50_ms", "concurrency")
        net_mib = single(nr, "goodput_mib_s", "concurrency")
        h2d_group = single(h, "host_p50_ms", "concurrency")
        h2d_device = single(h, "gpu_p50_ms", "concurrency")
        gpu_req = single(g, "request_p50_ms", "con")
        gpu_group = single(g, "group_p50_ms", "con")
        gpu_kernel = single(g, "kernel_p50_ms", "con")
        gpu_rate = single(g, "gpu_group_requests_s", "con")
        gpu_sources = single(g, "source_files", "con")
        gpu_req_slow = single(g, "source_paired_request_slowdown_x", "con")
        gpu_group_slow = single(g, "source_paired_group_slowdown_x", "con")
        mode_rows = g.loc[g.con.eq(con), "transfer_modes"] if "transfer_modes" in g else pd.Series(dtype=str)
        gpu_modes = str(mode_rows.iloc[0]) if len(mode_rows) == 1 and pd.notna(mode_rows.iloc[0]) else "unknown"
        # Avoid adding an independently measured H2D stage on top of GPU timings
        # that already include the H2D transfer, or whose transfer mode is unknown.
        gpu_preloaded = (gpu_modes == "preloaded")
        parts = [net_group, h2d_group, gpu_group]
        ready = gpu_preloaded and all(pd.notna(x) and x > 0 for x in parts)
        net_rate = net_mib * MIB / payload if net_mib > 0 else np.nan
        h2d_rate = 1000 * con / h2d_group if h2d_group > 0 else np.nan
        rates = [net_rate, h2d_rate, gpu_rate]
        ready_rates = all(pd.notna(x) and x > 0 for x in rates)
        names = ["Network", "Host to GPU", "GPU"]
        rows.append(dict(CON=con, network_h2d_payload_bytes=payload,
                         gpu_input_bytes=gpu_input_bytes, gpu_sources=gpu_sources,
                         gpu_transfer_modes=gpu_modes, gpu_preloaded=gpu_preloaded,
                         network_request_receive_p50_ms=net_recv,
                         network_request_rtt_p50_ms=net_req,
                         network_group_p50_ms=net_group,
                         network_goodput_mib_s=net_mib, network_requests_s=net_rate,
                         h2d_batch_host_p50_ms=h2d_group,
                         h2d_batch_gpu_p50_ms=h2d_device,
                         h2d_requests_s=h2d_rate,
                         gpu_request_p50_ms=gpu_req,
                         gpu_group_p50_ms=gpu_group,
                         gpu_kernel_p50_ms=gpu_kernel,
                         gpu_group_requests_s=gpu_rate,
                         gpu_request_slowdown_x=gpu_req_slow,
                         gpu_group_slowdown_x=gpu_group_slow,
                         independent_group_time_sum_ms=(sum(parts) if ready else np.nan),
                         largest_standalone_group_time=(names[int(np.argmax(parts))] if ready else "N/A"),
                         lowest_standalone_rate=(names[int(np.argmin(rates))] if ready_rates else "N/A"),
                         complete_group_times=ready, complete_stage_rates=ready_rates))
    result = pd.DataFrame(rows)
    for col, name in [("network_group_p50_ms", "network_group_slowdown_x"),
                      ("h2d_batch_host_p50_ms", "h2d_group_slowdown_x")]:
        baseline = result.loc[result.CON.eq(1), col].iloc[0]
        result[name] = result[col] / baseline if pd.notna(baseline) and baseline > 0 else np.nan
    return result


# ---- Node A local-only vs. offloading comparison (Section 4) ----------------
# Independent Stage 1/2/3 benchmarks cannot establish measured end-to-end
# latency. The measured Node A / offloaded E2E values must be user-supplied.

BASELINE_COLUMNS = ["network_payload_bytes", "gpu_input_bytes", "operator", "CON", "node_a_local_group_p50_ms"]


def prepare_node_a_comparison_input(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Validate an optional, measured Node A / offload E2E group timing CSV.

    A row describes an EXACT benchmark workload selection and CON condition.
    Missing cells are permitted; missing or nonpositive times are not used.
    Duplicated workload+CON keys are rejected rather than silently averaged.
    """
    columns = BASELINE_COLUMNS + ["offload_e2e_group_p50_ms"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    require(frame, BASELINE_COLUMNS, "Node A local comparison CSV")
    d = frame.copy()
    if "offload_e2e_group_p50_ms" not in d:
        d["offload_e2e_group_p50_ms"] = np.nan
    d = numeric(d, "network_payload_bytes", "gpu_input_bytes", "CON",
                "node_a_local_group_p50_ms", "offload_e2e_group_p50_ms")
    d["operator"] = d["operator"].astype("string").str.strip().str.lower()
    if d["operator"].isna().any() or d["operator"].eq("").any():
        raise ValueError("Node A baseline: operator must be specified (e.g. conv2d or ALL).")
    keys = ["network_payload_bytes", "gpu_input_bytes", "operator", "CON"]
    if d[keys].isna().any().any() or not d["CON"].isin(CONS).all() or \
       not d["network_payload_bytes"].gt(0).all() or not d["gpu_input_bytes"].gt(0).all():
        raise ValueError("Node A baseline: invalid sizes or CON; use measured positive sizes and CON 1/2/4/8/16/32.")
    if not d["network_payload_bytes"].eq(d["network_payload_bytes"].astype("int64")).all() or \
       not d["gpu_input_bytes"].eq(d["gpu_input_bytes"].astype("int64")).all():
        raise ValueError("Node A baseline: sizes must be integer byte counts.")
    d["network_payload_bytes"] = d["network_payload_bytes"].astype("int64")
    d["gpu_input_bytes"] = d["gpu_input_bytes"].astype("int64")
    d["CON"] = d["CON"].astype("int64")
    if d.duplicated(keys).any():
        raise ValueError("Node A baseline: duplicate (network bytes, GPU bytes, operator, CON) rows.")
    for value in ["node_a_local_group_p50_ms", "offload_e2e_group_p50_ms"]:
        bad = d[value].notna() & (~np.isfinite(d[value]) | ~d[value].gt(0))
        if bad.any():
            raise ValueError(f"Node A baseline: {value} must be a positive measured duration or blank.")
    return d[columns].sort_values(keys).reset_index(drop=True)


def compare_node_a_vs_offload(combo: pd.DataFrame,
                              baseline: pd.DataFrame | None, *,
                              operator: str) -> pd.DataFrame:
    """Return CON sweep with *measured* E2E if available, or a labeled proxy.

    Stage proxy (independent P50s, NOT actual E2E):
        GPU transfer_mode == 'h2d': Network batch + GPU group (GPU includes H2D)
        GPU transfer_mode == 'preloaded': Network + H2D + GPU group
        unknown or mixed mode: no proxy (do not risk transfer double count).

    Optional true offload E2E P50 and local Node A P50 come exclusively from
    user-provided observations. No missing values are fabricated.
    """
    df = combo.copy()
    b = prepare_node_a_comparison_input(baseline)
    b = b[b.operator.eq(str(operator).lower())].copy()
    # The combo belongs to one exact size selection, and one operator filter.
    if not df.empty:
        size_net = int(df.network_h2d_payload_bytes.iloc[0])
        size_gpu = int(df.gpu_input_bytes.iloc[0])
        b = b[b.network_payload_bytes.eq(size_net) & b.gpu_input_bytes.eq(size_gpu)]
    df = df.merge(b[["CON", "node_a_local_group_p50_ms", "offload_e2e_group_p50_ms"]],
                  on="CON", how="left", validate="one_to_one")
    proxy = []
    basis = []
    for row in df.itertuples():
        mode = str(row.gpu_transfer_modes).strip().lower()
        net = row.network_group_p50_ms
        gpu = row.gpu_group_p50_ms
        h2d = row.h2d_batch_host_p50_ms
        network_ok = pd.notna(net) and np.isfinite(net) and net > 0
        gpu_ok = pd.notna(gpu) and np.isfinite(gpu) and gpu > 0
        h2d_ok = pd.notna(h2d) and np.isfinite(h2d) and h2d > 0
        if mode == "h2d" and network_ok and gpu_ok:
            proxy.append(float(net + gpu))
            basis.append("Network + GPU group (GPU includes H2D)")
        elif mode == "preloaded" and network_ok and h2d_ok and gpu_ok:
            proxy.append(float(net + h2d + gpu))
            basis.append("Network + standalone H2D + preloaded GPU group")
        else:
            proxy.append(np.nan)
            basis.append("Unavailable: missing times or unknown/mixed GPU transfer mode")
    df["offload_standalone_proxy_ms"] = proxy
    df["offload_proxy_components"] = basis
    df["offload_comparison_ms"] = np.where(
        df["offload_e2e_group_p50_ms"].notna(),
        df["offload_e2e_group_p50_ms"], df["offload_standalone_proxy_ms"])
    df["offload_comparison_kind"] = np.select(
        [df.offload_e2e_group_p50_ms.notna(), df.offload_standalone_proxy_ms.notna()],
        ["MEASURED E2E", "ESTIMATED FROM STANDALONE STAGES"], default="UNAVAILABLE")
    valid = (df.node_a_local_group_p50_ms.gt(0) & df.offload_comparison_ms.gt(0))
    df["node_a_vs_offload_speedup_x"] = np.where(
        valid, df.node_a_local_group_p50_ms / df.offload_comparison_ms, np.nan)
    df["node_a_vs_offload_time_saved_pct"] = np.where(
        valid, 100. * (df.node_a_local_group_p50_ms - df.offload_comparison_ms) /
        df.node_a_local_group_p50_ms, np.nan)
    return df

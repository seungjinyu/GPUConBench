"""GPU Con Bench — English dashboard for existing Network, H2D, and GPU data.

Read-only. Run from ~/GPUConBench using the existing dashboard conda environment.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pi_gpu_section4 import render_pi_gpu_section4
from local_cpu_comparison import MATCHES, load_pi_cpu_baselines, build_local_vs_gpu_group
from stage_diagnostics import (NETWORK_ORDER, NETWORK_LABELS, GPU_ORDER, GPU_LABELS,
                               network_details, network_metric_pivot, h2d_details,
                               h2d_request_details, gpu_details, physical_request_details)
from step4_core import (CONS, read_csv, network_summary, h2d_summary,
                        gpu_summary, gpu_size_summary, network_batch_summary, combined_sweep, payload_label,
                        prepare_node_a_comparison_input, compare_node_a_vs_offload)

st.set_page_config(page_title="GPU Con Bench", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

HERE = Path(__file__).resolve()
UNIFIED = HERE.parents[1]  # ~/GPUConBench/analysis/unified
ANALYSIS = UNIFIED.parent
PATHS = {
    "Network latency": UNIFIED / "integrated" / "unified_latency.csv",
    "Network rounds": UNIFIED / "normalized" / "physical_rounds.csv",
    "Network request diagnostics": UNIFIED / "normalized" / "physical_requests.csv",
    "H2D conditions": ANALYSIS / "h2d_step2" / "results" / "merged_step2" / "condition_summary.csv",
    "H2D coverage": ANALYSIS / "h2d_step2" / "results" / "merged_step2" / "coverage_matrix.csv",
    "GPU Nsight requests": UNIFIED / "normalized" / "nsys_gpu_requests.csv",
    "H2D request diagnostics": ANALYSIS / "h2d_step2" / "h2d_request_summary.csv",
}
COLOR = {"Network": "#4c98e0", "Host to GPU": "#18a887", "GPU": "#b48be6"}
PI_CPU_ROOT = Path.home() / "linear_test" / "edge_gpu_profiler_v3_20261019_1557" / "results" / "integrated_fx320_all" / "cases"


def snapshot():
    return tuple((str(path), path.stat().st_mtime_ns if path.is_file() else -1)
                 for path in PATHS.values())


@st.cache_data(show_spinner="Loading existing experiment data...")
def load_all(file_signature):
    path = {name: p for name, p in PATHS.items()}
    def optional(key):
        return read_csv(path[key]) if path[key].is_file() else None
    lat = read_csv(path["Network latency"])
    gpu_raw = read_csv(path["GPU Nsight requests"])
    h2d = h2d_summary(optional("H2D conditions"), optional("H2D coverage"))
    h2d_request = h2d_request_details(optional("H2D request diagnostics"))
    net_phase = physical_request_details(optional("Network request diagnostics"))
    gpu_diag = gpu_details(gpu_raw)
    gpu, op = gpu_summary(gpu_raw)
    rounds = optional("Network rounds")
    del gpu_raw
    return lat, rounds, h2d, gpu, op, h2d_request, gpu_diag, net_phase


def selected_data_frame(df, con_field="concurrency"):
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df[con_field].isin(CONS)].sort_values(con_field)


def line_chart(frame: pd.DataFrame, series, title: str, y_title: str, height=350):
    if frame is None or frame.empty:
        st.info("No measurements are available for this selection.")
        return
    fig = go.Figure()
    plotted = 0
    for col, name, color in series:
        if col not in frame or not frame[col].notna().any():
            continue
        fig.add_trace(go.Scatter(
            x=frame["CON"].astype(int).astype(str), y=frame[col], mode="lines+markers",
            name=name, connectgaps=False, line=dict(color=color, width=2.5),
            marker=dict(size=7), customdata=frame["CON"],
            hovertemplate="CON %{customdata}<br>%{y:,.4f}<extra>" + name + "</extra>"))
        plotted += 1
    if not plotted:
        st.info("No measurements are available for this selection.")
        return
    fig.update_layout(title=title, height=height,
        xaxis=dict(title="Concurrent requests (CON)", type="category",
                   categoryorder="array", categoryarray=[str(c) for c in CONS]),
        yaxis=dict(title=y_title, rangemode="tozero"),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified", margin=dict(l=8, r=8, t=78, b=24))
    st.plotly_chart(fig, use_container_width=True)


def heatmap(frame: pd.DataFrame, *, row: str, col: str, value: str,
            row_formatter, title: str, colorbar_title: str, max_height=510):
    if frame is None or frame.empty or value not in frame:
        st.info("No data available for this heatmap.")
        return
    keep = frame[[row, col, value]].copy()
    keep[col] = pd.to_numeric(keep[col], errors="coerce")
    keep[value] = pd.to_numeric(keep[value], errors="coerce")
    keep = keep.dropna(subset=[row, col]).drop_duplicates([row, col])
    if keep.empty:
        st.info("No data available for this heatmap.")
        return
    pivot = keep.pivot(index=row, columns=col, values=value).sort_index()
    col_order = [c for c in CONS if c in pivot.columns]
    pivot = pivot.reindex(columns=col_order)
    labels = [row_formatter(x) for x in pivot.index]
    z = pivot.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        x=[str(c) for c in col_order], y=labels, z=z, colorscale="Viridis",
        colorbar=dict(title=colorbar_title), hoverongaps=False,
        hovertemplate="%{y}<br>CON %{x}<br>%{z:,.4f}<extra></extra>"))
    fig.update_layout(title=title, height=min(max_height, max(310, 28 * len(labels) + 140)),
                      xaxis_title="Concurrent requests (CON)", yaxis_title="Workload",
                      margin=dict(l=8, r=8, t=56, b=25))
    fig.update_yaxes(autorange="reversed", type="category")
    st.plotly_chart(fig, use_container_width=True)


def size_curve(frame: pd.DataFrame, *, size_col: str, con_col: str, value_col: str,
               title: str, y_title: str, height=370):
    """Show the measured data-size trend at EACH concurrency (no interpolation)."""
    if frame is None or frame.empty or value_col not in frame:
        st.info("No measurements are available for this size sweep.")
        return
    fig = go.Figure()
    for c in CONS:
        sub = frame[frame[con_col].eq(c)].dropna(subset=[size_col, value_col]).sort_values(size_col)
        sub = sub[sub[size_col] > 0]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[size_col], y=sub[value_col], mode="lines+markers", name=f"CON {c}",
            customdata=[payload_label(v) for v in sub[size_col]],
            hovertemplate="Input size %{customdata}<br>%{y:,.4f} ms<extra>CON " + str(c) + "</extra>"))
    if not fig.data:
        st.info("No measurements are available for this size sweep.")
        return
    unique_sizes = sorted(int(v) for v in frame[size_col].dropna().unique() if v > 0)
    # Avoid cluttering the x axis with hundreds of source-specific input sizes.
    ticks = unique_sizes if len(unique_sizes) <= 12 else [unique_sizes[i] for i in
          sorted({round(i * (len(unique_sizes) - 1) / 11) for i in range(12)})]
    fig.update_layout(title=title, height=height,
        xaxis=dict(title="Measured bytes per request", type="log", tickvals=ticks,
                   ticktext=[payload_label(v) for v in ticks]),
        yaxis=dict(title=y_title, rangemode="tozero"),
        legend=dict(orientation="h", y=1.15, x=0),
        hovermode="closest", margin=dict(l=8, r=8, t=79, b=25))
    st.plotly_chart(fig, use_container_width=True)


def stacked_illustration(df: pd.DataFrame):
    complete = df[df.complete_group_times].copy()
    if complete.empty:
        st.info("A complete three-stage timing comparison is not available for these conditions.")
        return
    fig = go.Figure()
    for col, stage in [("network_group_p50_ms", "Network"),
                       ("h2d_batch_host_p50_ms", "Host to GPU"),
                       ("gpu_group_p50_ms", "GPU")]:
        fig.add_bar(x=complete.CON.astype(int).astype(str), y=complete[col],
                    name=stage, marker_color=COLOR[stage],
                    hovertemplate="CON %{x}<br>%{y:,.4f} ms<extra>" + stage + "</extra>")
    fig.update_layout(barmode="stack", title="Sum of independent group completion P50 values — NOT measured end-to-end",
                      xaxis=dict(title="Concurrent requests (CON)", type="category"),
                      yaxis_title="Independent group completion time (ms)", height=370,
                      legend=dict(orientation="h", y=1.14),
                      margin=dict(l=8, r=8, t=75, b=25))
    st.plotly_chart(fig, use_container_width=True)



def local_vs_offload_chart(comp: pd.DataFrame):
    """Two scenario totals with measured and proxy offload shown separately."""
    fig = go.Figure()
    x = comp.CON.astype(int).astype(str)
    fig.add_trace(go.Scatter(
        x=x, y=comp.node_a_local_group_p50_ms, mode="lines+markers",
        name="Node A local only (measured)", connectgaps=False,
        line=dict(color="#3a82b6", width=3), marker=dict(size=8),
        hovertemplate="CON %{x}<br>Local %{y:,.3f} ms<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=comp.offload_e2e_group_p50_ms, mode="lines+markers",
        name="Offloading end-to-end (measured)", connectgaps=False,
        line=dict(color="#18a887", width=3), marker=dict(size=8),
        hovertemplate="CON %{x}<br>Measured offload %{y:,.3f} ms<extra></extra>"))
    # Never join an estimated point to a measured E2E point: different evidence.
    proxy = comp.offload_standalone_proxy_ms.where(comp.offload_e2e_group_p50_ms.isna())
    fig.add_trace(go.Scatter(
        x=x, y=proxy, mode="lines+markers",
        name="Offloading sum of standalone stages (estimate)", connectgaps=False,
        line=dict(color="#df9d3d", width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="CON %{x}<br>Stage estimate %{y:,.3f} ms<extra>Not measured E2E</extra>"))
    fig.update_layout(
        title="Node A local vs offloading — total CON-group completion",
        height=430, xaxis=dict(title="Concurrent requests (CON)", type="category",
                              categoryorder="array", categoryarray=[str(c) for c in CONS]),
        yaxis=dict(title="Completion time (ms)", rangemode="tozero"),
        legend=dict(orientation="h", y=1.22, x=0),
        hovermode="x unified", margin=dict(l=8, r=8, t=100, b=24))
    st.plotly_chart(fig, use_container_width=True)


def local_vs_offload_speedup_chart(comp: pd.DataFrame):
    d = comp[comp.node_a_vs_offload_speedup_x.notna()]
    if d.empty:
        st.info("Speedup appears after a measured Node A local-only baseline is supplied for this workload.")
        return
    fig = go.Figure()
    for kind, label, color, dash in [
        ("MEASURED E2E", "Speedup vs measured offloading", "#18a887", "solid"),
        ("ESTIMATED FROM STANDALONE STAGES", "Speedup vs estimated offloading", "#df9d3d", "dash")
    ]:
        vals = comp.node_a_vs_offload_speedup_x.where(comp.offload_comparison_kind.eq(kind))
        if not vals.notna().any():
            continue
        fig.add_trace(go.Scatter(x=comp.CON.astype(int).astype(str), y=vals,
                       name=label, mode="lines+markers", connectgaps=False,
                       line=dict(color=color, width=2.5, dash=dash),
                       hovertemplate="CON %{x}<br>Speedup %{y:,.3f}×<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color="gray", dash="dot"),
                  annotation_text="1× = no speedup", annotation_position="top left")
    fig.update_layout(title="Node A / offload completion ratio", height=355,
                      xaxis=dict(title="Concurrent requests (CON)", type="category",
                                 categoryorder="array", categoryarray=[str(c) for c in CONS]),
                      yaxis=dict(title="Speedup (×)"), legend=dict(orientation="h", y=1.2),
                      margin=dict(l=8, r=8, t=89, b=24))
    st.plotly_chart(fig, use_container_width=True)

st.title("GPU Con Bench")
st.caption("Existing experiments • Network → Host to GPU → GPU request and group completion")
st.markdown("**Concurrency sweep:** CON = 1, 2, 4, 8, 16, 32. "
            "Every section uses these same concurrency levels; payload size remains a per-request workload setting.")

critical = [PATHS["Network latency"], PATHS["GPU Nsight requests"]]
missing = [p for p in critical if not p.is_file()]
if missing:
    st.error("Required experiment files are not available:\n" + "\n".join(f"- `{x}`" for x in missing))
    st.stop()
try:
    lat, rounds, h2d, gpu, op_slowdowns, h2d_request, gpu_diag_all, net_phase = load_all(snapshot())
except Exception as ex:
    st.error(f"Failed to read or validate the experiment data: {ex}")
    st.stop()

network_base = lat[(lat.dataset.astype(str) == "physical") &
                   (lat.metric.astype(str) == "payload_recv_ms")].copy()
network_base["date"] = network_base.source_file.astype(str).str.extract(r"^(\d{8})_")[0]
dates = sorted(network_base.date.dropna().astype(str).unique().tolist())
network_payloads = {int(p) for p in pd.to_numeric(network_base.payload_bytes, errors="coerce").dropna().unique() if p > 0}
h2d_payloads = {int(p) for p in h2d.payload_bytes.dropna().unique() if p > 0}
common_payloads = sorted(network_payloads & h2d_payloads)
all_payloads = sorted(network_payloads | h2d_payloads)
if not all_payloads:
    st.error("No Network or H2D payload measurements were found.")
    st.stop()

with st.sidebar:
    st.header("Experiment settings")
    st.caption("All plots retain CON = 1, 2, 4, 8, 16, 32.")
    date = st.selectbox("Network run date", ["ALL"] + dates,
                        help="ALL aggregates existing physical network file summaries.")
    payloads = common_payloads if common_payloads else all_payloads
    pref = 262144 if 262144 in payloads else 65536 if 65536 in payloads else payloads[0]
    payload = int(st.selectbox("Network / H2D bytes per request", payloads,
                               index=payloads.index(pref), format_func=payload_label))
    ops = sorted(gpu.op.dropna().astype(str).unique().tolist())
    if not ops:
        st.error("No completed GPU request measurements were found.")
        st.stop()
    operator = st.selectbox("GPU operator filter (optional)", ["ALL"] + ops,
                            format_func=lambda x: "All operators" if x == "ALL" else x)
    gpu_by_size = gpu_size_summary(gpu, operator=operator)
    gpu_sizes = sorted(int(x) for x in gpu_by_size.input_bytes.dropna().unique() if x > 0)
    if not gpu_sizes:
        st.error("No GPU input-size measurements for this operator.")
        st.stop()
    # Favor exactly matched native input bytes, otherwise nearest measured size.
    # This changes ONLY the selection, never an underlying byte value.
    selected_default = min(gpu_sizes, key=lambda x: (abs(np.log2(x/payload)), x))
    gpu_bytes = int(st.selectbox("GPU input bytes per request", gpu_sizes,
                                index=gpu_sizes.index(selected_default), format_func=payload_label))
    st.divider()
    st.caption("Read-only: physical network, dedicated H2D, and existing Nsight requests.")

net, net_rates = network_summary(lat, rounds, date)
net_diag = (net_phase if date == "ALL" else physical_request_details(
    read_csv(PATHS["Network request diagnostics"]), date)) if not net_phase.empty else network_details(lat, date)
h2d_diag = h2d_details(h2d)
gpu_diag = gpu_diag_all if operator == "ALL" else gpu_details(read_csv(PATHS["GPU Nsight requests"]), operator)
network_batches = network_batch_summary(rounds, date)
selected_nb = network_batches[network_batches.payload_bytes.eq(payload)]
selected_nr = net_rates[net_rates.payload_bytes.eq(payload)]
selected_h2d = h2d[h2d.payload_bytes.eq(payload)]
selected_gpu = gpu_by_size[gpu_by_size.input_bytes.eq(gpu_bytes)]
combo = combined_sweep(net, net_rates, h2d, gpu_by_size, payload=payload,
                       gpu_input_bytes=gpu_bytes, network_batches=network_batches)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Network / H2D size", payload_label(payload))
with c2:
    st.metric("Concurrency conditions", "1, 2, 4, 8, 16, 32")
with c3:
    st.metric("GPU input size", payload_label(gpu_bytes))
if gpu_bytes != payload:
    st.caption(f"GPU input size ({payload_label(gpu_bytes)}) differs from Network/H2D payload "
               f"({payload_label(payload)}). The CON trends are comparable; identical end-to-end work "
               "or an exact payload match is not implied.")
else:
    st.caption("The selected stages share the same nominal input-byte count. Their original workload "
               "and timing definitions are still independent.")

tabs = st.tabs(["1. Network", "2. Host to GPU", "3. GPU Execution", "4. Combined Comparison"])

# ------------------------------------------------------------
# 1. Physical network
# ------------------------------------------------------------
with tabs[0]:
    st.header("1. Network — NIC to Host")
    st.caption("Physical network experiment; latency values aggregate the available file-level P50/P95 summaries. "
               "Goodput is calculated from valid physical network rounds.")
    metrics = sorted(net.metric.dropna().astype(str).unique().tolist())
    metric_labels = {"payload_recv_ms": "Payload receive", "request_rtt_ms": "Request RTT",
                     "host_ready_ms": "Host ready", "validation_ms": "Validation"}
    chosen_metric = st.selectbox("Network latency metric", metrics,
                                 index=metrics.index("request_rtt_ms") if "request_rtt_ms" in metrics else 0,
                                 format_func=lambda m: metric_labels.get(m, m))
    net_metric = net[net.payload_bytes.eq(payload) & net.metric.eq(chosen_metric)].rename(columns={"concurrency": "CON"})
    nr_plot = selected_nr.rename(columns={"concurrency": "CON"})
    lcol, rcol = st.columns(2)
    with lcol:
        line_chart(net_metric, [("p50_ms", "P50", COLOR["Network"]),
                                ("p95_ms", "P95", "#85b9f3")],
                   f"{metric_labels.get(chosen_metric, chosen_metric)} vs CON · {payload_label(payload)}", "Latency (ms)")
    with rcol:
        line_chart(nr_plot, [("goodput_mib_s", "Measured goodput", COLOR["Network"])],
                   f"Network goodput vs CON · {payload_label(payload)}", "MiB/s")
    size_curve(net[net.metric.eq(chosen_metric)], size_col="payload_bytes", con_col="concurrency",
               value_col="p50_ms", title="Network latency vs measured payload size · one line per CON",
               y_title="Network request P50 (ms)")
    heatmap(net[net.metric.eq(chosen_metric)], row="payload_bytes", col="concurrency",
            value="p50_ms", row_formatter=payload_label,
            title=f"Network payload × CON · {metric_labels.get(chosen_metric, chosen_metric)} P50", colorbar_title="ms")
    st.subheader("Network group completion")
    st.caption("Valid physical rounds containing exactly CON requests. Median of per-file batch P50s; "
               "a round with a different number of requests is not treated as a CON-sized group.")
    line_chart(selected_nb.rename(columns={"concurrency": "CON"}),
               [("batch_p50_ms", "Network group P50", COLOR["Network"]),
                ("batch_p95_ms", "Network group P95 (file median)", "#85b9f3")],
               f"Network group completion vs CON · {payload_label(payload)}", "Group elapsed (ms)")

    st.subheader("Network process breakdown — existing instrumented steps")
    st.caption("Source: normalized per-request physical_requests.csv" if not net_phase.empty
               else "Source: unified_latency.csv file-level phase summaries (per-request data unavailable)")
    st.caption("Measured phase labels: connection, accept/handler, header receive/parse, allocation, "
               "payload receive, host ready, validation, and request RTT. Exact timing boundaries "
               "follow the original instrumentation, not this display order. Only recorded metrics "
               "are shown. Some phases overlap or enclose others: DO NOT stack or add them.")
    network_pivot = network_metric_pivot(net_diag, payload)
    available_steps = [c for c,_ in NETWORK_ORDER if c in network_pivot.columns and network_pivot[c].notna().any()]
    other_steps = [c for c in network_pivot.columns if c != "CON" and c not in available_steps and network_pivot[c].notna().any()]
    chosen_steps = st.multiselect("Network phases to compare", available_steps + other_steps,
        default=available_steps if available_steps else other_steps,
        format_func=lambda x: NETWORK_LABELS.get(x, x), key="net_phase_multiselect")
    if chosen_steps:
        line_chart(network_pivot, [(c, NETWORK_LABELS.get(c,c),
                   ["#f6b44a", "#e17980", "#9f89df", "#4c98e0", "#65b29c", "#c0ad66"][i % 6])
                   for i,c in enumerate(chosen_steps)],
                   f"Network phase P50 vs CON · {payload_label(payload)}", "Measured phase latency (ms)", height=430)
        detail_metric = st.selectbox("Network phase for Payload × CON heatmap", chosen_steps,
                        format_func=lambda x: NETWORK_LABELS.get(x,x), key="net_phase_heatmap")
        heatmap(net_diag[net_diag.metric.eq(detail_metric)], row="payload_bytes", col="CON",
                value="p50_ms", row_formatter=payload_label,
                title=f"{NETWORK_LABELS.get(detail_metric,detail_metric)} · Payload × CON", colorbar_title="ms")
        p95 = (net_diag[net_diag.payload_bytes.eq(payload) & net_diag.metric.isin(chosen_steps)]
               .pivot(index="CON", columns="metric", values="p95_ms").reset_index())
        with st.expander("Network P95 phase trends, source coverage, and measurement table"):
            p95_series = [(c, NETWORK_LABELS.get(c,c), "#4c98e0") for c in chosen_steps if c in p95 and p95[c].notna().any()]
            if p95_series: line_chart(p95, p95_series, "Phase P95 vs CON (file-level P95 median)", "Latency (ms)")
            st.dataframe(net_diag[net_diag.payload_bytes.eq(payload)].sort_values(["metric", "CON"]),
                         use_container_width=True, hide_index=True)
    else:
        st.info("The current normalized Network CSV does not expose process-level timings for this payload.")

    with st.expander("Network measurements"):
        st.dataframe(net_metric[["CON", "p50_ms", "p95_ms", "file_rows"]],
                     hide_index=True, use_container_width=True)
        st.dataframe(selected_nb, hide_index=True, use_container_width=True)
        st.caption("P50/P95 here are medians of existing file-level quantile summaries, not pooled raw-request quantiles.")

# ------------------------------------------------------------
# 2. H2D batches
# ------------------------------------------------------------
with tabs[1]:
    st.header("2. Host to GPU — H2D Transfer")
    st.caption("Dedicated H2D batch experiment; one batch contains CON concurrent transfers. "
               "These results are not taken from the Nsight kernel dataset.")
    hd = selected_h2d.rename(columns={"concurrency": "CON"})
    lcol, rcol = st.columns(2)
    with lcol:
        line_chart(hd, [("gpu_p50_ms", "GPU batch P50", COLOR["Host to GPU"]),
                        ("gpu_p95_ms", "GPU batch P95", "#88d2c3"),
                        ("host_p50_ms", "Host wall P50", "#d8ab51")],
                   f"H2D batch completion vs CON · {payload_label(payload)}", "Batch time (ms)")
    with rcol:
        line_chart(hd, [("goodput_mib_s", "H2D effective bandwidth", COLOR["Host to GPU"])],
                   f"H2D bandwidth vs CON · {payload_label(payload)}", "MiB/s")
    size_curve(h2d, size_col="payload_bytes", con_col="concurrency", value_col="gpu_p50_ms",
               title="H2D batch latency vs measured payload size · one line per CON",
               y_title="H2D GPU batch P50 (ms)")
    heatmap(h2d, row="payload_bytes", col="concurrency", value="gpu_p50_ms",
            row_formatter=payload_label, title="H2D payload × CON · GPU batch P50", colorbar_title="ms")
    st.caption("For comparison later, H2D batch P50 / CON is an amortized transfer cost per request. "
               "It is not a separately measured request P50.")

    st.subheader("H2D process diagnostics — host wall vs GPU timeline")
    st.caption("Two measured clocks for the same CON-sized H2D batch. Their difference is a "
               "difference of independently computed P50 summaries, NOT a directly measured "
               "CUDA scheduling or queueing duration. It must not be stacked with either clock.")
    hd_detail = h2d_diag[h2d_diag.payload_bytes.eq(payload)].rename(columns={"concurrency":"CON"})
    line_chart(hd_detail, [("host_p50_ms", "Host wall P50", "#d8ab51"),
                           ("gpu_p50_ms", "GPU timeline P50", "#18a887")],
               f"H2D completion clocks vs CON · {payload_label(payload)}", "Batch completion (ms)")
    line_chart(hd_detail, [("host_minus_gpu_p50_ms", "Host P50 − GPU P50", "#df9d3d")],
               f"Host-wall minus GPU-timeline P50 · {payload_label(payload)}", "Difference of P50s (ms)")
    hreq = h2d_request[h2d_request.h2d_bytes.eq(payload)] if not h2d_request.empty else pd.DataFrame()
    if not hreq.empty:
        st.caption("Separate Step 2 per-request H2D summaries. These are not components of the dedicated "
                   "H2D batch timing; a request may have multiple H2D copies.")
        hreq_plot = hreq[hreq.h2d_count.eq(1)] if hreq.h2d_count.eq(1).any() else hreq
        if hreq_plot.CON.duplicated().any():
            st.info("Multiple H2D request groups exist per CON; see the raw summary below.")
        else:
            line_chart(hreq_plot, [("h2d_p50_ms", "Per-request H2D P50", "#18a887"),
                                   ("h2d_p95_ms", "Per-request H2D P95", "#88d2c3")],
                      f"Per-request H2D vs CON · {payload_label(payload)}", "Copy execution (ms)")
        with st.expander("H2D request statistics and exact bytes"):
            st.dataframe(hreq, hide_index=True, use_container_width=True)
    else:
        st.info("No exact-payload per-request H2D summary in existing Step 2 CSV; batch timing remains valid.")

    with st.expander("H2D conditions and coverage"):
        st.dataframe(hd[["CON", "status", "n_rounds", "gpu_p50_ms", "gpu_p95_ms",
                         "host_p50_ms", "amortized_ms_per_request", "goodput_mib_s"]],
                     hide_index=True, use_container_width=True)
        st.caption("Unmeasured, OOM, failed, and duplicate configurations remain missing; never zero-filled.")

# ------------------------------------------------------------
# 3. GPU request / group completion (primary analysis)
# ------------------------------------------------------------
with tabs[2]:
    st.header("3. GPU — Request and Group Completion")
    st.caption("Primary measurements: `global_latency_ms` (CON-group start → each request finishes) "
               "and `group_makespan_ms` (group start → final request finishes). "
               "Only complete CON-request repeats are used for group completion. "
               "The kernel-only metric is available in diagnostics below.")
    if operator == "ALL":
        st.info("All measured GPU operators are included. Equal input sizes do not imply equal computation; "
                "the input-size plot is exploratory and aggregates source-level summaries. "
                "Filter by operator to control the computation type.")
    gp = selected_gpu.rename(columns={"con": "CON"})
    lc, rc = st.columns(2)
    with lc:
        line_chart(gp, [("request_p50_ms", "Request completion P50", COLOR["GPU"]),
                        ("request_p95_ms", "Request completion P95", "#d6b8ff")],
                   f"GPU request completion vs CON · input {payload_label(gpu_bytes)}",
                   "Group start to request finish (ms)", height=405)
    with rc:
        line_chart(gp, [("group_p50_ms", "Group completion P50", COLOR["GPU"]),
                        ("group_p95_ms", "Group completion P95", "#d6b8ff")],
                   f"GPU group completion vs CON · input {payload_label(gpu_bytes)}",
                   "Group start to last request finish (ms)", height=405)
    line_chart(gp, [("gpu_group_requests_s", "GPU group rate", COLOR["GPU"])],
               f"GPU completed requests/s vs CON · input {payload_label(gpu_bytes)}",
               "Completed requests/s")
    st.subheader("Completion time vs input data size")
    size_curve(gpu_by_size, size_col="input_bytes", con_col="con",
               value_col="request_p50_ms",
               title="GPU request completion vs measured input size · one line per CON",
               y_title="Request completion (median of source P50s, ms)", height=435)
    size_curve(gpu_by_size, size_col="input_bytes", con_col="con",
               value_col="group_p50_ms",
               title="GPU group completion vs measured input size · one line per CON",
               y_title="Group completion (median of source P50s, ms)", height=435)
    lc, rc = st.columns(2)
    with lc:
        heatmap(gpu_by_size, row="input_bytes", col="con", value="request_p50_ms",
                row_formatter=payload_label, title="GPU input size × CON · request completion P50",
                colorbar_title="ms", max_height=730)
    with rc:
        heatmap(gpu_by_size, row="input_bytes", col="con", value="group_p50_ms",
                row_formatter=payload_label, title="GPU input size × CON · group completion P50",
                colorbar_title="ms", max_height=730)
    st.subheader("Completion slowdown vs CON (paired to each source's CON=1)")
    line_chart(gp, [("source_paired_request_slowdown_x", "Request completion slowdown", COLOR["GPU"]),
                    ("source_paired_group_slowdown_x", "Group completion slowdown", "#d6b8ff")],
               f"GPU completion slowdown · {payload_label(gpu_bytes)}", "Relative completion time (×)")

    st.subheader("GPU request path — measured process timings")
    st.caption("Host submission → CUDA API → H2D / GPU work → request completion. "
               "The diagnostic durations have different timing boundaries and may overlap; "
               "they are plotted side by side, never summed as a latency breakdown. "
               "GPU internal hardware scheduling is not inferred from these timings.")
    d = gpu_diag[gpu_diag.input_bytes.eq(gpu_bytes)]
    available_gpu_steps = [k for k,_ in GPU_ORDER if k in d and d[k].notna().any()]
    default_gpu = [k for k in ("host_submit_ms", "first_gpu_delay_ms", "h2d_exec_ms",
                               "kernel_exec_ms", "total_gpu_idle_between_ops_ms") if k in available_gpu_steps]
    chosen_gpu_steps = st.multiselect("GPU process metrics", available_gpu_steps,
                default=default_gpu, format_func=lambda k: GPU_LABELS.get(k,k), key="gpu_process_steps")
    if chosen_gpu_steps:
        line_chart(d, [(k, GPU_LABELS.get(k,k),
                         ["#4c98e0", "#18a887", "#b48be6", "#df9d3d", "#e17980", "#85b9f3"][i % 6])
                         for i,k in enumerate(chosen_gpu_steps)],
                   f"GPU process timings vs CON · {payload_label(gpu_bytes)}", "Observed diagnostic time (ms)", height=455)
        gpu_heat_metric = st.selectbox("GPU process metric for Input Size × CON heatmap",
                            chosen_gpu_steps, format_func=lambda k: GPU_LABELS.get(k,k),
                            key="gpu_process_heatmap")
        heatmap(gpu_diag, row="input_bytes", col="CON", value=gpu_heat_metric,
                row_formatter=payload_label,
                title=f"{GPU_LABELS.get(gpu_heat_metric,gpu_heat_metric)} · Input Size × CON",
                colorbar_title="ms", max_height=650)
        with st.expander("GPU phase coverage and source-aggregated values"):
            st.dataframe(d[["CON", "source_files", "requests"] + chosen_gpu_steps],
                         hide_index=True, use_container_width=True)
            st.caption("For each source file and CON: median across its request measurements; "
                       "then median across source files with the same input bytes. "
                       "Absent kernel timings are missing, not zero.")
    else:
        st.info("This selection has no measured GPU process diagnostics in nsys_gpu_requests.csv.")

    with st.expander("Kernel-only diagnostics (not the primary GPU outcome)"):
        line_chart(gp, [("kernel_p50_ms", "Kernel-only P50", COLOR["GPU"]),
                        ("kernel_p95_ms", "Kernel-only P95", "#d6b8ff")],
                   f"CUDA kernel execution vs CON · {payload_label(gpu_bytes)}",
                   "Kernel execution only (ms)")
        st.caption("`kernel_exec_ms` excludes surrounding host submission, synchronization, and other "
                   "completion delays. Requests with no CUDA kernel still contribute to request completion.")
    with st.expander("GPU source coverage and measurement definitions"):
        st.dataframe(gp[["CON", "input_bytes", "source_files", "operators", "requests",
                         "complete_repeats", "request_p50_ms", "request_p95_ms",
                         "group_p50_ms", "group_p95_ms", "gpu_group_requests_s"]],
                     hide_index=True, use_container_width=True)
        matching = gpu[gpu.input_bytes.eq(gpu_bytes)]
        if operator != "ALL":
            matching = matching[matching.op.eq(operator)]
        st.dataframe(matching[["op", "source_file", "layer", "input_bytes"]].drop_duplicates().sort_values(
            ["op", "source_file"]), hide_index=True, use_container_width=True)
        st.caption("Request P50 = median of source-level request P50s. Group P50 = median of "
                   "per-source median of complete repeat makespans. These are not pooled P50s. "
                   "Missing/failed repeats are never zero-filled.")

# ------------------------------------------------------------
# 4. Compare the three independently measured stages
# ------------------------------------------------------------
with tabs[3]:
    st.header("4. Combined Comparison")
    st.caption("All three stages are plotted for CON = 1, 2, 4, 8, 16, 32. "
               "Network and H2D use the selected payload; GPU uses its native measured input size.")
    if gpu_bytes != payload:
        st.warning(f"Data sizes differ: Network/H2D {payload_label(payload)}, "
                   f"GPU {payload_label(gpu_bytes)}. This is a comparison of independent "
                   "CON trends, not one identical end-to-end workload.")
    st.subheader("Measured completion times — existing data")
    line_chart(combo, [("network_group_p50_ms", "Network: CON requests finish", COLOR["Network"]),
                       ("h2d_batch_host_p50_ms", "H2D: CON transfers finish", COLOR["Host to GPU"]),
                       ("gpu_group_p50_ms", "GPU: CON requests finish", COLOR["GPU"])],
               "Measured standalone group-completion times vs CON", "Group completion P50 (ms)", height=450)
    st.caption("Network group: `physical_rounds.csv` valid rounds with exactly CON requests. "
               "H2D group: host-wall batch P50. GPU group: `group_makespan_ms`. "
               "All are standalone measurements with different work and synchronization boundaries.")

    measured = combo[["CON", "network_group_p50_ms", "h2d_batch_host_p50_ms",
                      "gpu_group_p50_ms"]].copy()
    measured.columns = ["CON", "Network group P50 (ms)", "H2D group P50 (ms)",
                        "GPU group P50 (ms)"]
    st.dataframe(measured, hide_index=True, use_container_width=True)
    st.caption("These measured standalone-stage results load automatically from the existing CSV files; "
               "no upload is required.")

    # BEGIN: Pi/GPU matched-source Section 4
    render_pi_gpu_section4()
    # END: Pi/GPU matched-source Section 4

    st.subheader("Full local vs offloading E2E (requires matching measurements)")
    st.caption("Compare **the total time to finish all CON requests** on Node A alone against "
               "offloading the same workload. A true local-only baseline and a true offloaded "
               "end-to-end time must be measured separately. Independent stage measurements "
               "can only produce a labeled estimate, not a measured end-to-end total.")

    baseline_path = HERE.parent / "data" / "node_a_local_comparison.csv"
    with st.expander("Optional: add measured Node A local / offloading E2E CSV", expanded=False):
        baseline_upload = st.file_uploader(
            "Measured Node A baseline / optional measured offloading E2E (CSV)",
            type=["csv"], key="node_a_local_csv",
            help="Upload measured group completion P50 for matching payload, GPU input, operator, and CON. "
                 "The optional measured offloading E2E column overrides the estimated curve.")
    raw_baseline = None
    data_origin = "No measured baseline loaded"
    if baseline_upload is not None:
        try:
            raw_baseline = pd.read_csv(baseline_upload)
            data_origin = "Uploaded CSV (in memory; source files unchanged)"
        except Exception as ex:
            st.error(f"Could not read uploaded CSV: {ex}")
    elif baseline_path.is_file():
        try:
            raw_baseline = pd.read_csv(baseline_path)
            data_origin = str(baseline_path)
        except Exception as ex:
            st.error(f"Could not read baseline file: {ex}")

    try:
        local_comparison = compare_node_a_vs_offload(combo, raw_baseline, operator=operator)
    except ValueError as ex:
        st.error(f"Invalid Node A comparison data: {ex}")
        local_comparison = compare_node_a_vs_offload(combo, None, operator=operator)
    st.caption(f"Baseline source: {data_origin}. Selected workload: "
               f"Network/H2D {payload_label(payload)}; GPU input {payload_label(gpu_bytes)}; "
               f"operator {operator}.")
    if local_comparison.node_a_local_group_p50_ms.notna().any():
        local_vs_offload_chart(local_comparison)
    else:
        st.info("No matched full Node A local benchmark is supplied for the selected three-stage "
                "workload. The measured 5 KiB operator comparison above is available automatically, "
                "but it is not a measured full offloading E2E comparison.")
    if not local_comparison.node_a_local_group_p50_ms.notna().any():
        st.caption("Node A local-only times are not in the existing stage CSVs, so the measured "
                   "local-only line and speedup are unavailable. Add measured baseline data using "
                   "the optional CSV field above; no data is filled in automatically.")
    if not local_comparison.offload_standalone_proxy_ms.notna().any() and not local_comparison.offload_e2e_group_p50_ms.notna().any():
        st.caption("No offload total can be calculated for this selection without an unambiguous "
                   "GPU transfer mode or a measured E2E offload duration. Existing stage lines "
                   "above remain available.")
    if operator == "ALL":
        st.caption("GPU size grouping may combine different operators. For a controlled "
                   "local-vs-offload comparison, use a matching operator/workload.")
    if local_comparison.node_a_local_group_p50_ms.notna().any():
        local_vs_offload_speedup_chart(local_comparison)
    st.caption("Solid curves = directly measured group completion; dashed curve = a **sum of independent "
               "stage P50s**, NOT a measured end-to-end P50. Offload stage proxy uses Network + GPU "
               "when GPU already includes H2D, or Network + H2D + GPU when GPU used preloaded inputs; "
               "unknown/mixed GPU transfer modes stay missing. It does not establish actual "
               "Node A→Node B→Node A return time, overlap, or pipelined execution.")
    with st.expander("Node A comparison CSV template and numeric table", expanded=False):
        template = ("network_payload_bytes,gpu_input_bytes,operator,CON,"
                    "node_a_local_group_p50_ms,offload_e2e_group_p50_ms\n")
        st.download_button("Download blank measured-data CSV template", template,
                           file_name="node_a_local_comparison.csv", mime="text/csv")
        st.markdown("- `network_payload_bytes`: bytes per request used in Network and H2D benchmarks.\n"
                    "- `gpu_input_bytes`: bytes per GPU request.\n"
                    "- `operator`: same GPU operator as the baseline, e.g. `conv2d` (or `ALL` only "
                    "for an intentionally matched aggregate).\n"
                    "- `CON`: 1, 2, 4, 8, 16, 32.\n"
                    "- `node_a_local_group_p50_ms`: **observed** time for Node A alone "
                    "to finish all CON requests.\n"
                    "- `offload_e2e_group_p50_ms` (optional): **observed** complete offload "
                    "time with the same start/finish boundary. Leave blank if not measured.")
        st.dataframe(local_comparison[["CON", "node_a_local_group_p50_ms",
                                       "offload_e2e_group_p50_ms", "offload_standalone_proxy_ms",
                                       "offload_proxy_components", "offload_comparison_kind",
                                       "node_a_vs_offload_speedup_x", "node_a_vs_offload_time_saved_pct"]],
                     hide_index=True, use_container_width=True)
        st.download_button("Download Node A / offload comparison (CSV)",
                           local_comparison.to_csv(index=False).encode("utf-8"),
                           file_name="gpu_con_bench_node_a_vs_offload.csv", mime="text/csv")

    st.subheader("Request completion time vs CON")
    line_chart(combo, [("network_request_rtt_p50_ms", "Network: request RTT", COLOR["Network"]),
                       ("gpu_request_p50_ms", "GPU: request completion", COLOR["GPU"])],
               "Request completion latency vs CON · standalone measurements", "Request latency P50 (ms)", height=420)
    st.caption("H2D was measured as a CON-sized batch, so its host-wall completion is shown "
               "in the group chart above—not falsely represented as an observed per-request P50.")

    st.subheader("Concurrency sensitivity (each stage paired with CON=1)")
    line_chart(combo, [("network_group_slowdown_x", "Network group", COLOR["Network"]),
                       ("h2d_group_slowdown_x", "H2D group", COLOR["Host to GPU"]),
                       ("gpu_group_slowdown_x", "GPU group (source-paired)", COLOR["GPU"])],
               "Group completion slowdown vs CON=1", "Relative completion time (×)", height=410)
    st.caption("GPU slowdown = median of source-specific CON/CON=1 completion ratios; "
               "other curves use the selected stage's group P50 baseline.")

    st.subheader("Sum of independently measured group completion times")
    if not combo.gpu_preloaded.all():
        st.info("The standalone time sum is disabled for GPU configurations with `transfer_mode=h2d`, "
                "mixed or unknown transfer modes. Their GPU completion time may already include H2D, "
                "so adding a separate H2D time could double-count the transfer.")
    lc, rc = st.columns(2)
    with lc:
        line_chart(combo, [("independent_group_time_sum_ms", "Sum of standalone group P50s", "#e3aa59")],
                   "Standalone stage-time sum vs CON (hypothetical)", "Sum of group P50s (ms)")
    with rc:
        stacked_illustration(combo)
    st.warning("When available, the displayed sum is **not measured end-to-end latency**. "
               "It combines independent group-completion P50s from Network, H2D, and GPU (preloaded input only). "
               "It ignores overlap/pipelining and cannot be interpreted as an E2E P50 or definitive bottleneck.")

    st.subheader("Standalone throughput / capacity screening")
    line_chart(combo, [("network_requests_s", "Network goodput / payload", COLOR["Network"]),
                       ("h2d_requests_s", "H2D CON / host batch P50", COLOR["Host to GPU"]),
                       ("gpu_group_requests_s", "GPU CON / group P50", COLOR["GPU"])],
               "Standalone completed request rate vs CON", "Requests/s", height=420)
    st.caption("Throughput is calculated for each independent workload; operator work or payload may differ. "
               "The lowest standalone rate is a candidate capacity constraint, not a proven E2E bottleneck.")
    valid = combo[combo.complete_stage_rates]
    if not valid.empty:
        st.markdown("**Lowest standalone request rate by CON (screening only)**")
        st.dataframe(valid[["CON", "network_requests_s", "h2d_requests_s", "gpu_group_requests_s",
                            "lowest_standalone_rate"]], hide_index=True, use_container_width=True)
    else:
        st.info("No fully comparable three-stage rate row exists for this selection; missing values are not zero.")
    st.markdown("**Measured CON comparison table**")
    st.dataframe(combo, hide_index=True, use_container_width=True)
    st.download_button("Download combined CON table (CSV)",
                       combo.to_csv(index=False).encode("utf-8"),
                       file_name="gpu_con_bench_combined.csv", mime="text/csv")
    with st.expander("Measurement definitions, coverage, and file provenance"):
        st.markdown("- **Network request**: `request_rtt_ms`, median of file-level P50s. "
                    "**Network group**: valid round `batch_elapsed_ms` when `total_requests == network_con`.\n"
                    "- **Host to GPU group**: dedicated `condition_summary.csv` `host_p50_ms` "
                    "(host-wall completion); the separate GPU event P50 is visible in Section 2.\n"
                    "- **GPU request**: `global_latency_ms`, group start to individual request completion.\n"
                    "- **GPU group**: `group_makespan_ms`, group start to last request completion, "
                    "one observation per complete repeat.\n"
                    "- **CUDA kernel**: `kernel_exec_ms` is a secondary diagnostic only.\n"
                    "- **Combined sum**: sum of independent standalone group P50s only when GPU input "
                    "was preloaded; disabled when the GPU time may already include H2D. NOT observed E2E.\n"
                    "- **Missing and OOM**: no zero filling or interpolation.")
        st.json({name: str(path) for name, path in PATHS.items()})
        st.json({"selected_network_h2d_payload_bytes": payload,
                 "selected_gpu_input_bytes": gpu_bytes, "selected_gpu_operator_filter": operator,
                 "network_date": date, "CON_values": list(CONS)})

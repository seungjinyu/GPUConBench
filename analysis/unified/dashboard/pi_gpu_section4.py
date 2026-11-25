"""GPU Con Bench / Section 4: matched Raspberry Pi local vs GPU+H2D requests.

Read-only: uses output from analysis/unified/compare_pi_gpu.py, never reruns
benchmarks or manufactures GPU/Pi concurrency measurements.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CON_VALUES = (1, 2, 4, 8, 16, 32)
HERE = Path(__file__).resolve().parent
COMPARISON_DIR = HERE.parent / "integrated" / "pi_vs_gpu"
PI_COLOR = "#428dcc"
GPU_COLOR = "#9872d2"


@st.cache_data(show_spinner=False)
def _read_csv(path_string: str, modified_ns: int) -> pd.DataFrame:
    """Cache invalidation tracks the file's modification timestamp."""
    return pd.read_csv(path_string)


def _load_results(root: Path):
    file = root / "pi_vs_gpu_by_layer_con.csv"
    if not file.is_file():
        return pd.DataFrame(), None, file
    d = _read_csv(str(file), file.stat().st_mtime_ns).copy()
    expected = {"source_file", "op", "layer", "input_bytes", "con",
                "pi_p50_ms", "gpu_p50_ms", "pi_torch_threads", "gpu_requests"}
    missing = expected - set(d.columns)
    if missing:
        raise ValueError(f"Comparison CSV is missing columns: {sorted(missing)}")
    if d.duplicated(["source_file", "con"]).any():
        raise ValueError("Multiple entries for the same source_file × CON; comparison is ambiguous")
    for col in ("input_bytes", "con", "pi_p50_ms", "gpu_p50_ms", "pi_torch_threads",
                "gpu_requests", "pi_p95_ms", "gpu_p95_ms", "pi_p99_ms", "gpu_p99_ms"):
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d["op"] = d.op.astype(str).str.strip().str.lower()
    d["source_file"] = d.source_file.astype(str).str.strip()
    d = d[d.con.isin(CON_VALUES)].copy()
    d["is_complete"] = d.pi_p50_ms.gt(0) & d.gpu_p50_ms.gt(0)
    d["winner_computed"] = np.select(
        [d.is_complete & (d.gpu_p50_ms < d.pi_p50_ms),
         d.is_complete & (d.pi_p50_ms < d.gpu_p50_ms),
         d.is_complete & (d.gpu_p50_ms == d.pi_p50_ms)],
        ["GPU", "Pi", "Tie"], default="Missing")
    d["gpu_over_pi_ratio"] = np.where(d.is_complete, d.gpu_p50_ms / d.pi_p50_ms, np.nan)
    bfile = root / "pi_vs_gpu_breakeven_by_layer.csv"
    breaks = _read_csv(str(bfile), bfile.stat().st_mtime_ns).copy() if bfile.is_file() else None
    return d, breaks, file


def _counts(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame.is_complete]
    counts = valid.groupby(["con", "winner_computed"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=list(CON_VALUES), fill_value=0)
    for col in ["GPU", "Pi", "Tie"]:
        if col not in counts:
            counts[col] = 0
    counts.index.name = "CON"
    return counts[["GPU", "Pi", "Tie"]].reset_index()


def _break_even_from_detail(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for source, group in d.groupby("source_file", sort=True):
        group = group.sort_values("con")
        row = group.iloc[0]
        complete = group[group.is_complete]
        coverage = set(complete.con.astype(int)) == set(CON_VALUES)
        first = complete[complete.winner_computed.isin(["Pi", "Tie"])]
        first_non_gpu = int(first.iloc[0].con) if not first.empty else np.nan
        con1 = complete[complete.con.eq(1)]
        if not coverage:
            status = "Incomplete coverage"
        elif con1.empty:
            status = "Incomplete coverage"
        elif con1.iloc[0].winner_computed != "GPU":
            status = "Pi not slower at CON=1"
        elif pd.notna(first_non_gpu):
            status = "GPU→Pi crossover"
        else:
            status = "GPU faster at all CON"
        out.append({"source_file": source, "op": row.op, "layer": row.layer,
                    "input_bytes": row.input_bytes, "pi_p50_ms": row.pi_p50_ms,
                    "first_non_gpu_CON": first_non_gpu, "status": status})
    return pd.DataFrame(out)


def render_pi_gpu_section4() -> None:
    """Call this from the existing `with tabs[3]:` block."""
    st.subheader("MobileNetV2 — Raspberry Pi Local vs Shared GPU")
    st.caption("**Measured** local Pi CPU execution: one 160×160, BS=1 request (CON=1). "
               "**Measured** GPU request completion: CON=1, 2, 4, 8, 16, 32; "
               "GPU measurements include H2D but exclude NIC→GPU network transfer. "
               "This is NOT a measured full offloading end-to-end comparison.")
    try:
        detail, _, file = _load_results(COMPARISON_DIR)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load Pi/GPU comparison results: {exc}")
        return
    if detail.empty:
        st.info("Missing comparison file. Copy the Oryx output directory to "
                f"`{COMPARISON_DIR}`. Expected: `{file.name}`")
        return

    valid = detail[detail.is_complete]
    counts = _counts(detail)
    be = _break_even_from_detail(detail)
    expected = int(detail.source_file.nunique()) * len(CON_VALUES)
    nvalid = len(valid)
    all_matched = detail.source_file.nunique()
    cross = int((be.status == "GPU→Pi crossover").sum())
    initially_pi = int((be.status == "Pi not slower at CON=1").sum())
    full_gpu = int((be.status == "GPU faster at all CON").sum())
    incomplete = int((be.status == "Incomplete coverage").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pi operators", f"{all_matched}")
    c2.metric("Valid layer × CON", f"{nvalid} / {expected}")
    c3.metric("Pi not slower at CON=1", f"{initially_pi}")
    c4.metric("GPU→Pi crossovers", f"{cross}")
    if incomplete:
        st.warning(f"{incomplete} layer(s) have missing CON measurements; break-even cannot be fully classified.")
    if full_gpu:
        st.caption(f"GPU remained faster through CON=32 for {full_gpu} layer(s).")

    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.markdown("**Winner counts by GPU concurrency**")
        fig = go.Figure()
        for name, color in [("GPU", GPU_COLOR), ("Pi", PI_COLOR), ("Tie", "#999999")]:
            if name == "Tie" and counts[name].sum() == 0:
                continue
            fig.add_bar(x=counts.CON.astype(str), y=counts[name], name=f"{name} faster" if name != "Tie" else "Tied",
                        marker_color=color,
                        hovertemplate="CON %{x}<br>Layers %{y}<extra>" + name + "</extra>")
        fig.update_layout(barmode="stack", height=365,
                          xaxis_title="Shared GPU concurrent requests (CON)",
                          yaxis_title="Number of source layers",
                          legend=dict(orientation="h", y=1.12),
                          margin=dict(l=10, r=10, t=50, b=35))
        st.plotly_chart(fig, use_container_width=True)
    with c_right:
        st.markdown("**GPU wins / Pi wins by CON**")
        st.dataframe(counts.rename(columns={"GPU": "GPU faster", "Pi": "Pi faster", "Tie": "Ties"}),
                     hide_index=True, use_container_width=True)
        st.caption("Each source is compared only to the matching Pi operator (same source filename, "
                   "layer, operator and input-byte count). Lower P50 determines the winner.")

    st.markdown("**Operator-level comparison**")
    ops = sorted(valid.op.unique().tolist())
    if not ops:
        st.warning("There are no valid matched layer × CON rows.")
        return
    selected_op = st.selectbox("Operator", ops,
                               index=ops.index("conv2d") if "conv2d" in ops else 0,
                               key="pi_gpu_s4_op")
    op_valid = valid[valid.op.eq(selected_op)]
    op_summary = op_valid.groupby("con", as_index=False).agg(
        matched_layers=("source_file", "nunique"),
        gpu_faster_layers=("winner_computed", lambda x: int(x.eq("GPU").sum())),
        pi_faster_layers=("winner_computed", lambda x: int(x.eq("Pi").sum())),
        median_gpu_over_pi_ratio=("gpu_over_pi_ratio", "median"))
    st.caption("Median GPU/Pi ratio is the **median of each layer's GPU P50 / Pi P50**, "
               "not a ratio of pooled operator medians. Ratio < 1 favors GPU; > 1 favors Pi.")
    st.dataframe(op_summary.round(4).rename(columns={
        "con": "GPU CON", "matched_layers": "Matched layers", "gpu_faster_layers": "GPU faster",
        "pi_faster_layers": "Pi faster", "median_gpu_over_pi_ratio": "Median GPU/Pi P50 ratio"}),
        hide_index=True, use_container_width=True)

    layer_choices = (op_valid[["source_file", "layer", "input_bytes"]]
                     .drop_duplicates().sort_values(["layer", "source_file"]))
    layers = layer_choices.source_file.tolist()
    labels = {r.source_file: f"{r.layer} · {int(r.input_bytes):,} bytes · {r.source_file}"
              for r in layer_choices.itertuples(index=False)}
    chosen = st.selectbox("Specific layer / workload", layers,
                          format_func=lambda source: labels[source], key="pi_gpu_s4_layer")
    one = detail[detail.source_file.eq(chosen)].sort_values("con")
    pi_val = float(one.iloc[0].pi_p50_ms)
    chosen_threads = one.iloc[0].pi_torch_threads
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[str(c) for c in CON_VALUES], y=[pi_val] * len(CON_VALUES),
                             name="Pi CPU local P50 (CON=1 baseline)", mode="lines+markers",
                             line=dict(color=PI_COLOR, dash="dash", width=3),
                             marker=dict(size=7),
                             hovertemplate="GPU CON %{x}<br>Pi local %{y:.4f} ms<extra>Measured once at Pi CON=1</extra>"))
    fig.add_trace(go.Scatter(x=one.con.astype(int).astype(str), y=one.gpu_p50_ms,
                             name="GPU request + H2D P50", mode="lines+markers",
                             line=dict(color=GPU_COLOR, width=3), marker=dict(size=8),
                             connectgaps=False,
                             hovertemplate="GPU CON %{x}<br>GPU request %{y:.4f} ms<extra>Measured</extra>"))
    fig.update_layout(height=430, xaxis=dict(title="Shared GPU concurrent requests (CON)",
                         type="category", categoryorder="array", categoryarray=[str(c) for c in CON_VALUES]),
                      yaxis=dict(title="Individual request completion P50 (ms)", rangemode="tozero"),
                      legend=dict(orientation="h", y=1.15),
                      hovermode="x unified", margin=dict(l=12, r=12, t=70, b=45))
    st.plotly_chart(fig, use_container_width=True)
    if pd.notna(chosen_threads):
        st.caption(f"Pi CPU best thread setting for this source: {int(chosen_threads)}. "
                   "The horizontal line repeats the SAME Pi CON=1 measurement for reference; "
                   "it does not imply that Pi was benchmarked at GPU CON=2…32.")

    table_columns = ["con", "pi_p50_ms", "gpu_p50_ms", "gpu_over_pi_ratio", "winner_computed", "gpu_requests"]
    if "gpu_p95_ms" in one and "pi_p95_ms" in one:
        table_columns = ["con", "pi_p50_ms", "pi_p95_ms", "gpu_p50_ms", "gpu_p95_ms",
                         "gpu_over_pi_ratio", "winner_computed", "gpu_requests"]
    st.dataframe(one[table_columns].round(4).rename(columns={
        "con": "GPU CON", "pi_p50_ms": "Pi P50 (ms)", "pi_p95_ms": "Pi P95 (ms)",
        "gpu_p50_ms": "GPU+H2D P50 (ms)", "gpu_p95_ms": "GPU+H2D P95 (ms)",
        "gpu_over_pi_ratio": "GPU/Pi P50 ratio", "winner_computed": "Faster device",
        "gpu_requests": "GPU request samples"}), hide_index=True, use_container_width=True)

    with st.expander("All-layer break-even details and CSV downloads", expanded=False):
        st.caption("Break-even is the FIRST MEASURED CON at which GPU P50 is no longer "
                   "strictly faster than Pi P50; no interpolation is performed. "
                   "When Pi already wins at CON=1, there is no GPU→Pi crossover.")
        st.dataframe(be.sort_values(["op", "layer"]), hide_index=True, use_container_width=True)
        st.download_button("Download matched layer × CON comparisons", detail.drop(columns=["is_complete"]).to_csv(index=False),
                           file_name="pi_vs_gpu_by_layer_con_dashboard.csv", mime="text/csv",
                           key="pi_gpu_s4_download_detail")
        st.download_button("Download break-even classifications", be.to_csv(index=False),
                           file_name="pi_vs_gpu_breakeven_dashboard.csv", mime="text/csv",
                           key="pi_gpu_s4_download_be")
    st.caption("Timing boundaries differ: Pi measures prepared CPU tensor → operator result, while GPU measures "
               "the request completion path including H2D. Pi generated activations and original GPU "
               "Imagenette activations are not value-identical; matching was verified by layer/operator/bytes, "
               "not by input tensor values. Network transmission and return are not included.")

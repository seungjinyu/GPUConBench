"""Process-level diagnostics from existing GPU Con Bench measurements.

The functions never synthesize timings or zero-fill missing/OOM conditions.
Inputs are existing standalone experiment data, not a traced E2E pipeline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NETWORK_ORDER = [
    ("dial_ms", "Connection establishment"),
    ("accept_to_handler_ms", "Accept → handler"),
    ("header_recv_ms", "Header receive"),
    ("header_parse_ms", "Header parse"),
    ("buffer_alloc_ms", "Buffer allocation"),
    ("payload_recv_ms", "Payload receive"),
    ("validation_ms", "Validation"),
    ("host_ready_ms", "Host ready (may include earlier stages)"),
    ("request_rtt_ms", "Request RTT (enclosing duration)"),
]
NETWORK_LABELS = dict(NETWORK_ORDER)
GPU_ORDER = [
    ("submit_delay_ms", "Submission delay"),
    ("host_submit_ms", "Host submission"),
    ("first_gpu_delay_ms", "First GPU delay"),
    ("h2d_api_to_gpu_ms", "H2D API → GPU"),
    ("h2d_exec_ms", "H2D execution"),
    ("kernel_api_to_gpu_ms", "Kernel API → GPU"),
    ("kernel_exec_ms", "Kernel execution"),
    ("total_gpu_idle_between_ops_ms", "Within-request GPU activity gaps"),
    ("gpu_window_ms", "GPU window"),
    ("nvtx_duration_ms", "NVTX duration"),
    ("stream_elapsed_ms", "Stream elapsed"),
]
GPU_LABELS = dict(GPU_ORDER)


def network_details(latency: pd.DataFrame, date: str = 'ALL', cons=(1,2,4,8,16,32)) -> pd.DataFrame:
    """Existing source-file quantiles, median across independent files.

    This *is not* a pooled request-level P50/P95.
    """
    cols = ['payload_bytes','CON','metric','p50_ms','p95_ms','p99_ms','files']
    need = {'dataset','source_file','metric','payload_bytes','concurrency','p50_ms'}
    if latency is None or latency.empty or not need.issubset(latency.columns):
        return pd.DataFrame(columns=cols)
    d = latency[latency.dataset.astype(str).eq('physical')].copy()
    if date != 'ALL':
        d = d[d.source_file.astype(str).str.startswith(date + '_')]
    for c in ['payload_bytes','concurrency','p50_ms','p95_ms','p99_ms']:
        if c in d: d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[d.concurrency.isin(cons) & d.payload_bytes.gt(0) &
          d.metric.astype(str).str.endswith('_ms')]
    # All available latency metrics are retained, not just the four top-level metrics.
    aggregations = {'p50_ms': ('p50_ms','median'), 'files': ('source_file','nunique')}
    for c in ['p95_ms','p99_ms']:
        if c in d.columns: aggregations[c] = (c,'median')
    if d.empty: return pd.DataFrame(columns=cols)
    out = d.groupby(['payload_bytes','concurrency','metric'], as_index=False).agg(**aggregations)
    for c in ['p95_ms','p99_ms']:
        if c not in out: out[c] = np.nan
    return out.rename(columns={'concurrency':'CON'})[cols]


def network_metric_pivot(detail: pd.DataFrame, payload: int, field: str='p50_ms') -> pd.DataFrame:
    if detail is None or detail.empty: return pd.DataFrame(columns=['CON'])
    x = detail[detail.payload_bytes.eq(payload)]
    if x.empty: return pd.DataFrame(columns=['CON'])
    return x.pivot(index='CON', columns='metric', values=field).reset_index().sort_values('CON')


def h2d_details(h2d: pd.DataFrame) -> pd.DataFrame:
    if h2d is None or h2d.empty: return pd.DataFrame()
    d = h2d.copy()
    for c in ['gpu_p50_ms','gpu_p95_ms','host_p50_ms','host_p95_ms']:
        if c in d: d[c] = pd.to_numeric(d[c], errors='coerce')
    if 'host_p50_ms' in d and 'gpu_p50_ms' in d:
        d['host_minus_gpu_p50_ms'] = d.host_p50_ms - d.gpu_p50_ms
        d['host_over_gpu_p50_x'] = (d.host_p50_ms / d.gpu_p50_ms.where(d.gpu_p50_ms.gt(0)))
    if 'gpu_p95_ms' in d and 'gpu_p50_ms' in d:
        d['gpu_tail_p95_p50_x'] = d.gpu_p95_ms / d.gpu_p50_ms.where(d.gpu_p50_ms.gt(0))
    if 'host_p95_ms' in d and 'host_p50_ms' in d:
        d['host_tail_p95_p50_x'] = d.host_p95_ms / d.host_p50_ms.where(d.host_p50_ms.gt(0))
    # Derived contrasts of two independently computed quantiles, not a raw per-round delta.
    return d


def h2d_request_details(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Optional single-request transfer summaries from existing Step 2 analysis."""
    cols = ['h2d_bytes','CON','h2d_count','requests','source_files','h2d_p50_ms','h2d_p95_ms','h2d_p99_ms']
    if raw is None or raw.empty or not {'h2d_bytes','con','h2d_p50_ms'}.issubset(raw.columns):
        return pd.DataFrame(columns=cols)
    d = raw.copy().rename(columns={'con':'CON'})
    for c in cols:
        if c not in d: d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d[d.CON.isin((1,2,4,8,16,32))][cols].sort_values(['h2d_bytes','CON'])


def gpu_details(requests: pd.DataFrame, operator: str='ALL', cons=(1,2,4,8,16,32)) -> pd.DataFrame:
    """Median of source-level medians by exact (input bytes, CON).

    Overlapping diagnostics are kept distinct and never summed.
    """
    base = ['source_file','op','layer','input_bytes','con']
    outcols = ['input_bytes','CON','source_files','requests'] + [k for k,_ in GPU_ORDER]
    if requests is None or requests.empty or not set(base).issubset(requests.columns):
        return pd.DataFrame(columns=outcols)
    d = requests.copy()
    for c in ['input_bytes','con']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[d.input_bytes.gt(0) & d.con.isin(cons)]
    if operator != 'ALL': d = d[d.op.eq(operator)]
    measures = [key for key,_ in GPU_ORDER if key in d.columns]
    if not measures or d.empty: return pd.DataFrame(columns=outcols)
    for c in measures: d[c] = pd.to_numeric(d[c], errors='coerce')
    # First summarize across requests/repeats within each identical source workload.
    sf = d.groupby(base, dropna=False, as_index=False)[measures].median()
    sf['source_rows'] = d.groupby(base, dropna=False).size().values
    rules = {c:(c,'median') for c in measures}
    rules.update(source_files=('source_file','nunique'), requests=('source_rows','sum'))
    result = sf.groupby(['input_bytes','con'], as_index=False).agg(**rules)
    result = result.rename(columns={'con':'CON'})
    for c in outcols:
        if c not in result: result[c] = np.nan
    return result[outcols].sort_values(['input_bytes','CON']).reset_index(drop=True)


def physical_request_details(raw: pd.DataFrame | None, date: str = 'ALL', cons=(1,2,4,8,16,32)) -> pd.DataFrame:
    """NIC-to-host phase P50/P95/P99 from normalized *request-level* data.

    Valid request rows only; per-source_file quantiles first, then medians across
    files for a size/CON condition. This avoids bias from uneven file sizes.
    """
    cols = ['payload_bytes','CON','metric','p50_ms','p95_ms','p99_ms','files','requests']
    need = {'source_file','payload_bytes','network_con'}
    if raw is None or raw.empty or not need.issubset(raw.columns):
        return pd.DataFrame(columns=cols)
    d = raw.copy()
    for c in ['payload_bytes','network_con']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[d.payload_bytes.gt(0) & d.network_con.isin(cons)].copy()
    if date != 'ALL': d = d[d.source_file.astype(str).str.startswith(date + '_')]
    if 'status' in d:
        d = d[d.status.astype(str).str.lower().isin(('ok','success','verified'))]
    if 'verified' in d:
        d = d[d.verified.astype(str).str.lower().isin(('true','1','yes'))]
    phases = [x for x,_ in NETWORK_ORDER if x in d.columns]
    if not phases or d.empty: return pd.DataFrame(columns=cols)
    for x in phases: d[x] = pd.to_numeric(d[x], errors='coerce')
    melted = d.melt(id_vars=['source_file','payload_bytes','network_con'],
                    value_vars=phases, var_name='metric',value_name='duration_ms')
    melted = melted.dropna(subset=['duration_ms'])
    melted = melted[melted.duration_ms.ge(0)]
    if melted.empty: return pd.DataFrame(columns=cols)
    by_file = melted.groupby(['source_file','payload_bytes','network_con','metric'],as_index=False).agg(
        p50_ms=('duration_ms','median'),
        p95_ms=('duration_ms',lambda x:x.quantile(.95)),
        p99_ms=('duration_ms',lambda x:x.quantile(.99)),
        requests=('duration_ms','size'))
    result = by_file.groupby(['payload_bytes','network_con','metric'],as_index=False).agg(
        p50_ms=('p50_ms','median'),p95_ms=('p95_ms','median'),p99_ms=('p99_ms','median'),
        files=('source_file','nunique'),requests=('requests','sum'))
    return result.rename(columns={'network_con':'CON'})[cols]

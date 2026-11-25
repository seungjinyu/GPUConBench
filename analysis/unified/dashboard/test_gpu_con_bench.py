"""Synthetic regression tests; no real experiment values are fabricated."""
from __future__ import annotations
import math
import pandas as pd
import numpy as np
from step4_core import (CONS, network_summary, network_batch_summary,
                        h2d_summary, gpu_summary, gpu_size_summary, combined_sweep)


def fixtures():
    nlat = pd.DataFrame([
        {"dataset":"physical", "metric":metric, "payload_bytes":65536,
         "concurrency":con, "p50_ms":0.12*con if metric=="request_rtt_ms" else 0.02*con,
         "p95_ms":0.2*con, "source_file":"20261124_demo.csv"}
        for metric in ("request_rtt_ms", "payload_recv_ms") for con in CONS
    ])
    nrounds = pd.DataFrame([
        {"source_file":"20261124_demo.csv", "payload_bytes":65536,
         "network_con":con, "total_requests":con, "batch_elapsed_ms":0.10*con,
         "valid":True} for con in CONS for _ in range(3)
    ] + [{"source_file":"20261124_demo.csv", "payload_bytes":65536,
          "network_con":8, "total_requests":64, "batch_elapsed_ms":999.0,
          "valid":True}])
    h2d = pd.DataFrame([
        {"payload_bytes":65536, "con":con, "gpu_p50_ms":.05*con,
         "gpu_p95_ms":.07*con, "host_p50_ms":.08*con,
         "host_p95_ms":.11*con, "effective_gib_s_from_gpu_p50":5.0,
         "n_rounds":30} for con in CONS
    ])
    cov = pd.DataFrame([{"payload_bytes":65536,"con":con,"status":"measured"} for con in CONS])
    gpu = []
    for source,op,input_size in [("s1.csv","conv2d",65536),("s2.csv","relu6",65536),
                                  ("s3.csv","flatten",65536),("s4.csv","conv2d",262144)]:
        for con in CONS:
            for repeat in range(3):
                # A group has CON uniquely identified requests, and a shared finish time.
                makespan = (0.20 + repeat*0.02) * con * (1 if source != "s2.csv" else 2)
                for rid in range(con):
                    gpu.append(dict(source_file=source,op=op,layer=source,input_bytes=input_size,
                        con=con,repeat=repeat,request_id=rid,
                        global_latency_ms=makespan*(.5+.5*(rid+1)/con),
                        group_makespan_ms=makespan, transfer_mode="preloaded",
                        kernel_count=0 if op=="flatten" else 1,
                        kernel_exec_ms=np.nan if op=="flatten" else .004*con))
    return nlat,nrounds,h2d,cov,pd.DataFrame(gpu)


def build():
    lat, rounds, h2d, cov, gpu = fixtures()
    net, rates = network_summary(lat, rounds)
    batches = network_batch_summary(rounds)
    hs = h2d_summary(h2d,cov)
    gs, slow = gpu_summary(gpu)
    sizes = gpu_size_summary(gs)
    combo = combined_sweep(net,rates,hs,sizes,payload=65536,gpu_input_bytes=65536,network_batches=batches)
    return lat,rounds,gpu,net,rates,batches,hs,gs,slow,sizes,combo


def test_full_request_and_group_concurrency():
    *_,gs,slow,sizes,combo = build()
    assert combo.CON.tolist()==list(CONS)
    assert combo.complete_group_times.all()
    assert (combo.gpu_group_p50_ms >= combo.gpu_request_p50_ms).all()
    assert (combo.gpu_request_p50_ms > combo.gpu_kernel_p50_ms).all()
    assert math.isclose(combo.loc[combo.CON.eq(32),"gpu_group_p50_ms"].iloc[0], .22*32)
    assert len(slow)==len(CONS)*3


def test_group_makespan_not_multiplied_by_con():
    *_,gs,_,sizes,_ = build()
    # Each source has exactly three repeated groups, not 3 × CON group samples.
    assert gs.complete_repeats.eq(3).all()
    assert gs.loc[gs.source_file.eq("s1.csv") & gs.con.eq(32),"group_p50_ms"].iloc[0] == .22*32
    assert sizes.loc[sizes.input_bytes.eq(65536)&sizes.con.eq(32),"complete_repeats"].iloc[0]==9


def test_cpu_only_no_kernel_requests_are_kept():
    *_,gs,_,sizes,_=build()
    fl=gs[gs.op.eq("flatten")]
    assert len(fl)==6
    assert fl.request_p50_ms.notna().all()
    assert fl.group_p50_ms.notna().all()
    assert fl.kernel_p50_ms.isna().all()


def test_missing_repeat_and_no_zero_fill():
    lat,rounds,h,c,g=fixtures()
    g=g[~((g.source_file=="s1.csv") & (g.con==16) & (g.repeat==1) & (g.request_id==0))]
    gs,_=gpu_summary(g)
    s=gs[(gs.source_file=="s1.csv")&(gs.con==16)]
    assert s.complete_repeats.iloc[0]==2
    g=g[~((g.source_file=="s2.csv") & (g.con==32))]
    gs,_=gpu_summary(g)
    assert not ((gs.source_file=="s2.csv")&(gs.con==32)).any()
    sized=gpu_size_summary(gs,operator="relu6")
    assert 32 not in sized.con.tolist()


def test_network_rounds_require_exact_con_requests():
    _,rounds,*_=fixtures()
    r=network_batch_summary(rounds)
    assert r.loc[r.concurrency.eq(8),"batch_p50_ms"].iloc[0] == .8
    assert r.loc[r.concurrency.eq(8),"rounds"].iloc[0] == 3


def test_composite_sum_group_not_kernel():
    *_,combo = build()
    row=combo[combo.CON.eq(4)].iloc[0]
    expected=row.network_group_p50_ms+row.h2d_batch_host_p50_ms+row.gpu_group_p50_ms
    assert math.isclose(row.independent_group_time_sum_ms,expected)
    assert row.independent_group_time_sum_ms > row.gpu_kernel_p50_ms


def test_no_network_batch_no_fake_sum():
    lat,rounds,h,c,g=fixtures()
    net,rates=network_summary(lat,rounds)
    hs=h2d_summary(h,c)
    gs,_=gpu_summary(g)
    comp=combined_sweep(net,rates,hs,gpu_size_summary(gs),payload=65536,gpu_input_bytes=65536)
    assert comp.network_group_p50_ms.isna().all()
    assert comp.independent_group_time_sum_ms.isna().all()
    assert not comp.complete_group_times.any()


def test_operator_filter_and_size_profiles():
    *_,gs,_,sizes,combo=build()
    assert set(sizes.input_bytes)=={65536,262144}
    assert set(sizes.con)==set(CONS)
    conv=gpu_size_summary(gs,"conv2d")
    assert conv.operators.max()==1
    assert conv.loc[conv.input_bytes.eq(65536),"source_files"].eq(1).all()
    assert (conv.request_p50_ms > conv.kernel_p50_ms).all()


def test_source_paired_slowdown_group():
    *_,gs,slow,sizes,combo=build()
    one=combo.loc[combo.CON.eq(1),"gpu_group_slowdown_x"].iloc[0]
    assert math.isclose(one,1.0)
    assert math.isclose(combo.loc[combo.CON.eq(32),"gpu_group_slowdown_x"].iloc[0],32)
    assert 1 < combo.loc[combo.CON.eq(32),"gpu_request_slowdown_x"].iloc[0] < 32


def test_h2d_missing_not_assumed_zero():
    lat,rounds,h,c,g=fixtures()
    c.loc[c.con.eq(8),"status"]="oom"
    hs=h2d_summary(h,c)
    assert hs.loc[hs.concurrency.eq(8),"status"].iloc[0]=="OOM"
    assert np.isnan(hs.loc[hs.concurrency.eq(8),"host_p50_ms"].iloc[0])


def test_gpu_h2d_transfer_mode_disables_double_counting():
    lat,rounds,h,c,g=fixtures()
    g["transfer_mode"]="h2d"
    net,rates=network_summary(lat,rounds)
    nb=network_batch_summary(rounds)
    hs=h2d_summary(h,c)
    source,_=gpu_summary(g)
    comp=combined_sweep(net,rates,hs,gpu_size_summary(source),
                        payload=65536,gpu_input_bytes=65536,network_batches=nb)
    assert comp.gpu_preloaded.eq(False).all()
    assert comp.independent_group_time_sum_ms.isna().all()
    assert comp.gpu_group_p50_ms.notna().all()

# Node A local only vs offload completion (measured data never fabricated)
from step4_core import prepare_node_a_comparison_input, compare_node_a_vs_offload


def node_a_fixture(con=1, operator="conv2d", network_size=65536, gpu_size=65536,
                   local_ms=1.25, offload_e2e_ms=np.nan):
    return pd.DataFrame([dict(network_payload_bytes=network_size,
                              gpu_input_bytes=gpu_size, operator=operator, CON=con,
                              node_a_local_group_p50_ms=local_ms,
                              offload_e2e_group_p50_ms=offload_e2e_ms)])


def test_node_a_baseline_missing_does_not_fabricate_speedup():
    *_, combo = build()
    d = compare_node_a_vs_offload(combo, None, operator="conv2d")
    assert d.node_a_local_group_p50_ms.isna().all()
    assert d.node_a_vs_offload_speedup_x.isna().all()
    assert d.offload_standalone_proxy_ms.notna().all()  # test fixture uses preloaded GPU
    assert d.offload_comparison_kind.eq("ESTIMATED FROM STANDALONE STAGES").all()


def test_node_a_local_vs_preloaded_proxy_ratio():
    *_, combo = build()
    local = node_a_fixture(con=2, local_ms=9.0)
    d = compare_node_a_vs_offload(combo, local, operator="conv2d")
    r = d[d.CON.eq(2)].iloc[0]
    est = r.network_group_p50_ms + r.h2d_batch_host_p50_ms + r.gpu_group_p50_ms
    assert math.isclose(r.offload_standalone_proxy_ms, est)
    assert math.isclose(r.node_a_vs_offload_speedup_x, 9.0 / est)
    assert r.offload_comparison_kind == "ESTIMATED FROM STANDALONE STAGES"
    assert d.loc[d.CON.eq(32), "node_a_vs_offload_speedup_x"].isna().all()


def test_node_a_measured_offload_preferred_to_proxy():
    *_, combo = build()
    local = node_a_fixture(con=8, local_ms=12.0, offload_e2e_ms=6.0)
    d = compare_node_a_vs_offload(combo, local, operator="conv2d")
    r = d[d.CON.eq(8)].iloc[0]
    assert r.offload_comparison_kind == "MEASURED E2E"
    assert math.isclose(r.node_a_vs_offload_speedup_x, 2.0)
    assert math.isclose(r.node_a_vs_offload_time_saved_pct, 50.)
    assert math.isclose(r.offload_comparison_ms, 6.0)


def test_node_a_proxy_gpu_h2d_is_not_doubled():
    lat, rounds, h, c, g = fixtures()
    g["transfer_mode"] = "h2d"
    net, rates = network_summary(lat, rounds)
    nb = network_batch_summary(rounds)
    hs = h2d_summary(h, c)
    sources, _ = gpu_summary(g)
    combo = combined_sweep(net, rates, hs, gpu_size_summary(sources),
                           payload=65536, gpu_input_bytes=65536, network_batches=nb)
    d = compare_node_a_vs_offload(combo, node_a_fixture(con=1), operator="conv2d")
    r = d[d.CON.eq(1)].iloc[0]
    assert math.isclose(r.offload_standalone_proxy_ms,
                        r.network_group_p50_ms + r.gpu_group_p50_ms)
    assert r.independent_group_time_sum_ms != r.independent_group_time_sum_ms  # NaN
    assert "includes H2D" in r.offload_proxy_components


def test_node_a_unknown_gpu_mode_refuses_stage_sum():
    *_, combo = build()
    combo["gpu_transfer_modes"] = "unknown"
    d = compare_node_a_vs_offload(combo, node_a_fixture(con=1), operator="conv2d")
    assert d.offload_standalone_proxy_ms.isna().all()
    assert d.offload_comparison_kind.eq("UNAVAILABLE").all()
    assert d.node_a_vs_offload_speedup_x.isna().all()


def test_node_a_baseline_matching_and_duplicates():
    *_, combo = build()
    local = node_a_fixture(con=1, network_size=262144)
    d = compare_node_a_vs_offload(combo, local, operator="conv2d")
    assert d.node_a_local_group_p50_ms.isna().all()
    local = node_a_fixture(con=1)
    wrong_op = compare_node_a_vs_offload(combo, local, operator="relu6")
    assert wrong_op.node_a_local_group_p50_ms.isna().all()
    with __import__("pytest").raises(ValueError, match="duplicate"):
        prepare_node_a_comparison_input(pd.concat([local, local], ignore_index=True))


def test_node_a_invalid_zero_baseline_not_filled():
    local = node_a_fixture(con=1, local_ms=0.0)
    with __import__("pytest").raises(ValueError, match="positive"):
        prepare_node_a_comparison_input(local)

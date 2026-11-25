import unittest
import pandas as pd
import numpy as np
from stage_diagnostics import (physical_request_details, network_details, network_metric_pivot,
    h2d_details, h2d_request_details, gpu_details)

class StageDiagnosticsTests(unittest.TestCase):
    def test_network_raw_each_phase_and_quantiles(self):
        raw=pd.DataFrame([
            dict(source_file='20261124_a',payload_bytes=65536,network_con=1,
                 status='ok', verified=True,dial_ms=1,header_recv_ms=0.2,payload_recv_ms=3,request_rtt_ms=5),
            dict(source_file='20261124_a',payload_bytes=65536,network_con=1,
                 status='ok', verified=True,dial_ms=3,header_recv_ms=0.4,payload_recv_ms=5,request_rtt_ms=7),
            dict(source_file='20261124_a',payload_bytes=65536,network_con=1,
                 status='failed', verified=False,dial_ms=999,header_recv_ms=999,payload_recv_ms=999,request_rtt_ms=999),
            dict(source_file='20261124_b',payload_bytes=65536,network_con=1,
                 status='ok',verified=True,dial_ms=5,header_recv_ms=0.6,payload_recv_ms=7,request_rtt_ms=9),
        ])
        out=physical_request_details(raw)
        x=out[out.metric.eq('dial_ms')].iloc[0]
        self.assertEqual(x.files,2)
        self.assertEqual(x.requests,3)
        self.assertEqual(x.p50_ms, 3.5) # median of source A median 2, source B median 5
        self.assertEqual(len(out.metric.unique()),4)
        self.assertTrue(network_metric_pivot(out,65536).shape[1] == 5)
        self.assertTrue(physical_request_details(raw,date='20261123').empty)

    def test_network_unified_fallback_retains_extra_phases(self):
        df=pd.DataFrame([dict(dataset='physical', source_file='20261124_a',
            metric='header_parse_ms',payload_bytes=65536,concurrency=1,p50_ms=.04,p95_ms=.08)])
        x=network_details(df)
        self.assertEqual(x.iloc[0].metric, 'header_parse_ms')

    def test_h2d_clock_differences_not_bogus_decomposition(self):
        inp=pd.DataFrame([dict(payload_bytes=65536,concurrency=2,gpu_p50_ms=.08,
                    gpu_p95_ms=.1,host_p50_ms=.1,host_p95_ms=.12)])
        x=h2d_details(inp).iloc[0]
        self.assertAlmostEqual(x.host_minus_gpu_p50_ms,.02)
        self.assertAlmostEqual(x.gpu_tail_p95_p50_x,1.25)

    def test_per_request_h2d_exact_conditions(self):
        x=h2d_request_details(pd.DataFrame([{'h2d_bytes':65536,'con':32,'h2d_count':1,
                                            'h2d_p50_ms':.003}]))
        self.assertEqual(x.iloc[0].CON,32)
        self.assertTrue(np.isnan(x.iloc[0].h2d_p99_ms))

    def test_gpu_paired_source_medians(self):
        data=[]
        for name,offset in [('source_a',0),('source_b',2)]:
            for ms in [1+offset,3+offset]:
                data.append(dict(source_file=name,op='conv2d',layer='x',input_bytes=307200,
                    con=2,repeat=0,request_id=len(data)%2,host_submit_ms=ms,
                    kernel_exec_ms=ms*.1,first_gpu_delay_ms=np.nan))
        d=gpu_details(pd.DataFrame(data))
        x=d.iloc[0]
        self.assertEqual(x.source_files,2)
        self.assertEqual(x.requests,4)
        self.assertEqual(x.CON,2)
        self.assertAlmostEqual(x.host_submit_ms,3)
        self.assertTrue(pd.isna(x.first_gpu_delay_ms))
        self.assertTrue(gpu_details(pd.DataFrame(data),'relu6').empty)

if __name__ == '__main__': unittest.main()

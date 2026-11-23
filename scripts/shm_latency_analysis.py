
import csv
import math
from pathlib import Path

ROOT = Path.cwd()
LOG_ROOT = ROOT / "results/shm/sweep_logs"

METRICS = [
    "sender_write_ms",
    "receiver_allocation_ms",
    "receiver_copy_ms",
    "receiver_hostready_ms",
    "receiver_validation_ms",
    "ready_to_ack_ms",
    "end_to_end_ms",
]


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


# Find the latest sweep containing both configurations
runs = [
    log.parent
    for log in LOG_ROOT.glob("*/sender_100MiB_con1.log")
    if (log.parent / "sender_100MiB_con8.log").is_file()
]

if not runs:
    raise SystemExit("No matching Large Payload sweep logs found")

run_dir = max(runs, key=lambda p: p.stat().st_mtime)
print("Analyzing:", run_dir)

for con in [1, 8]:
    log_path = run_dir / f"sender_100MiB_con{con}.log"

    # Use exact CSV paths recorded by the Sender
    paths = []
    for line in log_path.read_text().splitlines():
        if line.startswith("CSV saved: "):
            paths.append(ROOT / line.removeprefix("CSV saved: ").strip())

    if len(paths) != 5:
        raise SystemExit(f"CON={con}: expected 5 trials, found {len(paths)}")

    all_rows = []

    print(f"\n=== 100MiB / CON={con} ===")

    for trial, path in enumerate(paths, 1):
        if f"_trial{trial:02d}_" not in path.name:
            raise SystemExit(f"Unexpected trial order: {path}")

        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = [
                {metric: float(row[metric]) for metric in METRICS}
                for row in reader
            ]

        expected = con * 100
        if len(rows) != expected:
            raise SystemExit(
                f"{path.name}: {len(rows)} rows, expected {expected}"
            )

        all_rows.extend(rows)

        e2e = [row["end_to_end_ms"] for row in rows]
        print(
            f"Trial {trial}: requests={len(rows)}, "
            f"E2E P50={percentile(e2e, 50):.3f} ms"
        )

    print(f"\nTotal requests: {len(all_rows)}")
    print(f"{'Metric':<29} {'P50':>11} {'P95':>11} {'P99':>11}")
    print("-" * 66)

    for metric in METRICS:
        values = [row[metric] for row in all_rows]

        print(
            f"{metric:<29}"
            f"{percentile(values, 50):>11.3f}"
            f"{percentile(values, 95):>11.3f}"
            f"{percentile(values, 99):>11.3f}"
        )

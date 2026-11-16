"""Select an idle GPU UUID without initializing CUDA."""
import argparse
import csv
import io
import subprocess
import sys


def select_idle_gpus(gpu_csv, process_csv, count=1):
    if count < 1:
        raise ValueError("GPU count must be positive")
    busy = {row[0].strip() for row in csv.reader(io.StringIO(process_csv)) if row}
    candidates = []
    for row in csv.reader(io.StringIO(gpu_csv)):
        if not row:
            continue
        index, uuid, total, used, free, utilization = [value.strip() for value in row]
        total, used, free, utilization = map(float, (total, used, free, utilization))
        # A process-free device may still be reserved by graphics or an unlisted
        # workload. Require low usage and at least 90% free as an extra check.
        if uuid not in busy and free >= 0.9 * total and utilization <= 10:
            candidates.append((free, uuid, index))
    if len(candidates) < count:
        raise RuntimeError(f'Need {count} idle GPUs, found {len(candidates)}; model loading has not started.')
    return sorted(candidates, reverse=True)[:count]


def select_idle_gpu(gpu_csv, process_csv):
    return select_idle_gpus(gpu_csv, process_csv, 1)[0]


def query_gpus(count=1):
    common = ['nvidia-smi', '--format=csv,noheader,nounits']
    gpu_csv = subprocess.check_output(
        common + ['--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu'],
        text=True, timeout=15,
    )
    process_csv = subprocess.check_output(
        common + ['--query-compute-apps=gpu_uuid'], text=True, timeout=15,
    )
    return select_idle_gpus(gpu_csv, process_csv, count)


def query_gpu():
    return query_gpus(1)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1)
    args = parser.parse_args()
    try:
        selected = query_gpus(args.count)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'GPU selection failed: {error}', file=sys.stderr)
        raise SystemExit(1)
    for free, uuid, index in selected:
        print(f'Selected physical GPU {index}: {free:.0f} MiB free ({uuid})', file=sys.stderr)
    print(','.join(uuid for free, uuid, index in selected))

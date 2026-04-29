from __future__ import annotations

import subprocess
from typing import Dict, List


def get_free_gpu_indices() -> List[int]:
    """Return sorted CUDA GPU indices with no active compute process."""
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()

    uuid_to_idx: Dict[str, int] = {}
    for row in gpu_rows:
        idx_str, uuid = [x.strip() for x in row.split(",", 1)]
        uuid_to_idx[uuid] = int(idx_str)

    running_rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    busy_gpu_uuids = {r.strip() for r in running_rows if r.strip() and r.strip() in uuid_to_idx}

    free = [idx for uuid, idx in uuid_to_idx.items() if uuid not in busy_gpu_uuids]
    free.sort()
    return free

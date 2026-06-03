"""CSV 保存、默认实验规模、结果目录探测和可选绘图工具。"""

from __future__ import annotations

import csv
import platform
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CASES: Tuple[Tuple[int, int, int], ...] = (
    (1, 1024, 1024),
    (8, 1024, 1024),
    (1, 2048, 2048),
    (8, 2048, 2048),
    (1, 4096, 4096),
    (8, 4096, 4096),
)


def project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parents[3]


def default_output_dir() -> Path:
    """根据运行环境返回统一 results 目录。

    Windows 原生环境优先保存到 ``D:\\TritonLowbitBench\\results``；
    WSL 环境优先保存到 ``/mnt/d/TritonLowbitBench/results``；如果这些路径不可用，
    则退回当前项目内的 ``results``。
    """
    candidates: List[Path]
    if platform.system().lower() == "windows":
        candidates = [Path("D:/TritonLowbitBench/results"), project_root() / "results"]
    else:
        candidates = [Path("/mnt/d/TritonLowbitBench/results"), project_root() / "results"]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    return Path("results")


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回该路径。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_cases(
    m_size: Optional[int],
    k_size: Optional[int],
    n_size: Optional[int],
) -> List[Tuple[int, int, int]]:
    """解析命令行矩阵规模。

    如果用户同时提供 ``--m``、``--k``、``--n``，则只运行这一组规模；
    否则运行默认六组规模。
    """
    if m_size is not None or k_size is not None or n_size is not None:
        if m_size is None or k_size is None or n_size is None:
            raise ValueError("--m, --k and --n must be provided together.")
        return [(m_size, k_size, n_size)]
    return list(DEFAULT_CASES)


def save_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    """把结果行保存为 UTF-8-SIG CSV，便于 Windows Excel 直接打开。"""
    if not rows:
        print(f"No rows to save for {path}.")
        return
    ensure_dir(path.parent)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {path}")


def plot_latency_bars(
    rows: Sequence[Dict[str, object]],
    route_keys: Iterable[str],
    output_path: Path,
    title: str,
) -> None:
    """如果 matplotlib 可用，则保存横向 latency 柱状图；否则只提示跳过。"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"matplotlib unavailable, skip plotting: {exc}")
        return

    if not rows:
        return
    ensure_dir(output_path.parent)
    labels = [f"M{r['M']}-K{r['K']}-N{r['N']}" for r in rows]
    route_keys = list(route_keys)
    y_pos = np.arange(len(labels))
    height = 0.8 / max(len(route_keys), 1)

    fig, ax = plt.subplots(figsize=(11, max(4, len(labels) * 0.55)))
    for index, key in enumerate(route_keys):
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        ax.barh(y_pos + index * height, values, height=height, label=key)
    ax.set_yticks(y_pos + height * (len(route_keys) - 1) / 2)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Latency (ms)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved figure: {output_path}")

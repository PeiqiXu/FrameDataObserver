from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

DEFAULT_WORKERS = min(16, os.cpu_count() or 4)

LAYOUT_AXES = {
    "tyx": ("t", "y", "x"),
    "txy": ("t", "x", "y"),
    "yxt": ("y", "x", "t"),
    "xyt": ("x", "y", "t"),
}

MIN_DISPLAY_POINTS = 500
DEFAULT_DISPLAY_POINTS = 1200


@dataclass(frozen=True)
class CategoryMeta:
    name: str
    prepared_dir: Path
    cube_path: Path
    block_sums_path: Path
    preview_path: Path
    source_path: Path
    layout: str
    dtype: str
    time_steps: int
    height: int
    width: int
    pixel_count: int
    block_size: int
    block_count: int
    lod_levels: int = 0                         # 0 = no LOD available
    block_sums_lod_path: Path | None = None     # concatenated LOD file
    lod_offsets: Tuple[int, ...] = ()            # row offsets for each LOD level in the file


@dataclass
class SelectionPlan:
    selection_size: int
    full_blocks: np.ndarray
    partial_ranges: np.ndarray   # shape (N, 2), each row is [start, stop) of contiguous pixel indices
    partial_pixel_count: int


@dataclass
class QueryResult:
    category_name: str
    start: int
    stop: int
    raw_time: np.ndarray | None
    raw_values: np.ndarray | None
    display_time: np.ndarray
    display_values: np.ndarray
    elapsed_ms: float
    aggregated: bool


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _axis_sizes(shape: Sequence[int], layout: str) -> Dict[str, int]:
    axes = LAYOUT_AXES[layout]
    return {axis: int(shape[idx]) for idx, axis in enumerate(axes)}


def _time_axis(layout: str) -> int:
    return LAYOUT_AXES[layout].index("t")


def _slice_time_chunk(source: np.ndarray, layout: str, start: int, stop: int) -> np.ndarray:
    time_axis = _time_axis(layout)
    if time_axis == 0:
        chunk = source[start:stop, :, :]
    elif time_axis == 1:
        chunk = source[:, start:stop, :]
    else:
        chunk = source[:, :, start:stop]

    axes = LAYOUT_AXES[layout]
    order = [axes.index("t"), axes.index("y"), axes.index("x")]
    return np.transpose(chunk, order)


def _default_time_chunk(height: int, width: int, dtype: np.dtype) -> int:
    bytes_per_frame = max(1, height * width * dtype.itemsize)
    target_bytes = 128 * 1024 * 1024
    return max(1, target_bytes // bytes_per_frame)


def _resolve_target_dtype(source: np.ndarray, dtype: str | None) -> np.dtype:
    if dtype is None:
        return source.dtype
    normalized = str(dtype).strip().lower()
    if normalized in {"", "auto", "source"}:
        return source.dtype
    return np.dtype(dtype)


def _query_time_chunk(column_count: int, itemsize: int) -> int:
    bytes_per_frame = max(1, column_count * max(1, itemsize))
    target_bytes = 64 * 1024 * 1024
    return max(1, target_bytes // bytes_per_frame)


def _build_preview(cube: np.ndarray, max_frames: int) -> np.ndarray:
    frame_count = min(max_frames, cube.shape[0])
    if frame_count <= 0:
        raise ValueError("Cube must contain at least one frame.")
    sample_idx = np.unique(np.linspace(0, cube.shape[0] - 1, frame_count, dtype=int))
    return cube[sample_idx].mean(axis=0, dtype=np.float64).astype(np.float32)


def _block_lengths(pixel_count: int, block_size: int) -> np.ndarray:
    if pixel_count <= 0:
        return np.array([], dtype=np.int64)
    block_count = math.ceil(pixel_count / block_size)
    lengths = np.full(block_count, block_size, dtype=np.int64)
    lengths[-1] = pixel_count - (block_count - 1) * block_size
    return lengths


def _indices_to_ranges(indices: np.ndarray) -> np.ndarray:
    """Convert sorted unique indices into an (N, 2) array of [start, stop) ranges."""
    if indices.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    breaks = np.flatnonzero(np.diff(indices) != 1)
    n = breaks.size + 1
    starts = np.empty(n, dtype=np.int64)
    stops = np.empty(n, dtype=np.int64)
    starts[0] = indices[0]
    stops[-1] = indices[-1] + 1
    if breaks.size:
        starts[1:] = indices[breaks + 1]
        stops[:-1] = indices[breaks] + 1
    return np.column_stack((starts, stops))


def _build_lod_pyramid(block_sums: np.ndarray, min_frames: int = MIN_DISPLAY_POINTS) -> Tuple[np.ndarray, List[int]]:
    """Build time-LOD pyramid by iteratively halving the time dimension.

    Level k stores frame-group sums: LOD frame j at level k = sum of
    original block_sums rows [j*2^k, min((j+1)*2^k, T)).

    Returns (concatenated_levels, offsets) where offsets[i] is the row
    offset of level i+1 inside the concatenated array.
    """
    levels: List[np.ndarray] = []
    current = np.asarray(block_sums, dtype=np.float64)  # materialise from mmap
    while current.shape[0] > min_frames:
        n = current.shape[0]
        even_n = n - n % 2
        paired = current[:even_n].reshape(even_n // 2, 2, -1).sum(axis=1)
        if n % 2:
            next_level = np.empty((even_n // 2 + 1, current.shape[1]), dtype=np.float64)
            next_level[: even_n // 2] = paired
            next_level[-1] = current[-1]
        else:
            next_level = paired
        levels.append(next_level)
        current = next_level
    if not levels:
        return np.empty((0, block_sums.shape[1]), dtype=np.float64), []
    offsets = [0]
    for lvl in levels:
        offsets.append(offsets[-1] + lvl.shape[0])
    return np.concatenate(levels, axis=0), offsets


def _choose_lod_level(span: int, max_points: int, max_level: int) -> int:
    """Pick the coarsest LOD level where frame count >= max_points."""
    point_cap = max(MIN_DISPLAY_POINTS, max_points)
    level = 0
    while level < max_level:
        next_frames = math.ceil(span / (1 << (level + 1)))
        if next_frames < point_cap:
            break
        level += 1
    return level


def _chunk_block_sums(flat_chunk: np.ndarray, pixel_count: int, block_size: int, block_count: int) -> np.ndarray:
    frame_count = flat_chunk.shape[0]
    result = np.zeros((frame_count, block_count), dtype=np.float64)
    full_block_count, remainder = divmod(pixel_count, block_size)
    if full_block_count:
        full_pixels = full_block_count * block_size
        result[:, :full_block_count] = flat_chunk[:, :full_pixels].reshape(
            frame_count, full_block_count, block_size
        ).sum(axis=2, dtype=np.float64)
    if remainder:
        result[:, full_block_count] = flat_chunk[:, full_block_count * block_size :].sum(
            axis=1, dtype=np.float64
        )
    return result


def list_prepared_categories(root_dir: str | Path) -> list[CategoryMeta]:
    root = Path(root_dir)
    if not root.exists():
        return []

    results = []
    for child in sorted(root.iterdir()):
        meta_path = child / "meta.json"
        if child.is_dir() and meta_path.exists():
            results.append(load_prepared_category(child))
    return results


def load_prepared_category(prepared_dir: str | Path) -> CategoryMeta:
    prepared_path = Path(prepared_dir)
    payload = _load_json(prepared_path / "meta.json")
    lod_levels = int(payload.get("lod_levels", 0))
    lod_file = str(payload.get("block_sums_lod_file", ""))
    lod_offsets = tuple(int(v) for v in payload.get("lod_offsets", []))
    return CategoryMeta(
        name=str(payload["name"]),
        prepared_dir=prepared_path,
        cube_path=prepared_path / str(payload["cube_file"]),
        block_sums_path=prepared_path / str(payload["block_sums_file"]),
        preview_path=prepared_path / str(payload["preview_file"]),
        source_path=Path(str(payload["source_path"])),
        layout=str(payload["source_layout"]),
        dtype=str(payload["dtype"]),
        time_steps=int(payload["time_steps"]),
        height=int(payload["height"]),
        width=int(payload["width"]),
        pixel_count=int(payload["pixel_count"]),
        block_size=int(payload["block_size"]),
        block_count=int(payload["block_count"]),
        lod_levels=lod_levels,
        block_sums_lod_path=prepared_path / lod_file if lod_file else None,
        lod_offsets=lod_offsets,
    )


def prepare_category(
    source_path: str | Path,
    prepared_root: str | Path,
    name: str | None = None,
    layout: str = "tyx",
    dtype: str | None = "auto",
    block_size: int = 4096,
    preview_frames: int = 64,
    workers: int = DEFAULT_WORKERS,
    lod: bool = True,
) -> CategoryMeta:
    source_path = Path(source_path)
    prepared_root = Path(prepared_root)
    if layout not in LAYOUT_AXES:
        raise ValueError(f"Unsupported layout: {layout}")

    source = np.load(source_path, mmap_mode="r")
    if source.ndim != 3:
        raise ValueError("Source array must be a 3D .npy cube.")

    dims = _axis_sizes(source.shape, layout)
    time_steps = dims["t"]
    height = dims["y"]
    width = dims["x"]
    pixel_count = height * width
    block_count = math.ceil(pixel_count / block_size)

    category_name = name or source_path.stem
    prepared_dir = prepared_root / category_name
    _ensure_dir(prepared_dir)

    cube_path = prepared_dir / "cube_tyx.npy"
    block_sums_path = prepared_dir / "block_sums.npy"
    target_dtype = _resolve_target_dtype(source, dtype)
    cube_memmap = np.lib.format.open_memmap(
        cube_path,
        mode="w+",
        dtype=target_dtype,
        shape=(time_steps, height, width),
    )
    block_sums_memmap = np.lib.format.open_memmap(
        block_sums_path,
        mode="w+",
        dtype=np.float64,
        shape=(time_steps, block_count),
    )

    time_chunk = _default_time_chunk(height, width, target_dtype)

    def _prep_chunk(start: int) -> None:
        stop = min(start + time_chunk, time_steps)
        chunk = _slice_time_chunk(source, layout, start, stop).astype(target_dtype, copy=False)
        cube_memmap[start:stop] = chunk
        block_sums_memmap[start:stop] = _chunk_block_sums(
            chunk.reshape(stop - start, pixel_count),
            pixel_count=pixel_count,
            block_size=block_size,
            block_count=block_count,
        )

    chunk_starts = list(range(0, time_steps, time_chunk))
    if workers > 1 and len(chunk_starts) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_prep_chunk, chunk_starts))
    else:
        for cs in chunk_starts:
            _prep_chunk(cs)

    del cube_memmap
    del block_sums_memmap

    # --- build time-LOD pyramid for block_sums ---
    lod_levels = 0
    lod_offsets: list[int] = []
    lod_file = ""
    if lod:
        bs_for_lod = np.load(block_sums_path, mmap_mode="r")
        lod_data, lod_offsets = _build_lod_pyramid(bs_for_lod, min_frames=MIN_DISPLAY_POINTS)
        lod_levels = len(lod_offsets) - 1 if lod_offsets else 0
        if lod_levels > 0:
            lod_file = "block_sums_lod.npy"
            np.save(prepared_dir / lod_file, lod_data)

    cube = np.load(cube_path, mmap_mode="r")
    preview = _build_preview(cube, preview_frames)
    np.save(prepared_dir / "preview.npy", preview)

    meta_payload = {
        "name": category_name,
        "cube_file": cube_path.name,
        "block_sums_file": block_sums_path.name,
        "preview_file": "preview.npy",
        "source_path": str(source_path.resolve()),
        "source_layout": layout,
        "dtype": str(target_dtype),
        "time_steps": time_steps,
        "height": height,
        "width": width,
        "pixel_count": pixel_count,
        "block_size": block_size,
        "block_count": block_count,
        "lod_levels": lod_levels,
        "block_sums_lod_file": lod_file,
        "lod_offsets": lod_offsets,
    }
    _save_json(prepared_dir / "meta.json", meta_payload)
    return load_prepared_category(prepared_dir)


def coords_to_flat_indices(coords: Iterable[Tuple[int, int]], width: int, height: int) -> np.ndarray:
    rows = []
    for x, y in coords:
        if 0 <= x < width and 0 <= y < height:
            rows.append(y * width + x)
    if not rows:
        return np.array([], dtype=np.int64)
    return np.unique(np.asarray(rows, dtype=np.int64))


def mask_to_flat_indices(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D boolean array.")
    return np.flatnonzero(mask.reshape(-1))


def decimate_series(
    time_index: np.ndarray,
    values: np.ndarray,
    max_points: int = DEFAULT_DISPLAY_POINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    point_cap = max(MIN_DISPLAY_POINTS, int(max_points))
    total = values.size
    if total <= point_cap:
        return time_index.astype(np.float64), values.astype(np.float64)

    target = min(point_cap, total)
    edges = np.linspace(0, total, target + 1, dtype=int)
    display_time = np.empty(target, dtype=np.float64)
    display_values = np.empty(target, dtype=np.float64)
    for idx in range(target):
        start = edges[idx]
        stop = max(start + 1, edges[idx + 1])
        display_time[idx] = time_index[start:stop].mean(dtype=np.float64)
        display_values[idx] = values[start:stop].mean(dtype=np.float64)
    return display_time, display_values


def _display_edges(start: int, stop: int, max_points: int) -> np.ndarray:
    point_cap = max(MIN_DISPLAY_POINTS, int(max_points))
    target = min(point_cap, stop - start)
    return np.linspace(start, stop, target + 1, dtype=np.int64)


class QueryEngine:
    def __init__(self) -> None:
        self._cube_cache: Dict[Path, np.ndarray] = {}
        self._block_cache: Dict[Path, np.ndarray] = {}
        self._lod_cache: Dict[Path, np.ndarray] = {}
        self._preview_cache: Dict[Path, np.ndarray] = {}
        self._selection_cache: Dict[str, SelectionPlan] = {}

    def cube(self, meta: CategoryMeta) -> np.ndarray:
        cube = self._cube_cache.get(meta.cube_path)
        if cube is None:
            cube = np.load(meta.cube_path, mmap_mode="r")
            self._cube_cache[meta.cube_path] = cube
        return cube

    def block_sums(self, meta: CategoryMeta) -> np.ndarray:
        block_sums = self._block_cache.get(meta.block_sums_path)
        if block_sums is None:
            block_sums = np.load(meta.block_sums_path, mmap_mode="r")
            self._block_cache[meta.block_sums_path] = block_sums
        return block_sums

    def preview(self, meta: CategoryMeta) -> np.ndarray:
        preview = self._preview_cache.get(meta.preview_path)
        if preview is None:
            preview = np.load(meta.preview_path, mmap_mode="r")
            self._preview_cache[meta.preview_path] = preview
        return preview

    def block_sums_lod(self, meta: CategoryMeta) -> np.ndarray | None:
        if meta.block_sums_lod_path is None or meta.lod_levels == 0:
            return None
        arr = self._lod_cache.get(meta.block_sums_lod_path)
        if arr is None:
            arr = np.load(meta.block_sums_lod_path, mmap_mode="r")
            self._lod_cache[meta.block_sums_lod_path] = arr
        return arr

    def _lod_block_sums_slice(self, meta: CategoryMeta, level: int) -> np.ndarray:
        """Return the block_sums array for a given LOD level (1-indexed).

        Level 0 is the original block_sums; levels 1..lod_levels come from
        the concatenated LOD file.
        """
        if level <= 0:
            return self.block_sums(meta)
        lod = self.block_sums_lod(meta)
        row_start = meta.lod_offsets[level - 1]
        row_stop = meta.lod_offsets[level]
        return lod[row_start:row_stop]

    def selection_plan(self, meta: CategoryMeta, flat_indices: Sequence[int]) -> SelectionPlan:
        indices = np.unique(np.asarray(flat_indices, dtype=np.int64))
        if indices.size == 0:
            raise ValueError("Please select at least one pixel.")
        if indices.min() < 0 or indices.max() >= meta.pixel_count:
            raise ValueError("Selection contains out-of-range pixels.")

        signature = self._selection_signature(meta, indices)
        cached = self._selection_cache.get(signature)
        if cached is not None:
            return cached

        block_lengths = _block_lengths(meta.pixel_count, meta.block_size)
        block_ids = indices // meta.block_size
        counts = np.bincount(block_ids, minlength=meta.block_count)
        full_blocks = np.flatnonzero(counts == block_lengths)
        full_mask = counts[block_ids] == block_lengths[block_ids]
        partial_indices = indices[~full_mask]

        plan = SelectionPlan(
            selection_size=int(indices.size),
            full_blocks=full_blocks.astype(np.int64, copy=False),
            partial_ranges=_indices_to_ranges(partial_indices),
            partial_pixel_count=int(partial_indices.size),
        )
        self._selection_cache[signature] = plan
        return plan

    def _query_chunk_frames(self, meta: CategoryMeta, plan: SelectionPlan) -> int:
        partial_itemsize = self.cube(meta).dtype.itemsize if plan.partial_pixel_count else 0
        columns = int(plan.full_blocks.size) + plan.partial_pixel_count
        itemsize = max(np.dtype(np.float64).itemsize, partial_itemsize)
        return _query_time_chunk(columns, itemsize)

    def _frame_sums(
        self,
        plan: SelectionPlan,
        flat_cube: np.ndarray,
        block_sums: np.ndarray,
        start: int,
        stop: int,
    ) -> np.ndarray:
        values = np.zeros(stop - start, dtype=np.float64)
        if plan.full_blocks.size:
            values += np.asarray(
                block_sums[start:stop, plan.full_blocks].sum(axis=1, dtype=np.float64),
                dtype=np.float64,
            ).reshape(-1)
        for i in range(plan.partial_ranges.shape[0]):
            rs, re = int(plan.partial_ranges[i, 0]), int(plan.partial_ranges[i, 1])
            values += flat_cube[start:stop, rs:re].sum(axis=1, dtype=np.float64)
        return values

    def _window_total(
        self,
        plan: SelectionPlan,
        flat_cube: np.ndarray,
        block_sums: np.ndarray,
        start: int,
        stop: int,
    ) -> float:
        total = 0.0
        if plan.full_blocks.size:
            total += float(block_sums[start:stop, plan.full_blocks].sum(dtype=np.float64))
        for i in range(plan.partial_ranges.shape[0]):
            rs, re = int(plan.partial_ranges[i, 0]), int(plan.partial_ranges[i, 1])
            total += float(flat_cube[start:stop, rs:re].sum(dtype=np.float64))
        return total

    @staticmethod
    def _partial_pixel_total(
        plan: SelectionPlan,
        flat_cube: np.ndarray,
        start: int,
        stop: int,
    ) -> float:
        """Sum partial-pixel values over [start, stop) from cube_tyx."""
        total = 0.0
        for i in range(plan.partial_ranges.shape[0]):
            rs, re = int(plan.partial_ranges[i, 0]), int(plan.partial_ranges[i, 1])
            total += float(flat_cube[start:stop, rs:re].sum(dtype=np.float64))
        return total

    def _query_aggregated_lod(
        self,
        meta: CategoryMeta,
        plan: SelectionPlan,
        flat_cube: np.ndarray | None,
        start: int,
        stop: int,
        max_points: int,
        workers: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """LOD-accelerated aggregated query.  Display points = LOD frames."""
        lod_level = _choose_lod_level(stop - start, max_points, meta.lod_levels)
        stride = 1 << lod_level
        lod_start = start // stride
        lod_stop = math.ceil(stop / stride)
        n_points = lod_stop - lod_start

        # --- complete blocks: one read from the LOD level ---
        if plan.full_blocks.size and lod_level > 0:
            lod_bs = self._lod_block_sums_slice(meta, lod_level)
            block_totals = np.asarray(
                lod_bs[lod_start:lod_stop, plan.full_blocks], dtype=np.float64,
            ).sum(axis=1).reshape(-1)
        elif plan.full_blocks.size:
            bs = self.block_sums(meta)
            block_totals = np.asarray(
                bs[start:stop, plan.full_blocks], dtype=np.float64,
            ).sum(axis=1).reshape(-1)
        else:
            block_totals = np.zeros(n_points, dtype=np.float64)

        # --- boundary correction for complete blocks at LOD edges ---
        # The first / last LOD frame may extend beyond [start, stop).
        # Subtract the out-of-range portion from block_totals.
        if lod_level > 0 and plan.full_blocks.size:
            head_trim_start = lod_start * stride
            head_trim_stop = start
            if head_trim_stop > head_trim_start:
                bs = self.block_sums(meta)
                block_totals[0] -= float(
                    bs[head_trim_start:head_trim_stop, plan.full_blocks].sum(dtype=np.float64)
                )
            tail_trim_start = stop
            tail_trim_stop = min(lod_stop * stride, meta.time_steps)
            if tail_trim_stop > tail_trim_start:
                bs = self.block_sums(meta)
                block_totals[-1] -= float(
                    bs[tail_trim_start:tail_trim_stop, plan.full_blocks].sum(dtype=np.float64)
                )

        # --- partial pixels: read from cube_tyx, accumulate per LOD frame ---
        pixel_totals = np.zeros(n_points, dtype=np.float64)
        if plan.partial_pixel_count and flat_cube is not None:
            frame_chunk = self._query_chunk_frames(meta, plan)

            def _partial_lod_frame(idx: int) -> Tuple[int, float]:
                orig_start = max(start, (lod_start + idx) * stride)
                orig_stop = min(stop, (lod_start + idx + 1) * stride)
                total = 0.0
                for cs in range(orig_start, orig_stop, frame_chunk):
                    ce = min(cs + frame_chunk, orig_stop)
                    total += self._partial_pixel_total(plan, flat_cube, cs, ce)
                return idx, total

            use_threads = workers > 1 and n_points > 1
            if use_threads:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for idx, val in pool.map(_partial_lod_frame, range(n_points)):
                        pixel_totals[idx] = val
            else:
                for idx in range(n_points):
                    _, val = _partial_lod_frame(idx)
                    pixel_totals[idx] = val

        # --- assemble display arrays ---
        display_time = np.empty(n_points, dtype=np.float64)
        display_values = np.empty(n_points, dtype=np.float64)
        for idx in range(n_points):
            orig_start = max(start, (lod_start + idx) * stride)
            orig_stop = min(stop, (lod_start + idx + 1) * stride)
            count = orig_stop - orig_start
            display_time[idx] = 0.5 * float(orig_start + orig_stop - 1)
            display_values[idx] = (block_totals[idx] + pixel_totals[idx]) / float(
                plan.selection_size * count
            )
        return display_time, display_values

    def query_category(
        self,
        meta: CategoryMeta,
        flat_indices: Sequence[int],
        start: int = 0,
        stop: int | None = None,
        max_points: int = DEFAULT_DISPLAY_POINTS,
        workers: int = DEFAULT_WORKERS,
    ) -> QueryResult:
        plan = self.selection_plan(meta, flat_indices)
        stop = meta.time_steps if stop is None else int(stop)
        start = int(start)
        if start < 0 or stop > meta.time_steps or start >= stop:
            raise ValueError("Time range is invalid.")

        clock_start = time.perf_counter()
        span = stop - start
        flat_cube = None
        if plan.partial_pixel_count:
            flat_cube = self.cube(meta).reshape(meta.time_steps, meta.pixel_count)
        block_sums = self.block_sums(meta)
        frame_chunk = self._query_chunk_frames(meta, plan)
        display_edges = _display_edges(start, stop, max_points)
        aggregated = display_edges.size - 1 < span
        use_threads = workers > 1

        lod_level = _choose_lod_level(span, max_points, meta.lod_levels) if meta.lod_levels > 0 else 0

        if aggregated and lod_level > 0:
            # ---- LOD fast path ----
            raw_time = None
            raw_values = None
            display_time, display_values = self._query_aggregated_lod(
                meta, plan, flat_cube, start, stop, max_points, workers,
            )
        elif aggregated:
            # ---- legacy aggregated path (no LOD data) ----
            raw_time = None
            raw_values = None
            n_bins = display_edges.size - 1
            display_time = np.empty(n_bins, dtype=np.float64)
            display_values = np.empty(n_bins, dtype=np.float64)

            def _agg_bin(idx: int) -> Tuple[int, float, float]:
                bin_start = int(display_edges[idx])
                bin_stop = max(bin_start + 1, int(display_edges[idx + 1]))
                bin_total = 0.0
                for cs in range(bin_start, bin_stop, frame_chunk):
                    ce = min(cs + frame_chunk, bin_stop)
                    bin_total += self._window_total(plan, flat_cube, block_sums, cs, ce)
                dt = 0.5 * float(bin_start + bin_stop - 1)
                dv = bin_total / float(plan.selection_size * (bin_stop - bin_start))
                return idx, dt, dv

            if use_threads:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for idx, dt, dv in pool.map(_agg_bin, range(n_bins)):
                        display_time[idx] = dt
                        display_values[idx] = dv
            else:
                for idx in range(n_bins):
                    _, dt, dv = _agg_bin(idx)
                    display_time[idx] = dt
                    display_values[idx] = dv
        else:
            raw_time = np.arange(start, stop, dtype=np.int64)
            raw_values = np.empty(span, dtype=np.float64)

            def _raw_chunk(chunk_start: int) -> Tuple[int, int, np.ndarray]:
                chunk_stop = min(chunk_start + frame_chunk, stop)
                sums = self._frame_sums(plan, flat_cube, block_sums, chunk_start, chunk_stop)
                return chunk_start - start, chunk_stop - start, sums / float(plan.selection_size)

            chunk_starts = list(range(start, stop, frame_chunk))
            if use_threads:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for offset, end, vals in pool.map(_raw_chunk, chunk_starts):
                        raw_values[offset:end] = vals
            else:
                for cs in chunk_starts:
                    offset, end, vals = _raw_chunk(cs)
                    raw_values[offset:end] = vals

            display_time = raw_time.astype(np.float64)
            display_values = raw_values.copy()
        elapsed_ms = (time.perf_counter() - clock_start) * 1000.0

        return QueryResult(
            category_name=meta.name,
            start=start,
            stop=stop,
            raw_time=raw_time,
            raw_values=raw_values,
            display_time=display_time,
            display_values=display_values,
            elapsed_ms=elapsed_ms,
            aggregated=aggregated,
        )

    def query_categories(
        self,
        metas: List[CategoryMeta],
        flat_indices: Sequence[int],
        start: int = 0,
        stop: int | None = None,
        max_points: int = DEFAULT_DISPLAY_POINTS,
        workers: int = DEFAULT_WORKERS,
    ) -> List[QueryResult]:
        """Query multiple categories in parallel using threads."""
        if not metas:
            return []
        if len(metas) == 1:
            return [self.query_category(metas[0], flat_indices, start, stop, max_points, workers)]

        # Distribute workers: outer threads for categories, inner threads for chunks.
        cat_workers = min(len(metas), workers)
        inner_workers = max(1, workers // cat_workers)

        def _query_one(meta: CategoryMeta) -> QueryResult:
            return self.query_category(meta, flat_indices, start, stop, max_points, inner_workers)

        with ThreadPoolExecutor(max_workers=cat_workers) as pool:
            return list(pool.map(_query_one, metas))

    @staticmethod
    def _selection_signature(meta: CategoryMeta, indices: np.ndarray) -> str:
        payload = hashlib.blake2b(digest_size=16)
        payload.update(meta.name.encode("utf-8"))
        payload.update(str(meta.block_size).encode("ascii"))
        payload.update(indices.tobytes())
        return payload.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and query 3D data cubes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build prepared files from a raw .npy cube.")
    prepare_parser.add_argument("--source", required=True, help="Path to raw .npy file.")
    prepare_parser.add_argument("--prepared-root", required=True, help="Output directory for prepared datasets.")
    prepare_parser.add_argument("--name", help="Category name. Defaults to the file stem.")
    prepare_parser.add_argument(
        "--layout",
        default="tyx",
        choices=sorted(LAYOUT_AXES),
        help="Axis order in the source cube.",
    )
    prepare_parser.add_argument(
        "--dtype",
        default="auto",
        help="Data type for the prepared cube. Use auto to keep the source dtype.",
    )
    prepare_parser.add_argument("--block-size", type=int, default=4096, help="Pixel count per spatial block.")
    prepare_parser.add_argument(
        "--preview-frames",
        type=int,
        default=64,
        help="Sampled frame count used to build the spatial preview.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        meta = prepare_category(
            source_path=args.source,
            prepared_root=args.prepared_root,
            name=args.name,
            layout=args.layout,
            dtype=args.dtype,
            block_size=args.block_size,
            preview_frames=args.preview_frames,
        )
        print(
            f"Prepared {meta.name}: "
            f"time={meta.time_steps}, height={meta.height}, width={meta.width}, "
            f"block_size={meta.block_size}, block_count={meta.block_count}"
        )


if __name__ == "__main__":
    main()

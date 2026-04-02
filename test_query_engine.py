from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from query_engine import QueryEngine, prepare_category


class QueryEngineTests(unittest.TestCase):
    def test_query_matches_bruteforce_for_tyx_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cube = np.arange(12 * 5 * 4, dtype=np.float32).reshape(12, 5, 4)
            source = temp / "cube.npy"
            np.save(source, cube)

            meta = prepare_category(source, temp / "prepared", name="demo", layout="tyx", block_size=6)
            engine = QueryEngine()

            selected = np.array([0, 3, 9, 10, 19], dtype=np.int64)
            result = engine.query_category(meta, selected, start=2, stop=10, max_points=1000)

            expected = cube[2:10].reshape(8, -1)[:, selected].mean(axis=1)
            self.assertFalse(result.aggregated)
            self.assertIsNotNone(result.raw_time)
            self.assertIsNotNone(result.raw_values)
            np.testing.assert_allclose(result.raw_values, expected, rtol=0.0, atol=1e-5)
            np.testing.assert_allclose(result.display_values, expected, rtol=0.0, atol=1e-5)

    def test_prepare_supports_xyt_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cube_tyx = np.arange(9 * 3 * 4, dtype=np.float32).reshape(9, 3, 4)
            cube_xyt = np.transpose(cube_tyx, (2, 1, 0))
            source = temp / "cube_xyt.npy"
            np.save(source, cube_xyt)

            meta = prepare_category(source, temp / "prepared", name="xyt_demo", layout="xyt", block_size=5)
            engine = QueryEngine()

            selected = np.array([0, 4, 5, 10], dtype=np.int64)
            result = engine.query_category(meta, selected, start=1, stop=9, max_points=1000)

            expected = cube_tyx[1:9].reshape(8, -1)[:, selected].mean(axis=1)
            np.testing.assert_allclose(result.raw_values, expected, rtol=0.0, atol=1e-5)

    def test_large_time_range_is_aggregated_without_raw_series(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cube = np.arange(900 * 3 * 2, dtype=np.float32).reshape(900, 3, 2)
            source = temp / "large_cube.npy"
            np.save(source, cube)

            meta = prepare_category(source, temp / "prepared", name="large_demo", layout="tyx", block_size=3)
            engine = QueryEngine()

            selected = np.array([0, 2, 4], dtype=np.int64)
            result = engine.query_category(meta, selected, start=100, stop=900, max_points=200)

            self.assertTrue(result.aggregated)
            self.assertIsNone(result.raw_time)
            self.assertIsNone(result.raw_values)

            raw = cube[100:900].reshape(800, -1)[:, selected].mean(axis=1)
            edges = np.linspace(100, 900, 501, dtype=np.int64)
            expected_time = np.empty(500, dtype=np.float64)
            expected_values = np.empty(500, dtype=np.float64)
            for idx in range(500):
                bin_start = int(edges[idx])
                bin_stop = max(bin_start + 1, int(edges[idx + 1]))
                local_start = bin_start - 100
                local_stop = bin_stop - 100
                expected_time[idx] = 0.5 * float(bin_start + bin_stop - 1)
                expected_values[idx] = raw[local_start:local_stop].mean(dtype=np.float64)

            np.testing.assert_allclose(result.display_time, expected_time, rtol=0.0, atol=1e-8)
            np.testing.assert_allclose(result.display_values, expected_values, rtol=0.0, atol=1e-5)

    def test_prepare_auto_dtype_preserves_source_precision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            engine = QueryEngine()
            selected = np.array([0, 1, 3], dtype=np.int64)

            for dtype in (np.float32, np.int16, np.int8):
                with self.subTest(dtype=np.dtype(dtype).name):
                    cube = (np.arange(6 * 2 * 2).reshape(6, 2, 2) - 8).astype(dtype)
                    source = temp / f"{np.dtype(dtype).name}.npy"
                    np.save(source, cube)

                    meta = prepare_category(
                        source,
                        temp / "prepared",
                        name=np.dtype(dtype).name,
                        layout="tyx",
                        dtype="auto",
                        block_size=2,
                    )
                    result = engine.query_category(meta, selected, start=1, stop=6, max_points=1000)

                    self.assertEqual(meta.dtype, np.dtype(dtype).name)
                    expected = cube[1:6].reshape(5, -1)[:, selected].mean(axis=1)
                    np.testing.assert_allclose(result.raw_values, expected, rtol=0.0, atol=1e-5)


    def test_lod_query_matches_bruteforce(self) -> None:
        """LOD-accelerated aggregated query should match brute-force mean."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            # Large enough T so LOD levels are generated and used.
            T, H, W = 2000, 4, 5
            cube = np.arange(T * H * W, dtype=np.float32).reshape(T, H, W)
            source = temp / "cube.npy"
            np.save(source, cube)

            meta = prepare_category(source, temp / "prepared", name="lod_demo", layout="tyx", block_size=6)
            self.assertGreater(meta.lod_levels, 0, "LOD levels should be generated for T=2000")

            engine = QueryEngine()
            # Select a mix of full-block and partial pixels.
            selected = np.array([0, 1, 2, 3, 4, 5, 8, 12, 19], dtype=np.int64)
            # Query a large range that triggers aggregation + LOD.
            result = engine.query_category(meta, selected, start=100, stop=1900, max_points=500)

            self.assertTrue(result.aggregated)
            self.assertIsNone(result.raw_time)
            self.assertGreaterEqual(result.display_time.size, 500)

            # Brute-force: compute mean for each LOD display bin.
            raw = cube[100:1900].reshape(1800, -1)[:, selected].astype(np.float64)
            from query_engine import _choose_lod_level
            lod_level = _choose_lod_level(1800, 500, meta.lod_levels)
            for idx in range(result.display_time.size):
                dv = result.display_values[idx]
                stride = 1 << lod_level
                lod_start = 100 // stride
                orig_start = max(100, (lod_start + idx) * stride)
                orig_stop = min(1900, (lod_start + idx + 1) * stride)
                expected = raw[orig_start - 100 : orig_stop - 100].mean()
                self.assertAlmostEqual(dv, expected, places=5,
                                       msg=f"Mismatch at display point {idx}")

    def test_lod_pyramid_stored_in_meta(self) -> None:
        """Verify LOD metadata and file exist after prepare."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cube = np.arange(1200 * 3 * 4, dtype=np.float32).reshape(1200, 3, 4)
            source = temp / "cube.npy"
            np.save(source, cube)

            meta = prepare_category(source, temp / "prepared", name="lod_meta_test", layout="tyx", block_size=4)
            self.assertGreater(meta.lod_levels, 0)
            self.assertIsNotNone(meta.block_sums_lod_path)
            self.assertTrue(meta.block_sums_lod_path.exists())
            self.assertEqual(len(meta.lod_offsets), meta.lod_levels + 1)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Small deterministic tests for generic edit-program control maps."""

from __future__ import annotations

import unittest

import numpy as np

from PIL import Image

from edit_program import (
    adaptive_writeback_map,
    build_trimap,
    rasterize_creation_map,
    rasterize_preservation_hint,
    weighted_map,
)


class EditProgramTests(unittest.TestCase):
    def test_rasterizes_category_agnostic_shapes(self) -> None:
        program = {
            "layout": {
                "coordinate_space": "normalized",
                "primitives": [
                    {"type": "ellipse", "bbox": [0.1, 0.2, 0.4, 0.6], "stroke_width": 0.02},
                    {"type": "polyline", "points": [[0.4, 0.4], [0.6, 0.4]], "stroke_width": 0.02},
                ],
            }
        }
        creation = rasterize_creation_map(program, (100, 80))
        values = np.asarray(creation)
        self.assertGreater(int((values > 0).sum()), 30)
        self.assertLess(int((values > 0).sum()), values.size // 2)

    def test_trimap_is_bounded_and_preservation_complements_edit(self) -> None:
        program = {
            "layout": {
                "coordinate_space": "normalized",
                "primitives": [
                    {"type": "rectangle", "bbox": [0.3, 0.3, 0.7, 0.7], "stroke_width": 0.03}
                ],
            }
        }
        creation = rasterize_creation_map(program, (96, 96))
        trimap = build_trimap(creation, transition_dilation_px=4, transition_feather_px=1.0)
        c = np.asarray(trimap["creation"], dtype=np.float32) / 255.0
        t = np.asarray(trimap["transition"], dtype=np.float32) / 255.0
        p = np.asarray(trimap["preservation"], dtype=np.float32) / 255.0
        self.assertTrue(np.all(c + t + p <= 1.01))
        self.assertGreater(float(t.sum()), 0.0)
        self.assertGreater(float(p.sum()), 0.0)
        weight = np.asarray(weighted_map(trimap, 0.35), dtype=np.float32) / 255.0
        self.assertAlmostEqual(float(weight[c > 0.99].mean()), 1.0, places=2)
        self.assertLess(float(weight[t > 0.1].mean()), 0.5)

    def test_adaptive_writeback_preserves_explicit_hint(self) -> None:
        program = {
            "layout": {
                "coordinate_space": "normalized",
                "primitives": [
                    {"type": "rectangle", "bbox": [0.2, 0.2, 0.8, 0.8], "stroke_width": 0.04}
                ],
            },
            "preservation_layout": {
                "coordinate_space": "normalized",
                "primitives": [
                    {"type": "rectangle", "bbox": [0.45, 0.45, 0.55, 0.55], "fill": True}
                ],
            },
        }
        reference_array = np.full((64, 64, 3), 128, dtype=np.uint8)
        proposal_array = reference_array.copy()
        proposal_array[12:52, 12:52] = 20
        reference = Image.fromarray(reference_array, mode="RGB")
        proposal = Image.fromarray(proposal_array, mode="RGB")
        anchor = rasterize_creation_map(program, reference.size)
        preservation = rasterize_preservation_hint(program, reference.size)
        maps = adaptive_writeback_map(
            reference,
            proposal,
            anchor,
            preservation,
            support_dilation_px=8,
            delta_quantile=40,
            detail_quantile=40,
            high_delta_quantile=70,
        )
        alpha = np.asarray(maps["alpha"], dtype=np.uint8)
        protected = np.asarray(preservation, dtype=np.uint8) > 0
        self.assertGreater(int((alpha > 0).sum()), 0)
        self.assertTrue(np.all(alpha[protected] == 0))


if __name__ == "__main__":
    unittest.main()

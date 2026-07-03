import unittest

from experiments_l131.build_splits import both_unseen, split_audit
from experiments_l131.common import canonicalize_pairs, expected_calibration_error


class PairManifestTests(unittest.TestCase):
    def test_label_conflict_is_permanently_excluded(self):
        rows = [
            {"pair_id": "p1", "protein_A": "A", "protein_B": "B", "label": "1"},
            {"pair_id": "p2", "protein_A": "B", "protein_B": "A", "label": "0"},
            {"pair_id": "p3", "protein_A": "A", "protein_B": "B", "label": "1"},
        ]
        accepted, rejected = canonicalize_pairs(rows)
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 3)
        self.assertTrue(all(row["exclusion_reason"] == "label_conflict" for row in rejected))

    def test_both_unseen_has_no_cross_split_protein_overlap(self):
        rows = [
            {
                "pair_id": f"pair_{index}",
                "protein_A": f"A{index}",
                "protein_B": f"B{index}",
                "label": str(index % 2),
            }
            for index in range(30)
        ]
        split_rows, _ = both_unseen(rows, seed=1337, ratios=(0.6, 0.2, 0.2))
        audit = split_audit(split_rows)
        for values in audit["overlap"].values():
            self.assertEqual(values, [])

    def test_ece_is_zero_for_exact_bin_calibration(self):
        self.assertAlmostEqual(
            expected_calibration_error([0.0, 1.0], [0, 1], n_bins=2),
            0.0,
        )

if __name__ == "__main__":
    unittest.main()

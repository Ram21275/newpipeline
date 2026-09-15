import unittest

from lger.layer_decoding_analysis import (
    LayerDecodingAnalysisError,
    METHOD_PREFIXES,
    analyze_layer_decoding,
)


def rows() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for decision, image, target in (("d0", "i0", 0), ("d1", "i1", 1)):
        for control in ("image", "prompt_only", "image_shuffled"):
            sign = 1 if target else -1
            ordinary_raw = 0.2 if target else -0.2
            row: dict[str, object] = {
                "decision_id": decision,
                "image_id": image,
                "class_id": f"c{target}",
                "attribute_id": f"a{target}",
                "target": target,
                "control": control,
                "binary_premature_layer": 20 + target,
                "full_vocab_premature_layer": 21 + target,
                "binary_deco_anchor_layer": 22 + target,
                "full_vocab_deco_anchor_layer": 23 + target,
            }
            for method in METHOD_PREFIXES:
                gain = 0.0
                if method == "full_vocab_deco" and control == "image":
                    gain = 0.5
                raw = ordinary_raw + sign * gain
                row[f"{method}_semantic_margin"] = raw
                row[f"{method}_correct_margin"] = sign * raw
                row[f"{method}_correct"] = 1
            output.append(row)
    return output


class LayerDecodingAnalysisTests(unittest.TestCase):
    def test_paired_primary_and_visual_specificity(self) -> None:
        result = analyze_layer_decoding(rows(), samples=100, seed=4)
        primary = {
            row["metric"]: row
            for row in result["primary_promotion_snapshot"]
        }
        self.assertAlmostEqual(primary["correct_margin_difference"]["estimate"], 0.5)
        visual = next(
            row
            for row in result["visual_specificity_inference"]
            if row["method"] == "full_vocab_deco"
            and row["contrast"] == "image_gain_minus_prompt_only_gain"
        )
        self.assertAlmostEqual(visual["estimate"], 0.5)
        self.assertEqual(result["decision_count"], 2)
        self.assertEqual(result["image_count"], 2)

    def test_incomplete_coverage_is_rejected(self) -> None:
        with self.assertRaises(LayerDecodingAnalysisError):
            analyze_layer_decoding(rows()[:-1], samples=100)


if __name__ == "__main__":
    unittest.main()

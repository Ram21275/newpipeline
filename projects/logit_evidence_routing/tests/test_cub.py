import tempfile
import unittest
from pathlib import Path

from lger.cub import (
    CubBoundingBox,
    discover_cub_root,
    load_cub_bounding_boxes,
    load_cub_records,
    make_balanced_pilot_split,
    map_bbox_to_center_crop,
)


class CubTests(unittest.TestCase):
    def build_cub(self, parent: Path) -> Path:
        root = parent / "dataset" / "CUB_200_2011"
        (root / "images").mkdir(parents=True)
        images = []
        labels = []
        splits = []
        bounding_boxes = []
        image_id = 1
        for label in range(1, 4):
            class_name = f"{label:03d}.class_{label}"
            (root / "images" / class_name).mkdir()
            for example in range(5):
                relative = f"{class_name}/image_{example}.jpg"
                (root / "images" / relative).touch()
                images.append(f"{image_id} {relative}")
                labels.append(f"{image_id} {label}")
                splits.append(f"{image_id} {1 if example < 4 else 0}")
                bounding_boxes.append(f"{image_id} 10 20 30 40")
                image_id += 1
        (root / "images.txt").write_text("\n".join(images) + "\n")
        (root / "image_class_labels.txt").write_text("\n".join(labels) + "\n")
        (root / "train_test_split.txt").write_text("\n".join(splits) + "\n")
        (root / "bounding_boxes.txt").write_text(
            "\n".join(bounding_boxes) + "\n"
        )
        (root / "classes.txt").write_text(
            "1 001.class_1\n2 002.class_2\n3 003.class_3\n"
        )
        return root

    def test_discovery_parsing_and_split_exclude_official_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            search_root = Path(temporary)
            root = self.build_cub(search_root)
            self.assertEqual(discover_cub_root(search_root), root.resolve())
            records = load_cub_records(root)
            pilot = make_balanced_pilot_split(
                records,
                num_classes=2,
                train_per_class=2,
                val_per_class=1,
                seed=7,
            )
            self.assertEqual(len(pilot), 6)
            official_test_ids = {
                record.image_id for record in records if record.official_split == "test"
            }
            self.assertTrue(official_test_ids.isdisjoint({row.image_id for row in pilot}))
            repeated = make_balanced_pilot_split(
                records,
                num_classes=2,
                train_per_class=2,
                val_per_class=1,
                seed=7,
            )
            self.assertEqual(pilot, repeated)

    def test_bounding_boxes_load_and_follow_center_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.build_cub(Path(temporary))
            boxes = load_cub_bounding_boxes(root)
            self.assertEqual(boxes[1], CubBoundingBox(10.0, 20.0, 30.0, 40.0))

        mapped = map_bbox_to_center_crop(
            CubBoundingBox(25.0, 0.0, 50.0, 50.0),
            original_size=(100, 50),
            output_size=(40, 40),
        )
        self.assertEqual(mapped, (0.0, 0.0, 40.0, 40.0))


if __name__ == "__main__":
    unittest.main()

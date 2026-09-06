import tempfile
import unittest
from pathlib import Path

from lger.cub import (
    CubBoundingBox,
    center_crop_transform,
    discover_cub_root,
    load_cub_attributes,
    load_cub_certainties,
    load_cub_bounding_boxes,
    load_cub_image_attribute_labels,
    load_cub_part_locations,
    load_cub_records,
    make_balanced_pilot_split,
    materialize_attribute_targets,
    map_bbox_to_center_crop,
    map_point_to_center_crop,
    select_training_attribute_subset,
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
        (root / "parts").mkdir()
        (root / "parts" / "part_locs.txt").write_text(
            "\n".join(
                f"{current_id} 1 20 30 {1 if current_id % 2 else 0}"
                for current_id in range(1, image_id)
            )
            + "\n"
        )
        (root / "attributes").mkdir()
        (root / "attributes" / "attributes.txt").write_text(
            "1 has_crown_color::red\n"
            "2 has_wing_pattern::striped\n"
        )
        (root / "attributes" / "certainties.txt").write_text(
            "1 not visible\n"
            "2 guess\n"
            "3 probably\n"
            "4 definitely\n"
        )
        (root / "attributes" / "image_attribute_labels.txt").write_text(
            "1 1 1 4 1.0\n"
            "2 1 0 3 1.1\n"
            "3 1 1 2 1.2\n"
            "4 1 0 4 1.3\n"
            "1 2 1 1 1.4\n"
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

        point = map_point_to_center_crop(
            (50.0, 25.0), original_size=(100, 50), output_size=(40, 40)
        )
        self.assertEqual(point, (20.0, 20.0))
        self.assertIsNone(
            map_point_to_center_crop(
                (0.0, 25.0), original_size=(100, 50), output_size=(40, 40)
            )
        )

    def test_part_locations_load_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.build_cub(Path(temporary))
            locations = load_cub_part_locations(root)
            self.assertTrue(locations[1][0].visible)
            self.assertFalse(locations[2][0].visible)

    def test_attribute_labels_preserve_uncertainty_and_missingness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.build_cub(Path(temporary))
            attributes = load_cub_attributes(root)
            certainties = load_cub_certainties(root)
            labels = load_cub_image_attribute_labels(root, image_ids={1, 2})
            targets = materialize_attribute_targets(
                attributes, certainties, labels[1]
            )
            self.assertEqual(attributes[0].group, "has_crown_color")
            self.assertEqual(targets[0]["state"], "present")
            self.assertTrue(targets[0]["primary_target"])
            self.assertEqual(targets[1]["state"], "not_visible")
            self.assertIsNone(targets[1]["primary_target"])
            self.assertEqual(len(labels[2]), 1)

    def test_attribute_subset_uses_only_requested_training_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.build_cub(Path(temporary))
            attributes = load_cub_attributes(root)
            certainties = load_cub_certainties(root)
            labels = load_cub_image_attribute_labels(root, image_ids={1, 2, 3, 4})
            selected = select_training_attribute_subset(
                attributes,
                certainties,
                labels,
                train_image_ids={1, 2, 3, 4},
                groups={"has_crown_color", "has_wing_pattern"},
                allowed_certainty_names={"probably", "definitely"},
                min_positive=1,
                min_negative=1,
                max_missing_fraction=0.25,
            )
            self.assertEqual([row["attribute_id"] for row in selected], [1])
            self.assertEqual(selected[0]["positive_train"], 1)
            self.assertEqual(selected[0]["negative_train"], 2)

    def test_center_crop_transform_is_explicit(self) -> None:
        transform = center_crop_transform(
            original_size=(100, 50), output_size=(40, 40)
        )
        self.assertEqual(transform.resized_size, (80, 40))
        self.assertEqual(transform.crop_left, 20)
        self.assertEqual(transform.crop_top, 0)


if __name__ == "__main__":
    unittest.main()

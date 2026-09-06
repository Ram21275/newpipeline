"""Exercise result-audit failures without loading a model or fitting probes."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bundle_review', ROOT / 'scripts/review_development_bundles.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)
EVIDENCE = ROOT / 'reports/development_20260906'


class BundleReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ('phase3_review_bundle', 'phase4_cached_localizer_smoke_bundle'):
            shutil.copytree(EVIDENCE / name, self.root / name)
        self.p3 = self.root / 'phase3_review_bundle'
        self.p4 = self.root / 'phase4_cached_localizer_smoke_bundle'

    def change_csv_and_rehash(self, name, edit):
        path = self.p3 / name
        rows = review.read_csv(path)
        edit(rows)
        review.write_csv(path, rows)
        manifest = review.read_json(self.p3 / 'bundle_manifest.json')
        manifest[name] = {'bytes': path.stat().st_size, 'sha256': review.sha256(path)}
        (self.p3 / 'bundle_manifest.json').write_text(json.dumps(manifest))

    def test_returned_evidence_and_all_attribute_aggregation(self):
        _, attributes, audit = review.validate_phase3(self.p3)
        smoke = review.validate_phase4(self.p4, audit['experiment_commit'])
        self.assertEqual(smoke['image_id'], 544)
        self.assertFalse(smoke['phase4_complete'])
        attribute_rows, groups, transitions = review.aggregate_attributes(attributes)
        self.assertEqual((len(attribute_rows), len(groups), len(transitions)), (234, 45, 52))
        group_sizes = {r['attribute_group']: r['attributes'] for r in groups}
        self.assertEqual(sum(group_sizes.values()), 26)
        bill = {r['stage']: r['auroc_mean'] for r in groups if r['attribute_group'] == 'has_bill_shape'}
        self.assertAlmostEqual(bill['llm.final'] - bill['vision.late'], -0.0749, places=4)

    def test_hash_mismatch_is_rejected(self):
        path = self.p3 / 'phase3_run_report.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Hash/size mismatch'):
            review.validate_phase3(self.p3)

    def test_duplicate_attribute_is_rejected_even_with_updated_hash(self):
        def edit(rows):
            rows[1] = rows[0].copy()
        self.change_csv_and_rehash('per_attribute_metrics.csv', edit)
        with self.assertRaisesRegex(ValueError, 'Missing/repeated attribute'):
            review.validate_phase3(self.p3)

    def test_nonfinite_fitted_loss_is_rejected(self):
        self.change_csv_and_rehash('per_attribute_metrics.csv', lambda rows: rows[0].update(train_loss='inf'))
        with self.assertRaisesRegex(ValueError, 'Invalid fitted loss'):
            review.validate_phase3(self.p3)

    def test_wrong_macro_metric_is_rejected(self):
        self.change_csv_and_rehash('attribute_probe_by_stage.csv', lambda rows: rows[0].update(macro_auroc='0.9'))
        with self.assertRaisesRegex(ValueError, 'disagrees with per-attribute'):
            review.validate_phase3(self.p3)

    def test_wrong_paired_sd_is_rejected(self):
        self.change_csv_and_rehash('paired_delta_summary.csv', lambda rows: rows[0].update(auroc_delta_sd='0.9'))
        with self.assertRaisesRegex(ValueError, 'Paired summary mean/SD'):
            review.validate_phase3(self.p3)

    def test_smoke_cannot_be_mislabeled_as_full_phase4(self):
        path = self.p4 / 'phase4_cached_localizer_smoke_report.json'
        report = review.read_json(path)
        report['phase4_complete'] = True
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'scope/provenance'):
            review.validate_phase4(self.p4, report['git_commit'])

    def test_wrong_agreement_pairs_are_rejected(self):
        path = self.p4 / 'one_image_selector_agreement.csv'
        rows = review.read_csv(path)
        rows[1] = rows[0].copy()
        review.write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, 'Agreement pairs'):
            review.validate_phase4(self.p4, '1a6e0c96681f250659ce703207931289cd9112c2')


if __name__ == '__main__':
    unittest.main()

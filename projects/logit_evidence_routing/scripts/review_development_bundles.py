#!/usr/bin/env python3
"""Audit returned Phase 3/4 tables and summarize all attributes; no model imports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

STAGES = ('vision.early', 'vision.middle', 'vision.late', 'vision.final',
          'projector.output', 'llm.early', 'llm.middle', 'llm.late', 'llm.final')
CONTROLS = ('primary', 'prevalence', 'shuffled_labels', 'random_projection')
SEEDS = (0, 1, 2)
DIGEST = '63cf0e80ec0a24533682467b6f3b23ccded8625ef25d2fb43d012aa0e72179d8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_key(row):
    return row['stage'], row['pooling'], row['control'], int(row['seed'])


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def check_hash_manifest(directory):
    manifest = read_json(directory / 'bundle_manifest.json')
    required = {'phase3_run_report.json', 'phase3_aggregation_report.json',
                'evaluation_config.json', 'attribute_probe_by_stage.csv',
                'per_attribute_metrics.csv', 'stage_seed_summary.csv',
                'paired_seed_deltas.csv', 'paired_delta_summary.csv'}
    require(required <= manifest.keys(), 'Hash manifest is missing required result files')
    for name, identity in manifest.items():
        require(Path(name).name == name, 'Manifest paths must be plain filenames')
        path = directory / name
        require(path.is_file() and path.stat().st_size == identity['bytes']
                and sha256(path) == identity['sha256'], f'Hash/size mismatch: {name}')
    return len(manifest)


def validate_phase3(directory):
    hashed_files = check_hash_manifest(directory)
    report = read_json(directory / 'phase3_run_report.json')
    config = read_json(directory / 'evaluation_config.json')
    gate = read_json(directory / 'phase3_aggregation_report.json')
    expected = dict(status='PASS', cache_config_digest=DIGEST, images=240,
                    train_images=160, validation_images=80, selected_attributes=26,
                    stages=list(STAGES), pooling=['mean'], controls=list(CONTROLS),
                    seeds=list(SEEDS), result_rows=90, per_attribute_rows=2340,
                    official_test_images_used=0)
    for key, value in expected.items():
        require(report.get(key) == value, f'Phase 3 report mismatch: {key}')
    for key in ('status', 'cache_config_digest', 'result_rows', 'per_attribute_rows',
                'official_test_images_used'):
        require(gate.get(key) == expected[key], f'Aggregation gate mismatch: {key}')
    require(gate.get('official_test_split_untouched') is True, 'Test split gate is absent')
    config_expected = dict(stages=list(STAGES), pooling=['mean'], controls=list(CONTROLS),
                           seeds=list(SEEDS), epochs=300, learning_rate=0.01,
                           weight_decay=0.0001, device='cuda', random_projection_dim=256,
                           normalization='training_feature_mean_std_v1',
                           threshold_policy='training_f1_only_v1', official_test_images_used=0)
    for key, value in config_expected.items():
        require(config.get(key) == value, f'Evaluation config mismatch: {key}')
    require(config['git_commit'] == gate['git_commit'], 'Run/gate commit differs')
    summary = read_csv(directory / 'attribute_probe_by_stage.csv')
    attributes = read_csv(directory / 'per_attribute_metrics.csv')
    expected_keys = {(s, 'mean', c, seed) for s in STAGES for c in CONTROLS
                     for seed in ((-1,) if c == 'prevalence' else SEEDS)}
    require(len(summary) == 90 and Counter(map(run_key, summary)) == Counter(expected_keys),
            'Summary contains missing, duplicate or unexpected run keys')
    require(len(attributes) == 2340, 'Wrong per-attribute row count')
    by_run = defaultdict(list)
    identities = {}
    supports = {}
    for row in attributes:
        key, attribute_id = run_key(row), int(row['attribute_id'])
        by_run[key].append(row)
        identity = row['attribute_name'], row['attribute_group']
        require(identity == identities.setdefault(attribute_id, identity), 'Attribute identity changed')
        counts = tuple(int(row[f'{split}_{field}']) for split in ('train', 'validation')
                       for field in ('observed', 'positive', 'negative'))
        require(counts == supports.setdefault(attribute_id, counts), 'Attribute support changed')
        for offset, limit in ((0, 160), (3, 80)):
            observed, positive, negative = counts[offset:offset + 3]
            require(positive > 0 and negative > 0 and observed == positive + negative
                    and observed <= limit, 'Invalid or unevaluable attribute support')
        for metric in ('auroc', 'f1'):
            value = float(row[metric])
            require(math.isfinite(value) and 0 <= value <= 1, f'Invalid {metric}')
        require(math.isfinite(float(row['threshold_selected_on_train'])), 'Nonfinite threshold')
        require(math.isclose(float(row['train_prevalence']), counts[1] / counts[0],
                             rel_tol=1e-6, abs_tol=1e-8), 'Prevalence/support mismatch')
        loss = float(row['train_loss'])
        require(math.isnan(loss) if key[2] == 'prevalence' else math.isfinite(loss) and loss >= 0,
                'Invalid fitted loss or unexpected prevalence loss')
    require(set(by_run) == expected_keys and len(identities) == 26, 'Attribute run identities differ')
    for key, rows in by_run.items():
        require(Counter(int(r['attribute_id']) for r in rows) == Counter(identities.keys()),
                f'Missing/repeated attribute in {key}')
        if key[2] != 'prevalence':
            require(len({float(r['train_loss']) for r in rows}) == 1, 'Shared probe losses differ')
    for row in summary:
        key = run_key(row)
        expected_dim = 256 if key[2] == 'random_projection' else (1024 if key[0].startswith('vision.') else 4096)
        for field, expected_value in (('selected_attributes', 26), ('evaluable_auroc_attributes', 26),
                                      ('evaluable_f1_attributes', 26), ('train_examples', 160),
                                      ('validation_examples', 80), ('feature_dim', expected_dim)):
            require(int(row[field]) == expected_value, f'Unexpected {field}')
        for metric in ('auroc', 'f1'):
            recomputed = statistics.mean(float(r[metric]) for r in by_run[key])
            require(math.isclose(float(row['macro_' + metric]), recomputed, abs_tol=1e-10),
                    f'Macro {metric} disagrees with per-attribute rows')
        if key[2] == 'prevalence':
            require(float(row['macro_auroc']) == 0.5, 'Prevalence AUROC differs from 0.5')
    # Independently reproduce exported stage summaries and paired seed differences.
    stage_rows = read_csv(directory / 'stage_seed_summary.csv')
    require(Counter((r['stage'], r['control']) for r in stage_rows)
            == Counter(itertools.product(STAGES, CONTROLS)), 'Stage summary keys differ')
    for row in stage_rows:
        selected = [r for r in summary if (r['stage'], r['control']) == (row['stage'], row['control'])]
        require(int(row['n']) == len(selected), 'Stage-summary seed count differs')
        for metric in ('auroc', 'f1'):
            mean, sd = mean_sd([float(r['macro_' + metric]) for r in selected])
            require(math.isclose(float(row[metric + '_mean']), mean, abs_tol=1e-10), 'Seed mean differs')
            require(not row[metric + '_sd'] if sd is None else
                    math.isclose(float(row[metric + '_sd']), sd, abs_tol=1e-10), 'Seed SD differs')
    indexed = {run_key(r): r for r in summary}
    paired = read_csv(directory / 'paired_seed_deltas.csv')
    comparisons = ('shuffled_labels', 'random_projection')
    require(Counter((r['stage'], r['pooling'], r['comparison'], int(r['seed'])) for r in paired)
            == Counter((s, 'mean', 'primary-minus-' + c, seed)
                       for s in STAGES for c in comparisons for seed in SEEDS), 'Paired delta keys differ')
    for row in paired:
        control = row['comparison'].removeprefix('primary-minus-')
        for metric in ('auroc', 'f1'):
            p = indexed[(row['stage'], 'mean', 'primary', int(row['seed']))]
            c = indexed[(row['stage'], 'mean', control, int(row['seed']))]
            require(math.isclose(float(row[metric + '_delta']),
                                 float(p['macro_' + metric]) - float(c['macro_' + metric]), abs_tol=1e-10),
                    'Paired difference differs')
    delta_summary = read_csv(directory / 'paired_delta_summary.csv')
    require(Counter((r['stage'], r['comparison']) for r in delta_summary)
            == Counter((s, 'primary-minus-' + c) for s in STAGES for c in comparisons),
            'Paired summary keys differ')
    for row in delta_summary:
        selected = [r for r in paired if (r['stage'], r['comparison'])
                    == (row['stage'], row['comparison'])]
        require(int(row['n']) == 3, 'Paired summary seed count differs')
        for metric in ('auroc', 'f1'):
            mean, sd = mean_sd([float(r[metric + '_delta']) for r in selected])
            require(math.isclose(float(row[metric + '_delta_mean']), mean, abs_tol=1e-10)
                    and math.isclose(float(row[metric + '_delta_sd']), sd, abs_tol=1e-10),
                    'Paired summary mean/SD differs')
    return summary, attributes, dict(status='PASS', hashed_files=hashed_files,
                                     result_rows=90, per_attribute_rows=2340,
                                     experiment_commit=config['git_commit'], cache_config_digest=DIGEST)


def validate_phase4(directory, experiment_commit):
    report = read_json(directory / 'phase4_cached_localizer_smoke_report.json')
    require(report['status'] == 'PASS' and report['phase4_complete'] is False
            and report['scope'] == 'one_training_image_existing_cached_localizers_only'
            and report['official_test_images_used'] == 0 and report['cache_config_digest'] == DIGEST
            and report['git_commit'] == experiment_commit, 'Phase 4 smoke scope/provenance differs')
    require(report['K'] == [16, 32] and report['random_seeds'] == [0, 1, 2], 'Smoke settings differ')
    expected_stages = dict(zip(STAGES, (6, 12, 23, 24, None, 8, 16, 31, 32)))
    require(report['stage_indices'] == expected_stages, 'Cached layer indices differ')
    metrics = read_csv(directory / 'one_image_localization.csv')
    agreement = read_csv(directory / 'one_image_selector_agreement.csv')
    methods = ('vision_cls_attention', 'llm_attention', 'logit_concept', 'attention_logit_fusion',
               'random_seed_0', 'random_seed_1', 'random_seed_2')
    require(len(metrics) == report['metric_rows'] == 14 and len(agreement) == report['agreement_rows'] == 42,
            'Smoke row counts differ')
    require(Counter((r['method'], int(r['K'])) for r in metrics)
            == Counter(itertools.product(methods, (16, 32))), 'Smoke method/K keys differ')
    require(Counter((frozenset((r['selector_a'], r['selector_b'])), int(r['K'])) for r in agreement)
            == Counter((frozenset(pair), k) for pair in itertools.combinations(methods, 2) for k in (16, 32)),
            'Agreement pairs differ')
    for row in metrics + agreement:
        require(int(row['image_id']) == report['image_id'], 'Smoke image ID differs')
    for row in metrics:
        require(row['split'] == 'train', 'Smoke unexpectedly evaluates validation/test')
        for field in ('inside_fraction', 'bbox_patch_recall', 'bbox_patch_iou', 'pointing_game'):
            require(math.isfinite(float(row[field])) and 0 <= float(row[field]) <= 1, 'Invalid box metric')
        if report['part_metrics_available']:
            for field in ('part_patch_recall', 'any_part_hit', 'top1_part_hit'):
                require(math.isfinite(float(row[field])) and 0 <= float(row[field]) <= 1, 'Invalid part metric')
            require(math.isfinite(float(row['top1_nearest_part_distance_patches']))
                    and float(row['top1_nearest_part_distance_patches']) >= 0, 'Invalid distance')
    require(all(math.isfinite(float(r['jaccard'])) and 0 <= float(r['jaccard']) <= 1 for r in agreement),
            'Invalid selector agreement')
    return dict(status='PASS', phase4_complete=False, image_id=report['image_id'], metric_rows=14,
                agreement_rows=42, stage_indices=report['stage_indices'],
                file_sha256={p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()})


def aggregate_attributes(attributes):
    grouped = defaultdict(list)
    per_seed_group = defaultdict(list)
    for row in attributes:
        if row['control'] != 'primary':
            continue
        key = int(row['attribute_id']), row['attribute_name'], row['attribute_group'], row['stage']
        grouped[key].append(row)
        per_seed_group[(row['attribute_group'], row['stage'], int(row['seed']))].append(row)
    attribute_summary = []
    for (aid, name, group, stage), rows in sorted(grouped.items()):
        summary = dict(attribute_id=aid, attribute_name=name, attribute_group=group, stage=stage,
                       validation_positive=int(rows[0]['validation_positive']),
                       validation_negative=int(rows[0]['validation_negative']))
        for metric in ('auroc', 'f1'):
            summary[metric + '_mean'], summary[metric + '_sd'] = mean_sd([float(r[metric]) for r in rows])
        attribute_summary.append(summary)
    group_summary = []
    for group in sorted({k[0] for k in per_seed_group}):
        for stage in STAGES:
            rows = [per_seed_group[group, stage, seed] for seed in SEEDS]
            summary = dict(attribute_group=group, stage=stage, attributes=len(rows[0]), seeds=3)
            for metric in ('auroc', 'f1'):
                values = [statistics.mean(float(r[metric]) for r in seed_rows) for seed_rows in rows]
                summary[metric + '_mean'], summary[metric + '_sd'] = mean_sd(values)
            group_summary.append(summary)
    # Pair each attribute across stages using the same seed; keep all 26 attributes.
    indexed = {(int(r['attribute_id']), r['stage'], int(r['seed'])): r
               for r in attributes if r['control'] == 'primary'}
    transitions = []
    for aid in sorted({key[0] for key in indexed}):
        for before, after in (('vision.late', 'projector.output'), ('vision.late', 'llm.final')):
            source = indexed[aid, before, 0]
            row = dict(attribute_id=aid, attribute_name=source['attribute_name'],
                       attribute_group=source['attribute_group'], before=before, after=after,
                       validation_positive=int(source['validation_positive']),
                       validation_negative=int(source['validation_negative']))
            for metric in ('auroc', 'f1'):
                values = [float(indexed[aid, after, s][metric]) - float(indexed[aid, before, s][metric])
                          for s in SEEDS]
                row[metric + '_delta_mean'], row[metric + '_delta_sd'] = mean_sd(values)
            transitions.append(row)
    return attribute_summary, group_summary, transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase3-bundle', type=Path, required=True)
    parser.add_argument('--phase4-bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    _, attributes, phase3 = validate_phase3(args.phase3_bundle)
    phase4 = validate_phase4(args.phase4_bundle, phase3['experiment_commit'])
    summaries = aggregate_attributes(attributes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(('primary_attribute_trajectory.csv', 'primary_group_trajectory.csv',
                           'primary_attribute_transitions.csv'), summaries):
        write_csv(args.output_dir / name, rows)
    audit = dict(status='PASS', phase3=phase3, phase4_cached_smoke=phase4,
                 official_test_images_used_reported=0,
                 uncertainty='sample SD across probe seeds, not image-sampling confidence intervals',
                 scope='Returned-table/hash consistency; source tensor caches and images are not reloaded')
    (args.output_dir / 'returned_bundle_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

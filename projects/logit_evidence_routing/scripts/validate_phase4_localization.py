#!/usr/bin/env python3
"""Validate Phase 4 result coverage, summarize development results and bundle reports."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from lger.phase4 import (BOX_METRICS, METRICS, PART_METRICS, macro_attribute_summary,
                         paired_selector_deltas, read_json, require, sha256, summarize_metrics, summarize_agreements)
from lger.stage_cache import atomic_json_write, config_digest


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def metric_rows(path):
    rows = read_csv(path)
    for row in rows:
        for field in ('image_id', 'attribute_id', 'K', 'selection_seed', 'visible_part_instances'):
            row[field] = int(row[field])
        for field in (*METRICS, 'visible_part_patches'):
            row[field] = float(row[field]) if row[field] else None
    return rows


def compare_numeric_csv(path, expected):
    actual = read_csv(path)
    require(len(actual) == len(expected), f'Summary row count differs: {path.name}')
    for row, target in zip(actual, expected):
        require(row.keys() == target.keys(), f'Summary columns differ: {path.name}')
        for key, value in target.items():
            if value is None:
                require(row[key] == '', f'Unexpected missing-value representation: {key}')
            elif isinstance(value, (float, int)):
                require(math.isclose(float(row[key]), value, rel_tol=1e-9, abs_tol=1e-10),
                        f'Summary value differs: {path.name}/{key}')
            else:
                require(row[key] == value, f'Summary identity differs: {path.name}/{key}')


def validate(output_dir):
    output_dir = Path(output_dir)
    report = read_json(output_dir / 'phase4_run_report.json')
    cfg = read_json(output_dir / 'evaluation_config.json')
    policy = cfg['policy']
    require(report['status'] == 'PASS' and report['official_test_images_used'] == 0
            and report['official_test_split_untouched'] is True and report['global_projection_checked'] is True,
            'Run gate has not passed')
    base = {key: value for key, value in cfg.items() if key not in ('mode', 'image_ids', 'protocol_digest')}
    require(config_digest(base) == report['protocol_digest'] == cfg['protocol_digest'], 'Protocol digest differs')
    require(report['mode'] == cfg['mode'] and report['mode'] in ('smoke', 'development'), 'Mode differs')
    require(report['quantitative_evaluation_complete'] is (report['mode'] == 'development')
            and report['selected_attributes'] == len(policy['attributes']) == 26,
            'Quantitative completion or attribute count differs')
    ids = cfg['image_ids']
    expected_n = 1 if cfg['mode'] == 'smoke' else 240
    require(len(ids) == len(set(ids)) == report['images'] == expected_n and ids == report['image_ids'],
            'Image identity/count differs')
    for name, identity in report['artifact_manifest'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Invalid artifact path')
        path = output_dir / name
        require(path.is_file() and path.stat().st_size == identity['bytes'] and sha256(path) == identity['sha256'],
                f'Artifact hash/size differs: {name}')
    required = {'evaluation_config.json', 'object_metrics.csv', 'attribute_metrics.csv',
                'attribute_eligibility.csv', 'selector_agreement.csv', 'source_records.csv',
                'localization_summary.csv', 'attribute_macro_summary.csv', 'paired_selector_deltas.csv',
                'attribute_support.csv', 'localization_overview.png', 'selector_agreement_summary.csv',
                'selector_agreement.png'}
    require(required <= report['artifact_manifest'].keys(), 'Missing manifested result files')
    sources = read_csv(output_dir / 'source_records.csv')
    require([int(r['image_id']) for r in sources] == ids, 'Source index differs')
    splits = {int(r['image_id']): r['split'] for r in sources}
    expected_splits = {'train': 1} if cfg['mode'] == 'smoke' else {'train': 160, 'val': 80}
    require(Counter(splits.values()) == expected_splits == report['split_counts'], 'Split counts differ')
    # Score tensors are intentionally excluded from review ZIPs; check them when present.
    score_hashes_verified = 0
    for row in sources:
        name = Path(row['dense_score_file'])
        require(not name.is_absolute() and '..' not in name.parts, 'Invalid score path')
        if (output_dir / name).exists():
            require(sha256(output_dir / name) == row['dense_score_sha256'], 'Dense score hash differs')
            score_hashes_verified += 1
    attributes = {a['attribute_id']: a for a in policy['attributes']}
    eligibility = read_csv(output_dir / 'attribute_eligibility.csv')
    require(Counter((int(r['image_id']), int(r['attribute_id'])) for r in eligibility)
            == Counter(itertools.product(ids, attributes)), 'Eligibility coverage differs')
    reasons = ('eligible', 'observed_negative', 'uncertain_or_missing', 'no_visible_in_crop_relevant_part')
    for row in eligibility:
        require(row['reason'] in reasons and row['split'] == splits[int(row['image_id'])], 'Eligibility split/reason differs')
        require(row['attribute_name'] == attributes[int(row['attribute_id'])]['name'], 'Attribute name differs')
        require(row['cached_primary_target'] in ('', 'True', 'False')
                and row['certainty_policy_approved'] in ('True', 'False')
                and row['unapproved_cached_target_masked'] in ('0', '1'),
                'Invalid certainty-policy audit fields')
        approved = row['certainty_name'].strip().lower() in ('probably', 'definitely')
        require((row['certainty_policy_approved'] == 'True') is approved,
                'Certainty-policy approval flag differs')
        masked = row['unapproved_cached_target_masked'] == '1'
        require(masked is (bool(row['cached_primary_target']) and not approved),
                'Certainty-policy override flag differs')
        counts = [int(row[name]) for name in ('relevant_annotated_parts', 'relevant_visible_parts',
                                            'relevant_visible_in_crop_parts')]
        require(0 <= counts[2] <= counts[1] <= counts[0], 'Invalid eligibility part counts')
        if row['reason'] == 'eligible':
            require(counts[2] > 0, 'Eligible pair lacks an in-crop relevant landmark')
        if row['reason'] in ('eligible', 'no_visible_in_crop_relevant_part'):
            require(approved and row['cached_primary_target'] == 'True', 'Positive eligibility policy differs')
        elif row['reason'] == 'observed_negative':
            require(approved and row['cached_primary_target'] == 'False', 'Negative eligibility policy differs')
        else:
            require(row['cached_primary_target'] == '' or masked, 'Uncertain eligibility policy differs')
    eligible = {(int(r['image_id']), int(r['attribute_id'])) for r in eligibility if r['reason'] == 'eligible'}
    objects = metric_rows(output_dir / 'object_metrics.csv')
    attrs = metric_rows(output_dir / 'attribute_metrics.csv')
    base_methods = [(name, -1) for name in policy['cached_selectors'] + ['dense_object']] + [('random', s) for s in (0, 1, 2)]
    for scope, rows, pairs, methods in [('object', objects, {(i, 0) for i in ids}, base_methods),
                                       ('attribute', attrs, eligible, base_methods + [('dense_attribute', -1)])]:
        expected = Counter((image_id, aid, k, method, seed) for image_id, aid in pairs
                           for k in (16, 32) for method, seed in methods)
        require(Counter((r['image_id'], r['attribute_id'], r['K'], r['selector'], r['selection_seed']) for r in rows)
                == expected, f'Matched {scope} metric coverage differs')
        for row in rows:
            require(row['scope'] == scope and row['split'] == splits[row['image_id']], 'Metric scope/split differs')
            for metric in METRICS:
                value = row[metric]
                if value is None:
                    require(scope == 'object' and metric in PART_METRICS and row['visible_part_instances'] == 0,
                            'Unexpected missing metric')
                else:
                    require(math.isfinite(value) and value >= 0
                            and (metric == 'top1_nearest_part_distance_patches' or value <= 1), 'Invalid metric value')
    agreements = read_csv(output_dir / 'selector_agreement.csv')
    expected_agreements = Counter()
    for scope, pairs, methods in [('object', {(i, 0) for i in ids}, base_methods),
                                  ('attribute', eligible, base_methods + [('dense_attribute', -1)])]:
        names = [f'random.{seed}' if name == 'random' else name for name, seed in methods]
        expected_agreements.update((scope, i, aid, k, frozenset(pair)) for i, aid in pairs
                                   for k in (16, 32) for pair in itertools.combinations(names, 2))
    require(Counter((r['scope'], int(r['image_id']), int(r['attribute_id']), int(r['K']),
                     frozenset((r['selection_a'], r['selection_b']))) for r in agreements) == expected_agreements,
            'Matched selector agreement coverage differs')
    require(all(math.isfinite(float(r['jaccard'])) and 0 <= float(r['jaccard']) <= 1 for r in agreements),
            'Invalid Jaccard value')
    compare_numeric_csv(output_dir / 'selector_agreement_summary.csv', summarize_agreements(agreements))
    require(len(objects) == report['object_metric_rows'] and len(attrs) == report['attribute_metric_rows']
            and len(eligibility) == report['eligibility_rows'] and len(eligible) == report['eligible_image_attribute_pairs']
            and len(agreements) == report['agreement_rows'], 'Report/table counts differ')
    masked_targets = sum(r['unapproved_cached_target_masked'] == '1' for r in eligibility)
    require(report['unapproved_cached_targets_masked'] == masked_targets
            and report['phase3_revalidation_required'] is (masked_targets > 0),
            'Certainty-policy report counts differ')
    summary = summarize_metrics(objects + attrs)
    macro = macro_attribute_summary(summary, policy)
    compare_numeric_csv(output_dir / 'localization_summary.csv', summary)
    compare_numeric_csv(output_dir / 'attribute_macro_summary.csv', macro)
    compare_numeric_csv(output_dir / 'paired_selector_deltas.csv', paired_selector_deltas(objects + attrs))
    support = read_csv(output_dir / 'attribute_support.csv')
    require(Counter((r['split'], int(r['attribute_id'])) for r in support)
            == Counter(itertools.product(expected_splits, attributes)), 'Attribute support table coverage differs')
    for row in support:
        counts = Counter(r['reason'] for r in eligibility if r['split'] == row['split']
                         and r['attribute_id'] == row['attribute_id'])
        require(all(int(row[reason]) == counts[reason] for reason in reasons), 'Attribute support totals differ')
    return report, summary, macro, support, score_hashes_verified


def render_review(output_dir, report, summary, macro, support):
    split = 'val' if report['mode'] == 'development' else 'train'
    lines = ['# Phase 4 localization results', '', f"Computation status: PASS; mode: {report['mode']}.",
             f"Git commit: `{report['git_commit']}`. Protocol: `{report['protocol_digest']}`.", '',
             'Quantitative completion is not a causal claim. Review these results and qualitative panels before Phase 5.', '',
             f"Certainty audit: {report['unapproved_cached_targets_masked']} non-null cached targets with "
             'unapproved certainty were masked as uncertain for Phase 4. '
             + ('Rerun Phase 3 with the corrected consumer-side policy before cross-phase interpretation.'
                if report['phase3_revalidation_required'] else
                'The cached primary targets agree with the fixed probably/definitely policy.'), '',
             f'## Object localization, development {split}, K=32', '',
             '| Selector | Inside box | Top-1 in box | Visible-part patch recall | Top-1 part distance |',
             '|---|---:|---:|---:|---:|']
    selected = [r for r in summary if r['scope']=='object' and r['split']==split and r['K']==32]
    lookup = {(r['selector'], r['metric']): r for r in selected}
    def number(row, key='mean'):
        return 'N/A' if row is None else f"{row[key]:.4f}"
    for method in sorted({r['selector'] for r in selected}):
        values = [number(lookup.get((method, m))) for m in ('inside_fraction', 'pointing_game', 'part_patch_recall',
                                                           'top1_nearest_part_distance_patches')]
        lines.append('| ' + ' | '.join([method, *values]) + ' |')
    lines += ['', f'## Attribute landmark proxies, development {split}, K=32', '',
              '| Selector | Macro part-patch recall | Macro top-1 distance | Evaluable attributes |',
              '|---|---:|---:|---:|']
    selected = [r for r in macro if r['split']==split and r['K']==32 and r['attribute_group']=='ALL_SELECTED_ATTRIBUTES']
    lookup = {(r['selector'],r['metric']):r for r in selected}
    for method in sorted({r['selector'] for r in selected}):
        recall = lookup.get((method,'part_patch_recall'))
        distance = lookup.get((method,'top1_nearest_part_distance_patches'))
        lines.append(f"| {method} | {number(recall,'mean_across_evaluable_attributes')} | "
                     f"{number(distance,'mean_across_evaluable_attributes')} | "
                     f"{recall['evaluable_attributes'] if recall else 0}/26 |")
    lines += ['', '## Coverage and interpretation', '',
              'Attribute localization uses observed positives with at least one relevant visible in-crop landmark. '
              'All 26 selected attributes remain in the support table; an attribute with zero eligible images is unavailable, not zero-scoring.', '',
              'Random seeds are averaged within image before image aggregation. Attribute macro scores weight evaluable attributes equally. '
              'SD in CSVs describes variation across images; it is not a confidence interval.', '',
              'Generic bird/birds Logit Lens and prompt attention are object-level maps. Dense-attribute queries are conditional on the named attribute; '
              'compare dense-attribute with dense-object to inspect query specificity. This comparison does not isolate scoring method from query content.', '',
              'Dense CLIP uses the paired frozen visual/text projections applied to contextual patch states. '
              'The projection is globally trained, not a dense segmentation head. Landmark proxies are not attribute masks. '
              'High box overlap, landmark proximity, semantic relevance and causal VLM utilization are distinct measurements.', '',
              'Top-1 distances are in 14-pixel patch units; lower is better. Top-K recall counts distinct part-containing patches. '
              'The attention/concept fusion is a diagnostic. The official CUB test split remains untouched.', '',
              '## Selector agreement', '',
              'See `selector_agreement.png` and `selector_agreement_summary.csv` for image-matched Top-32 Jaccard. '
              'Random variants are averaged within image; attribute panels then average over evaluable attributes. '
              'The raw table retains all K=16/32 pairs, including distinct random seeds. '
              'Low agreement is not evidence that a selector is incorrect or that evidence has moved between representation stages.', '',
              '## Attribute eligibility counts', '',
              '| Attribute | Eligible | Negative | Uncertain/missing | No relevant visible in-crop part |',
              '|---|---:|---:|---:|---:|']
    for row in support:
        if row['split'] == split:
            lines.append('| ' + ' | '.join(row[k] for k in ('attribute_name','eligible','observed_negative',
                                                           'uncertain_or_missing','no_visible_in_crop_relevant_part')) + ' |')
    (output_dir / 'PHASE4_RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, help='ZIP of reports/plots only; excludes cached score tensors')
    args = parser.parse_args()
    report, summary, macro, support, score_hashes = validate(args.output_dir)
    render_review(args.output_dir, report, summary, macro, support)
    audit = dict(status='PASS', mode=report['mode'], git_commit=report['git_commit'],
                 protocol_digest=report['protocol_digest'], images=report['images'],
                 official_test_images_used=0, dense_score_hashes_verified=score_hashes,
                 unapproved_cached_targets_masked=report['unapproved_cached_targets_masked'],
                 phase3_revalidation_required=report['phase3_revalidation_required'],
                 quantitative_evaluation_complete=report['mode']=='development', scientific_review_pending=True)
    atomic_json_write(audit, args.output_dir / 'phase4_validation_report.json')
    if args.bundle:
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        names = list(report['artifact_manifest']) + ['phase4_run_report.json', 'phase4_validation_report.json', 'PHASE4_RESULTS.md']
        with zipfile.ZipFile(args.bundle, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            identities = {}
            for name in names:
                path = args.output_dir / name
                bundle.write(path, arcname=name)
                identities[name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
            bundle.writestr('bundle_manifest.json', json.dumps(identities, indent=2))
        print('Report/plot bundle:', args.bundle)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

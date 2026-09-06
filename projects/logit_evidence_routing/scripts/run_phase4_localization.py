#!/usr/bin/env python3
"""Run the fixed Phase 4 semantic smoke or full development localization on Kaggle."""

from __future__ import annotations

import argparse
import csv
import itertools
import importlib.metadata
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

import torch

from lger.dense_clip import FrozenDenseClip
from lger.phase4 import (METRICS, evaluate_image, join_cached_record, load_development_metadata,
                         load_part_vocabulary, load_policy, macro_attribute_summary,
                         paired_selector_deltas, read_json, require, sha256, summarize_metrics, summarize_agreements,
                         validate_corrected_cache, write_csv)
from lger.reproducibility import current_git_commit
from lger.stage_cache import atomic_json_write, config_digest, write_or_validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'development'), required=True)
    parser.add_argument('--policy', type=Path, default=PROJECT / 'configs/phase4_localization.json')
    parser.add_argument('--stage-cache', type=Path, required=True)
    parser.add_argument('--localizer-cache', type=Path, required=True)
    parser.add_argument('--phase1-gate', type=Path, required=True)
    parser.add_argument('--phase3-bundle', type=Path,
                        default=PROJECT / 'reports/development_20260906/phase3_review_bundle')
    parser.add_argument('--part-vocabulary', type=Path,
                        help='CUB parts/parts.txt; defaults to the dataset root recorded by Phase 2')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke-dir', type=Path, help='Required passing semantic smoke for development mode')
    args = parser.parse_args()
    require(Path.cwd().resolve() == PROJECT.resolve(), 'Run from projects/logit_evidence_routing')
    policy = load_policy(args.policy)
    cfg2, all_records = load_development_metadata(args.stage_cache, policy)
    cfg1 = validate_corrected_cache(args.localizer_cache, args.phase1_gate, cfg2)
    phase3 = read_json(args.phase3_bundle / 'phase3_run_report.json')
    require(phase3['status'] == 'PASS' and phase3['cache_config_digest'] == policy['cache_config_digest']
            and phase3['official_test_images_used'] == 0 and phase3['selected_attributes'] == 26
            and phase3['result_rows'] == 90 and phase3['per_attribute_rows'] == 2340,
            'Complete Phase 3 gate is required')
    vocabulary_path = args.part_vocabulary or Path(cfg2['dataset']['root']) / 'parts/parts.txt'
    part_names = load_part_vocabulary(vocabulary_path, policy)
    training = sorted((r for r in all_records if r['image']['development_split'] == 'train'),
                      key=lambda r: r['image']['image_id'])
    smoke_id = training[0]['image']['image_id']
    versions = {'python': sys.version.split()[0], 'torch': str(torch.__version__)}
    for package in ('transformers', 'numpy', 'matplotlib'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unavailable'
    run_identity = dict(schema_version=1, git_commit=current_git_commit(PROJECT.parents[1]), policy=policy,
                        runtime_versions=versions,
                        input_paths=dict(stage_cache=str(args.stage_cache.resolve()),
                                         localizer_cache=str(args.localizer_cache.resolve()),
                                         phase1_gate=str(args.phase1_gate.resolve()),
                                         phase3_bundle=str(args.phase3_bundle.resolve()),
                                         part_vocabulary=str(vocabulary_path.resolve())),
                        policy_sha256=sha256(args.policy), stage_config_digest=policy['cache_config_digest'],
                        stage_index_sha256=sha256(args.stage_cache / 'index.json'),
                        stage_validation_sha256=sha256(args.stage_cache / 'validation_report.json'),
                        localizer_config_sha256=sha256(args.localizer_cache / 'extraction_config.json'),
                        phase1_gate_sha256=sha256(args.phase1_gate),
                        phase3_report_sha256=sha256(args.phase3_bundle / 'phase3_run_report.json'),
                        part_vocabulary_sha256=sha256(vocabulary_path), part_names=part_names,
                        input_rgb='cached_uint8_crop_no_spatial_resampling_v1',
                        precision='float32_eager_tf32_disabled', official_test_images_used=0,
                        generic_logit_query='bird/birds; object-level only',
                        llm_attention='cached heads-mean final prompt text query at attention layer offset -2',
                        uncertainty='SD across images; random seeds averaged within image; not confidence intervals')
    require(run_identity['git_commit'] != 'unknown', 'A Git checkout is required for reproducibility')
    digest = config_digest(run_identity)
    if args.mode == 'development':
        require(args.smoke_dir is not None, '--smoke-dir is required before development evaluation')
        smoke = read_json(args.smoke_dir / 'phase4_run_report.json')
        require(smoke['status'] == 'PASS' and smoke['mode'] == 'smoke'
                and smoke['protocol_digest'] == digest and smoke['image_ids'] == [smoke_id]
                and smoke['global_projection_checked'] is True
                and smoke['attribute_metric_rows'] > 0, 'Passing matching semantic smoke is required')
        for name, identity in smoke['artifact_manifest'].items():
            require(sha256(args.smoke_dir / name) == identity['sha256'], f'Smoke artifact changed: {name}')
    records = [training[0]] if args.mode == 'smoke' else all_records
    run_cfg = dict(run_identity, protocol_digest=digest, mode=args.mode,
                   image_ids=[r['image']['image_id'] for r in records])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(args.output_dir / 'evaluation_config.json', run_cfg)
    report_path = args.output_dir / 'phase4_run_report.json'
    report_path.unlink(missing_ok=True)  # A partial rerun must never retain a stale PASS.
    score_dir = args.output_dir / 'dense_scores'
    score_dir.mkdir(exist_ok=True)
    # CUDA determinism requires this before a context is created.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    require(torch.cuda.is_available(), 'Run the real Phase 4 model evaluation on Kaggle CUDA')
    torch.cuda.reset_peak_memory_stats()
    scorer = None
    objects, attributes, eligibility, agreements, source_rows = [], [], [], [], []
    start = time.monotonic()
    global_checked = False
    figures = []
    plot_ids = {smoke_id} if args.mode == 'smoke' else {
        r['image']['image_id'] for r in sorted(all_records, key=lambda r: r['image']['image_id'])
        if r['image']['development_split'] == 'val'}
    plot_ids = set(sorted(plot_ids)[:6])
    for position, record in enumerate(records, 1):
        image = record['image']
        pixels, cached_scores, source_hash = join_cached_record(args.localizer_cache, record)
        score_path = score_dir / f"{image['image_id']:05d}.pt"
        if score_path.exists():
            saved = torch.load(score_path, map_location='cpu', weights_only=True)
            require(saved['protocol_digest'] == digest and saved['image_id'] == image['image_id']
                    and saved['source_sha256'] == source_hash, 'Dense-score resume identity differs')
            dense, diagnostics = saved['scores'], saved['diagnostics']
        else:
            if scorer is None:
                print('Loading frozen paired CLIP on Kaggle:', policy['dense_model'], policy['dense_revision'], flush=True)
                scorer = FrozenDenseClip.from_pretrained(policy, cfg2['spatial_preprocessing'])
            dense, diagnostics = scorer.score(pixels, validate_global=(position == 1))
            saved = dict(protocol_digest=digest, image_id=image['image_id'], source_sha256=source_hash,
                         scores=dense, diagnostics=diagnostics)
            temporary = score_path.with_suffix('.pt.tmp')
            torch.save(saved, temporary)
            temporary.replace(score_path)
        if position == 1:
            require(diagnostics['global_projection_checked'] is True, 'First-image projection audit is missing')
        global_checked |= diagnostics['global_projection_checked']
        result = evaluate_image(record, cached_scores, dense, policy, part_names)
        for target, rows in zip((objects, attributes, eligibility, agreements), result):
            target.extend(rows)
        source_rows.append(dict(image_id=image['image_id'], split=image['development_split'],
                                localizer_record_sha256=source_hash,
                                dense_score_file=str(score_path.relative_to(args.output_dir)),
                                dense_score_sha256=sha256(score_path)))
        if image['image_id'] in plot_ids:
            from lger.phase4_plots import plot_image
            figures.extend(plot_image(args.output_dir, record, pixels, cached_scores, dense, policy, part_names))
        print(f"[{position}/{len(records)}] image={image['image_id']} split={image['development_split']} "
              f"eligible attributes={len(result[1]) // 18} elapsed={time.monotonic()-start:.1f}s", flush=True)
    require(global_checked and bool(attributes), 'No validated projection or eligible attribute rows')
    eligible_pairs = sum(r['reason'] == 'eligible' for r in eligibility)
    require(len(objects) == len(records) * 16 and len(attributes) == eligible_pairs * 18
            and len(eligibility) == len(records) * 26
            and len(agreements) == len(records) * 56 + eligible_pairs * 72, 'Unexpected result row counts')
    outputs = {'object_metrics.csv': objects, 'attribute_metrics.csv': attributes,
               'attribute_eligibility.csv': eligibility, 'selector_agreement.csv': agreements,
               'source_records.csv': source_rows}
    summary = summarize_metrics(objects + attributes)
    outputs['localization_summary.csv'] = summary
    outputs['attribute_macro_summary.csv'] = macro_attribute_summary(summary, policy)
    outputs['paired_selector_deltas.csv'] = paired_selector_deltas(objects + attributes)
    outputs['selector_agreement_summary.csv'] = summarize_agreements(agreements)
    support = []
    for split in sorted({r['image']['development_split'] for r in records}):
        for attribute in policy['attributes']:
            counts = Counter(r['reason'] for r in eligibility
                             if r['split'] == split and r['attribute_id'] == attribute['attribute_id'])
            support.append(dict(split=split, attribute_id=attribute['attribute_id'], attribute_name=attribute['name'],
                                **{reason: counts[reason] for reason in ('eligible', 'observed_negative',
                                   'uncertain_or_missing', 'no_visible_in_crop_relevant_part')}))
    outputs['attribute_support.csv'] = support
    for name, rows in outputs.items():
        write_csv(args.output_dir / name, rows)
    from lger.phase4_plots import plot_summary, plot_agreements
    figures.extend(plot_summary(args.output_dir, summary, outputs['attribute_macro_summary.csv'], args.mode))
    figures.extend(plot_agreements(args.output_dir, outputs['selector_agreement_summary.csv'], args.mode))
    artifact_names = ['evaluation_config.json', *outputs, *figures]
    manifest = {name: dict(bytes=(args.output_dir / name).stat().st_size,
                           sha256=sha256(args.output_dir / name)) for name in artifact_names}
    report = dict(schema_version=1, status='PASS', mode=args.mode, protocol_digest=digest,
                  git_commit=run_identity['git_commit'], official_test_images_used=0,
                  official_test_split_untouched=True, images=len(records), image_ids=run_cfg['image_ids'],
                  split_counts=dict(Counter(r['image']['development_split'] for r in records)),
                  selected_attributes=26, object_metric_rows=len(objects), attribute_metric_rows=len(attributes),
                  eligibility_rows=len(eligibility), eligible_image_attribute_pairs=eligible_pairs,
                  agreement_rows=len(agreements), global_projection_checked=global_checked,
                  runtime_seconds=time.monotonic()-start,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                  quantitative_evaluation_complete=args.mode == 'development',
                  phase4_review_required=True, artifact_manifest=manifest,
                  interpretation='Contextual patch similarity and landmark-proxy localization; not causal evidence')
    atomic_json_write(report, report_path)
    print(json.dumps(report, indent=2))
    print('Phase 4 computation PASS. Review summaries, support counts and qualitative panels before Phase 5.')


if __name__ == '__main__':
    main()

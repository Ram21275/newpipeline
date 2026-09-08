"""Phase 4 cache joining, fixed eligibility and matched localization metrics."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .cub import PRIMARY_TARGET_CERTAINTY_NAMES, certainty_policy_target
from .localization import patch_centers_in_box, selection_localization_metrics, selection_part_metrics
from .phase1b import feature_key
from .scoring import stable_topk
from .stage_cache import REQUIRED_STAGE_NAMES, config_digest

BOX_METRICS = ('inside_fraction', 'bbox_patch_recall', 'bbox_patch_iou', 'pointing_game')
PART_METRICS = ('part_patch_recall', 'any_part_hit', 'top1_part_hit',
                'top1_nearest_part_distance_patches')
METRICS = BOX_METRICS + PART_METRICS
CACHED_SELECTORS = ('vision_cls_attention', 'llm_attention', 'logit_concept', 'attention_logit_fusion')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        require(bool(rows), f'No rows/columns for {path}')
        fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_policy(path):
    policy = read_json(path)
    require(policy['schema_version'] == 1 and policy['k_values'] == [16, 32]
            and policy['random_seeds'] == [0, 1, 2], 'Unsupported Phase 4 policy')
    require(policy['cached_selectors'] == list(CACHED_SELECTORS), 'Cached selector policy differs')
    require(policy['eligibility'] == 'observed_positive_and_at_least_one_relevant_visible_in_crop_part'
            and policy['dense_method'] == 'clip_final_patch_post_layernorm_visual_projection_cosine_v1',
            'Unsupported eligibility or dense scoring method')
    require(set(policy.get('approved_certainty_names', ())) == PRIMARY_TARGET_CERTAINTY_NAMES,
            'Phase 4 must use the fixed probably/definitely certainty policy')
    require(len(policy['dense_revision']) == 40
            and all(c in '0123456789abcdef' for c in policy['dense_revision']), 'Pin the dense model revision')
    attributes = policy['attributes']
    require(len(attributes) == len({a['attribute_id'] for a in attributes}) == 26,
            'Exactly 26 distinct selected attributes are required')
    require(all(a['text'].strip() and a['relevant_parts'] and a['landmark_proxy_note'] for a in attributes),
            'Attribute descriptions and landmark proxies must be explicit')
    require(len({a['text'] for a in attributes}) == 26, 'Repeated attribute query')
    return policy


def load_part_vocabulary(path, policy):
    names = {}
    for line in Path(path).read_text().splitlines():
        key, name = line.split(maxsplit=1)
        normalized = ' '.join(name.replace('_', ' ').lower().split())
        require(normalized not in names and int(key) not in names.values(), 'Duplicate part vocabulary')
        names[normalized] = int(key)
    requested = {name for a in policy['attributes'] for name in a['relevant_parts']}
    require(requested <= names.keys(), f'Unknown part names: {requested - names.keys()}')
    return names


def load_development_metadata(cache_dir, policy):
    """Read JSON only; enforce official-training-only identity on all 240 records."""
    cache_dir = Path(cache_dir)
    cfg = read_json(cache_dir / 'run_config.json')
    index = read_json(cache_dir / 'index.json')
    gate = read_json(cache_dir / 'validation_report.json')
    digest = config_digest(cfg)
    require(digest == policy['cache_config_digest'] == index['config_digest'] == gate['config_digest'],
            'Frozen Phase 2 cache digest differs')
    require(cfg['purpose'] == 'full_240_image_development_pilot_stage_cache'
            and cfg['official_test_images'] == 0 and index['complete'] is True,
            'Phase 2 is not a complete development cache')
    require(gate['status'] == 'PASS' and gate['images'] == gate['reload_validated_records'] == 240
            and gate['official_test_images'] == 0 and gate['official_test_split_untouched'] is True
            and gate['development_split_counts'] == {'train': 160, 'val': 80}
            and gate['stage_count_per_record'] == 9 and len(index['shards']) == gate['shards'] == 12,
            'Phase 2 gate differs')
    expected_attributes = {a['attribute_id']: (a['name'], a['group']) for a in policy['attributes']}
    records = []
    for shard in index['shards']:
        path = cache_dir / shard['metadata_path']
        require(sha256(path) == shard['metadata_sha256'], f'Shard metadata hash differs: {path}')
        metadata = read_json(path)
        require(metadata['complete'] is True and metadata['config_digest'] == digest, 'Invalid shard metadata')
        for item in metadata['records']:
            record = item['packed_record']
            image = record['image']
            require(image['official_split'] == 'train', 'Official test record encountered')
            require(record['complete'] is True and record['config_digest'] == digest, 'Invalid record provenance')
            require(set(record['stages']) == set(REQUIRED_STAGE_NAMES), 'Missing stages')
            require(set(image['selected_attribute_ids']) == set(expected_attributes)
                    and len(image['selected_attribute_ids']) == 26, 'Selected attributes changed')
            by_id = {a['attribute_id']: a for a in image['attributes']}
            require(len(by_id) == len(image['attributes']), 'Repeated attribute metadata')
            require(len({p['part_id'] for p in image['parts']}) == len(image['parts']), 'Repeated part ID')
            for aid, identity in expected_attributes.items():
                require(aid in by_id and (by_id[aid]['name'], by_id[aid]['group']) == identity,
                        'Attribute identity differs')
            records.append(record)
    ids = [r['image']['image_id'] for r in records]
    require(len(ids) == len(set(ids)) == 240 and ids == cfg['manifest_image_ids']
            == [r['image_id'] for r in index['records']], 'Frozen image IDs/order changed')
    require(Counter(r['image']['development_split'] for r in records) == {'train': 160, 'val': 80},
            'Frozen development split differs')
    return cfg, records


def validate_corrected_cache(cache_dir, gate_path, cfg2):
    cfg = read_json(Path(cache_dir) / 'extraction_config.json')
    gate = read_json(gate_path)
    require(gate['passed'] is True and gate['status'] in ('PASS', 'PASS WITH ANOMALY')
            and not gate['blocking_findings'], 'Phase 1 sanity gate is not passed')
    require(cfg['concept_tokenization_policy'] == 'single_lexical_token_v1'
            and cfg['concept_token_ids'] == [11199, 17952] and cfg['layer_offset'] == -2,
            'Corrected bird/birds cache with the audited late-layer setting is required')
    require(cfg['concept_tokens'] == ['▁bird', '▁birds'], 'Concept token identity differs')
    for key in ('model', 'resolved_revision', 'quantization', 'prompt'):
        require(cfg[key] == cfg2[key], f'Phase 1/2 provenance mismatch: {key}')
    return cfg


def join_cached_record(cache_dir, record2):
    image = record2['image']
    path = Path(cache_dir) / 'records' / f"{image['image_id']:05d}.pt"
    # These are the user's own trusted extraction artifacts, never arbitrary uploads.
    row = torch.load(path, map_location='cpu', weights_only=False)
    require(row['schema_version'] == 2 and row['image_id'] == image['image_id'], 'Cache record mismatch')
    for left, right in (('split', 'development_split'), ('relative_path', 'relative_path'),
                        ('label', 'class_id'), ('class_name', 'class_name')):
        require(row[left] == image[right], f'Cache identity mismatch: {left}')
    spatial = record2['spatial']
    require(tuple(row['grid_size']) == tuple(spatial['grid_size']) == (24, 24)
            and tuple(row['processed_image_size']) == tuple(spatial['processed_image_size_hw']) == (336, 336)
            and row['patch_count'] == spatial['patch_count'] == 576
            and tuple(row['original_image_size']) == tuple(spatial['original_image_size_wh']),
            'Cached geometry differs')
    require(torch.allclose(torch.tensor(row['bbox_xyxy_model'], dtype=torch.float64),
                           torch.tensor(image['bbox_model_xyxy'], dtype=torch.float64), atol=1e-3, rtol=0),
            'Mapped boxes differ')
    pixels = row['processed_image']
    require(pixels.dtype == torch.uint8 and tuple(pixels.shape) == (3, 336, 336),
            'Expected the audited uint8 RGB crop, not normalized pixel_values')
    scores = {}
    for selector in CACHED_SELECTORS:
        score = row['score_maps'][selector].detach().cpu().float()
        require(tuple(score.shape) == (576,) and bool(torch.isfinite(score).all()), 'Invalid cached score map')
        for k in (16, 32):
            require(torch.equal(stable_topk(score, k), row['selections'][feature_key(selector, k)].cpu()),
                    'Stored selector ranking differs')
        scores[selector] = score
    return pixels, scores, sha256(path)


def attribute_eligibility(image, attribute, part_names):
    labels = {a['attribute_id']: a for a in image['attributes']}
    label = labels[attribute['attribute_id']]
    ids = {part_names[name] for name in attribute['relevant_parts']}
    related = [p for p in image['parts'] if p['part_id'] in ids]
    visible = [p for p in related if p['visible']]
    points = [tuple(p['model_xy']) for p in visible if p['model_xy'] is not None]
    target, certainty_policy_override = certainty_policy_target(label)
    certainty_name = label.get('certainty_name')
    normalized_certainty = (str(certainty_name).strip().lower()
                              if certainty_name is not None else '')
    if target is None:
        reason = 'uncertain_or_missing'
    elif not target:
        reason = 'observed_negative'
    else:
        reason = 'eligible' if points else 'no_visible_in_crop_relevant_part'
    return reason, points, dict(
        certainty_name=certainty_name,
        cached_primary_target=label.get('primary_target'),
        certainty_policy_approved=normalized_certainty in PRIMARY_TARGET_CERTAINTY_NAMES,
        unapproved_cached_target_masked=int(certainty_policy_override),
        relevant_annotated_parts=len(related),
        relevant_visible_parts=len(visible),
        relevant_visible_in_crop_parts=len(points),
    )


def selection_maps(scores, k, image_id, seeds):
    result = {name: stable_topk(score, k) for name, score in scores.items()}
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed * 1_000_003 + image_id)
        result[f'random.{seed}'] = torch.randperm(576, generator=generator)[:k]
    return result


def selector_identity(name):
    return ('random', int(name.split('.')[1])) if name.startswith('random.') else (name, -1)


def evaluate_image(record, cached_scores, dense_scores, policy, part_names):
    """Labels only determine evaluation eligibility; score maps already exist for every query."""
    image, spatial = record['image'], record['spatial']
    image_id, split = image['image_id'], image['development_split']
    require(image['official_split'] == 'train' and split in ('train', 'val'), 'Non-development image')
    require(tuple(dense_scores.shape) == (27, 576) and bool(torch.isfinite(dense_scores).all()),
            'Dense scores must cover all 27 fixed queries on 576 patches')
    require(bool((dense_scores.abs() <= 1.0001).all()), 'Dense cosine scores fall outside [-1,1]')
    grid, size = tuple(spatial['grid_size']), tuple(spatial['processed_image_size_hw'])
    box = patch_centers_in_box(grid, size, tuple(image['bbox_model_xyxy']))
    all_parts = [tuple(p['model_xy']) for p in image['parts'] if p['visible'] and p['model_xy'] is not None]
    objects, attributes, eligibility, agreements = [], [], [], []
    generic = {**cached_scores, 'dense_object': dense_scores[0]}

    def score_rows(scope, aid, attribute_name, group, scores, points, destination):
        for k in policy['k_values']:
            selected = selection_maps(scores, k, image_id, policy['random_seeds'])
            base = dict(scope=scope, image_id=image_id, split=split, attribute_id=aid,
                        attribute_name=attribute_name, attribute_group=group, K=k)
            for name, indices in selected.items():
                method, seed = selector_identity(name)
                part = selection_part_metrics(indices, points, grid_size=grid, image_size=size) if points else {}
                metrics = selection_localization_metrics(indices, box)
                metrics.update({metric: part.get(metric) for metric in PART_METRICS})
                destination.append(dict(base, selector=method, selection_seed=seed,
                                        visible_part_instances=len(points),
                                        visible_part_patches=part.get('visible_part_patches', 0), **metrics))
            for a, b in itertools.combinations(selected, 2):
                sa, sb = set(selected[a].tolist()), set(selected[b].tolist())
                agreements.append(dict(base, selection_a=a, selection_b=b, jaccard=len(sa & sb) / len(sa | sb)))

    score_rows('object', 0, 'bird', 'object', generic, all_parts, objects)
    for column, attribute in enumerate(policy['attributes'], 1):
        reason, points, counts = attribute_eligibility(image, attribute, part_names)
        eligibility.append(dict(image_id=image_id, split=split, attribute_id=attribute['attribute_id'],
                                attribute_name=attribute['name'], attribute_group=attribute['group'],
                                reason=reason, **counts))
        if reason == 'eligible':
            score_rows('attribute', attribute['attribute_id'], attribute['name'], attribute['group'],
                       {**generic, 'dense_attribute': dense_scores[column]}, points, attributes)
    return objects, attributes, eligibility, agreements


def summarize_metrics(rows):
    """Average random seeds within image first, then describe variation across images."""
    within_image = defaultdict(list)
    for row in rows:
        for metric in METRICS:
            value = row[metric]
            if value is None:
                continue
            require(math.isfinite(float(value)), f'Nonfinite defined metric: {metric}')
            key = tuple(row[k] for k in ('scope', 'split', 'attribute_id', 'attribute_name',
                                         'attribute_group', 'selector', 'K', 'image_id')) + (metric,)
            within_image[key].append(float(value))
    grouped = defaultdict(list)
    for key, values in within_image.items():
        grouped[key[:7] + (key[-1],)].append(statistics.mean(values))
    summary = []
    for key, values in sorted(grouped.items()):
        row = dict(zip(('scope', 'split', 'attribute_id', 'attribute_name', 'attribute_group',
                        'selector', 'K', 'metric'), key))
        summary.append(dict(row, n_images=len(values), mean=statistics.mean(values),
                            sd_across_images=statistics.stdev(values) if len(values) > 1 else None))
    return summary


def macro_attribute_summary(summary, policy):
    grouped = defaultdict(list)
    for row in summary:
        if row['scope'] != 'attribute':
            continue
        for group in (row['attribute_group'], 'ALL_SELECTED_ATTRIBUTES'):
            grouped[(row['split'], group, row['selector'], row['K'], row['metric'])].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        group = key[1]
        selected_count = len(policy['attributes']) if group == 'ALL_SELECTED_ATTRIBUTES' else sum(
            a['group'] == group for a in policy['attributes'])
        result.append(dict(zip(('split', 'attribute_group', 'selector', 'K', 'metric'), key),
                           mean_across_evaluable_attributes=statistics.mean(r['mean'] for r in values),
                           evaluable_attributes=len(values), selected_attributes=selected_count,
                           eligible_image_attribute_pairs=sum(r['n_images'] for r in values)))
    return result


def paired_selector_deltas(rows):
    """Compare each selector with image-matched random and dense-object controls."""
    values = defaultdict(list)
    for row in rows:
        base = tuple(row[k] for k in ('scope', 'split', 'attribute_id', 'attribute_name', 'K', 'image_id'))
        for metric in METRICS:
            if row[metric] is not None:
                values[base + (row['selector'], metric)].append(float(row[metric]))
    means = {key: statistics.mean(v) for key, v in values.items()}
    grouped = defaultdict(list)
    for key, value in means.items():
        base, method, metric = key[:-2], key[-2], key[-1]
        for reference in ('random', 'dense_object'):
            other = base + (reference, metric)
            if method != reference and other in means:
                grouped[base[:-1] + (method, reference, metric)].append(value - means[other])
    output = []
    for key, values in sorted(grouped.items()):
        output.append(dict(zip(('scope', 'split', 'attribute_id', 'attribute_name', 'K',
                                'selector', 'reference', 'metric'), key),
                           n_images=len(values), mean_delta=statistics.mean(values),
                           sd_delta_across_images=statistics.stdev(values) if len(values) > 1 else None))
    return output


def summarize_agreements(rows):
    """Average random-selection variants within image; preserve attribute-specific coverage."""
    within_image = defaultdict(list)
    for row in rows:
        a, b = sorted((selector_identity(row['selection_a'])[0], selector_identity(row['selection_b'])[0]))
        if a == b:  # Distinct random seeds are retained in the raw table, not a method diagonal.
            continue
        key = (row['scope'], row['split'], int(row['attribute_id']), row['attribute_name'],
               row['attribute_group'], int(row['K']), int(row['image_id']), a, b)
        within_image[key].append(float(row['jaccard']))
    grouped = defaultdict(list)
    for key, values in within_image.items():
        grouped[key[:6] + key[7:]].append(statistics.mean(values))
    output = []
    for key, values in sorted(grouped.items()):
        output.append(dict(zip(('scope', 'split', 'attribute_id', 'attribute_name', 'attribute_group',
                                'K', 'selector_a', 'selector_b'), key), n_images=len(values),
                           mean_jaccard=statistics.mean(values),
                           sd_across_images=statistics.stdev(values) if len(values)>1 else None))
    return output

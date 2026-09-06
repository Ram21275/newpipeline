"""Readable Phase 4 diagnostics with fixed image/query selection and explicit scope."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .phase4 import attribute_eligibility
from .scoring import stable_topk


def plot_image(output_dir, record, pixels, cached_scores, dense_scores, policy, part_names):
    image = record['image']
    image_id = image['image_id']
    rgb = pixels.permute(1, 2, 0).numpy()
    tasks = [('object', None, {**cached_scores, 'dense_object': dense_scores[0]},
              [tuple(p['model_xy']) for p in image['parts'] if p['visible'] and p['model_xy'] is not None])]
    for column, attribute in enumerate(policy['attributes'], 1):
        reason, points, _ = attribute_eligibility(image, attribute, part_names)
        if reason == 'eligible':
            tasks.append(('attribute', attribute, {**cached_scores, 'dense_object': dense_scores[0],
                                                   'dense_attribute': dense_scores[column]}, points))
            break  # Fixed first eligible attribute by policy order; never select by score.
    paths = []
    for scope, attribute, scores, points in tasks:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, (name, score) in zip(axes.flat, scores.items()):
            ax.imshow(rgb, extent=(0, 336, 336, 0))
            values = score.float().reshape(24, 24).numpy()
            lo, hi = float(values.min()), float(values.max())
            ax.imshow(values, extent=(0, 336, 336, 0), cmap='magma', alpha=0.48,
                      interpolation='nearest', vmin=lo, vmax=hi)
            indices = stable_topk(score, 32).tolist()
            ax.scatter([(i % 24 + .5) * 14 for i in indices], [(i // 24 + .5) * 14 for i in indices],
                       s=9, c='cyan', linewidths=0)
            top = indices[0]
            ax.scatter([(top % 24 + .5) * 14], [(top // 24 + .5) * 14],
                       s=95, c='yellow', marker='*', edgecolors='black', linewidths=.5)
            if points:
                ax.scatter([p[0] for p in points], [p[1] for p in points],
                           s=25, c='lime', marker='x', linewidths=1)
            x1, y1, x2, y2 = image['bbox_model_xyxy']
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=1))
            ax.set_title(name.replace('_', ' '), fontsize=10)
        title = 'bird / all visible parts' if attribute is None else attribute['name']
        fig.suptitle(f"Image {image_id} · development {image['development_split']} · {title}\n"
                     'Top-32 cyan · top-1 yellow star · landmark proxies green × · each heatmap independently scaled',
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .93))
        suffix = 'object' if attribute is None else f"attribute_{attribute['attribute_id']:03d}"
        name = f'qualitative/{image_id:05d}_{suffix}.png'
        path = Path(output_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(name)
    return paths


def plot_summary(output_dir, summary, macro, mode):
    split = 'train' if mode == 'smoke' else 'val'
    methods = ['random', 'llm_attention', 'vision_cls_attention', 'logit_concept',
               'attention_logit_fusion', 'dense_object', 'dense_attribute']
    panels = [('object', 'inside_fraction', 'Bird box: selected centers inside'),
              ('object', 'top1_nearest_part_distance_patches', 'All visible parts: top-1 distance (lower is better)'),
              ('attribute', 'part_patch_recall', 'Attribute landmark proxies: Top-K patch recall'),
              ('attribute', 'top1_nearest_part_distance_patches', 'Attribute landmark proxies: top-1 distance')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (scope, metric, title) in zip(axes.flat, panels):
        shown = methods if scope == 'attribute' else methods[:-1]
        x = np.arange(len(shown))
        for offset, k in [(-.18, 16), (.18, 32)]:
            values, labels = [], []
            for method in shown:
                if scope == 'object':
                    candidates = [r for r in summary if r['scope']=='object' and r['split']==split
                                  and r['selector']==method and r['K']==k and r['metric']==metric]
                    values.append(candidates[0]['mean'] if candidates else np.nan)
                    labels.append(f"n={candidates[0]['n_images']}" if candidates else 'N/A')
                else:
                    candidates = [r for r in macro if r['split']==split and r['attribute_group']=='ALL_SELECTED_ATTRIBUTES'
                                  and r['selector']==method and r['K']==k and r['metric']==metric]
                    values.append(candidates[0]['mean_across_evaluable_attributes'] if candidates else np.nan)
                    labels.append(f"a={candidates[0]['evaluable_attributes']}/26" if candidates else 'N/A')
            bars = ax.bar(x+offset, values, width=.34, label=f'K={k}')
            ax.bar_label(bars, labels=labels, fontsize=6, rotation=90, padding=3)
        ax.set_xticks(x, [m.replace('_', '\n') for m in shown], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'Phase 4 · development {split} · {"ONE-IMAGE ENGINEERING SMOKE" if mode=="smoke" else "fixed protocol"}\n'
                 'Object means over images; attribute means over evaluable attributes; no confidence intervals', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    name = 'localization_overview.png'
    fig.savefig(Path(output_dir)/name, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return [name]


def plot_agreements(output_dir, summary, mode):
    split = 'train' if mode == 'smoke' else 'val'
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    order = ['random', 'llm_attention', 'vision_cls_attention', 'logit_concept',
             'attention_logit_fusion', 'dense_object', 'dense_attribute']
    for ax, scope in zip(axes, ('object', 'attribute')):
        methods = order[:-1] if scope == 'object' else order
        values = np.full((len(methods), len(methods)), np.nan)
        selected = [r for r in summary if r['scope'] == scope and r['split'] == split and r['K'] == 32]
        for i, a in enumerate(methods):
            for j, b in enumerate(methods):
                rows = [r for r in selected if {r['selector_a'], r['selector_b']} == {a, b}]
                if i != j and rows:
                    values[i, j] = np.mean([r['mean_jaccard'] for r in rows])
        im = ax.imshow(values, vmin=0, vmax=1, cmap='viridis')
        ax.set_facecolor('#eeeeee')
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, '—' if np.isnan(value) else f'{value:.2f}', ha='center', va='center',
                    fontsize=8, color='black' if np.isnan(value) or value>.55 else 'white')
        ax.set_xticks(range(len(methods)), [m.replace('_','\n') for m in methods], fontsize=7)
        ax.set_yticks(range(len(methods)), [m.replace('_',' ') for m in methods], fontsize=8)
        ax.set_title('Object query' if scope=='object' else 'Attribute queries: mean over evaluable attributes', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label='Top-32 Jaccard')
    fig.suptitle(f'Phase 4 · development {split} · matched selector agreement\n'
                 'Random variants averaged within image; diagonals omitted', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.90))
    name='selector_agreement.png'
    fig.savefig(Path(output_dir)/name,dpi=150,bbox_inches='tight')
    plt.close(fig)
    return [name]

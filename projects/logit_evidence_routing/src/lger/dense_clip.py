"""A frozen paired CLIP patch/text diagnostic; real loading is Kaggle CUDA only."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .phase4 import require


def normalize_cached_rgb(pixels, mean, std):
    require(pixels.dtype == torch.uint8 and tuple(pixels.shape) == (3, 336, 336),
            'Dense baseline requires the cached uint8 336x336 RGB crop')
    mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    require(bool(torch.isfinite(mean).all()) and bool(torch.isfinite(std).all())
            and bool((std > 0).all()), 'Invalid CLIP normalization')
    # Crop is already spatially aligned: do not resize/crop it again.
    return ((pixels.float() / 255.0 - mean) / std).unsqueeze(0)


def paired_patch_cosine(last_hidden_state, post_layernorm, visual_projection, text_features):
    """Apply paired CLIP heads to contextual patch states, excluding CLS."""
    require(last_hidden_state.ndim == 3 and last_hidden_state.shape[0] == 1
            and last_hidden_state.shape[1] == 577, 'Unexpected CLIP patch layout')
    patches = visual_projection(post_layernorm(last_hidden_state[:, 1:, :]))[0].float()
    text = text_features.float()
    require(patches.shape[1] == text.shape[1] and text.ndim == 2, 'Paired projection dimension differs')
    require(bool(torch.isfinite(patches).all()) and bool(torch.isfinite(text).all())
            and bool((patches.norm(dim=-1) > 1e-8).all()) and bool((text.norm(dim=-1) > 1e-8).all()),
            'Invalid/zero projected embedding')
    return F.normalize(text, dim=-1) @ F.normalize(patches, dim=-1).T


class FrozenDenseClip:
    def __init__(self, model, processor, texts, spatial_preprocessing):
        self.model = model.eval().requires_grad_(False)
        self.processor = processor
        self.device = next(model.parameters()).device
        vision = model.config.vision_config
        require(vision.image_size == 336 and vision.patch_size == 14 and vision.hidden_size == 1024
                and model.config.projection_dim == 768, 'Unexpected paired CLIP architecture')
        ip = processor.image_processor
        self.mean, self.std = list(ip.image_mean), list(ip.image_std)
        require(all(abs(a - b) < 1e-7 for a, b in zip(self.mean, spatial_preprocessing['image_mean']))
                and all(abs(a - b) < 1e-7 for a, b in zip(self.std, spatial_preprocessing['image_std']))
                and len(spatial_preprocessing['image_mean']) == len(spatial_preprocessing['image_std']) == 3,
                'CLIP and cached RGB normalization provenance differs')
        require(spatial_preprocessing['crop_size'] == {'height': 336, 'width': 336},
                'Cache crop geometry differs')
        tokenized = processor.tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        require(tokenized['input_ids'].shape[1] <= model.config.text_config.max_position_embeddings,
                'A fixed semantic query exceeds CLIP context; do not truncate silently')
        self.tokens = {key: value.to(self.device) for key, value in tokenized.items()
                       if key in ('input_ids', 'attention_mask')}
        with torch.inference_mode():
            output = model.text_model(**self.tokens, return_dict=True)
            self.text_features = model.text_projection(output.pooler_output).detach()

    @classmethod
    def from_pretrained(cls, policy, spatial_preprocessing):
        require(torch.cuda.is_available(), 'Real Phase 4 semantic scoring requires Kaggle CUDA')
        from transformers import CLIPModel, CLIPProcessor, __version__
        require(__version__ == '4.49.0', 'Use the existing pinned transformers==4.49.0 Kaggle environment')

        revision = policy['dense_revision']
        processor = CLIPProcessor.from_pretrained(policy['dense_model'], revision=revision)
        model = CLIPModel.from_pretrained(policy['dense_model'], revision=revision,
                                         torch_dtype=torch.float32, attn_implementation='eager').to('cuda')
        require(getattr(model.config, '_commit_hash', None) == revision, 'Resolved dense revision differs')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        texts = [policy['object_text']] + [attribute['text'] for attribute in policy['attributes']]
        return cls(model, processor, texts, spatial_preprocessing)

    def score(self, pixels, *, validate_global=False):
        inputs = normalize_cached_rgb(pixels, self.mean, self.std).to(self.device)
        with torch.inference_mode():
            output = self.model.vision_model(pixel_values=inputs, return_dict=True)
            scores = paired_patch_cosine(output.last_hidden_state, self.model.vision_model.post_layernorm,
                                         self.model.visual_projection, self.text_features)
            require(bool(torch.isfinite(scores).all()) and bool((scores.std(dim=1) > 1e-8).all()),
                    'Dense map is nonfinite or constant')
            diagnostics = dict(global_projection_checked=False, global_logit_max_abs_error=None)
            if validate_global:
                # This equality audits the normalization/head wiring against the official forward path.
                cls_post = self.model.vision_model.post_layernorm(output.last_hidden_state[:, 0, :])
                require(torch.allclose(cls_post, output.pooler_output, atol=1e-5, rtol=1e-5),
                        'Unexpected CLIP CLS post-normalization semantics')
                manual = F.normalize(self.model.visual_projection(output.pooler_output).float(), dim=-1)
                manual = manual @ F.normalize(self.text_features.float(), dim=-1).T
                manual = manual * self.model.logit_scale.exp()
                official = self.model(pixel_values=inputs, **self.tokens, return_dict=True).logits_per_image
                error = float((manual - official).abs().max())
                require(torch.allclose(manual, official, rtol=1e-4, atol=1e-4),
                        f'Paired global CLIP projection sanity failed (max error {error})')
                diagnostics = dict(global_projection_checked=True, global_logit_max_abs_error=error)
        return scores.detach().cpu().float(), diagnostics

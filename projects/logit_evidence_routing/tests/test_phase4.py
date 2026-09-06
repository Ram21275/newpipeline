"""Synthetic-only geometry, eligibility, aggregation and smoke/gate regression tests."""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from lger.dense_clip import normalize_cached_rgb, paired_patch_cosine
from lger.phase1b import feature_key
from lger.phase4 import (CACHED_SELECTORS, METRICS, attribute_eligibility, evaluate_image,
                         load_development_metadata, load_policy, macro_attribute_summary,
                         paired_selector_deltas, sha256, summarize_metrics)
from lger.scoring import stable_topk
from lger.stage_cache import REQUIRED_STAGE_NAMES, config_digest

PROJECT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


RUN = module('run_phase4_localization')
CHECK = module('validate_phase4_localization')


def fixture_image(policy):
    names = sorted({name for a in policy['attributes'] for name in a['relevant_parts']})
    part_names = {name: i + 1 for i, name in enumerate(names)}
    image = dict(image_id=0, official_split='train', development_split='train', relative_path='synthetic.jpg',
                 class_id=1, class_name='synthetic', selected_attribute_ids=[a['attribute_id'] for a in policy['attributes']],
                 bbox_model_xyxy=[0, 0, 336, 336],
                 parts=[dict(part_id=i, visible=True, model_xy=[7., 7.]) for i in part_names.values()],
                 attributes=[dict(attribute_id=a['attribute_id'], name=a['name'], group=a['group'],
                                  primary_target=True, certainty_name='probably') for a in policy['attributes']])
    record = dict(complete=True, image=image, stages=dict.fromkeys(REQUIRED_STAGE_NAMES),
                  spatial=dict(grid_size=[24,24], patch_count=576, processed_image_size_hw=[336,336],
                               original_image_size_wh=[336,336]))
    return record, part_names


class Phase4MathTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(PROJECT / 'configs/phase4_localization.json')
        self.record, self.names = fixture_image(self.policy)

    def test_policy_exactly_matches_frozen_attributes(self):
        import csv
        with (PROJECT / 'reports/development_20260906/phase3_review_bundle/per_attribute_metrics.csv').open() as handle:
            expected = {(int(r['attribute_id']), r['attribute_name'], r['attribute_group']) for r in csv.DictReader(handle)}
        actual = {(a['attribute_id'],a['name'],a['group']) for a in self.policy['attributes']}
        self.assertEqual(actual, expected)

    def test_rgb_normalization_does_not_resize_or_rescale_twice(self):
        pixels = torch.zeros((3,336,336), dtype=torch.uint8)
        pixels[:,0,0] = 255
        output = normalize_cached_rgb(pixels, [.5,.5,.5], [.5,.5,.5])
        self.assertEqual(tuple(output.shape), (1,3,336,336))
        self.assertEqual(output[0,0,0,0].item(), 1)
        self.assertEqual(output[0,0,1,1].item(), -1)
        with self.assertRaisesRegex(RuntimeError,'uint8'):
            normalize_cached_rgb(pixels.float(), [.5]*3, [.5]*3)

    def test_paired_heads_exclude_cls_and_normalize_vectors(self):
        states = torch.zeros(1,577,2)
        states[:,0,:] = 1000  # Must not enter dense patch scores.
        states[:,1:,0] = 3
        scores = paired_patch_cosine(states, nn.Identity(), nn.Identity(), torch.tensor([[1.,0.],[0.,2.]]))
        self.assertEqual(tuple(scores.shape),(2,576))
        torch.testing.assert_close(scores[0],torch.ones(576))
        torch.testing.assert_close(scores[1],torch.zeros(576))
        with self.assertRaisesRegex(RuntimeError,'zero projected'):
            paired_patch_cosine(torch.zeros_like(states),nn.Identity(),nn.Identity(),torch.eye(2))

    def test_eligibility_excludes_negative_uncertain_occluded_and_out_of_crop(self):
        image = self.record['image']
        a = self.policy['attributes'][0]
        label = image['attributes'][0]
        self.assertEqual(attribute_eligibility(image,a,self.names)[0],'eligible')
        label['primary_target'] = False
        self.assertEqual(attribute_eligibility(image,a,self.names)[0],'observed_negative')
        label['primary_target'] = None
        self.assertEqual(attribute_eligibility(image,a,self.names)[0],'uncertain_or_missing')
        label['primary_target'] = True
        for p in image['parts']: p['visible'] = False
        self.assertEqual(attribute_eligibility(image,a,self.names)[0],'no_visible_in_crop_relevant_part')
        for p in image['parts']: p.update(visible=True,model_xy=None)
        self.assertEqual(attribute_eligibility(image,a,self.names)[0],'no_visible_in_crop_relevant_part')

    def test_equal_k_coverage_and_explicit_unavailable_parts(self):
        generic = {name:torch.linspace(1,0,576) for name in CACHED_SELECTORS}
        dense = torch.linspace(-1,1,576).repeat(27,1)
        objects,attrs,eligibility,agreement = evaluate_image(self.record,generic,dense,self.policy,self.names)
        self.assertEqual((len(objects),len(attrs),len(eligibility),len(agreement)),(16,26*18,26,56+26*72))
        self.assertTrue(all(r['inside_fraction']==1 for r in objects+attrs))
        self.assertTrue(all(0<=r['jaccard']<=1 for r in agreement))
        for p in self.record['image']['parts']: p['visible']=False
        objects,attrs,eligibility,_ = evaluate_image(self.record,generic,dense,self.policy,self.names)
        self.assertEqual(len(attrs),0)
        self.assertTrue(all(r['part_patch_recall'] is None for r in objects))
        self.assertEqual(len(eligibility),26)

    def test_random_seeds_are_averaged_within_image(self):
        rows=[]
        for image_id,values in [(1,[0.,.3,.6]),(2,[.9,.9,.9])]:
            for seed,value in enumerate(values):
                row=dict(scope='object',split='val',attribute_id=0,attribute_name='bird',attribute_group='object',
                         selector='random',selection_seed=seed,K=16,image_id=image_id)
                row.update({m:None for m in METRICS});row['inside_fraction']=value;rows.append(row)
        summary=summarize_metrics(rows)[0]
        self.assertEqual(summary['n_images'],2)
        self.assertAlmostEqual(summary['mean'],.6)
        self.assertAlmostEqual(summary['sd_across_images'],.6/2**.5)

    def test_attribute_macro_retains_evaluable_count(self):
        summary=[dict(scope='attribute',split='val',attribute_group='has_bill_shape',selector='dense_attribute',
                      K=16,metric='part_patch_recall',mean=.4,n_images=2)]
        output=macro_attribute_summary(summary,self.policy)
        all_groups=next(r for r in output if r['attribute_group']=='ALL_SELECTED_ATTRIBUTES')
        self.assertEqual(all_groups['evaluable_attributes'],1)
        self.assertEqual(all_groups['selected_attributes'],26)


class Phase4PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.policy=load_policy(PROJECT/'configs/phase4_localization.json')
        self.record,self.names=fixture_image(self.policy)
        self.stage=self.root/'stage';self.stage.mkdir()
        self.localizer=self.root/'localizer';(self.localizer/'records').mkdir(parents=True)
        self.phase3=self.root/'phase3';self.phase3.mkdir()
        parts=self.root/'parts';parts.mkdir()
        (parts/'parts.txt').write_text('\n'.join(f'{i} {name}' for name,i in self.names.items()))
        cfg=dict(purpose='full_240_image_development_pilot_stage_cache',official_test_images=0,
                 manifest_image_ids=list(range(240)),model='synthetic',resolved_revision='synthetic',quantization='4bit',
                 prompt='synthetic',dataset={'root':str(self.root)},spatial_preprocessing={})
        digest=config_digest(cfg);self.policy['cache_config_digest']=digest
        self.dump(self.stage/'run_config.json',cfg)
        self.dump(self.root/'policy.json',self.policy)
        self.dump(self.phase3/'phase3_run_report.json',dict(status='PASS',cache_config_digest=digest,
                  official_test_images_used=0,selected_attributes=26,result_rows=90,per_attribute_rows=2340))
        self.dump(self.root/'gate.json',dict(status='PASS WITH ANOMALY',passed=True,blocking_findings=[]))
        cfg1={k:cfg[k] for k in ('model','resolved_revision','quantization','prompt')}
        cfg1.update(concept_tokenization_policy='single_lexical_token_v1',concept_token_ids=[11199,17952],
                    concept_tokens=['▁bird','▁birds'],layer_offset=-2)
        self.dump(self.localizer/'extraction_config.json',cfg1)
        shards=[];index=[]
        for shard in range(12):
            items=[]
            for i in range(shard*20,(shard+1)*20):
                record=copy.deepcopy(self.record);record['config_digest']=digest
                record['image'].update(image_id=i,development_split='train' if i<160 else 'val')
                items.append(dict(packed_record=record));index.append(dict(image_id=i))
            path=self.stage/f'{shard}.json'
            self.dump(path,dict(config_digest=digest,complete=True,records=items))
            shards.append(dict(metadata_path=path.name,metadata_sha256=sha256(path)))
        self.dump(self.stage/'index.json',dict(config_digest=digest,complete=True,records=index,shards=shards))
        self.dump(self.stage/'validation_report.json',dict(config_digest=digest,status='PASS',images=240,
                  reload_validated_records=240,official_test_images=0,official_test_split_untouched=True,
                  development_split_counts={'train':160,'val':80},stage_count_per_record=9,shards=12))
        scores={name:torch.linspace(1,0,576) for name in CACHED_SELECTORS}
        row=dict(schema_version=2,image_id=0,split='train',relative_path='synthetic.jpg',label=1,class_name='synthetic',
                 grid_size=(24,24),processed_image_size=(336,336),original_image_size=(336,336),patch_count=576,
                 bbox_xyxy_model=[0,0,336,336],processed_image=torch.tensor([0],dtype=torch.uint8).expand(3,336,336),
                 score_maps=scores,selections={feature_key(name,k):stable_topk(score,k)
                                               for name,score in scores.items() for k in (16,32)})
        torch.save(row,self.localizer/'records/00000.pt')
        self.plot_stub=types.ModuleType('lger.phase4_plots')
        self.plot_stub.plot_image=lambda *args:[]
        def plot_summary(out,*args):
            (out/'localization_overview.png').write_bytes(b'synthetic-only plot placeholder')
            return ['localization_overview.png']
        self.plot_stub.plot_summary=plot_summary
        def plot_agreements(out,*args):
            (out/'selector_agreement.png').write_bytes(b'synthetic-only agreement placeholder')
            return ['selector_agreement.png']
        self.plot_stub.plot_agreements=plot_agreements
        self.fake_scorer=types.SimpleNamespace(score=lambda pixels,validate_global:
             (torch.linspace(-1,1,576).expand(27,-1),dict(global_projection_checked=True,global_logit_max_abs_error=0.)))

    def dump(self,path,value):path.write_text(json.dumps(value))

    def invoke(self,mode='smoke',extra=()):
        argv=['run_phase4_localization.py','--mode',mode,'--policy',str(self.root/'policy.json'),
              '--stage-cache',str(self.stage),'--localizer-cache',str(self.localizer),
              '--phase1-gate',str(self.root/'gate.json'),'--phase3-bundle',str(self.phase3),
              '--output-dir',str(self.root/mode),*extra]
        with mock.patch.object(sys,'argv',argv),mock.patch.dict(sys.modules,{'lger.phase4_plots':self.plot_stub}), \
             mock.patch.object(RUN.FrozenDenseClip,'from_pretrained',return_value=self.fake_scorer), \
             mock.patch.object(torch.cuda,'is_available',return_value=True), \
             mock.patch.object(torch.cuda,'reset_peak_memory_stats'), \
             mock.patch.object(torch.cuda,'max_memory_allocated',return_value=0),mock.patch('builtins.print'):
            RUN.main()

    def test_synthetic_smoke_verifies_and_resumes_without_model_loading(self):
        self.invoke()
        report,*_=CHECK.validate(self.root/'smoke')
        self.assertEqual(report['images'],1)
        self.assertEqual(report['attribute_metric_rows'],468)
        self.fake_scorer.score=mock.Mock(side_effect=AssertionError('Resume must reuse score cache'))
        self.invoke()
        CHECK.validate(self.root/'smoke')
        bundle=self.root/'reports.zip'
        with mock.patch.object(sys,'argv',['validate_phase4_localization.py','--output-dir',str(self.root/'smoke'),
                                          '--bundle',str(bundle)]),mock.patch('builtins.print'):
            CHECK.main()
        import zipfile
        with zipfile.ZipFile(bundle) as archive:
            self.assertIn('PHASE4_RESULTS.md',archive.namelist())
            self.assertIn('selector_agreement.png',archive.namelist())
            self.assertFalse(any(name.endswith('.pt') for name in archive.namelist()))

    def test_development_requires_matching_semantic_smoke(self):
        with self.assertRaisesRegex(RuntimeError,'smoke-dir is required'):
            self.invoke('development')
        self.invoke()
        policy=self.policy.copy();policy['object_text']='changed text'
        self.dump(self.root/'policy.json',policy)
        with self.assertRaisesRegex(RuntimeError,'matching semantic smoke'):
            self.invoke('development',('--smoke-dir',str(self.root/'smoke')))

    def test_official_test_record_is_rejected_before_scoring(self):
        path=self.stage/'0.json';data=json.loads(path.read_text())
        data['records'][0]['packed_record']['image']['official_split']='test';self.dump(path,data)
        index=json.loads((self.stage/'index.json').read_text());index['shards'][0]['metadata_sha256']=sha256(path)
        self.dump(self.stage/'index.json',index)
        with self.assertRaisesRegex(RuntimeError,'Official test record'):
            self.invoke()

    def test_validator_rejects_missing_selector_even_after_rehash(self):
        self.invoke();out=self.root/'smoke';path=out/'object_metrics.csv'
        rows=CHECK.read_csv(path);rows[1]=rows[0].copy()
        RUN.write_csv(path,rows)
        report=json.loads((out/'phase4_run_report.json').read_text())
        report['artifact_manifest'][path.name]={'bytes':path.stat().st_size,'sha256':sha256(path)}
        self.dump(out/'phase4_run_report.json',report)
        with self.assertRaisesRegex(RuntimeError,'Matched object metric coverage'):
            CHECK.validate(out)

    def test_full_development_coverage_and_zero_support_attributes(self):
        # Small synthetic score maps and metadata, not real images/features/model inference.
        index=json.loads((self.stage/'index.json').read_text())
        source=torch.load(self.localizer/'records/00000.pt',weights_only=True)
        for shard in index['shards']:
            path=self.stage/shard['metadata_path'];data=json.loads(path.read_text())
            for item in data['records']:
                image=item['packed_record']['image'];image_id=image['image_id']
                if image_id!=0:
                    for label in image['attributes']:label['primary_target']=False
                if image_id==160:image['attributes'][0]['primary_target']=True
                row=dict(source,image_id=image_id,split=image['development_split'])
                torch.save(row,self.localizer/'records'/f'{image_id:05d}.pt')
            self.dump(path,data);shard['metadata_sha256']=sha256(path)
        self.dump(self.stage/'index.json',index)
        self.invoke()
        self.invoke('development',('--smoke-dir',str(self.root/'smoke')))
        report,summary,macro,support,_=CHECK.validate(self.root/'development')
        self.assertEqual(report['images'],240)
        self.assertEqual(report['object_metric_rows'],3840)
        self.assertEqual(report['eligibility_rows'],6240)
        self.assertEqual(report['eligible_image_attribute_pairs'],27)
        val=[r for r in macro if r['split']=='val' and r['attribute_group']=='ALL_SELECTED_ATTRIBUTES']
        self.assertTrue(val and all(r['evaluable_attributes']==1 for r in val))
        self.assertEqual(len([r for r in support if r['split']=='val' and r['eligible']=='0']),25)


if __name__=='__main__':unittest.main()

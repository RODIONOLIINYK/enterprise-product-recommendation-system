"""Profile views and memory lengths must preserve mask/statistic alignment."""

import copy
import pytest
import torch

from recommender.ranker import UPSCANRRanker, RankerLoss
from recommender.experiments.candidate_context.model import CandidateContextRanker
from recommender.experiments.candidate_token_enrichment.model import CandidateTokenEnrichmentRanker
from recommender.experiments.separate_profile.model import SeparateProfileRanker, save_checkpoint, load_checkpoint


def inputs(length):
    b, c = 2, 4
    mask = torch.ones(b, length, dtype=torch.long); mask[0, length//2:] = 0
    return dict(text_embedding=torch.randn(b,32), city_id=torch.zeros(b,dtype=torch.long),
                region_id=torch.zeros(b,dtype=torch.long), industry_id=torch.zeros(b,dtype=torch.long),
                tabular_features=torch.rand(b,8), sequence=torch.rand(b,length,9), attention_mask=mask,
                seq_product_ids=torch.ones(b,length,dtype=torch.long), seq_category_ids=torch.ones(b,length,dtype=torch.long),
                seq_subcategory_ids=torch.ones(b,length,dtype=torch.long), seq_business_line_ids=torch.ones(b,length,dtype=torch.long),
                seq_purchase_dates=torch.full((b,length), 90./365.25), prediction_day=torch.full((b,),100.),
                candidate_vectors=torch.randn(b,c,128), pairwise_features=torch.rand(b,c,41),
                candidate_mask=torch.ones(b,c,dtype=torch.bool))


def control():
    torch.manual_seed(42)
    base=UPSCANRRanker(text_dim=32,num_products=8,num_categories=4,num_subcategories=4,num_business_lines=4,
                       num_cities=4,num_regions=4,num_industries=4,statistics_attention=True,
                       excluded_statistics=('history_repeat',),cached_only=True)
    return CandidateTokenEnrichmentRanker.from_dynamic_context_ranker(CandidateContextRanker(base,'dynamic_features'))


def test_fused1_reconstruction_preserves_control_and_split_shapes():
    source=control().eval(); x=inputs(32)
    clone=SeparateProfileRanker.from_control(source,profile_layout='fused1',history_length=32).eval()
    with torch.no_grad():
        torch.testing.assert_close(source(**x).logits, clone(**x).logits,rtol=0,atol=0)
    for layout in ('fused2','split2'):
        m=SeparateProfileRanker.from_control(source,profile_layout=layout,history_length=32).eval()
        out=m(**x)
        assert out.memory.shape == (2,34,128)
        assert out.attn_weights.shape == (2,4,34)
        torch.testing.assert_close(out.token_0_weight,out.attn_weights[...,:2].sum(-1))
        torch.testing.assert_close(out.history_weights,out.attn_weights[...,2:].sum(-1))
        stats=m.base.build_memory_statistics(x['tabular_features'],x['sequence'])
        assert stats.shape == (2,34,17)
        assert torch.count_nonzero(stats[:,0]) == 0
        torch.testing.assert_close(stats[:,1,9:],x['tabular_features'])
        padding=m.base.build_key_padding_mask(x['attention_mask'])
        assert not padding[:,:2].any()
        assert torch.equal(padding[:,2:],x['attention_mask']==0)


def test_profile_tokens_are_separate_before_self_attention():
    m=SeparateProfileRanker.from_control(control(),profile_layout='split2',history_length=32)
    text, attributes=torch.randn(2,128),torch.randn(2,128)
    first=m.base.build_profile_tokens(text,attributes)
    second=m.base.build_profile_tokens(text+1,attributes)
    torch.testing.assert_close(first[:,1],second[:,1],rtol=0,atol=0)
    torch.testing.assert_close(second[:,0]-first[:,0],torch.ones(2,128))
    assert not any(p.requires_grad for p in m.base.profile_fusion.parameters())


def test_64_events_are_retained_and_self_attended_without_pooling():
    m=SeparateProfileRanker.from_control(control(),profile_layout='split2',history_length=64)
    x=inputs(64); y=torch.zeros(2,4);y[:,0]=1
    out=m(**x)
    assert out.memory.shape == (2,66,128)
    assert out.attn_weights.shape == (2,4,66)
    loss=RankerLoss()(out.logits,y,candidate_mask=x['candidate_mask'])['loss']
    assert torch.isfinite(loss);loss.backward()
    assert m.base.memory_encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    assert m.base.cross_attention.in_proj_weight.grad.abs().sum()>0
    m.eval()
    with torch.no_grad():
        ref=m(**x).logits
        changed=copy.deepcopy(x); changed['sequence'][x['attention_mask']==0]=float('nan')
        changed['seq_product_ids'][x['attention_mask']==0]=9999
        torch.testing.assert_close(ref,m(**changed).logits,rtol=0,atol=0)
    with pytest.raises(ValueError,match='64 event positions'):
        m(**inputs(32))


def test_split_checkpoint_reload_and_contract_tamper(tmp_path):
    m=SeparateProfileRanker.from_control(control(),profile_layout='split2',history_length=64).eval()
    path=tmp_path/'model.pt'
    provenance={'statistics_policy':'rolling_observed_v2','teacher_score_included':False}
    save_checkpoint(path,m,epoch=1,metrics={},selection_criterion='test',provenance=provenance,feature_contract_digest='test')
    loaded,_=load_checkpoint(path,expected_provenance=provenance,expected_feature_contract_digest='test');loaded.eval()
    x=inputs(64)
    with torch.no_grad():torch.testing.assert_close(m(**x).logits,loaded(**x).logits,rtol=0,atol=0)
    payload=torch.load(path,weights_only=False);payload['model_config']['profile_layout']='fused2'
    torch.save(payload,path)
    with pytest.raises(ValueError,match='mismatch'):
        load_checkpoint(path)

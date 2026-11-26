"""Deterministic exploratory stratification; inclusion probabilities are not claimed."""

from collections import Counter
import numpy as np


def select_cases(frame,*,size=50,seed=42):
    if not 1<=size<=len(frame):raise ValueError('Invalid sample size.')
    required={'case_id','customer_id','month','case_class','history_bin','industry','frequency_bin'}
    if not required<=set(frame) or frame.case_id.duplicated().any():raise ValueError('Unique, fully stratified cases required.')
    work=frame.copy();rng=np.random.default_rng(seed)
    work['_tie']=rng.random(len(work));work['_cell']=work.month+'|'+work.case_class
    cells=sorted(work['_cell'].unique());rng.shuffle(cells)
    chosen=[];used=set();history=Counter();industry=Counter();frequency=Counter();by_cell=Counter()
    while len(chosen)<size:
        progress=False
        for cell in sorted(cells,key=lambda c:(by_cell[c],cells.index(c))):
            rows=work.loc[work['_cell'].eq(cell)&~work.index.isin(chosen)&~work.customer_id.isin(used)]
            if rows.empty:continue
            score=rows.apply(lambda r:10*history[r.history_bin]+2*industry[r.industry]+frequency[r.frequency_bin]+r['_tie'],axis=1)
            idx=score.idxmin();r=work.loc[idx];chosen.append(idx);used.add(r.customer_id)
            history[r.history_bin]+=1;industry[r.industry]+=1;frequency[r.frequency_bin]+=1;by_cell[cell]+=1;progress=True
            if len(chosen)==size:break
        if not progress:raise ValueError('Not enough diverse customers for requested sample; do not silently repeat them.')
    selected=work.loc[chosen].drop(columns=['_tie','_cell']).copy()
    # Order itself is deterministic and intentionally nonchronological.
    selected=selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    def composition(df):
        return {k:{str(a):int(b) for a,b in df[k].value_counts().sort_index().items()}
                for k in ('month','case_class','history_bin','frequency_bin')}
    metadata={'seed':seed,'size':size,'population_size':len(frame),'unique_customers':int(selected.customer_id.nunique()),
              'sampling':'round-robin month × novelty/coverage; unique customer, balanced history/industry/frequency tie-breaking',
              'population_composition':composition(frame),'sample_composition':composition(selected),
              'sample_industries':int(selected.industry.nunique()),
              'oversampling':'Discovery/coverage strata deliberately oversampled; unweighted sample results are not population estimates.',
              'inclusion_probability':'not available for this constrained adaptive stratification; no inverse-probability population claim'}
    return selected,metadata

"""Complete historical market access through fixed-date, paginated, deterministic tools.

Response limits bound display only. Every statistic uses the complete eligible
history, and every query reapplies date<t. No chronological mutable cursor exists.
"""

from collections import Counter,defaultdict
import numpy as np
import pandas as pd

from recommender.split import TEST_START
from .protocol import PRODUCT_BASIC, PRODUCT_TEXT, COMPANY_FIELDS, digest


def json_value(value):
    if value is None or (not isinstance(value,(list,dict)) and pd.isna(value)):return None
    return value.item() if isinstance(value,np.generic) else value


class HistoryTools:
    def __init__(self,events,catalogue,companies,*,as_of,customer_id,allow_global=True,semantic=True):
        frame=pd.DataFrame(events).copy()
        required={'customer_id','purchase_date','product_id','quantity'}
        if not required<=set(frame):raise ValueError('History requires dated company/product/quantity events.')
        frame['purchase_date']=pd.to_datetime(frame.purchase_date,format='mixed')
        date=pd.Timestamp(as_of)
        if pd.isna(date) or date.tzinfo is not None or date!=date.normalize() or date>=TEST_START:
            raise ValueError('A valid development prediction day is required.')
        if frame.purchase_date.isna().any() or not frame.purchase_date.lt(TEST_START).all():
            raise ValueError('Held-out/invalid transaction fields are forbidden.')
        frame['purchase_date']=frame.purchase_date.dt.normalize()
        for name in ('customer_id','product_id'):frame[name]=frame[name].astype(str)
        frame['quantity']=pd.to_numeric(frame.quantity,errors='raise')
        if not np.isfinite(frame.quantity).all() or not frame.quantity.gt(0).all():raise ValueError('Paid positive finite quantities required.')
        if frame.duplicated(['customer_id','purchase_date','product_id']).any():raise ValueError('History must use merged purchase events.')
        self._events=frame.sort_values(['purchase_date','customer_id','product_id'],kind='stable')
        self._date=date;self.customer_id=str(customer_id);self.allow_global=bool(allow_global);self.semantic=bool(semantic)
        self._catalogue={str(k):dict(v) for k,v in catalogue.items()}
        self._companies={str(k):dict(v) for k,v in companies.items()}

    def _past(self,*,global_access=False,company_id=None):
        # Never rely on the snapshot/export alone to enforce the boundary.
        df=self._events.loc[self._events.purchase_date.lt(self._date)]
        if not self.allow_global:
            if global_access or (company_id is not None and str(company_id)!=self.customer_id):
                raise PermissionError('This arm permits only the focal company history.')
            df=df.loc[df.customer_id.eq(self.customer_id)]
        if company_id is not None:df=df.loc[df.customer_id.eq(str(company_id))]
        return df.copy()

    def _page(self,rows,*,offset=0,limit=20):
        if isinstance(offset,bool) or not isinstance(offset,int) or offset<0 or not isinstance(limit,int) or not 1<=limit<=500:
            raise ValueError('Nonnegative integer offset and limit in1..500 required.')
        enriched=[]
        for record in rows[offset:offset+limit]:
            clean={k:json_value(v) for k,v in record.items()}
            clean['evidence_id']='E-'+digest({'as_of':str(self._date.date()),'record':clean})[:20]
            enriched.append(clean)
        return {'rows':enriched,'total_rows':len(rows),'offset':offset,'limit':limit,
                'next_offset':offset+limit if offset+limit<len(rows) else None,
                'prediction_date':str(self._date.date()),'strictly_before':True}

    def catalogue(self,product_ids=None,query=None,offset=0,limit=100):
        known=set(self._past().product_id)
        # Every displayed candidate has a prior global sale, but customer-only
        # arms may inspect its catalogue metadata without learning other buyers.
        selected=set(map(str,product_ids)) if product_ids is not None else known
        fields=(*PRODUCT_BASIC,*PRODUCT_TEXT) if self.semantic else PRODUCT_BASIC
        records=[]
        for pid in sorted(selected):
            if pid not in self._catalogue:continue
            record={'product_id':pid,**{k:json_value(self._catalogue[pid].get(k)) for k in fields if k!='product_id'}}
            if query and str(query).casefold() not in ' '.join(str(v or '') for v in record.values()).casefold():continue
            records.append(record)
        return self._page(records,offset=offset,limit=limit)

    def company_history(self,company_id=None,offset=0,limit=100):
        cid=str(company_id or self.customer_id);df=self._past(company_id=cid)
        rows=[{'company_id':cid,'purchase_date':str(r.purchase_date.date()),'product_id':r.product_id,'quantity':float(r.quantity)}
              for r in df.itertuples()]
        return self._page(rows,offset=offset,limit=limit)

    def product_history(self,product_id,offset=0,limit=100):
        df=self._past(global_access=True);df=df.loc[df.product_id.eq(str(product_id))]
        rows=[{'company_id':r.customer_id,'purchase_date':str(r.purchase_date.date()),'product_id':r.product_id,'quantity':float(r.quantity)}
              for r in df.itertuples()]
        result=self._page(rows,offset=offset,limit=limit)
        result.update(distinct_companies=int(df.customer_id.nunique()),observed_events=len(df))
        return result

    def similar_companies(self,industry_match=False,offset=0,limit=20):
        df=self._past(global_access=True);histories=df.groupby('customer_id').product_id.agg(set).to_dict()
        focal=histories.get(self.customer_id,set());industry=self._companies.get(self.customer_id,{}).get('industry_branch')
        rows=[]
        for cid,products in histories.items():
            if cid==self.customer_id:continue
            same=bool(industry) and self._companies.get(cid,{}).get('industry_branch')==industry
            if industry_match and not same:continue
            overlap=products&focal
            rows.append({'company_id':cid,'jaccard':len(overlap)/max(len(products|focal),1),
                         'same_industry':same,'industry_branch':self._companies.get(cid,{}).get('industry_branch'),
                         'shared_products':sorted(overlap),'other_products':sorted(products-focal),
                         'observed_skus':len(products)})
        rows.sort(key=lambda r:(-r['jaccard'],-r['same_industry'],r['company_id']))
        return self._page(rows,offset=offset,limit=limit)

    def relations(self,anchor_ids,relation='next_basket',match='any',exclude_focal=True,group_by='product_id',candidate_ids=None,offset=0,limit=30):
        if relation not in ('next_basket','same_basket') or match not in ('any','all') or group_by not in ('product_id','product_category','brand','product_family','business_line'):
            raise ValueError('Unsupported relation/match/aggregation.')
        anchors=set(map(str,anchor_ids))
        if not anchors:raise ValueError('At least one anchor is required.')
        df=self._past(global_access=True)
        if exclude_focal:df=df.loc[~df.customer_id.eq(self.customer_id)]
        counts=Counter();buyers=defaultdict(set);examples=defaultdict(list);source_count=0;total_pairs=0;background=Counter()
        def mapped(skus):
            return set(skus) if group_by=='product_id' else {self._catalogue.get(p,{}).get(group_by) for p in skus}-{None,''}
        for cid,g in df.groupby('customer_id',sort=True):
            baskets=[(date,set(rows.product_id)) for date,rows in g.groupby('purchase_date',sort=True)]
            pairs=list(zip(baskets,baskets[1:])) if relation=='next_basket' else [(b,b) for b in baskets]
            for (source_day,source),(dest_day,dest) in pairs:
                total_pairs+=1;background.update(mapped(dest))
                matches=anchors<=source if match=='all' else bool(anchors&source)
                if not matches:continue
                source_count+=1
                values=mapped(dest-anchors if relation=='same_basket' else dest)
                for value in values:
                    counts[value]+=1;buyers[value].add(cid)
                    if len(examples[value])<3:
                        examples[value].append({'company_id':cid,'source_date':str(source_day.date()),'destination_date':str(dest_day.date())})
        allowed_dest=mapped(set(map(str,candidate_ids))) if candidate_ids is not None else None
        rows=[{'destination':value,'group_by':group_by,'support_baskets':count,'distinct_companies':len(buyers[value]),
               'source_baskets':source_count,'observed_conditional_frequency':count/source_count,
               'background_baskets':background[value],'all_observed_pairs':total_pairs,'examples':examples[value]}
              for value,count in counts.items() if allowed_dest is None or value in allowed_dest]
        rows.sort(key=lambda r:(-r['support_baskets'],str(r['destination'])))
        result=self._page(rows,offset=offset,limit=limit)
        result.update(source_baskets=source_count,all_observed_pairs=total_pairs,relation=relation,anchor_ids=sorted(anchors),match=match,
                      excludes_focal_company=bool(exclude_focal),filtered_to_candidates=bool(candidate_ids is not None),
                      scope='all companies observed strictly before prediction date')
        return result

    def trajectories(self,anchor_ids,destination_id,offset=0,limit=20):
        anchors=set(map(str,anchor_ids));destination=str(destination_id)
        if not anchors:raise ValueError('At least one anchor required.')
        df=self._past(global_access=True);rows=[];qualifying=0
        for cid,g in df.groupby('customer_id',sort=True):
            if cid==self.customer_id:continue
            first=g.groupby('product_id').purchase_date.min().to_dict()
            if not anchors<=set(first):continue
            anchor_day=max(first[a] for a in anchors)
            if destination in first and first[destination]<=anchor_day:continue
            qualifying+=1
            if destination not in first:continue
            after=g.loc[g.purchase_date.ge(anchor_day)&g.purchase_date.le(first[destination])]
            baskets=[{'purchase_date':str(day.date()),'product_ids':sorted(set(b.product_id))} for day,b in after.groupby('purchase_date',sort=True)]
            rows.append({'company_id':cid,'industry_branch':self._companies.get(cid,{}).get('industry_branch'),
                         'anchor_completion_date':str(anchor_day.date()),'destination_first_date':str(first[destination].date()),
                         'elapsed_days':int((first[destination]-anchor_day).days),'observed_trajectory':baskets})
        rows.sort(key=lambda r:(r['elapsed_days'],r['company_id']))
        result=self._page(rows,offset=offset,limit=limit)
        result.update(qualifying_companies=qualifying,later_adopting_companies=len(rows),
                      right_censoring_warning='Non-adoption before the cutoff is not a confirmed negative preference.')
        return result

    def intervals(self,company_id=None,product_id=None,offset=0,limit=100):
        df=self._past(company_id=str(company_id or self.customer_id))
        if product_id is not None:df=df.loc[df.product_id.eq(str(product_id))]
        rows=[]
        for pid,g in df.groupby('product_id',sort=True):
            dates=np.sort(g.purchase_date.unique());gaps=np.diff(dates)/np.timedelta64(1,'D')
            rows.append({'product_id':pid,'purchase_events':len(g),'quantity_sum':float(g.quantity.sum()),
                         'first_date':str(pd.Timestamp(dates[0]).date()),'last_date':str(pd.Timestamp(dates[-1]).date()),
                         'days_since_last':int((self._date-pd.Timestamp(dates[-1])).days),
                         'median_gap_days':float(np.median(gaps)) if len(gaps) else None,
                         'mean_gap_days':float(np.mean(gaps)) if len(gaps) else None,
                         'gap_std_days':float(np.std(gaps)) if len(gaps) else None,'last_quantity':float(g.iloc[-1].quantity)})
        return self._page(rows,offset=offset,limit=limit)

    def market_trends(self,product_ids=None,offset=0,limit=30):
        df=self._past(global_access=True);selected=set(map(str,product_ids)) if product_ids else set(df.product_id)
        rows=[]
        for pid,g in df.loc[df.product_id.isin(selected)].groupby('product_id',sort=True):
            row={'product_id':pid,'lifetime_events':len(g),'lifetime_buyers':int(g.customer_id.nunique())}
            for days in (7,30,90):
                recent=g.loc[g.purchase_date.ge(self._date-pd.Timedelta(days=days))]
                row[f'events_{days}d']=len(recent);row[f'buyers_{days}d']=int(recent.customer_id.nunique())
            row['monthly_events']={str(month):len(group) for month,group in g.groupby(g.purchase_date.dt.to_period('M'),sort=True)}
            rows.append(row)
        rows.sort(key=lambda r:(-r['events_30d'],r['product_id']))
        return self._page(rows,offset=offset,limit=limit)

    def context(self,candidate_ids):
        df=self._past(company_id=self.customer_id)
        baskets=[{'purchase_date':str(date.date()),'items':[[r.product_id,float(r.quantity)] for r in rows.itertuples()]}
                 for date,rows in df.groupby('purchase_date',sort=True)]
        pids=list(candidate_ids)+sorted(set(df.product_id)-set(candidate_ids))
        catalogue=self.catalogue(pids,limit=500)
        while catalogue['next_offset'] is not None:
            page=self.catalogue(pids,offset=catalogue['next_offset'],limit=500)
            catalogue['rows'].extend(page['rows']);catalogue['next_offset']=page['next_offset']
        by_id={r['product_id']:r for r in catalogue['rows']}
        catalogue['rows']=[by_id[p] for p in pids if p in by_id]
        company={k:json_value(self._companies.get(self.customer_id,{}).get(k)) for k in COMPANY_FIELDS}
        full=self._past()
        return {'company_id':self.customer_id,'company_profile':company,'metadata_policy':'snapshot_unverified',
                'complete_customer_baskets':baskets,'customer_history_events':len(df),
                'customer_history_evidence_id':'E-'+digest({'company_id':self.customer_id,'baskets':baskets,'as_of':str(self._date.date())})[:20],
                'customer_intervals':self.intervals(limit=500),'catalogue':catalogue,
                'visible_market':{'events':len(full),'companies':int(full.customer_id.nunique()),'products':int(full.product_id.nunique()),
                                  'last_observed_date':str(full.purchase_date.max().date()) if len(full) else None},
                'product_text_available':self.semantic,'global_history_available':self.allow_global}

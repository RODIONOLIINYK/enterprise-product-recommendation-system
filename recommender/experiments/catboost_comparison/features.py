"""Strictly earlier event features, preserving the historical 17-column contract."""

import numpy as np
import pandas as pd

from recommender.split import TEST_START

CATEGORICAL = ('product_id', 'product_category', 'business_line')
LEGACY_NUMERIC = (
    'previous_paid_purchase_count', 'previous_paid_quantity', 'last_paid_quantity',
    'days_since_last_paid_purchase', 'average_days_between_customer_product_purchases',
    'std_days_between_customer_product_purchases', 'expected_days_before_next_order',
    'previous_category_purchase_count', 'previous_category_purchase_share',
    'previous_business_line_purchase_count', 'previous_business_line_purchase_share',
    'historical_product_purchase_count', 'historical_product_unique_customer_count',
    'product_purchase_count_last_30_days',
)
LEGACY_FEATURES = CATEGORICAL + LEGACY_NUMERIC


def categorical_frame(product_ids, catalogue, indices):
    pids = np.asarray(product_ids, dtype=object)
    if len(set(pids)) != len(pids):
        raise ValueError('Product identities must be unique.')
    master = catalogue.set_index('product_id').reindex(pids)
    values = {'product_id': pids}
    for name in CATEGORICAL[1:]:
        values[name] = master[name].fillna('<NONE>').astype(str).to_numpy()
    return pd.DataFrame({name: pd.Categorical(vals[indices.reshape(-1)]) for name, vals in values.items()})


def build_legacy_numeric(purchases, catalogue, product_ids, customer_ids, prediction_dates, indices):
    """No labels are consumed; observations on the prediction day are excluded.

    Customer frequencies count purchase rows, as in the original notebook. Category/line
    metadata use the current snapshot, matching the current experiment's declared policy.
    """
    dates = pd.to_datetime(prediction_dates)
    if dates.isna().any() or (dates >= TEST_START).any():
        raise ValueError('Only dated development queries are permitted.')
    frame = purchases.copy()
    frame['purchase_date'] = pd.to_datetime(frame.purchase_date).dt.normalize()
    if frame.purchase_date.isna().any() or (frame.purchase_date >= TEST_START).any():
        raise ValueError('Purchases must be development-only.')
    if len(customer_ids) != len(indices) or len(dates) != len(indices):
        raise ValueError('Query identities and candidate rows must align.')
    pids = [str(p) for p in product_ids]
    index = {p: i for i, p in enumerate(pids)}
    master = catalogue.set_index('product_id')
    attrs = {k: master[k].fillna('<NONE>').astype(str).to_dict() for k in CATEGORICAL[1:]}
    cat_values = sorted(set(attrs['product_category'].values()) | {'<NONE>'})
    line_values = sorted(set(attrs['business_line'].values()) | {'<NONE>'})
    cat_index = {v: i for i, v in enumerate(cat_values)}
    line_index = {v: i for i, v in enumerate(line_values)}
    cats = np.array([cat_index[attrs['product_category'].get(p, '<NONE>')] for p in pids])
    lines = np.array([line_index[attrs['business_line'].get(p, '<NONE>')] for p in pids])
    frame['quantity'] = pd.to_numeric(frame.quantity, errors='raise')
    frame['_sku'] = frame.product_id.astype(str).map(index).fillna(-1).astype(int)
    frame['_cat'] = frame.product_id.astype(str).map(attrs['product_category']).fillna('<NONE>').map(cat_index)
    frame['_line'] = frame.product_id.astype(str).map(attrs['business_line']).fillna('<NONE>').map(line_index)
    frame['_day'] = frame.purchase_date.to_numpy(dtype='datetime64[D]').astype(np.int64)
    qdays = dates.to_numpy(dtype='datetime64[D]').astype(np.int64)
    out = np.zeros((*indices.shape, len(LEGACY_NUMERIC)), dtype=np.float32)
    customers = np.asarray(customer_ids, dtype=str)
    groups = {str(cid): rows.sort_values(['purchase_date', 'product_id'], kind='stable') for cid, rows in frame.groupby('customer_id', sort=False)}
    for cid in np.unique(customers):
        query_rows = np.flatnonzero(customers == cid)
        query_rows = query_rows[np.argsort(qdays[query_rows], kind='stable')]
        rows = groups.get(cid)
        events = [] if rows is None else list(rows[['_sku', '_cat', '_line', '_day', 'quantity']].itertuples(index=False, name=None))
        n = len(pids)
        count = np.zeros(n); qty = np.zeros(n); last_qty = np.zeros(n)
        last = np.zeros(n); gap_sum = np.zeros(n); gap_sq = np.zeros(n)
        cat_count = np.zeros(len(cat_values)); line_count = np.zeros(len(line_values))
        cursor, total = 0, 0
        for row in query_rows:
            day = qdays[row]
            while cursor < len(events) and events[cursor][3] < day:
                sku, cat, line, event_day, quantity = events[cursor]
                cat_count[cat] += 1; line_count[line] += 1; total += 1
                if sku >= 0:
                    if count[sku]:
                        gap = event_day - last[sku]
                        gap_sum[sku] += gap; gap_sq[sku] += gap * gap
                    count[sku] += 1; qty[sku] += quantity
                    last_qty[sku] = quantity; last[sku] = event_day
                cursor += 1
            selected = indices[row]
            c = count[selected]; q = qty[selected]; lq = last_qty[selected]
            gap_n = c - 1
            mean = np.divide(gap_sum[selected], gap_n, out=np.full(len(c), np.nan), where=gap_n > 0)
            variance = np.divide(gap_sq[selected], gap_n, out=np.full(len(c), np.nan), where=gap_n > 0) - mean ** 2
            std = np.sqrt(np.maximum(variance, 0))
            age = np.where(c > 0, day - last[selected], 0)
            avg_qty = np.divide(q, c, out=np.zeros(len(c)), where=c > 0)
            expected = np.full(len(c), np.nan)
            cycle = (mean > 0) & (lq > 0) & (avg_qty > 0)
            expected[cycle] = mean[cycle] * lq[cycle] / avg_qty[cycle] - age[cycle]
            out[row, :, :11] = np.stack([
                c, q, lq, age, mean, std, expected,
                cat_count[cats[selected]], cat_count[cats[selected]] / max(total, 1),
                line_count[lines[selected]], line_count[lines[selected]] / max(total, 1),
            ], axis=-1)
    # Global counts roll over observed purchases, with no fixed validation cutoff.
    for sku, rows in frame.loc[frame['_sku'] >= 0].groupby('_sku', sort=False):
        qr, qc = np.where(indices == sku)
        if not len(qr):
            continue
        event_days = np.sort(rows['_day'].to_numpy())
        stops = np.searchsorted(event_days, qdays[qr], side='left')
        starts = np.searchsorted(event_days, qdays[qr] - 30, side='left')
        first_buyer_days = np.sort(rows.groupby('customer_id')['_day'].min().to_numpy())
        buyers = np.searchsorted(first_buyer_days, qdays[qr], side='left')
        out[qr, qc, 11:] = np.stack([stops, buyers, stops - starts], axis=-1)
    return out


def legacy_frame(numeric, categorical):
    output = categorical.copy(deep=False)
    for i, name in enumerate(LEGACY_NUMERIC):
        output[name] = numeric[..., i].reshape(-1)
    return output

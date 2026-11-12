# Selected-checkpoint comparison

Mean ± sample standard deviation across all repeated seeds. A dash means single-seed screening, not zero uncertainty. No promotion.

| Configuration | n | New hits@10 | Recall@10 | NDCG@10 | Conditional new recall | Repeat recall (micro) | New targets in slate |
|---|---:|---:|---:|---:|---:|---:|---:|
| production50-control | 3 | 188.0000 ± 4.3589 | 0.7585 ± 0.0025 | 0.5919 ± 0.0015 | 0.2254 ± 0.0052 | 0.8875 ± 0.0030 | 834.0000 ± 0.0000 |
| fixed50-control | 1 | 197.0000 (screen) | 0.7492 (screen) | 0.5837 (screen) | 0.2231 (screen) | 0.8685 (screen) | 883.0000 (screen) |
| production50-no_repeat_flag | 1 | 190.0000 (screen) | 0.7607 (screen) | 0.5927 (screen) | 0.2278 (screen) | 0.8895 (screen) | 834.0000 (screen) |
| production50-no_repeat_group | 1 | 268.0000 (screen) | 0.7078 (screen) | 0.5423 (screen) | 0.3213 (screen) | 0.8119 (screen) | 834.0000 (screen) |
| production50-experts | 3 | 197.0000 ± 2.6458 | 0.7599 ± 0.0011 | 0.5928 ± 0.0009 | 0.2362 ± 0.0032 | 0.8875 ± 0.0023 | 834.0000 ± 0.0000 |
| production50-group_gate | 1 | 185.0000 (screen) | 0.7594 (screen) | 0.5903 (screen) | 0.2218 (screen) | 0.8898 (screen) | 834.0000 (screen) |
| production50-recent_sku | 1 | 194.0000 (screen) | 0.7580 (screen) | 0.5920 (screen) | 0.2326 (screen) | 0.8855 (screen) | 834.0000 (screen) |
| production50-hierarchy | 3 | 199.6667 ± 4.1633 | 0.7589 ± 0.0008 | 0.5935 ± 0.0005 | 0.2394 ± 0.0050 | 0.8871 ± 0.0021 | 834.0000 ± 0.0000 |
| production50-complements | 3 | 198.0000 ± 2.6458 | 0.7595 ± 0.0021 | 0.5926 ± 0.0007 | 0.2374 ± 0.0032 | 0.8879 ± 0.0019 | 834.0000 ± 0.0000 |
| production50-dynamic_features | 3 | 216.6667 ± 10.7858 | 0.7604 ± 0.0014 | 0.5949 ± 0.0006 | 0.2598 ± 0.0129 | 0.8851 ± 0.0008 | 834.0000 ± 0.0000 |
| production50-values_only | 1 | 190.0000 (screen) | 0.7590 (screen) | 0.5899 (screen) | 0.2278 (screen) | 0.8885 (screen) | 834.0000 (screen) |
| production50-bias_and_values | 1 | 188.0000 (screen) | 0.7579 (screen) | 0.5911 (screen) | 0.2254 (screen) | 0.8867 (screen) | 834.0000 (screen) |
| production50-context_direct | 1 | 193.0000 (screen) | 0.7561 (screen) | 0.5874 (screen) | 0.2314 (screen) | 0.8854 (screen) | 834.0000 (screen) |
| production50-context_dot | 1 | 195.0000 (screen) | 0.7576 (screen) | 0.5833 (screen) | 0.2338 (screen) | 0.8844 (screen) | 834.0000 (screen) |
| production50-context_residual | 1 | 191.0000 (screen) | 0.7588 (screen) | 0.5935 (screen) | 0.2290 (screen) | 0.8880 (screen) | 834.0000 (screen) |
| static-quota50_25-control | 3 | 202.0000 ± 4.0000 | 0.7616 ± 0.0007 | 0.5927 ± 0.0004 | 0.2319 ± 0.0051 | 0.8892 ± 0.0015 | 871.0000 ± 2.0000 |
| dynamic-quota50_25-control | 3 | 199.3333 ± 6.1101 | 0.7617 ± 0.0008 | 0.5937 ± 0.0006 | 0.2242 ± 0.0056 | 0.8893 ± 0.0003 | 889.0000 ± 11.5326 |
| relations-quota50_25-control | 3 | 204.6667 ± 6.0277 | 0.7606 ± 0.0017 | 0.5929 ± 0.0007 | 0.2275 ± 0.0056 | 0.8868 ± 0.0006 | 899.6667 ± 4.6188 |
| dynamic-quota50_10-control | 1 | 210.0000 (screen) | 0.7556 (screen) | 0.5893 (screen) | 0.2220 (screen) | 0.8807 (screen) | 946.0000 (screen) |
| dynamic-top50-control | 3 | 203.6667 ± 5.5076 | 0.7611 ± 0.0013 | 0.5930 ± 0.0010 | 0.2366 ± 0.0067 | 0.8880 ± 0.0011 | 861.0000 ± 11.7898 |
| dynamic-quota100_50-control | 3 | 215.6667 ± 9.2376 | 0.7623 ± 0.0014 | 0.5939 ± 0.0010 | 0.1685 ± 0.0073 | 0.8879 ± 0.0019 | 1279.6667 ± 1.5275 |
| combined-control | 3 | 212.6667 ± 7.6376 | 0.7629 ± 0.0029 | 0.5944 ± 0.0014 | 0.1662 ± 0.0061 | 0.8889 ± 0.0029 | 1279.6667 ± 1.5275 |
| combined-combined_direct | 3 | 239.0000 ± 5.2915 | 0.7613 ± 0.0022 | 0.5875 ± 0.0024 | 0.1868 ± 0.0043 | 0.8828 ± 0.0024 | 1279.6667 ± 1.5275 |
| dot100-control | 3 | 214.0000 ± 5.2915 | 0.7620 ± 0.0022 | 0.5929 ± 0.0005 | 0.1692 ± 0.0042 | 0.8885 ± 0.0019 | 1265.0000 ± 0.0000 |
| dot100-dynamic_features | 3 | 227.6667 ± 4.9329 | 0.7625 ± 0.0012 | 0.5958 ± 0.0014 | 0.1800 ± 0.0039 | 0.8860 ± 0.0012 | 1265.0000 ± 0.0000 |
| dynamic100-dynamic_features | 3 | 233.3333 ± 6.6583 | 0.7642 ± 0.0009 | 0.5961 ± 0.0005 | 0.1823 ± 0.0052 | 0.8872 ± 0.0029 | 1279.6667 ± 1.5275 |
| production50-pointwise | 1 | 188.0000 (screen) | 0.7541 (screen) | 0.5805 (screen) | 0.2254 (screen) | 0.8823 (screen) | 834.0000 (screen) |
| finetune50-control | 1 | 187.0000 (screen) | 0.7593 (screen) | 0.5908 (screen) | 0.2242 (screen) | 0.8897 (screen) | 834.0000 (screen) |
| finetune50-context_residual | 1 | 187.0000 (screen) | 0.7584 (screen) | 0.5919 (screen) | 0.2242 (screen) | 0.8883 (screen) | 834.0000 (screen) |

## Full-pool selectors

| Selector | n | Unseen hits@25 | @50 | @100 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| dot | 1 | 707.0000 (screen) | 1005.0000 (screen) | 1330.0000 (screen) | 0.3535 (screen) | 0.2162 (screen) |
| static | 3 | 708.6667 ± 6.5064 | 991.6667 ± 7.0946 | 1323.3333 ± 8.1445 | 0.7342 ± 0.0018 | 0.5553 ± 0.0017 |
| dynamic | 3 | 731.6667 ± 4.5092 | 1011.0000 ± 7.0000 | 1348.6667 ± 4.0415 | 0.7500 ± 0.0041 | 0.5759 ± 0.0042 |
| relations | 3 | 731.0000 ± 5.2915 | 1019.0000 ± 7.5498 | 1350.3333 ± 9.5044 | 0.7471 ± 0.0052 | 0.5757 ± 0.0058 |

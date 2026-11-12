# Fit time, parameter allocation and forward latency

Time is mean ± sample SD in seconds per completed fit, including per-epoch validation. Parameters exclude the shared frozen S0 encoder. Active means a gradient tensor was present on one diagnostic batch; it does not prove every entry has nonzero gradient. Latencies average per-seed batch-1 medians, in milliseconds; full-request timings are in telemetry.json.

| Configuration | n | Fit seconds | Allocated parameters | Gradient-active parameters | CPU forward median | MPS forward median |
|---|---:|---:|---:|---:|---:|---:|
| production50-control | 3 | 101.1757 ± 5.8282 | 866313 | 664329 | 0.461 | 1.283 |
| fixed50-control | 1 | 95.6446 (screen) | 866313 | 664329 | 0.496 | 1.197 |
| production50-no_repeat_flag | 1 | 100.1524 (screen) | 866313 | 664329 | 0.415 | 1.249 |
| production50-no_repeat_group | 1 | 114.7218 (screen) | 866313 | 664329 | 0.722 | 2.092 |
| production50-experts | 3 | 130.4149 ± 12.4805 | 1154315 | 808330 | 0.561 | 1.498 |
| production50-group_gate | 1 | 115.2840 (screen) | 866314 | 664330 | 0.482 | 1.206 |
| production50-recent_sku | 1 | 106.1833 (screen) | 1011082 | 665097 | 0.472 | 1.276 |
| production50-hierarchy | 3 | 107.1163 ± 3.6949 | 1012106 | 666121 | 0.475 | 1.297 |
| production50-complements | 3 | 105.2779 ± 4.4335 | 1010570 | 664585 | 0.427 | 1.264 |
| production50-dynamic_features | 3 | 114.7204 ± 12.5617 | 1017482 | 671497 | 0.508 | 1.401 |
| production50-values_only | 1 | 96.6067 (screen) | 865361 | 663377 | 0.425 | 1.087 |
| production50-bias_and_values | 1 | 102.5605 (screen) | 866313 | 664329 | 0.405 | 1.289 |
| production50-context_direct | 1 | 121.9507 (screen) | 1082762 | 736777 | 0.489 | 1.411 |
| production50-context_dot | 1 | 160.7299 (screen) | 1082761 | 736776 | 0.478 | 1.472 |
| production50-context_residual | 1 | 154.8538 (screen) | 1082762 | 880778 | 0.626 | 1.565 |
| static-quota50_25-control | 3 | 95.1719 ± 4.2636 | 866313 | 664329 | 0.431 | 1.267 |
| dynamic-quota50_25-control | 3 | 96.4225 ± 2.1390 | 866313 | 664329 | 0.471 | 1.348 |
| relations-quota50_25-control | 3 | 95.8311 ± 2.9293 | 866313 | 664329 | 0.473 | 1.329 |
| dynamic-quota50_10-control | 1 | 99.5656 (screen) | 866313 | 664329 | 0.466 | 1.302 |
| dynamic-top50-control | 3 | 95.5548 ± 2.5903 | 866313 | 664329 | 0.475 | 1.321 |
| dynamic-quota100_50-control | 3 | 113.3358 ± 1.8469 | 866313 | 664329 | 0.625 | 1.499 |
| combined-control | 3 | 118.0283 ± 6.4529 | 866313 | 664329 | 0.568 | 1.367 |
| combined-combined_direct | 3 | 158.3243 ± 20.9907 | 1086346 | 740361 | 0.824 | 2.224 |
| dot100-control | 3 | 114.3992 ± 0.9571 | 866313 | 664329 | 0.599 | 1.584 |
| dot100-dynamic_features | 3 | 119.0000 ± 1.3814 | 1017482 | 671497 | 0.682 | 1.722 |
| dynamic100-dynamic_features | 3 | 137.2239 ± 13.6641 | 1017482 | 671497 | 0.663 | 1.648 |
| production50-pointwise | 1 | 37.5353 (screen) | 75777 | 75777 | 0.045 | 0.259 |
| finetune50-control | 1 | 121.5738 (screen) | 866313 | 664329 | 0.497 | 1.203 |
| finetune50-context_residual | 1 | 170.1267 (screen) | 1082762 | 880778 | 0.630 | 1.646 |

| Full-pool selector | n | Fit seconds | Parameters |
|---|---:|---:|---:|
| dot | 1 | 0.0000 (screen) | 0 |
| static | 3 | 115.8766 ± 11.3246 | 79361 |
| dynamic | 3 | 113.6468 ± 12.6977 | 79361 |
| relations | 3 | 119.0126 ± 14.0332 | 79361 |

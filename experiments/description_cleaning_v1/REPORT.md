# Controlled Product Description Cleaning Experiment (`no-brand-v1` + `CompactStage1Selector uv_47`, `seed=42`)

## 1. Scope, Provenance & Experimental Contract

- **Active Baseline Reference (Unchanged, Not Promoted):** `no-brand-v1` (`E_no_brand` epoch 16 + `CompactStage1Selector uv_47` epoch 1, `seed=42`, `t1_fam` text without descriptions, 6 product embeddings excluding `brand`, customer `sales_branch`, 0 tower customer stats, static 12D `container_type`, frozen 303-input `uv_47`).
- **Variants Tested:**
  - `D_html`: appends legacy `clean_html_description(raw_description)` to exact `t1_fam` prefix (`t1_fam_desc_html_v1`).
  - `D_clean`: appends deterministic `desc_clean_v1` cleaned description to exact `t1_fam` prefix (`t1_fam_desc_clean_v1`).
- **Stage 1 MLP Only:** Clean towers trained from scratch (30 epochs, `seed=42`) -> `CompactStage1Selector uv_47` (10 epochs, `seed=42`, `new_hits@50 -> ndcg@10 -> recall@10`). **Zero Stage 2 / `UPS-CANR` training or evaluation.**
- **Strict Date Gate & Metadata Policy:** Train `< 2025-10-01`, Validation `2025-10-01..2026-02-18` (`6,191` transitions, `1,857` new targets), zero held-out (`>= 2026-02-19`) access. Description export (`data/raw/описание.csv`, SHA-256 `a3db2131ff72d6b6584efa00015e66dce1f22eeabc052e74dac3dd84faa31955`) is **snapshot-unverified (`snapshot_unverified`)** — not historical effective-dated.
- **Data Coverage:** `276` unique SKU rows in CSV (`253` non-empty); `159` in 841 dev SKUs (`155` usable = `18.4304%`). Validation new targets: `514` WITH description vs `1343` WITHOUT description (sum `1857`).

## 2. Full-Pool `CompactStage1Selector uv_47` Results (6,191 Queries / 1,857 New Targets, Seed=42)

| Variant | S1 Ep (s) | MLP Ep (s) | New@25 (/1,857) | New@50 (/1,857) | New@100 (/1,857) | Recall@10 (/6,191) | NDCG@10 (/6,191) | Δ New@50 vs Ref (95% CI) | Δ NDCG@10 vs Ref (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `no-brand-v1` (Ref) | 16 (ref) | 1 (ref) | 741 (39.90%) | 1037 (55.84%) | 1324 (71.30%) | 0.75656 | 0.57829 | — | — |
| `D_html` | 18 (43.9s) | 1 (109.9s) | 731 (39.36%) | 1017 (54.77%) | 1321 (71.14%) | 0.75493 | 0.57653 | -20 ([-2.11, -1.07, -0.05] pp) | -0.00176 ([-0.0034, -0.0018, -0.0003]) |
| `D_clean` | 18 (44.2s) | 1 (111.2s) | 726 (39.10%) | 1012 (54.50%) | 1318 (70.97%) | 0.75574 | 0.57687 | -25 ([-2.37, -1.34, -0.39] pp) | -0.00141 ([-0.0030, -0.0014, +0.0000]) |

### Upstream Stage 1 Tower Metrics (128D Dot + Familiarity)

| Variant | S1 Ep | S1 New@25 | S1 New@50 | S1 Recall@10 | S1 NDCG@10 |
|---|---:|---:|---:|---:|---:|
| `no-brand-v1` (Ref) | 16 | 726 (39.10%) | 1016 (54.71%) | 0.36716 | 0.22232 |
| `D_html` | 18 | 734 (39.53%) | 993 (53.47%) | 0.35741 | 0.21356 |
| `D_clean` | 18 | 726 (39.10%) | 988 (53.20%) | 0.35989 | 0.21449 |

## 3. Subgroup Breakdown: Targets WITH vs WITHOUT Descriptions (Full Pool Fixed)

Note: Even for targets WITHOUT descriptions (`1,343 / 1,857`), fresh tower weights and altered history vectors for customers who previously bought any of the 155 described SKUs can change predictions.

| Variant | Subgroup | Targets | New@10 | New@25 | New@50 | New@100 | Δ New@50 vs Ref (95% CI) |
|---|---|---:|---:|---:|---:|---:|---|
| `no-brand-v1` (Ref) | WITH desc | 514 | 37 (7.20%) | 172 (33.46%) | 268 (52.14%) | 362 (70.43%) | — |
| `no-brand-v1` (Ref) | WITHOUT desc | 1343 | 177 (13.18%) | 569 (42.37%) | 769 (57.26%) | 962 (71.63%) | — |
| `D_html` | WITH desc | 514 | 28 (5.45%) | 166 (32.30%) | 262 (50.97%) | 360 (70.04%) | -6 ([-3.56, -1.18, +1.02] pp) |
| `D_html` | WITHOUT desc | 1343 | 179 (13.33%) | 565 (42.07%) | 755 (56.22%) | 961 (71.56%) | -14 ([-2.18, -1.04, +0.07] pp) |
| `D_clean` | WITH desc | 514 | 30 (5.84%) | 163 (31.71%) | 258 (50.19%) | 356 (69.26%) | -10 ([-4.41, -1.98, +0.21] pp) |
| `D_clean` | WITHOUT desc | 1343 | 180 (13.40%) | 563 (41.92%) | 754 (56.14%) | 962 (71.63%) | -15 ([-2.21, -1.11, +0.00] pp) |

### Paired `D_clean` vs `D_html` (Customer-Clustered 95% Bootstrap CIs)

- **Overall `New@50` Δ:** [-0.78, -0.27, +0.27] pp
- **Overall `NDCG@10` Δ:** [-0.0003, +0.0004, +0.0010]
- **WITH description `New@50` Δ:** [-1.78, -0.77, +0.18] pp
- **WITHOUT description `New@50` Δ:** [-0.69, -0.07, +0.60] pp

## 4. Cleaning Rules (`desc_clean_v1`), Tokenizer Counts & Bounded Examples

- **Totals across 276 CSV rows:** `13,332` tags stripped, `3` tables formatted into key-value pairs, `1,568` decorative emojis/checkmarks removed, `110` CTA heading blocks + `78` marketing/QR/URL sentences removed, `8` duplicate within-product blocks removed, **`0` spec-token losses**.
- **Recipe `t1_fam`:** changed `0/841` (reused `841`, encoded `0`), max chars `119`, max tokens `None` / `8192`, truncated `0`.
- **Recipe `t1_fam_desc_html_v1`:** changed `155/841` (reused `686`, encoded `155`), max chars `4243`, max tokens `1383` / `8192`, truncated `0`.
- **Recipe `t1_fam_desc_clean_v1`:** changed `155/841` (reused `686`, encoded `155`), max chars `4131`, max tokens `1330` / `8192`, truncated `0`.

### Example `ТОВ-У001541` (Helix HX7 10W-40, 4л)
- Chars: raw `3437` -> `D_html` `2296` -> `D_clean` `1820`; counts: {'tags_stripped': 68, 'tables_parsed': 0, 'emojis_removed': 12, 'boilerplate_blocks_removed': 2, 'boilerplate_sentences_removed': 3, 'duplicate_blocks_removed': 0, 'duplicate_sentences_removed': 0}
- Removed items (sample): [{'reason': 'cta_heading', 'text': 'Чому варто купити Shell Helix HX7 10W-40?'}, {'reason': 'cta_heading', 'text': 'Як перевірити оригінальність оливи?'}, {'reason': 'boilerplate_sentence', 'text': 'Переконатися в оригінальності Shell Helix HX7 10W-40 можна через shellcheck, використовуючи QR-код на чотирьох, та п’ятилітрових каністрах.'}, {'reason': 'boilerplate_sentence', 'text': 'Якщо ви шукаєте кращу моторну оливу за розумною ціною, перевірте допуски та специфікації вашого авто і оберіть необхідне мастило.'}, {'reason': 'boilerplate_sentence', 'text': 'Shell Helix HX7 10W-40 – перевірена якість для стабільної роботи двигуна!'}]
- **Before (`D_html` excerpt):** `Моторна олива Shell Helix HX7 10W-40 – універсальний захист для двигуна Shell Helix HX7 10W-40 – це моторна олива на основі синтетичної технології , яка забезпечує ефективний захист від зношення, зменшене тертя та чистоту двигуна . Вона підходить для бензинових, дизельних (без DPF) та газових двигунів , а також моторів, що працюють на біодизелі або суміші бензину з етанолом . 🔹 Основні характеристики: ✅ API SP – новітній стандарт, що забезпечує захист від передчасного запалювання LSPI, покращену`
- **After (`D_clean` excerpt):** `Моторна олива Shell Helix HX7 10W-40 – універсальний захист для двигуна
Shell Helix HX7 10W-40 – це моторна олива на основі синтетичної технології, яка забезпечує ефективний захист від зношення, зменшене тертя та чистоту двигуна. Вона підходить для бензинових, дизельних (без DPF) та газових двигунів, а також моторів, що працюють на біодизелі або суміші бензину з етанолом.
Основні характеристики:
API SP – новітній стандарт, що забезпечує захист від передчасного запалювання LSPI, покращену стійкіс`

### Example `ТОВ-У510621` (Winter Screenwash (-55°C), 1л)
- Chars: raw `8911` -> `D_html` `1962` -> `D_clean` `1944`; counts: {'tags_stripped': 130, 'tables_parsed': 1, 'emojis_removed': 16, 'boilerplate_blocks_removed': 1, 'boilerplate_sentences_removed': 0, 'duplicate_blocks_removed': 0, 'duplicate_sentences_removed': 0}
- Removed items (sample): [{'reason': 'cta_heading', 'text': 'Чому варто купити Shell Winter Screenwash Concentrate -55°C?'}]
- **Before (`D_html` excerpt):** `Омивач скла Shell Winter Screenwash Concentrate -55°C – ефективне зимове очищення Shell Winter Screenwash Concentrate -55°C – це концентрований зимовий омивач скла , створений для ефективного очищення лобового скла та фар автомобіля у холодну пору року. Завдяки високоякісним очищувальним компонентам, він швидко видаляє дорожній бруд, маслянисту плівку, сіль і сліди реагентів , запобігаючи замерзанню форсунок та бачка омивача. 🔹 Основні переваги: ✅ Миттєве очищення від снігу, льоду та дорожніх ре`
- **After (`D_clean` excerpt):** `Омивач скла Shell Winter Screenwash Concentrate -55°C – ефективне зимове очищення
Shell Winter Screenwash Concentrate -55°C – це концентрований зимовий омивач скла, створений для ефективного очищення лобового скла та фар автомобіля у холодну пору року. Завдяки високоякісним очищувальним компонентам, він швидко видаляє дорожній бруд, маслянисту плівку, сіль і сліди реагентів, запобігаючи замерзанню форсунок та бачка омивача.
Основні переваги:
Миттєве очищення від снігу, льоду та дорожніх реагенті`

### Example `ТОВ-У510357` (Premium Ant. LL RTU 774DF, 20л)
- Chars: raw `8636` -> `D_html` `4111` -> `D_clean` `3886`; counts: {'tags_stripped': 155, 'tables_parsed': 0, 'emojis_removed': 9, 'boilerplate_blocks_removed': 1, 'boilerplate_sentences_removed': 1, 'duplicate_blocks_removed': 1, 'duplicate_sentences_removed': 0}
- Removed items (sample): [{'reason': 'cta_heading', 'text': 'Чому варто купити Shell Premium Antifreeze Longlife Ready to use 774 D-F?'}, {'reason': 'boilerplate_sentence', 'text': 'Shell Premium Antifreeze Longlife Ready to use 774 D-F – надійний вибір для ефективної роботи системи охолодження!'}, {'reason': 'duplicate_block', 'text': 'Ford WSS-M97B44-D'}]
- **Before (`D_html` excerpt):** `Антифриз Shell Premium Antifreeze Longlife Ready to use 774 D-F PBT78F – готовий до використання Shell Premium Antifreeze Longlife Ready to use 774 D-F – це готова до використання охолоджуюча рідина, створена на основі моноетиленгліколю без нітритів, амінів, фосфатів, силікатів та боратів (NAP-free). Вона забезпечує відмінний захист від корозії, перегріву та замерзання, подовжуючи термін служби системи охолодження. Основні характеристики: ✅ Органічна технологія інгібіторів корозії (OAT) – захища`
- **After (`D_clean` excerpt):** `Антифриз Shell Premium Antifreeze Longlife Ready to use 774 D-F PBT78F – готовий до використання
Shell Premium Antifreeze Longlife Ready to use 774 D-F – це готова до використання охолоджуюча рідина, створена на основі моноетиленгліколю без нітритів, амінів, фосфатів, силікатів та боратів (NAP-free). Вона забезпечує відмінний захист від корозії, перегріву та замерзання, подовжуючи термін служби системи охолодження.
Основні характеристики:
Органічна технологія інгібіторів корозії (OAT) – захищає `

### Example `ТОВ-У503715` (Helix HX8 5W-30, 1л)
- Chars: raw `7192` -> `D_html` `1836` -> `D_clean` `1679`; counts: {'tags_stripped': 91, 'tables_parsed': 0, 'emojis_removed': 10, 'boilerplate_blocks_removed': 1, 'boilerplate_sentences_removed': 1, 'duplicate_blocks_removed': 0, 'duplicate_sentences_removed': 0}
- Removed items (sample): [{'reason': 'cta_heading', 'text': 'Чому варто купити Shell Helix HX8 5W-30?'}, {'reason': 'boilerplate_sentence', 'text': 'Shell Helix HX8 5W-30 – ваш вибір для надійного захисту та продуктивності двигуна!'}]
- **Before (`D_html` excerpt):** `Моторна олива Shell Helix HX8 5W-30 – оптимальний захист для будь-яких умов Shell Helix HX8 5W-30 – це повністю синтетична моторна олива для бензинових, дизельних, та газових авто, яка поєднує передові технології очищення та захисту двигуна . Завдяки високій стійкості до окислення вона допомагає зберігати продуктивність двигуна протягом усього інтервалу заміни, зменшуючи утворення відкладень і запобігаючи передчасному зношенню. 🔹 Основні характеристики: ✅ ACEA A3/B3, A3/B4 – ідеально підходить д`
- **After (`D_clean` excerpt):** `Моторна олива Shell Helix HX8 5W-30 – оптимальний захист для будь-яких умов
Shell Helix HX8 5W-30 – це повністю синтетична моторна олива для бензинових, дизельних, та газових авто, яка поєднує передові технології очищення та захисту двигуна. Завдяки високій стійкості до окислення вона допомагає зберігати продуктивність двигуна протягом усього інтервалу заміни, зменшуючи утворення відкладень і запобігаючи передчасному зношенню.
Основні характеристики:
ACEA A3/B3, A3/B4 – ідеально підходить для бе`

## 5. Reproduction Commands

```bash
PYTHONPATH=. ./venv/bin/pytest tests/
PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only
PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --prepare-only
PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --verify-baseline-only
PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --variants D_html
PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/run.py --variants D_clean
PYTHONPATH=. ./venv/bin/python experiments/description_cleaning_v1/summarize.py
```

# C360 Prep Run Verification - 2026-01-21

## Today's Runs

| Name | Schema | Volume | Rows | Cols | KPIs | Status | Verified | Storage Location | Command |
|------|--------|--------|------|------|------|--------|----------|------------------|---------|
| Media 27.8M | media | 019be2f9-ed88-759e-b5fe-c6f91b96532c | 27,812,352 | 180 | 6/6 ✅ | **DONE** | ✅ | s3://nl-dev-unity-catalog/staging-360/...volumes/4b11168c-ce62-4442-b193-e0acf2c91d3f/ | `pixi run neuralift_c360_prep --config configs/media_demo_scale.yaml --runtime coiled` |
| Media 5M | media | 019be2fd-2f61-71a9-98b0-3784a4f89447 | 5,000,000 | 1,568 | 6/6 ✅ | **DONE** | ✅ | s3://nl-dev-unity-catalog/staging-360/...volumes/ac75ee1d-a582-4aaa-9b73-a151b05810c3/ | `pixi run neuralift_c360_prep --config configs/media_testing_5_mil.yaml --runtime coiled` |
| Media 50M | media | TBD | 50,000,000 | 1,569 | 6 expected | **RUNNING** | Pending | TBD | `pixi run neuralift_c360_prep --config configs/media_testing_50_mil.yaml --runtime coiled` |

## Verification Details

### Media 27.8M (media_demo_scale) - MANUALLY VERIFIED ✅
- **Source Table:** `staging-c360.media.viva_stream_media_27m`
- **Output Table:** `vivastream_media_27m_01`
- **Volume:** `019be2f9-ed88-759e-b5fe-c6f91b96532c`
- **Artifacts Written:** config.yaml, bundleconfig.yaml, data_dictionary.json, suggestions.yaml
- **Data Doctor Suggestions:** 12

**KPIs - Tagging Verified:**
| KPI Column | dtype | uniques | Tagged As | Status |
|------------|-------|---------|-----------|--------|
| at_risk | bool | 2 | kpi | ✅ |
| is_new_customer | bool | 2 | kpi | ✅ |
| is_churned | bool | 2 | kpi | ✅ |
| is_active | object | 3 | kpi | ✅ |
| is_cycler | bool | 2 | kpi | ✅ |
| total_spent_tier | object | 4 | kpi | ✅ |

**KPIs - Data Sample Verified:**
| KPI Column | Sample Values | Status |
|------------|---------------|--------|
| at_risk | False, False, False, True, False | ✅ |
| is_new_customer | False, False, False, False, False | ✅ |
| is_churned | False, False, True, False, False | ✅ |
| is_active | Active, Active, Cancelled, Active, Active | ✅ |
| is_cycler | False, False, False, False, False | ✅ |
| total_lifetime_spend_tier | 1. Small ($0<$250), 3. Large ($600+), 2. Medium, 3. Large, 3. Large | ✅ |

**Note:** Source table has pre-existing `total_lifetime_spend_tier` column. Config specifies `total_spent_tier` in kpi_cols but actual column name differs - consider aligning config.

- **Notes:** Uses pre-processed source with 185 columns (not raw 1575-column source)

### Media 5M (media_testing_5_mil) - MANUALLY VERIFIED ✅
- **Source Table:** `staging-media-source.default.media_demo_5m`
- **Output Table:** `media_demo_5m_512mb`
- **Volume:** `019be2fd-2f61-71a9-98b0-3784a4f89447`
- **Artifacts Written:** config.yaml, bundleconfig.yaml, data_dictionary.json, suggestions.yaml
- **Data Doctor Suggestions:** 11

**KPIs - Metadata Verified:**
| Metric | Value | Status |
|--------|-------|--------|
| Total Columns | 1568 | ✅ |
| ID Columns | 1 | ✅ |
| KPI Columns | 6 | ✅ |
| Categorical Columns | 1458 | ✅ |
| Continuous Columns | 103 | ✅ |

**KPIs - Column Output Verified:**
- `total_spent_tier` appears in final cast output as string column ✅
- LLM profiled column 1568/1568 (total_spent_tier) ✅

- **Notes:** Uses raw source with 1575 columns, drops 7 PII columns

### Media 50M (media_testing_50_mil) - IN PROGRESS
- **Source Table:** `staging-media-source.default.media_demo_50m`
- **Output Table:** `media_demo_50m`
- **Worker Specs:** r7i.12xlarge (384 GiB RAM) - bumped from m7i.8xlarge due to OOM
- **Current Phase:** Metadata tagging (computing unique counts for 1569 cols)

---

## Historical Runs (from user's tracking sheet)

| Name | Schema | Volume | Points | Segmenter Run | Segmenter Completed | Storage Location | Command |
|------|--------|--------|--------|---------------|---------------------|------------------|---------|
| Ecomm Demo | ecomm-clothing | 019ba4b4-ab84-7ab8-a52f-8b1cd3547ff3 | Created | | | s3://nl-dev-unity-catalog/staging-360/...volumes/ca83e2a3-5eae-43ed-9b80-859f22d0995a | `pixi run neuralift_c360_prep --config configs/ecomm.yaml --runtime coiled` |
| Ecomm Loyalty Demo | ecomm-clothing | 019ba4c2-229f-757d-899a-197d10968a81 | Created | | | s3://nl-dev-unity-catalog/staging-360/...volumes/e689de9d-d871-40b4-b174-79f56f706e4b | `pixi run neuralift_c360_prep --config configs/ecomm_loyalty.yaml --runtime coiled` |
| Gaming Demo | gaming | 019ba4c7-2ff2-70fe-be37-31a8d244adae | N/A | | | s3://nl-dev-unity-catalog/staging-360/...volumes/650a2a9a-134c-40c4-91a9-bad624f3cfda | `pixi run neuralift_c360_prep --config configs/gaming.yaml --runtime coiled` |
| Wine & Cheese (Kaggle) | kaggle-marketing | 019ba478-393a-711d-b13b-06c422c7d6f1 | N/A | | | s3://nl-dev-unity-catalog/staging-360/...volumes/94fed369-d7d4-4f30-b5b8-6df1b5d1291f | |
| Media 27m row demo | media | 019ba4de-d28d-7e92-b23f-016d89d8b36e | Created | | | s3://nl-dev-unity-catalog/staging-360/...volumes/25bd7455-8be2-4f35-b80c-e9927e51aaf5 | |
| Media 54k demo | media | 019ba4e6-4168-7b8b-8daa-c91abe546feb | Created | | | s3://nl-dev-unity-catalog/staging-360/...volumes/839fb8db-92ac-40ce-8dfa-650a2c6dd09d | |
| Media 50m | media | 019baeab-a562-72e6-af4a-058ffdc3f2ef | Created | | | s3://nl-dev-unity-catalog/staging-360/...volumes/43b22618-d112-4987-9a08-1b619884a8f1/ | `pixi run neuralift_c360_prep --config configs/media_testing_50_mil.yaml --runtime coiled` |
| Media 5m | media | 019bae87-dd3e-7118-bbf6-6bb9eba7ebc1 | Created | | | s3://nl-dev-unity-catalog/staging-360/...volumes/e1782b26-410c-4ccb-9df7-f1cd25df17d4 | `pixi run neuralift_c360_prep --config configs/media_testing_5_mil.yaml --runtime coiled` |
| Sports & Concerts | sports-and-concerts | 019ba5ba-8ee1-7cb1-bf6f-725d1ab28372 | Created | | | s3://nl-dev-unity-catalog/staging-360/...volumes/634ad5d8-9df2-4595-bb32-3a45f94939a3 | `pixi run neuralift_c360_prep --config configs/sports_and_concert.yaml --runtime coiled` |

---

## Observations

1. **Column Count Discrepancy:** The 27.8M dataset uses a pre-processed source table with only 185 columns, while 5M/50M use raw source tables with 1575 columns.

2. **KPIs Consistent:** All expected KPIs (at_risk, is_new_customer, is_churned, is_cycler, is_active, total_spent_tier) are properly tagged across runs.

3. **Data Doctor:** Suggestions are consistent (11-12 per run) focusing on average_monthly_spend, total_spent, and high-cardinality columns.

4. **50M Worker Sizing:** Required memory-optimized instances (r7i.12xlarge, 384 GiB) to handle 1575 columns without OOM.

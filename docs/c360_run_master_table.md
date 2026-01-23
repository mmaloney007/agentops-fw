# C360 Prep Master Run Table

*Last Updated: 2026-01-21 20:20*

## All Runs

| Name | Config | Rows | ID Field | KPIs (count) | KPI Fields (snake_case) | Type Counts (id/kpi/cat/cont) | Data Dict | KPIs Verified | Points | Runtime | UC Volume | S3 Location |
|------|--------|------|----------|--------------|-------------------------|-------------------------------|-----------|---------------|--------|---------|-----------|-------------|
| **Ecomm Demo** | `configs/ecomm.yaml` | ~500K | customer_id | 5 | returns_20_pct, is_loyal_customer, sales_20_pct, loyalty_program_tier, average_net_order_spend | TBD | ⏳ Pending | ⏳ Pending | ⏳ | TBD | `staging-c360.ecomm-clothing.019ba4b4-ab84-7ab8-a52f-8b1cd3547ff3` | s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/ca83e2a3-5eae-43ed-9b80-859f22d0995a/ |
| **Ecomm Loyalty** | `configs/ecomm_loyalty.yaml` | ~2.3M | customer_id | 6 | increased_spending_yoy, returns_20_pct, sales_20_pct, is_loyal_customer, loyalty_program_tier, average_net_order_spend | TBD | ⏳ Pending | ⏳ Pending | ⏳ | TBD | `staging-c360.ecomm-clothing.019ba4c2-229f-757d-899a-197d10968a81` | s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/e689de9d-d871-40b4-b174-79f56f706e4b/ |
| **Gaming Demo** | `configs/gaming.yaml` | TBD | player_id | 4 | kpi_6_m_net_worth, kpi_net_worth_daily, investment_target, net_worth_change | TBD | ⏳ Pending | ⏳ Pending | ⏳ | TBD | `staging-c360.gaming.019ba4c7-2ff2-70fe-be37-31a8d244adae` | s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/650a2a9a-134c-40c4-91a9-bad624f3cfda/ |
| **Sports & Concerts** | `configs/sports_and_concert.yaml` | TBD | customer_id | 4 | average_event_spend, parking_user, app_user, new_this_year | TBD | ⏳ Pending | ⏳ Pending | ⏳ | TBD | `staging-c360.sports-and-concerts.019ba5ba-8ee1-7cb1-bf6f-725d1ab28372` | s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/634ad5d8-9df2-4595-bb32-3a45f94939a3/ |
| **Wine & Cheese** | `configs/wine_and_cheese.yaml` | TBD | id | 2 | aov_tier, customer_complained_recently | TBD | ⏳ Pending | ⏳ Pending | ⏳ | TBD | `staging-c360.kaggle-marketing.019ba478-393a-711d-b13b-06c422c7d6f1` | s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/94fed369-d7d4-4f30-b5b8-6df1b5d1291f/ |
| **Media 54k** | `configs/media_small.yaml` | ~54K | customer_id | 6 | at_risk, is_new_customer, is_churned, is_cycler, is_active, total_spent_tier | TBD | ⏳ Pending | ⏳ Pending | ⏳ | TBD | `staging-c360.media.019ba4e6-4168-7b8b-8daa-c91abe546feb` | s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/839fb8db-92ac-40ce-8dfa-650a2c6dd09d/ |
| **Media 27.8M** | `configs/media_demo_scale.yaml` | 27,812,352 | customer_id | 6 | at_risk, is_new_customer, is_churned, is_cycler, is_active, total_spent_tier | 1/6/70/103 | ✅ Verified | ✅ 6/6 | ✅ Done | ~25 min | `staging-c360.media.019be2f9-ed88-759e-b5fe-c6f91b96532c` | s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/4b11168c-ce62-4442-b193-e0acf2c91d3f/ |
| **Media 5M** | `configs/media_testing_5_mil.yaml` | 5,000,000 | customer_id | 6 | at_risk, is_new_customer, is_churned, is_cycler, is_active, total_spent_tier | 1/6/1458/103 | ✅ Verified | ✅ 6/6 | ✅ Done | ~24 min | `staging-c360.media.019be2fd-2f61-71a9-98b0-3784a4f89447` | s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/ac75ee1d-a582-4aaa-9b73-a151b05810c3/ |
| **Media 50M** | `configs/media_testing_50_mil.yaml` | 50,000,000 | customer_id | 6 | at_risk, is_new_customer, is_churned, is_cycler, is_active, total_spent_tier | 1/6/1459/103 | 🔄 Running | 🔄 Running | ⏳ Pending | ~90 min | `staging-c360.media.TBD` | TBD (run in progress) |

---

## Detailed Run Information

### Today's Runs (2026-01-21) - Fully Verified

#### Media 27.8M (media_demo_scale)
| Field | Value |
|-------|-------|
| **Config** | `configs/media_demo_scale.yaml` |
| **Source Table** | `staging-c360.media.viva_stream_media_27m` |
| **Output Table** | `staging-c360.media.vivastream_media_27m_01` |
| **Volume** | `019be2f9-ed88-759e-b5fe-c6f91b96532c` |
| **S3 Location** | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/4b11168c-ce62-4442-b193-e0acf2c91d3f/` |
| **Rows** | 27,812,352 |
| **Columns** | 180 (after preprocessing) |
| **ID Field** | customer_id |
| **Type Counts** | 1 id, 6 kpi, 70 cat, 103 cont |
| **KPIs Verified** | ✅ at_risk, is_new_customer, is_churned, is_cycler, is_active, total_spent_tier |
| **Data Dictionary** | ✅ data_dictionary.json written |
| **Artifacts** | bundleconfig.yaml, data_dictionary.json, suggestions.yaml, input_data/ |
| **Data Doctor Suggestions** | 12 |
| **Points Status** | ✅ Complete (labels.npy, precomputed_points.npy) |
| **Runtime** | ~25 min |
| **Command** | `pixi run neuralift_c360_prep --config configs/media_demo_scale.yaml --runtime coiled` |

#### Media 5M (media_testing_5_mil)
| Field | Value |
|-------|-------|
| **Config** | `configs/media_testing_5_mil.yaml` |
| **Source Table** | `staging-media-source.default.media_demo_5m` |
| **Output Table** | `staging-c360.media.media_demo_5m_512mb` |
| **Volume** | `019be2fd-2f61-71a9-98b0-3784a4f89447` |
| **S3 Location** | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/ac75ee1d-a582-4aaa-9b73-a151b05810c3/` |
| **Rows** | 5,000,000 |
| **Columns** | 1,568 (after preprocessing) |
| **ID Field** | customer_id |
| **Type Counts** | 1 id, 6 kpi, 1458 cat, 103 cont |
| **KPIs Verified** | ✅ at_risk, is_new_customer, is_churned, is_cycler, is_active, total_spent_tier |
| **Data Dictionary** | ✅ data_dictionary.json written |
| **Artifacts** | bundleconfig.yaml, data_dictionary.json, suggestions.yaml, input_data/, segmented_data/ |
| **Data Doctor Suggestions** | 11 |
| **Points Status** | ✅ Complete (labels.npy, precomputed_points.npy) |
| **Runtime** | ~24 min |
| **Command** | `pixi run neuralift_c360_prep --config configs/media_testing_5_mil.yaml --runtime coiled` |

#### Media 50M (media_testing_50_mil) - IN PROGRESS
| Field | Value |
|-------|-------|
| **Config** | `configs/media_testing_50_mil.yaml` |
| **Source Table** | `staging-media-source.default.media_demo_50m` |
| **Output Table** | `staging-c360.media.media_demo_50m` |
| **Volume** | TBD |
| **S3 Location** | TBD |
| **Rows** | 50,000,000 |
| **Columns** | 1,569 (after preprocessing) |
| **ID Field** | Customer_ID |
| **Type Counts** | 1 id, 6 kpi, 1459 cat, 103 cont |
| **KPIs Expected** | at_risk, is_new_customer, is_churned, is_cycler, is_active, total_spent_tier |
| **Worker Specs** | r7i.12xlarge (384 GiB RAM) x 4 workers |
| **Status** | 🔄 Running (preprocessing complete, metadata in progress) |
| **Command** | `pixi run neuralift_c360_prep --config configs/media_testing_50_mil.yaml --runtime coiled` |

---

## Historical Runs (Need Verification)

These runs were completed previously. Row counts, type counts, and verification status need to be confirmed from run logs or by re-running.

| Name | Volume ID (from tracking) | Points Status | Segmenter Status |
|------|---------------------------|---------------|------------------|
| Ecomm Demo | 019ba4b4-ab84-7ab8-a52f-8b1cd3547ff3 | Created | - |
| Ecomm Loyalty | 019ba4c2-229f-757d-899a-197d10968a81 | Created | - |
| Gaming Demo | 019ba4c7-2ff2-70fe-be37-31a8d244adae | N/A | - |
| Wine & Cheese | 019ba478-393a-711d-b13b-06c422c7d6f1 | N/A | - |
| Media 27m (old) | 019ba4de-d28d-7e92-b23f-016d89d8b36e | Created | - |
| Media 54k | 019ba4e6-4168-7b8b-8daa-c91abe546feb | Created | - |
| Media 50m (old) | 019baeab-a562-72e6-af4a-058ffdc3f2ef | Created | - |
| Media 5m (old) | 019bae87-dd3e-7118-bbf6-6bb9eba7ebc1 | Created | - |
| Sports & Concerts | 019ba5ba-8ee1-7cb1-bf6f-725d1ab28372 | Created | - |

---

## Commands Quick Reference

```bash
# Ecomm
pixi run neuralift_c360_prep --config configs/ecomm.yaml --runtime coiled

# Ecomm Loyalty
pixi run neuralift_c360_prep --config configs/ecomm_loyalty.yaml --runtime coiled

# Gaming
pixi run neuralift_c360_prep --config configs/gaming.yaml --runtime coiled

# Sports & Concerts
pixi run neuralift_c360_prep --config configs/sports_and_concert.yaml --runtime coiled

# Wine & Cheese
pixi run neuralift_c360_prep --config configs/wine_and_cheese.yaml --runtime coiled

# Media Small (54k)
pixi run neuralift_c360_prep --config configs/media_small.yaml --runtime coiled

# Media 27.8M
pixi run neuralift_c360_prep --config configs/media_demo_scale.yaml --runtime coiled

# Media 5M
pixi run neuralift_c360_prep --config configs/media_testing_5_mil.yaml --runtime coiled

# Media 50M
pixi run neuralift_c360_prep --config configs/media_testing_50_mil.yaml --runtime coiled
```

---

## Notes

1. **Column Count Discrepancy**: Media 27.8M uses pre-processed source (185 cols) while 5M/50M use raw source (1575 cols)
2. **KPI Naming**: 27.8M has `total_lifetime_spend_tier` instead of `total_spent_tier` - source table difference
3. **Historical runs need verification**: Type counts and exact row counts not available without re-running or accessing logs

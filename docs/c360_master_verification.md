# C360 Prep Master Verification Table

*Last Updated: 2026-01-21 20:05 EST*

---

## Full Run Verification Table

| Name | Config | Rows | ID Field (snake_case) | KPI Count | KPI Fields (snake_case) | Types (id/kpi/cat/cont) | Data Dict ✓ | KPIs ✓ | Points ✓ | Runtime | UC Volume | S3 Location | Command |
|------|--------|------|----------------------|-----------|-------------------------|-------------------------|-------------|--------|----------|---------|-----------|-------------|---------|
| **Ecomm Demo** | `configs/ecomm.yaml` | ~500K | `customer_id` | 5 | `returns_20_pct`, `is_loyal_customer`, `sales_20_pct`, `loyalty_program_tier`, `average_net_order_spend` | TBV | TBV | TBV | Created | TBV | `staging-c360.ecomm-clothing.ecomm_demo_lp` (Vol: `019ba4b4-ab84-7ab8-a52f-8b1cd3547ff3`) | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/ca83e2a3-5eae-43ed-9b80-859f22d0995a` | `pixi run neuralift_c360_prep --config configs/ecomm.yaml --runtime coiled` |
| **Ecomm Loyalty** | `configs/ecomm_loyalty.yaml` | ~2.3M | `customer_id` | 6 | `increased_spending_yoy`, `returns_20_pct`, `sales_20_pct`, `is_loyal_customer`, `loyalty_program_tier`, `average_net_order_spend` | TBV | TBV | TBV | Created | TBV | `staging-c360.ecomm-clothing.loyalty_demo_lp` (Vol: `019ba4c2-229f-757d-899a-197d10968a81`) | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/e689de9d-d871-40b4-b174-79f56f706e4b` | `pixi run neuralift_c360_prep --config configs/ecomm_loyalty.yaml --runtime coiled` |
| **Gaming Demo** | `configs/gaming.yaml` | TBV | `player_id` | 4 | `kpi_6_m_net_worth`, `kpi_net_worth_daily`, `investment_target`, `net_worth_change` | TBV | TBV | TBV | N/A | TBV | `staging-c360.gaming.gaming_demo_lp` (Vol: `019ba4c7-2ff2-70fe-be37-31a8d244adae`) | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/650a2a9a-134c-40c4-91a9-bad624f3cfda` | `pixi run neuralift_c360_prep --config configs/gaming.yaml --runtime coiled` |
| **Sports & Concerts** | `configs/sports_and_concert.yaml` | TBV | `customer_id` | 4 | `average_event_spend`, `parking_user`, `app_user`, `new_this_year` | TBV | TBV | TBV | Created | TBV | `staging-c360.sports-and-concerts.sports_and_concert_demo_lp` (Vol: `019ba5ba-8ee1-7cb1-bf6f-725d1ab28372`) | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/634ad5d8-9df2-4595-bb32-3a45f94939a3` | `pixi run neuralift_c360_prep --config configs/sports_and_concert.yaml --runtime coiled` |
| **Wine & Cheese** | `configs/wine_and_cheese.yaml` | TBV | `id` | 2 | `aov_tier`, `customer_complained_recently` | TBV | TBV | TBV | N/A | TBV | `staging-c360.kaggle-marketing.wine_demo` (Vol: `019ba478-393a-711d-b13b-06c422c7d6f1`) | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/94fed369-d7d4-4f30-b5b8-6df1b5d1291f` | `pixi run neuralift_c360_prep --config configs/wine_and_cheese.yaml --runtime coiled` |
| **Media 54k** | `configs/media_small.yaml` | ~54K | `customer_id` | 6 | `at_risk`, `is_new_customer`, `is_churned`, `is_cycler`, `is_active`, `total_spent_tier` | TBV | TBV | TBV | Created | TBV | `staging-c360.media.media_dask_test_54k` (Vol: `019ba4e6-4168-7b8b-8daa-c91abe546feb`) | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/839fb8db-92ac-40ce-8dfa-650a2c6dd09d` | `pixi run neuralift_c360_prep --config configs/media_small.yaml --runtime coiled` |
| **Media 27.8M** ✅ | `configs/media_demo_scale.yaml` | 27,812,352 | `customer_id` | 6 | `at_risk`, `is_new_customer`, `is_churned`, `is_cycler`, `is_active`, `total_lifetime_spend_tier` | 1/6/~170/~5 | ✅ | ✅ 6/6 | Needed | ~25m | `staging-c360.media.vivastream_media_27m_01` (Vol: `019be2f9-ed88-759e-b5fe-c6f91b96532c`) | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/4b11168c-ce62-4442-b193-e0acf2c91d3f/` | `pixi run neuralift_c360_prep --config configs/media_demo_scale.yaml --runtime coiled` |
| **Media 5M** ✅ | `configs/media_testing_5_mil.yaml` | 5,000,000 | `customer_id` | 6 | `at_risk`, `is_new_customer`, `is_churned`, `is_cycler`, `is_active`, `total_spent_tier` | 1/6/1458/103 | ✅ | ✅ 6/6 | Needed | ~24m | `staging-c360.media.media_demo_5m_512mb` (Vol: `019be2fd-2f61-71a9-98b0-3784a4f89447`) | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/ac75ee1d-a582-4aaa-9b73-a151b05810c3/` | `pixi run neuralift_c360_prep --config configs/media_testing_5_mil.yaml --runtime coiled` |
| **Media 50M** 🔄 | `configs/media_testing_50_mil.yaml` | 50,000,000 | `customer_id` | 6 | `at_risk`, `is_new_customer`, `is_churned`, `is_cycler`, `is_active`, `total_spent_tier` | 1/6/1459/103 | 🔄 | 🔄 | Pending | ~90m | `staging-c360.media.media_demo_50m` | TBD (run in progress) | `pixi run neuralift_c360_prep --config configs/media_testing_50_mil.yaml --runtime coiled` |

**Legend:** TBV = To Be Verified, ✅ = Verified Today, 🔄 = In Progress

---

## UC Volume Locations (Full)

| Name | UC Catalog | UC Schema | UC Table | Volume ID |
|------|------------|-----------|----------|-----------|
| Ecomm Demo | `staging-c360` | `ecomm-clothing` | `ecomm_demo_lp` | `019ba4b4-ab84-7ab8-a52f-8b1cd3547ff3` |
| Ecomm Loyalty | `staging-c360` | `ecomm-clothing` | `loyalty_demo_lp` | `019ba4c2-229f-757d-899a-197d10968a81` |
| Gaming Demo | `staging-c360` | `gaming` | `gaming_demo_lp` | `019ba4c7-2ff2-70fe-be37-31a8d244adae` |
| Sports & Concerts | `staging-c360` | `sports-and-concerts` | `sports_and_concert_demo_lp` | `019ba5ba-8ee1-7cb1-bf6f-725d1ab28372` |
| Wine & Cheese | `staging-c360` | `kaggle-marketing` | `wine_demo` | `019ba478-393a-711d-b13b-06c422c7d6f1` |
| Media 54k | `staging-c360` | `media` | `media_dask_test_54k` | `019ba4e6-4168-7b8b-8daa-c91abe546feb` |
| **Media 27.8M** | `staging-c360` | `media` | `vivastream_media_27m_01` | `019be2f9-ed88-759e-b5fe-c6f91b96532c` |
| **Media 5M** | `staging-c360` | `media` | `media_demo_5m_512mb` | `019be2fd-2f61-71a9-98b0-3784a4f89447` |
| **Media 50M** | `staging-c360` | `media` | `media_demo_50m` | TBD |

---

## KPI Fields by Run (snake_case)

### Ecomm Demo
| KPI Field | Description |
|-----------|-------------|
| `returns_20_pct` | Returns in top 20% |
| `is_loyal_customer` | Loyalty flag |
| `sales_20_pct` | Sales in top 20% |
| `loyalty_program_tier` | Tier level |
| `average_net_order_spend` | Avg order value |

### Ecomm Loyalty
| KPI Field | Description |
|-----------|-------------|
| `increased_spending_yoy` | Year-over-year spending increase |
| `returns_20_pct` | Returns in top 20% |
| `sales_20_pct` | Sales in top 20% |
| `is_loyal_customer` | Loyalty flag |
| `loyalty_program_tier` | Tier level |
| `average_net_order_spend` | Avg order value |

### Gaming Demo
| KPI Field | Description |
|-----------|-------------|
| `kpi_6_m_net_worth` | 6-month net worth |
| `kpi_net_worth_daily` | Daily net worth |
| `investment_target` | Investment target flag |
| `net_worth_change` | Net worth delta |

### Sports & Concerts
| KPI Field | Description |
|-----------|-------------|
| `average_event_spend` | Avg spend per event |
| `parking_user` | Uses parking |
| `app_user` | Uses mobile app |
| `new_this_year` | New customer flag |

### Wine & Cheese
| KPI Field | Description |
|-----------|-------------|
| `aov_tier` | Average order value tier (ZSML) |
| `customer_complained_recently` | Recent complaint flag |

### Media (54k, 27.8M, 5M, 50M)
| KPI Field | Description |
|-----------|-------------|
| `at_risk` | At-risk subscriber |
| `is_new_customer` | New subscriber flag |
| `is_churned` | Churned flag |
| `is_cycler` | Subscription cycler |
| `is_active` | Active status |
| `total_spent_tier` / `total_lifetime_spend_tier` | Spend tier (ZSML) |

---

## S3 Storage Locations (Full)

| Name | S3 Base Path |
|------|-------------|
| Ecomm Demo | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/ca83e2a3-5eae-43ed-9b80-859f22d0995a` |
| Ecomm Loyalty | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/e689de9d-d871-40b4-b174-79f56f706e4b` |
| Gaming Demo | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/650a2a9a-134c-40c4-91a9-bad624f3cfda` |
| Sports & Concerts | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/634ad5d8-9df2-4595-bb32-3a45f94939a3` |
| Wine & Cheese | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/94fed369-d7d4-4f30-b5b8-6df1b5d1291f` |
| Media 54k | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/839fb8db-92ac-40ce-8dfa-650a2c6dd09d` |
| **Media 27.8M** | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/4b11168c-ce62-4442-b193-e0acf2c91d3f/` |
| **Media 5M** | `s3://nl-dev-unity-catalog/staging-360/__unitystorage/catalogs/152a3d9c-d65d-4555-a801-dbf118b0b12c/volumes/ac75ee1d-a582-4aaa-9b73-a151b05810c3/` |
| **Media 50M** | TBD |

---

## Points & Labels Status

| Name | Points Created | Command |
|------|----------------|---------|
| Ecomm Demo | ✅ Created | `pixi run points_and_labels --volume /Volumes/staging-c360/ecomm-clothing/019ba4b4-ab84-7ab8-a52f-8b1cd3547ff3 --runtime coiled` |
| Ecomm Loyalty | ✅ Created | `pixi run points_and_labels --volume /Volumes/staging-c360/ecomm-clothing/019ba4c2-229f-757d-899a-197d10968a81 --runtime coiled` |
| Gaming Demo | N/A | - |
| Sports & Concerts | ✅ Created | `pixi run points_and_labels --volume /Volumes/staging-c360/sports-and-concerts/019ba5ba-8ee1-7cb1-bf6f-725d1ab28372 --runtime coiled` |
| Wine & Cheese | N/A | - |
| Media 54k | ✅ Created | `pixi run points_and_labels --volume /Volumes/staging-c360/media/019ba4e6-4168-7b8b-8daa-c91abe546feb --runtime coiled` |
| **Media 27.8M** | ⏳ Needed | `pixi run points_and_labels --volume /Volumes/staging-c360/media/019be2f9-ed88-759e-b5fe-c6f91b96532c --runtime coiled` |
| **Media 5M** | ⏳ Needed | `pixi run points_and_labels --volume /Volumes/staging-c360/media/019be2fd-2f61-71a9-98b0-3784a4f89447 --runtime coiled` |
| **Media 50M** | ⏳ Pending | TBD |

---

## Commands Quick Reference

### C360 Prep Runs
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

### Points & Labels (for verified runs)
```bash
# Media 27.8M
pixi run points_and_labels --volume /Volumes/staging-c360/media/019be2f9-ed88-759e-b5fe-c6f91b96532c --runtime coiled

# Media 5M
pixi run points_and_labels --volume /Volumes/staging-c360/media/019be2fd-2f61-71a9-98b0-3784a4f89447 --runtime coiled
```

---

## Verification Summary

| Category | Verified | To Be Verified |
|----------|----------|----------------|
| C360 Prep Runs | 3 (27.8M, 5M, 50M in progress) | 6 (Ecomm, Loyalty, Gaming, Sports, Wine, 54k) |
| Data Dictionaries | 2 (27.8M, 5M) | 7 |
| KPIs in Data | 2 (27.8M, 5M) | 7 |
| Points & Labels | 0 new | 2 needed (27.8M, 5M) + 50M pending |

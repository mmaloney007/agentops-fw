# Neuralift C360 Prep

Private helper library, notebooks, and Databricks Asset Bundle for Neuralift's data-engineering team.
Proprietary - do not distribute.

## Dask CLI (primary)

### Quickstart

The --runtime flag controls where Dask tasks are run in the Cloud on Coiled or local on your machine.
The configs are capable of reading and writing to both local and remote sources/destinations.  That is controlled 100% by the yaml files.

Local run (Pixi) meaning local dask and file writing:

```bash
pixi run neuralift_c360_prep --config configs/data_prep.yaml --runtime local
```

Coiled run (requires Coiled auth + a software env; see setup below):

```bash
pixi run neuralift_c360_prep --config configs/data_prep.yaml --runtime coiled
```

### Coiled software env setup (regular + Docker)

Recommended names (lowercase required by Coiled):
- `neuralift_c360_prep` (regular packages)
- `neuralift_c360_prep_cpu` (Docker, CPU-only)

Regular env (pip + local code):

```bash
mkdir -p coiled
pixi run --environment default python scripts/lock_pip_requirements.py --output coiled/requirements.txt
pixi run coiled env create --workspace neuralift-dev --name neuralift_c360_prep --pip coiled/requirements.txt --include-local-code
```

Note: the lock resolves for your current platform. Run the lock step on linux-64 (or in a Linux container) for a Coiled-ready lock.

Docker env (CPU):

```bash
docker build -t ghcr.io/neuralift-ai/neuralift-c360-prep:cpu .
docker push ghcr.io/neuralift-ai/neuralift-c360-prep:cpu
pixi run coiled env create --workspace neuralift-dev --name neuralift_c360_prep_cpu --container ghcr.io/neuralift-ai/neuralift-c360-prep:cpu --ignore-container-entrypoint
```

### Points + labels helper (segmenter)

Generate `labels.npy` and `precomputed_points.npy` from parquet files that
contain a `segment` column. By default this reads from `<volume>/input_data`
and writes to `<volume>/segmented_data`, then updates `<volume>/config.yaml` by
adding `labels_file_name` and `ranked_points_file_name` at the top.

That said you will have to go into Databricks and get the actual `s3 storage location` or note it from the run.

```bash
# Coiled + S3 base
pixi run points_and_labels --volume s3://bucket/prefix --runtime coiled

# Override directories if your segment column lives elsewhere
pixi run points_and_labels --volume s3://bucket/prefix --input-subdir segmented_data --output-subdir segmented_data
```

### Environment credentials

Set these in a local `.env` file; they are loaded automatically and passed to Coiled workers.

Required for Coiled + Unity Catalog reads/writes:
- `DATABRICKS_HOST` (workspace host, without `https://`)
- `DATABRICKS_CLIENT_ID` (used for auth and to token sometimes for sql)
- `DATABRICKS_CLIENT_SECRET` (used for auth and to token sometimes for sql)
- `DATABRICKS_WAREHOUSE_ID` (SQL warehouse id)
- `AWS_ACCESS_KEY_ID` (only required if you are running locally, coiled should have these variables)
- `AWS_SECRET_ACCESS_KEY` (only required if you are running locally, coiled should have these variables)

Required for metadata generation:
- `OPENAI_API_KEY` (LLM data dictionary + table comment)

Optional:
- `WANDB_API_KEY` (only if `metadata.use_wandb: true`)
- `NL_SKIP_LLM=1` (skip LLM metadata; uses deterministic fallbacks)
- `NL_DEVICE_MEM_GB` or `NL_GPU_MEM_GB` (batch-size heuristic for config generation)
- `DASK_DATAFRAME__QUERY_PLANNING=0` (disable query planning for large Dask plans)

### Dask behavior highlights

- Supports parquet/csv/delta paths and UC tables (DBSQL fallback for logical column names).
- Pipeline order: preprocess → functions (KPIs + features) → drop columns → metadata → write.
- Metadata (creation of data dictionary) timeout defaults to 45m; parquet shards target ~512 MB unless overridden.
- If `output.s3_base` is omitted, a managed UC volume is created and used.

### Available configs and what they do

- `configs/data_prep.yaml`: local demo on `example-data/wine_cheese.parquet`
  that writes to `local-output`.
- `configs/ecomm.yaml`: Coiled UC table `staging-ecomm-clothing-source.default.ecomm_lp`
  with ecomm KPI tags; writes to `staging-c360.ecomm-clothing.ecomm_demo_lp`.
- `configs/ecomm_loyalty.yaml`: Coiled UC table
  `staging-ecomm-clothing-source.default.ecomm_loyalty_lp`; loyalty KPI tags; writes to
  `staging-c360.ecomm-clothing.loyalty_demo_lp`.
- `configs/gaming.yaml`: Coiled UC table `staging-gaming-source.default.gaming_demo`;
  gaming KPI tags; writes to `staging-c360.gaming.gaming_demo_lp`.
- `configs/marketing_kaggle_local.yaml`: local CSV at
  `/Users/maloney/neuralift_data/heuristics_data/marketing-kaggle/marketing-kaggle-dab.csv`
  with dtype overrides and feature functions; writes to a placeholder `s3_base`.
- `configs/media_demo_scale.yaml`: Coiled UC table `staging-c360.media.viva_stream_media_27m`;
  ZSML KPI for `total_spent`; drops PII-like columns; writes to
  `staging-c360.media.vivastream_media_27m_01` with 256 MB target partitions.
- `configs/media_small.yaml`: Coiled UC table `staging-media-source.default.media_54`;
  ZSML KPI; drops PII-like columns; writes to `staging-c360.media` (volume name
  `media_dask_test_54k`).
- `configs/media_testing_mils.yaml`: Coiled UC table `staging-media-source.default.media_demo_5m`;
  ZSML KPI; `use_approx_unique: false`; sets `DASK_DATAFRAME__QUERY_PLANNING=0`; writes to
  `staging-c360.media.media_demo_5m_512mb`.
- `configs/sports_and_concert.yaml`: Coiled UC table
  `staging-sports-and-concerts-source.default.sports_lp`; sports KPI tags; writes to
  `staging-c360.sports-and-concerts.sports_and_concert_demo_lp`.
- `configs/wine_and_cheese.yaml`: Coiled UC table
  `staging-c360.kaggle-marketing.wine_cheese_20250624_01`; adds age feature and drops
  `CustomerJoinDate`; writes to `staging-c360.kaggle-marketing.wine_demo`.

## Non-prod Databricks UC structure (kaggle_marketing)

### Top-level catalog
- Catalog: `staging-c360`

### Schemas and their roles

1) `media`
- Type: domain schema
- Purpose: curated, analysis-ready media datasets
- Objects: tables (4), volumes (12)
- Interpretation: authoritative production schema for media analytics

2) `sports-and-concerts`
- Type: domain schema
- Purpose: production data for sports and concerts use cases

3) `staging-ecomm-clothing-source`
- Type: staging schema
- Purpose: raw or lightly processed e-commerce clothing ingestion

4) `staging-gaming-source`
- Type: staging schema
- Purpose: raw gaming ingestion

5) `staging-kaggle-marketing-source`
- Type: staging schema
- Purpose: raw Kaggle marketing ingestion

6) `staging-media-source`
- Type: staging schema
- Purpose: raw + demo-scale media datasets for testing and scale experiments

Sub-schema:
- `default`

Tables in `staging-media-source.default`:
- `raw_media_demo_5m` (raw, ~5M rows)
- `media_demo_5m` (~5M rows)
- `media_demo_10m` (~10M rows)
- `media_demo_20m` (~20M rows)
- `media_demo_30m` (~30M rows)
- `media_demo_50m` (~50M rows)
- `media_demo_for_scale` (scale testing)
- `media_54` (experiment/snapshot)
- `media_lp` (variant)

Volumes:
- 1 volume for file-based assets (parquet, numpy, embeddings, checkpoints)

7) `staging-sports-and-concerts-source`
- Type: staging schema
- Purpose: raw sports and concerts ingestion

8) `information_schema`
- Type: system schema
- Purpose: metadata about tables, columns, permissions

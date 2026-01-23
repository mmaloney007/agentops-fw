# Config Generator from UC Table Tags

**Type:** Feature
**Priority:** Medium
**Labels:** `tooling`, `developer-experience`, `unity-catalog`

---

## Summary

Add a CLI tool to automatically generate pipeline YAML configs from existing tagged Databricks Unity Catalog tables. This enables rapid onboarding of new datasets by reading column metadata (ID, KPI, lift tags) and producing a ready-to-run config.

---

## Problem

When onboarding a new dataset or replicating an existing prepared table's config:
- Manually identifying ID columns, KPIs, and lift metadata is tedious
- Existing UC tables already have this information in column tags
- No automated way to bootstrap a config from production data

---

## Solution

Create `config_generator.py` module that:
1. Connects to Databricks UC via SDK
2. Reads table schema and column tags
3. Extracts columns by tag type: `id`, `kpi`, `cat`, `continuous`
4. Extracts lift metadata from KPI column tags
5. Generates a valid YAML config file

---

## Expected UC Column Tags

| Tag Key | Values | Description |
|---------|--------|-------------|
| `type` | `id`, `kpi`, `cat`, `continuous` | Column classification |
| `value_sum_column` | column name | Lift: value aggregation column |
| `value_sum_unit` | `USD`, `events`, etc. | Lift: value unit |
| `event_sum_column` | column name | Lift: event aggregation column |
| `event_sum_unit` | `events`, `days`, etc. | Lift: event unit |

---

## CLI Interface

```bash
# Print config to stdout
python -m neuralift_c360_prep.config_generator staging-c360.media.vivastream_27m

# Write to file
python -m neuralift_c360_prep.config_generator staging-c360.media.vivastream_27m \
  -o configs/staging/vivastream_generated.yaml

# Specify workspace/environment
python -m neuralift_c360_prep.config_generator staging-c360.media.vivastream_27m \
  --workspace neuralift-prod \
  --software-env neuralift_c360_prep
```

---

## Generated Config Structure

```yaml
# Generated from: staging-c360.media.vivastream_27m

logging:
  level: info
  show_progress: true

runtime:
  engine: coiled
  coiled:
    workspace: neuralift-dev
    software_env: neuralift_c360_prep
    n_workers: 4

input:
  source: uc_table
  uc_table: staging-c360.media.vivastream_27m

ids:
  columns: [customer_id]
  auto_detect: false

functions:
  - type: identity
    column: total_spent_tier
    kpi: true
    lift:
      value_sum_column: total_spent
      value_sum_unit: USD
      event_sum_column: months_subscribed
      event_sum_unit: months

  - type: identity
    column: is_churned
    kpi: true

preprocessing:
  rename_to_snake: true
  fill:
    categorical: "Unknown"
    continuous: "median"

output:
  uc_catalog: staging-c360
  uc_schema: media
  uc_table: vivastream_27m_prepared
```

---

## Acceptance Criteria

- [ ] `config_generator.py` module created with `generate_config_from_uc_table()` function
- [ ] CLI entry point via `python -m neuralift_c360_prep.config_generator`
- [ ] Reads UC column tags via Databricks SDK
- [ ] Extracts ID columns (type: id)
- [ ] Extracts KPI columns (type: kpi) with lift metadata
- [ ] Generates valid YAML that passes `load_config()` validation
- [ ] Supports `--output`, `--workspace`, `--software-env` flags
- [ ] Unit tests with mocked Databricks client
- [ ] Documentation in README

---

## Testing

```bash
# Integration test (requires UC access)
python -m neuralift_c360_prep.config_generator staging-c360.media.vivastream_27m -v

# Validate generated config loads
python -c "from neuralift_c360_prep.config import load_config; load_config('configs/generated.yaml')"
```

---

## Future Enhancements

- [ ] Infer function types from column naming patterns (e.g., `*_tier` → zsml, `*_bin` → binning)
- [ ] Support reading from parquet metadata
- [ ] Interactive mode to confirm/edit detected columns
- [ ] Diff mode: compare generated config vs existing config

# Data Doctor Enhancement Spec: Top-25 Ranked Suggestions

## Problem Statement
Currently, Data Doctor outputs all suggestions (11-12+ per run) without ranking. For large datasets with many columns, this can produce too many suggestions without clear prioritization.

## Proposed Enhancement

### 1. `max_suggestions` Config Option
Add a new config option to limit output:

```yaml
data_doctor:
  enabled: true
  save_yaml: true
  max_suggestions: 25          # NEW: Limit to top N suggestions (default: null = all)
  ranking_enabled: true        # NEW: Enable impact scoring (default: true)
```

### 2. Impact Ranking System

Each suggestion gets scored on a 0-100 scale based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Business Impact** | 40% | Is this a KPI candidate? Revenue/conversion related? |
| **Data Quality** | 25% | High nulls, high cardinality, type mismatches |
| **ML Readiness** | 20% | Would this improve model performance? |
| **Implementation Effort** | 15% | Inverse - easier = higher score |

### 3. Suggestion Categories with Default Priorities

| Priority | Category | Examples |
|----------|----------|----------|
| 🚨 HIGH (80-100) | KPI candidates, critical data quality issues | ZSML on revenue columns, high-null ID columns |
| 🧪 MEDIUM (50-79) | Feature engineering, moderate issues | Log transforms, bucketing high-cardinality |
| 💡 LOW (0-49) | Nice-to-have enhancements | Date part extraction, winsorization |

### 4. Ranking Logic (Pseudocode)

```python
def score_suggestion(suggestion: Suggestion, column_stats: dict) -> int:
    score = 0

    # Business Impact (40%)
    if suggestion.type == "kpi_candidate":
        score += 35
        if column_stats.get("is_monetary"):
            score += 5
    elif suggestion.type == "feature_engineering":
        score += 20

    # Data Quality (25%)
    null_pct = column_stats.get("null_pct", 0)
    if null_pct > 50:
        score += 25
    elif null_pct > 20:
        score += 15

    cardinality = column_stats.get("unique_count", 0)
    if cardinality > 1000:
        score += 10  # High cardinality needs attention

    # ML Readiness (20%)
    if suggestion.improves_distribution:
        score += 15
    if suggestion.reduces_dimensionality:
        score += 5

    # Implementation Effort (15%) - inverse
    effort_map = {"low": 15, "medium": 10, "high": 5}
    score += effort_map.get(suggestion.effort, 10)

    return min(score, 100)
```

### 5. Output Format Enhancement

```yaml
# suggestions.yaml (enhanced)
summary:
  total_analyzed: 180
  total_suggestions: 47
  showing_top: 25
  score_range: [45, 95]

suggestions:
  - rank: 1
    score: 95
    priority: HIGH
    column: average_monthly_spend
    type: kpi_candidate
    suggestion: "ZSML tiering for customer segmentation"
    reasoning: "Monetary column with business significance. ZSML creates actionable Low/Medium/High segments for targeting."

  - rank: 2
    score: 88
    priority: HIGH
    column: total_spent
    type: feature_engineering
    suggestion: "Log transform for normalization"
    reasoning: "Right-skewed monetary distribution (skew=4.2). Log transform improves ML model performance."

  # ... top 25
```

### 6. Implementation Tasks

1. [ ] Add `max_suggestions` and `ranking_enabled` to `DataDoctorConfig`
2. [ ] Create `SuggestionScorer` class with configurable weights
3. [ ] Add `score` and `reasoning` fields to suggestion model
4. [ ] Sort suggestions by score before output
5. [ ] Add summary stats to output (total vs shown)
6. [ ] Update YAML/JSON serialization
7. [ ] Add tests for scoring logic
8. [ ] Update documentation

### 7. Backward Compatibility

- Default `max_suggestions: null` preserves current behavior (show all)
- Existing configs work unchanged
- New fields are additive only

---

## Files to Modify

- `src/neuralift_c360_prep/data_doctor.py` - Main logic
- `src/neuralift_c360_prep/config.py` - Add new config options
- `tests/test_data_doctor.py` - Add scoring tests

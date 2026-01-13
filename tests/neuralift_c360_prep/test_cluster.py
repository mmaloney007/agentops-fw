from dask import config as dask_config

from neuralift_c360_prep.cluster import _apply_dask_perf_defaults


def test_apply_dask_perf_defaults_keeps_query_planning_enabled():
    with dask_config.set({"dataframe.query-planning": True}):
        _apply_dask_perf_defaults()
        assert dask_config.get("dataframe.query-planning") is True

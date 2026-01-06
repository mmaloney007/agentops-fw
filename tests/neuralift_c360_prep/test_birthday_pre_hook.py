import pandas as pd
import dask.dataframe as dd

from neuralift_c360_prep.features.birthday import add_age_from_year


def test_add_age_from_year():
    pdf = pd.DataFrame({"year_birth": [1980, 2000, None]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    out = ddf.map_partitions(add_age_from_year).compute()
    assert {"age", "age_range", "generation"} <= set(out.columns)
    # 1980 -> age_range 35-44 if current year >= 2024; just ensure not Unknown
    assert out.loc[0, "age_range"] != "Unknown"
    assert out.loc[1, "generation"] in {"Gen Z", "Millennial"}
    assert out.loc[2, "age_range"] == "Unknown"

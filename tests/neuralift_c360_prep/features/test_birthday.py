import pandas as pd

from neuralift_c360_prep.features.birthday import add_age_from_year, add_birth_month


def test_add_birth_month():
    df = pd.DataFrame({"birth_date": ["1980-01-15", None]})
    out = add_birth_month(df, date_col="birth_date")

    assert out.loc[0, "birth_month"] == 1
    assert out.loc[0, "birth_day"] == 15
    assert pd.isna(out.loc[1, "birth_month"])
    assert pd.isna(out.loc[1, "birth_day"])


def test_add_age_from_year():
    df = pd.DataFrame({"year_birth": [1980, 2000, 9999, None]})
    out = add_age_from_year(df, current_year=2024)

    assert out.loc[0, "age"] == 44
    assert out.loc[0, "age_range"] == "35-44"
    assert out.loc[0, "generation"] == "Gen X"

    assert out.loc[1, "age"] == 24
    assert out.loc[1, "age_range"] == "18-24"
    assert out.loc[1, "generation"] == "Gen Z"

    assert out.loc[2, "age_range"] == "Unknown"
    assert out.loc[3, "age_range"] == "Unknown"

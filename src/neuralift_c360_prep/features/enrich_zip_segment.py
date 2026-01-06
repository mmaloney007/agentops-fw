#!/usr/bin/env python3
"""
Dask-friendly ZIP enrichment.

Adds state/city/county/country/region/urban density/timezone for US ZIPs.
Intended for use as a pre-hook via map_partitions.
"""
from __future__ import annotations

import pandas as pd

from .map_state_and_region import REGION_MAP as REGION_MAPPING

try:
    from timezonefinder import TimezoneFinder
    from uszipcode import SearchEngine
except Exception:  # pragma: no cover - optional install
    TimezoneFinder = None
    SearchEngine = None


def classify_urban_density(population: float | int | None, land_area_sqmi: float | int | None) -> str:
    if not population or not land_area_sqmi or land_area_sqmi <= 0:
        return "Unknown"
    density = float(population) / float(land_area_sqmi)
    if density >= 3000:
        return "Urban"
    if density >= 1000:
        return "Suburban"
    return "Rural"


def enrich_zip_segment_dask(
    df: pd.DataFrame,
    *,
    id_col: str = "id",
    zip_col: str = "zip",
    default_country: str = "US",
    fill_value: str = "Unknown",
) -> pd.DataFrame:
    if SearchEngine is None or TimezoneFinder is None:
        return df  # skip silently if deps are absent
    if zip_col not in df.columns:
        return df

    out = df.copy()
    out[zip_col] = out[zip_col].astype(str).str.strip()
    unique_zips = out[zip_col].unique()

    try:
        search = SearchEngine(simple_zipcode=True)
    except TypeError:
        search = SearchEngine()
    tz_finder = TimezoneFinder()

    zip_info: dict[str, dict] = {}

    for z in unique_zips:
        z_str = str(z).strip()
        if not z_str.isdigit():
            zip_info[z_str] = {
                "State": fill_value,
                "City": fill_value,
                "County": fill_value,
                "Country": fill_value,
                "Region": fill_value,
                "UrbanDensity": fill_value,
                "Timezone": fill_value,
            }
            continue
        res = search.by_zipcode(z_str)
        if res is None or not res.zipcode:
            zip_info[z_str] = {
                "State": fill_value,
                "City": fill_value,
                "County": fill_value,
                "Country": fill_value,
                "Region": fill_value,
                "UrbanDensity": fill_value,
                "Timezone": fill_value,
            }
            continue
        st = res.state or fill_value
        city = res.major_city or fill_value
        county = res.county or fill_value
        region = REGION_MAPPING.get(st, fill_value)
        country = default_country if st != fill_value else fill_value
        density = classify_urban_density(res.population, res.land_area_in_sqmi)
        try:
            tz = tz_finder.timezone_at(lat=res.lat, lng=res.lng) or fill_value
        except Exception:
            tz = fill_value

        zip_info[z_str] = {
            "State": st,
            "City": city,
            "County": county,
            "Country": country,
            "Region": region,
            "UrbanDensity": density,
            "Timezone": tz,
        }

    zip_df = (
        pd.DataFrame.from_dict(zip_info, orient="index")
        .rename_axis("ZIP")
        .reset_index()
    )
    merged = out.merge(zip_df, left_on=zip_col, right_on="ZIP", how="left").drop(
        columns=["ZIP"]
    )
    return merged


__all__ = ["REGION_MAPPING", "classify_urban_density", "enrich_zip_segment_dask"]

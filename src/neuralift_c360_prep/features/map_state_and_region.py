#!/usr/bin/env python3
"""
Dask-friendly state/region mapper.

Adds USPS abbreviation, full state name, and Census region; intended for use
as a pre-hook via map_partitions.
"""
from __future__ import annotations

import pandas as pd


STATE_MAP = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "AS": "American Samoa",
    "GU": "Guam",
    "MP": "Northern Mariana Islands",
    "VI": "U.S. Virgin Islands",
}

REGION_MAP = {
    "CT": "Northeast",
    "ME": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "NE": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "AL": "South",
    "AR": "South",
    "DC": "South",
    "DE": "South",
    "FL": "South",
    "GA": "South",
    "MD": "South",
    "NC": "South",
    "SC": "South",
    "TN": "South",
    "KY": "South",
    "MS": "South",
    "WV": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    "AK": "West",
    "AZ": "West",
    "CA": "West",
    "CO": "West",
    "HI": "West",
    "ID": "West",
    "MT": "West",
    "NV": "West",
    "NM": "West",
    "UT": "West",
    "WY": "West",
    "OR": "West",
    "WA": "West",
}


def map_state_and_region(df: pd.DataFrame, state_col: str = "state") -> pd.DataFrame:
    if state_col not in df.columns:
        return df
    out = df.copy()
    upper = out[state_col].astype(str).str.upper().str.strip()
    name_to_abbrev = {abbr: abbr for abbr in STATE_MAP}
    name_to_abbrev.update({name.upper(): abbr for abbr, name in STATE_MAP.items()})
    out["usps_abbrev"] = upper.map(name_to_abbrev)
    out["full_state_name"] = out["usps_abbrev"].map(STATE_MAP)
    out["census_region"] = out["usps_abbrev"].map(REGION_MAP)
    return out


__all__ = ["map_state_and_region"]

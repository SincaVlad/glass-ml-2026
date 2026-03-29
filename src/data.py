from glasspy.data import SciGlass
import pandas as pd
import re

KELVIN_PROPERTIES = [
    "Tg",
    "Tmelt",
    "Tliquidus",
    "Tsoft",
    "TdilatometricSoftening",
    "TAnnealing",
    "Tstrain",
    "TLittletons",
]


def load_glass_data(
    target: str = "Tg",
    coverage_threshold: float = 90.0,
    oxide_min_presence: float = 0.05,
) -> pd.DataFrame:
    """
    Loads and filters SciGlass data for a given target property.

    Oxide features are selected dynamically based on the subset of glasses
    that have the target property measured — not the full database.

    Args:
        target: Property to predict. Must match a SciGlass column name.
                Common examples: 'Tg', 'Density293K', 'Tmelt', 'Tsoft'.
                To see all available properties, use:
                SciGlass(properties_cfg={"keep_all": True}).data["property"].columns
        coverage_threshold: Minimum % of composition explained by the
                            selected oxides. Filters out non-oxide systems.
        oxide_min_presence: Minimum fraction of rows (in the target subset)
                            where an oxide must be non-zero to be included.
                            Default 0.05 = present in at least 5% of rows.

    Returns:
        DataFrame with selected oxide features + 'other_compounds' + target.
        Rows with missing target values are dropped.
        Temperature properties are converted from Kelvin to Celsius.
    """
    source = SciGlass(
        properties_cfg={"keep_all": True},
        compounds_cfg={"keep_all": True},
    )
    df = source.data

    if target not in df["property"].columns:
        available = df["property"].columns.tolist()
        raise ValueError(f"'{target}' not found. Available: {available}")

    # Work only with rows that have the target measured
    target_mask = df["property"][target].notna()
    compounds_subset = df["compounds"][target_mask]

    # Select oxides present in >= oxide_min_presence of those rows
    non_zero_pct = (compounds_subset > 0).mean()
    selected_oxides = non_zero_pct[non_zero_pct >= oxide_min_presence].index.tolist()

    # Exclude non-oxide compounds (fluorides, elements, sulfides, etc.)
    _oxide_pattern = re.compile(r"^[A-Za-z0-9]+O\d*$")
    selected_oxides = [ox for ox in selected_oxides if _oxide_pattern.match(ox)]

    # Apply coverage threshold using the selected oxides
    coverage = compounds_subset[selected_oxides].sum(axis=1)
    coverage_mask = coverage >= coverage_threshold

    # Build result dataframe
    result = compounds_subset[selected_oxides][coverage_mask].copy()
    result["other_compounds"] = (100 - coverage[coverage_mask]).clip(lower=0)
    result[target] = df["property"][target][target_mask][coverage_mask]

    result = result.dropna(subset=[target])

    if target in KELVIN_PROPERTIES:
        result[target] = result[target] - 273.15

    return result

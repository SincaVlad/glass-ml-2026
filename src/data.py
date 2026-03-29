from glasspy.data import SciGlass
import pandas as pd

OXIDE_FEATURES = [
    "SiO2",
    "B2O3",
    "Al2O3",
    "Na2O",
    "CaO",
    "K2O",
    "MgO",
    "P2O5",
    "BaO",
    "Li2O",
    "ZnO",
    "TiO2",
    "PbO",
    "ZrO2",
    "SrO",
    "La2O3",
    "Fe2O3",
    "Nb2O5",
    "Sb2O3",
    "Bi2O3",
    "TeO2",
    "V2O5",
    "GeO2",
    "WO3",
    "SnO2",
    "Ta2O5",
    "Y2O3",
]

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


def get_oxide_features() -> list[str]:
    """Returns the list of oxide features used in the pipeline."""
    return OXIDE_FEATURES.copy()


def load_glass_data(
    target: str = "Tg",
    coverage_threshold: float = 90.0,
) -> pd.DataFrame:
    """
    Loads and filters SciGlass data for a given target property.

    Args:
        target: Property to predict. Must match a SciGlass property name.
        coverage_threshold: Minimum % of composition explained by known
                            oxides. Filters out non-oxide glass systems.

    Returns:
        DataFrame with oxide features + 'other_oxides' + target column.
        Rows with missing target values are dropped.
        Temperature properties (Tg, Tmelt etc.) are converted to Celsius.
    """
    source = SciGlass(
        properties_cfg={"keep_all": True},
        compounds_cfg={"keep_all": True},
    )
    df = source.data

    if target not in df["property"].columns:
        available = df["property"].columns.tolist()
        raise ValueError(f"'{target}' not found. Available: {available}")

    coverage = df["compounds"][OXIDE_FEATURES].sum(axis=1)
    oxide_mask = coverage >= coverage_threshold

    result = df["compounds"][OXIDE_FEATURES][oxide_mask].copy()
    result["other_oxides"] = (100 - coverage[oxide_mask]).clip(lower=0)
    result[target] = df["property"][target][oxide_mask]

    result = result.dropna(subset=[target])

    if target in KELVIN_PROPERTIES:
        result[target] = result[target] - 273.15

    return result

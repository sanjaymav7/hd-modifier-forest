"""Checks that each cohort is wired to the right samples. Needs the GEO files (run.py fetches them)."""
import numpy as np
from hdrf.data import load_gse24250, load_gse51799, load_gse64810


def test_gse51799_columns_follow_titles():
    # The supplementary matrix has no sample IDs; XIST (female-only) must track recorded sex.
    e, m = load_gse51799()
    assert np.corrcoef(e.loc["XIST"], m["female"])[0, 1] > 0.9


def test_gse24250_sex_matches_xist():
    e, m = load_gse24250()
    assert np.corrcoef(e.loc["XIST"], m["female"])[0, 1] > 0.6


def test_label_counts_match_geo():
    assert load_gse24250()[1]["label"].value_counts().to_dict() == {1: 8, 0: 6}
    assert load_gse51799()[1]["label"].value_counts().to_dict() == {1: 91, 0: 33}
    assert load_gse64810()[1]["label"].value_counts().to_dict() == {0: 49, 1: 20}


def test_brain_cag_only_in_hd():
    _, m = load_gse64810()
    assert m.loc[m.label == 1, "cag_repeats"].notna().all()
    assert m.loc[m.label == 0, "cag_repeats"].isna().all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

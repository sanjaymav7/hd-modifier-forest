"""Download and parse the three public Huntington's cohorts into one shape.

Each loader returns (expr, meta): expr is genes x samples on a log2 scale,
indexed by HGNC symbol; meta is one row per sample with at least
`label` (1 = HD / mutation carrier, 0 = control), `study`, `tissue`,
and whatever clinical covariates the study published.

The original notebook read expression from each GSM's own data table. That
works for microarrays, but GEO keeps RNA-seq expression in series-level
supplementary files, so three of the four studies came back empty and the
model trained on 14 samples. These loaders read the supplementary files.
"""
import gzip
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("HDRF_DATA", Path(__file__).resolve().parent.parent / "data"))
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
FILES = {
    "GSE24250_series_matrix.txt.gz": f"{GEO}/GSE24nnn/GSE24250/matrix/GSE24250_series_matrix.txt.gz",
    "GSE51799-GPL10999_series_matrix.txt.gz": f"{GEO}/GSE51nnn/GSE51799/matrix/GSE51799-GPL10999_series_matrix.txt.gz",
    "GSE51799-GPL11154_series_matrix.txt.gz": f"{GEO}/GSE51nnn/GSE51799/matrix/GSE51799-GPL11154_series_matrix.txt.gz",
    "GSE51799_Normalised_data.txt.gz": f"{GEO}/GSE51nnn/GSE51799/suppl/GSE51799_Normalised_data.txt.gz",
    "GSE64810_series_matrix.txt.gz": f"{GEO}/GSE64nnn/GSE64810/matrix/GSE64810_series_matrix.txt.gz",
    "GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz": f"{GEO}/GSE64nnn/GSE64810/suppl/GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz",
    "GPL96.txt": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL96&targ=self&form=text&view=data",
    "hgnc.txt": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
}


def fetch(name):
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name}")
        urllib.request.urlretrieve(FILES[name], path)
    return path


def series_meta(name):
    """Sample characteristics from a GEO series matrix, one column per 'key: value' field."""
    rows, ids, titles = {}, [], []
    with gzip.open(fetch(name), "rt", errors="replace") as f:
        for line in f:
            cells = [c.strip('"') for c in line.rstrip("\n").split("\t")[1:]]
            if line.startswith("!Sample_geo_accession"):
                ids = cells
            elif line.startswith("!Sample_title"):
                titles = cells
            elif line.startswith("!Sample_characteristics_ch1"):
                for i, v in enumerate(cells):
                    if ": " in v:
                        k, val = v.split(": ", 1)
                        rows.setdefault(i, {})[re.sub(r"\s*\[.*?\]", "", k).strip()] = val
    meta = pd.DataFrame([rows.get(i, {}) for i in range(len(ids))], index=ids)
    meta.insert(0, "title", titles)
    return meta


def _ensembl_to_symbol():
    hg = pd.read_csv(fetch("hgnc.txt"), sep="\t", usecols=["symbol", "ensembl_gene_id"], dtype=str)
    hg = hg.dropna().drop_duplicates("ensembl_gene_id")
    return hg.set_index("ensembl_gene_id")["symbol"]


def _collapse(expr, mapping):
    """Map row IDs to symbols and keep the most-expressed row per symbol."""
    expr = expr.copy()
    expr = expr[~expr.index.duplicated()]
    expr["symbol"] = expr.index.map(mapping)
    expr = expr.dropna(subset=["symbol"])
    expr["_mean"] = expr.drop(columns="symbol").mean(axis=1)
    expr = expr.sort_values("_mean", ascending=False).drop_duplicates("symbol")
    return expr.drop(columns="_mean").set_index("symbol")


def load_gse24250():
    """Whole blood, Affymetrix HG-U133A: 8 HD, 6 controls."""
    meta = series_meta("GSE24250_series_matrix.txt.gz")
    expr = pd.read_csv(fetch("GSE24250_series_matrix.txt.gz"), sep="\t", comment="!", index_col=0)
    expr.columns = [c.strip('"') for c in expr.columns]
    with open(fetch("GPL96.txt"), errors="replace") as f:
        start = next(i for i, line in enumerate(f) if line.startswith("ID\t"))
    gpl = pd.read_csv(fetch("GPL96.txt"), sep="\t", skiprows=start, usecols=["ID", "Gene Symbol"], dtype=str)
    gpl = gpl[~gpl["ID"].str.startswith("!")].dropna()
    gpl["Gene Symbol"] = gpl["Gene Symbol"].str.split(" /// ").str[0]
    expr = _collapse(expr, gpl.set_index("ID")["Gene Symbol"])
    if np.nanquantile(expr.values, 0.99) > 50:
        expr = np.log2(expr.clip(lower=1))
    meta["label"] = meta["title"].str.startswith("Huntington").astype(int)
    meta["age"] = pd.to_numeric(meta["age at blood draw"])
    meta["female"] = (meta["gender"] == "F").astype(int)
    meta["study"], meta["tissue"] = "GSE24250", "blood"
    return expr[meta.index], meta


def load_gse51799():
    """Whole blood, 3' digital gene expression: 91 mutation carriers, 33 controls.

    The supplementary matrix labels its columns 'Sample 1..124'. They follow the
    sample titles (Control 1..33, then Carrier 1..91), not GEO accession order:
    XIST against recorded sex correlates at r = 0.97 in title order and 0.01 in
    accession order. test_data.py keeps that check.
    """
    meta = pd.concat([series_meta("GSE51799-GPL10999_series_matrix.txt.gz"),
                      series_meta("GSE51799-GPL11154_series_matrix.txt.gz")])
    t = meta["title"].str.extract(r"(Control|Carrier) (\d+)")
    meta = meta.assign(_carrier=t[0].eq("Carrier"), _n=t[1].astype(int)).sort_values(["_carrier", "_n"])
    expr = pd.read_csv(fetch("GSE51799_Normalised_data.txt.gz"), sep="\t", index_col=0)
    expr.columns = meta.index
    expr = _collapse(expr, _ensembl_to_symbol())
    meta["label"] = meta["_carrier"].astype(int)
    for k in ("age", "cag_repeats", "motor_score", "hb_percentage", "tfc_score"):
        meta[k] = pd.to_numeric(meta[k], errors="coerce")
    meta["female"] = pd.to_numeric(meta["gender"])
    meta["study"], meta["tissue"] = "GSE51799", "blood"
    return expr, meta.drop(columns=["_carrier", "_n"])


def load_gse64810():
    """Post-mortem BA9 prefrontal cortex, mRNA-seq: 20 HD, 49 neurologically normal."""
    meta = series_meta("GSE64810_series_matrix.txt.gz")
    expr = pd.read_csv(fetch("GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz"), sep="\t", index_col=0)
    expr.index = expr.index.str.replace(r"\.\d+$", "", regex=True)
    expr = np.log2(_collapse(expr, _ensembl_to_symbol()) + 1)
    expr.columns = meta.index[pd.Index(meta["title"]).get_indexer(expr.columns)]
    meta["label"] = meta["diagnosis"].str.contains("Huntington").astype(int)
    for k, src in (("age", "age of death"), ("cag_repeats", "cag"), ("pmi", "pmi"), ("rin", "rin"),
                   ("vonsattel", "vonsattel grade")):
        meta[k] = pd.to_numeric(meta[src], errors="coerce") if src in meta else np.nan
    meta["study"], meta["tissue"] = "GSE64810", "brain"
    return expr[meta.index], meta

"""Download the NHANES August 2021-August 2023 files used in this project."""

import time
import urllib.request
from pathlib import Path

from functools import reduce
import pandas as pd

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"

# The 25 files used in this project, grouped by NHANES component
NHANES_FILES = {
    "Demographics": ["DEMO_L"],
    "Examination": ["BMX_L", "BPXO_L"],
    "Laboratory": ["GHB_L", "GLU_L", "HDL_L", "INS_L", "TCHOL_L", "TRIGLY_L",
                   "FASTQX_L"],
    "Questionnaire": ["ALQ_L", "BPQ_L", "DBQ_L", "DIQ_L", "DPQ_L", "FNQ_L",
                      "HIQ_L", "HSQ_L", "INQ_L", "MCQ_L", "PAQ_L", "RXQ_RX_L",
                      "SLQ_L", "SMQ_L", "WHQ_L"],
}


def download_file(name, data_dir):
    """Download one .xpt file from the CDC website, unless it already exists."""
    target = data_dir / f"{name}.xpt"
    if target.exists() and target.stat().st_size > 0:
        return "skipped (already downloaded)"

    url = f"{BASE_URL}{name}.xpt"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()

    # A real XPT file starts with this header. An HTML error page does not.
    if not content.startswith(b"HEADER RECORD"):
        raise ValueError(f"{name}: the server did not return an XPT file ({url})")

    target.write_bytes(content)
    return f"downloaded ({len(content) / 1024:.0f} KB)"


def download_all(data_dir="data"):
    """Download all 25 project files into data_dir and return its path."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for component, names in NHANES_FILES.items():
        for name in names:
            status = download_file(name, data_dir)
            print(f"{component:14s} {name:10s} {status}")
            if status.startswith("downloaded"):
                time.sleep(0.5)  # short pause between requests to the CDC server
    return data_dir



def load_all(data_dir):
    """Read every project file into a dictionary of DataFrames, keyed by file name."""
    data_dir = Path(data_dir)
    tables = {}
    for names in NHANES_FILES.values():
        for name in names:
            df = pd.read_sas(data_dir / f"{name}.xpt", format="xport")
            if not df["SEQN"].is_unique:
                raise ValueError(f"{name} has more than one row per participant")
            tables[name] = df
    return tables


def merge_on_seqn(tables):
    """Left-join all files onto DEMO_L by SEQN.

    Columns that already exist in the merged table (such as the survey
    weights repeated in several lab files) are not added again.
    """
    def add_new_columns(left, name):
        right = tables[name]
        new_cols = ["SEQN"] + [c for c in right.columns if c not in left.columns]
        return left.merge(right[new_cols], on="SEQN", how="left")

    others = [name for name in tables if name != "DEMO_L"]
    return reduce(add_new_columns, others, tables["DEMO_L"])

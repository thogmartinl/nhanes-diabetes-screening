"""Download the NHANES August 2021-August 2023 files used in this project."""

import time
import urllib.request
from pathlib import Path

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

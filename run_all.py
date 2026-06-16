"""Run the supplementary analysis for the three cancer datasets used in the study."""

import subprocess
import sys
from pathlib import Path


DATASETS = ["luad", "kirc", "ucec"]
DATA_ROOT = Path(r"C:\path\to\TCGA")
OUTPUT_ROOT = Path(r"C:\path\to\output")
PIPELINE = Path(__file__).resolve().parent / "run_pipeline.py"


for dataset in DATASETS:
    subprocess.run(
        [
            sys.executable,
            str(PIPELINE),
            "--expression",
            str(DATA_ROOT / dataset / "miRNA_rpm_matrix.csv"),
            "--metadata",
            str(DATA_ROOT / dataset / "metadata.csv"),
            "--output",
            str(OUTPUT_ROOT / dataset),
        ],
        check=True,
    )

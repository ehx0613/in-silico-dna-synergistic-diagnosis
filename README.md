# A synergistic in silico-DNA model for disease diagnosis

This code reproduces the three computational stages described in the manuscript
*A synergistic in silico-DNA model for disease diagnosis*.

## Files

- `01_statistical_screening.py`: adaptive statistical testing, Benjamini-Hochberg correction, differential-expression filtering and Pearson-correlation filtering.
- `02_svm_thresholds.py`: single-miRNA balanced linear SVM training and expression-threshold recovery.
- `03_or_logic_validation.py`: OR logic aggregation and diagnostic performance evaluation.
- `run_pipeline.py`: complete analysis for one cancer dataset.
- `run_all.py`: batch entry point for LUAD, KIRC and UCEC.
- `requirements.txt`: Python package dependencies.

## Input data

The expression CSV must contain miRNAs as rows and sample identifiers as columns.
Values must be RPM-normalised expression values.

The metadata CSV must contain a sample identifier column named `SampleID` and a
`Condition` column containing `Diseased` or `Healthy`.

## Run one dataset

```bash
python run_pipeline.py \
  --expression path/to/miRNA_rpm_matrix.csv \
  --metadata path/to/metadata.csv \
  --output path/to/output
```

The random 70:30 stratified split uses `random_state=42`. Statistical screening,
feature selection and SVM threshold determination use only the training set.

## OR-logic definition

Each selected miRNA is represented by a separately trained linear SVM. A sample
is classified as diseased when at least one single-miRNA SVM decision score is
greater than or equal to zero. This score-based definition correctly handles
both upregulated and downregulated biomarkers.

## Archiving and citation

For publication, deposit this folder as a versioned GitHub release and archive
the release in Zenodo or the journal's supplementary-file system. Cite the
permanent DOI or supplementary ZIP in the manuscript's Data accessibility
statement.

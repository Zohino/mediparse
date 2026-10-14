rule build_cohort:
    input:
        discharge="resources/mimic/parquet/discharge.parquet",
        diagnoses="resources/mimic/parquet/diagnoses_icd.parquet",
        candidates="config/diagnosis_candidates.json",
        tables="config/mimic_tables.json",
    output:
        cohort="resources/mimic/cohort.parquet",
        manifest="results/cohort/manifest.json",
    log:
        "logs/cohort/build_cohort.log",
    script:
        "../scripts/build_cohort.py"

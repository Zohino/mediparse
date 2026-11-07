rule build_cohort:
    input:
        discharge="resources/mimic/parquet/discharge.parquet",
        diagnoses="resources/mimic/parquet/diagnoses_icd.parquet",
        discharge_source="resources/mimic/parquet/discharge.source.json",
        diagnoses_source="resources/mimic/parquet/diagnoses_icd.source.json",
        candidates="config/diagnosis_candidates.json",
        corpus_rule="config/input_corpus.json",
    output:
        cohort="resources/mimic/cohort.parquet",
        manifest="results/cohort/manifest.json",
    log:
        "logs/cohort/build_cohort.log",
    script:
        "../scripts/build_cohort.py"

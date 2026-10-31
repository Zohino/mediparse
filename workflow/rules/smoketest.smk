rule synthetic_mimic:
    input:
        plans=SYNTHETIC / "plans.jsonl",
        notes=sorted((SYNTHETIC / "en").glob("*.txt")),
        candidates="config/diagnosis_candidates.json",
    output:
        discharge=SMOKETEST / "mimic/parquet/discharge.parquet",
        diagnoses=SMOKETEST / "mimic/parquet/diagnoses_icd.parquet",
        tables=SMOKETEST / "mimic/tables.json",
    log:
        SMOKETEST_LOGS / "synthetic_mimic.log",
    params:
        corpus=SYNTHETIC,
        # Manifest vytváří validate_mimic; input by do smoketestu přitáhl pravidla MIMIC.
        manifest="results/mimic/manifest.json",
    script:
        "../scripts/synthetic_mimic.py"


use rule build_cohort as build_smoketest_cohort with:
    input:
        discharge=SMOKETEST / "mimic/parquet/discharge.parquet",
        diagnoses=SMOKETEST / "mimic/parquet/diagnoses_icd.parquet",
        candidates="config/diagnosis_candidates.json",
        tables=SMOKETEST / "mimic/tables.json",
    output:
        cohort=SMOKETEST / "cohort.parquet",
        manifest=SMOKETEST / "cohort/manifest.json",
    log:
        SMOKETEST_LOGS / "build_cohort.log",


rule make_smoketest_splits:
    input:
        notes=SMOKETEST / "cohort.parquet",
        config="config/split.json",
    output:
        SMOKETEST / "splits.parquet",
    log:
        SMOKETEST_LOGS / "make_smoketest_splits.log",
    script:
        "../scripts/make_smoketest_splits.py"


rule train_smoketest_model:
    input:
        notes=SMOKETEST / "cohort.parquet",
        config="config/smoketest_training.json",
        candidates="config/diagnosis_candidates.json",
        lexicon="config/section_headers.json",
        split=SMOKETEST / "splits.parquet",
    output:
        model=SMOKETEST / "model.skops",
        predictions=SMOKETEST / "predictions.parquet",
        manifest=SMOKETEST / "run_manifest.json",
    log:
        SMOKETEST_LOGS / "train_smoketest_model.log",
    script:
        "../scripts/train_smoketest.py"


rule evaluate_smoketest_model:
    input:
        predictions=SMOKETEST / "predictions.parquet",
        notes=SMOKETEST / "cohort.parquet",
        split=SMOKETEST / "splits.parquet",
    output:
        SMOKETEST / "metrics.parquet",
    log:
        SMOKETEST_LOGS / "evaluate_smoketest_model.log",
    script:
        "../scripts/evaluate_smoketest.py"


rule infer_smoketest_model:
    input:
        notes=SMOKETEST / "cohort.parquet",
        config="config/smoketest_training.json",
        lexicon="config/section_headers.json",
        model=SMOKETEST / "model.skops",
        predictions=SMOKETEST / "predictions.parquet",
    output:
        SMOKETEST / "demo.parquet",
    log:
        SMOKETEST_LOGS / "infer_smoketest_model.log",
    script:
        "../scripts/infer_smoketest.py"

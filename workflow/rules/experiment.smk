rule make_splits:
    input:
        cohort="resources/mimic/cohort.parquet",
        config="config/split.json",
    output:
        "resources/mimic/splits.parquet",
    log:
        "logs/experiment/make_splits.log",
    script:
        "../scripts/make_splits.py"


# Config řádku pro MIMIC dodá S40; smoketest ho přebije vlastním configem.
rule train_model:
    input:
        cohort="resources/mimic/cohort.parquet",
        config="config/training.json",
        candidates="config/diagnosis_candidates.json",
        lexicon="config/section_headers.json",
        split="resources/mimic/splits.parquet",
    output:
        model="resources/mimic/model.skops",
        predictions="resources/mimic/predictions.parquet",
        manifest="results/experiment/run_manifest.json",
    log:
        "logs/experiment/train_model.log",
    script:
        "../scripts/train_model.py"


rule evaluate_model:
    input:
        predictions="resources/mimic/predictions.parquet",
        cohort="resources/mimic/cohort.parquet",
        split="resources/mimic/splits.parquet",
    output:
        "results/experiment/metrics.parquet",
    log:
        "logs/experiment/evaluate_model.log",
    script:
        "../scripts/evaluate_model.py"


# Config řádku pro MIMIC dodá S40; smoketest ho přebije vlastním configem.
rule infer_model:
    input:
        cohort="resources/mimic/cohort.parquet",
        config="config/training.json",
        lexicon="config/section_headers.json",
        model="resources/mimic/model.skops",
        predictions="resources/mimic/predictions.parquet",
    output:
        "resources/mimic/inference.parquet",
    log:
        "logs/experiment/infer_model.log",
    script:
        "../scripts/infer_model.py"

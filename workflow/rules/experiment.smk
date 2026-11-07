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


# config/training.json chybí záměrně: config řádku pro MIMIC dodá S40,
# smoketest ho přebije vlastním configem. config/smoketest_training.json sem
# nepatří, protože MIMIC by trénoval pod jeho row_id.
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
        costs="results/experiment/training_costs.json",
    log:
        "logs/experiment/train_model.log",
    benchmark:
        # repeat() by zopakovalo celý trénink, proto měříme jediný běh.
        "results/experiment/benchmarks/train_model.tsv"
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


# Config řádku: viz train_model.
rule infer_model:
    input:
        cohort="resources/mimic/cohort.parquet",
        config="config/training.json",
        lexicon="config/section_headers.json",
        model="resources/mimic/model.skops",
        predictions="resources/mimic/predictions.parquet",
    output:
        inference="resources/mimic/inference.parquet",
        costs="results/experiment/inference_costs.json",
    log:
        "logs/experiment/infer_model.log",
    benchmark:
        "results/experiment/benchmarks/infer_model.tsv"
    script:
        "../scripts/infer_model.py"

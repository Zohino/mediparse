from pathlib import Path

SYNTHETIC = Path("resources/synthetic")


rule prepare_smoketest_subset:
    input:
        plans=SYNTHETIC / "plans.jsonl",
        notes=sorted((SYNTHETIC / "en").glob("*.txt")),
    output:
        "build/smoketest/notes.parquet",
    log:
        "logs/smoketest/prepare_smoketest_subset.log",
    params:
        corpus=SYNTHETIC,
    script:
        "../scripts/prepare_smoketest.py"


rule train_smoketest_model:
    input:
        notes="build/smoketest/notes.parquet",
        config="config/smoketest_training.json",
    output:
        model="build/smoketest/model.skops",
        predictions="build/smoketest/predictions.parquet",
    log:
        "logs/smoketest/train_smoketest_model.log",
    script:
        "../scripts/train_smoketest.py"


rule evaluate_smoketest_model:
    input:
        predictions="build/smoketest/predictions.parquet",
    output:
        "build/smoketest/metrics.parquet",
    log:
        "logs/smoketest/evaluate_smoketest_model.log",
    script:
        "../scripts/evaluate_smoketest.py"

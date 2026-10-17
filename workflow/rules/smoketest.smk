rule prepare_smoketest_subset:
    input:
        plans=SYNTHETIC / "plans.jsonl",
        notes=sorted((SYNTHETIC / "en").glob("*.txt")),
    output:
        SMOKETEST / "notes.parquet",
    log:
        SMOKETEST_LOGS / "prepare_smoketest_subset.log",
    params:
        corpus=SYNTHETIC,
    script:
        "../scripts/prepare_smoketest.py"


rule train_smoketest_model:
    input:
        notes=SMOKETEST / "notes.parquet",
        config="config/smoketest_training.json",
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
    output:
        SMOKETEST / "metrics.parquet",
    log:
        SMOKETEST_LOGS / "evaluate_smoketest_model.log",
    script:
        "../scripts/evaluate_smoketest.py"


rule infer_smoketest_model:
    input:
        notes=SMOKETEST / "notes.parquet",
        model=SMOKETEST / "model.skops",
        predictions=SMOKETEST / "predictions.parquet",
    output:
        SMOKETEST / "demo.parquet",
    log:
        SMOKETEST_LOGS / "infer_smoketest_model.log",
    script:
        "../scripts/infer_smoketest.py"

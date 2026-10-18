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


rule make_smoketest_splits:
    input:
        notes=SMOKETEST / "notes.parquet",
        config="config/split.json",
    output:
        SMOKETEST / "splits.parquet",
    log:
        SMOKETEST_LOGS / "make_smoketest_splits.log",
    script:
        "../scripts/make_smoketest_splits.py"


rule train_smoketest_model:
    input:
        notes=SMOKETEST / "notes.parquet",
        config="config/smoketest_training.json",
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
        notes=SMOKETEST / "notes.parquet",
        split=SMOKETEST / "splits.parquet",
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

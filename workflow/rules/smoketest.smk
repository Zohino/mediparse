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

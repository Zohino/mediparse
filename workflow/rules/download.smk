import re

from mediparse.entrypoints.mimic_download import mimic_tables

MIMIC_TABLES = mimic_tables()


rule download_mimic:
    input:
        expand("resources/mimic/{file}", file=MIMIC_TABLES),


rule download_mimic_table:
    output:
        protected(
            ensure(
                "resources/mimic/{file}",
                sha256=lambda wildcards: MIMIC_TABLES[wildcards.file].sha256,
            )
        ),
    log:
        "logs/download/{file}.log",
    wildcard_constraints:
        file="|".join(map(re.escape, MIMIC_TABLES)),
    retries: 2
    resources:
        physionet=1,
    params:
        url=lambda wildcards: MIMIC_TABLES[wildcards.file].url,
    shell:
        """
        workflow/scripts/download.sh \
            .netrc \
            {params.url:q} \
            {output:q} \
            {log:q}
        """


rule validate_mimic:
    input:
        expand("resources/mimic/inventory/{file}.json", file=MIMIC_TABLES),
    output:
        manifest="results/mimic/manifest.json",
        flag=touch("resources/mimic/validated.flag"),
    shell:
        "jq -s '{{tables: .}}' {input:q} >{output.manifest:q}"


rule inventory_mimic_table:
    input:
        "resources/mimic/{file}",
    output:
        "resources/mimic/inventory/{file}.json",
    wildcard_constraints:
        file="|".join(map(re.escape, MIMIC_TABLES)),
    shell:
        "workflow/scripts/inventory.sh {input:q} {output:q}"


MIMIC_TABLE_NAMES = [name.removesuffix(".csv.gz") for name in MIMIC_TABLES]


rule parquet_mimic:
    input:
        expand("resources/mimic/parquet/{table}.parquet", table=MIMIC_TABLE_NAMES),


rule parquet_mimic_table:
    input:
        table="resources/mimic/{table}.csv.gz",
        inventory="resources/mimic/inventory/{table}.csv.gz.json",
        validated="resources/mimic/validated.flag",
    output:
        "resources/mimic/parquet/{table}.parquet",
    log:
        "logs/parquet/{table}.log",
    wildcard_constraints:
        table="|".join(map(re.escape, MIMIC_TABLE_NAMES)),
    script:
        "../scripts/convert_mimic.py"

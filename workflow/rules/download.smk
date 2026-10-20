import re

from mediparse.entrypoints.mimic_download import mimic_table_files, mimic_tables

MIMIC_TABLES = mimic_tables()
MIMIC_TABLE_FILES = mimic_table_files()


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


rule parquet_mimic:
    input:
        expand("resources/mimic/parquet/{table}.parquet", table=MIMIC_TABLE_FILES),


rule parquet_mimic_table:
    input:
        table=lambda wildcards: f"resources/mimic/{MIMIC_TABLE_FILES[wildcards.table]}",
        inventory=lambda wildcards: (
            f"resources/mimic/inventory/{MIMIC_TABLE_FILES[wildcards.table]}.json"
        ),
        validated="resources/mimic/validated.flag",
    output:
        "resources/mimic/parquet/{table}.parquet",
    log:
        "logs/parquet/{table}.log",
    wildcard_constraints:
        table="|".join(map(re.escape, MIMIC_TABLE_FILES)),
    script:
        "../scripts/convert_mimic.py"

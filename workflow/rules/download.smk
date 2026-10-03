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

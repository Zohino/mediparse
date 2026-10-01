from collections import Counter
import posixpath
import re


configfile: "config/mimic_tables.json"


filenames = [posixpath.basename(table["url"]) for table in config["mimic_tables"]]
duplicates = [name for name, count in Counter(filenames).items() if count > 1]

if duplicates:
    raise ValueError(
        "config/mimic_tables.json obsahuje duplicitní názvy souborů: "
        + ", ".join(duplicates)
    )

MIMIC_TABLES = {
    posixpath.basename(table["url"]): table for table in config["mimic_tables"]
}


rule download_mimic:
    input:
        expand("resources/mimic/{file}", file=MIMIC_TABLES),


rule download_mimic_table:
    output:
        protected(
            ensure(
                "resources/mimic/{file}",
                sha256=lambda wildcards: MIMIC_TABLES[wildcards.file]["sha256"],
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
        url=lambda wildcards: MIMIC_TABLES[wildcards.file]["url"],
    shell:
        """
        workflow/scripts/download.sh \
            .netrc \
            {params.url:q} \
            {output:q} \
            {log:q}
        """

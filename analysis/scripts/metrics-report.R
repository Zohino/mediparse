args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop("Použití: Rscript analysis/scripts/metrics-report.R <metrics> <manifest> <out> <candidates>")
}

# Sazba práce má jednu kopii v EDA; její umístění přehodnotí WP-9.
source("eda/R/dua.R")
source("eda/R/export.R")
source("analysis/R/contract.R")
source("analysis/R/metrics.R")
source("analysis/R/provenance.R")

# Rscript nesmí automaticky vypsat ggplot na výchozí pdf() bez písma práce.
invisible(write_metrics_report(args[[1]], args[[2]], args[[3]], args[[4]]))

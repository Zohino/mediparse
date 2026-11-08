args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop("Použití: Rscript analysis/scripts/metrics-report.R <metrics> <evaluation> <manifest> <out> <candidates>")
}

source("R/thesis/dua.R")
source("R/thesis/export.R")
source("R/thesis/contract.R")
source("analysis/R/metrics.R")
source("analysis/R/provenance.R")

# Rscript nesmí automaticky vypsat ggplot na výchozí pdf() bez písma práce.
invisible(write_metrics_report(args[[1]], args[[2]], args[[3]], args[[4]], args[[5]]))

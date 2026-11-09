RUN_MANIFEST_FIXTURE <- "../../../tests/fixtures/contracts/run_manifest.json"
EVALUATION_MANIFEST_FIXTURE <- "../../../tests/fixtures/contracts/evaluation_manifest.json"

coreutils_sha256 <- function(path) {
  strsplit(system2("sha256sum", shQuote(path), stdout = TRUE), " +")[[1]][1]
}

metrics_chain <- function(dir, row_id = "smoketest", metrics_path = METRICS_FIXTURE) {
  manifest <- jsonlite::fromJSON(RUN_MANIFEST_FIXTURE, simplifyVector = FALSE)
  evaluation <- jsonlite::fromJSON(EVALUATION_MANIFEST_FIXTURE, simplifyVector = FALSE)
  manifest$row_id <- row_id
  evaluation$row_id <- row_id
  manifest_path <- file.path(dir, "manifest běhu.json")
  evaluation_path <- file.path(dir, "manifest evaluace.json")
  jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE)
  evaluation$training_manifest_sha256 <- coreutils_sha256(manifest_path)
  evaluation$metrics_sha256 <- coreutils_sha256(metrics_path)
  jsonlite::write_json(evaluation, evaluation_path, auto_unbox = TRUE)
  list(evaluation = evaluation_path, manifest = manifest_path)
}

report_with_chain <- function(metrics_path, chain, out_dir, candidates_path) {
  write_metrics_report(metrics_path, chain$evaluation, chain$manifest, out_dir, candidates_path)
}

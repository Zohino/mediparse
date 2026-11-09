is_single_string <- function(x) {
  is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x)
}

ensure_evaluation_lineage <- function(metrics_path, evaluation_path, manifest_path, manifest) {
  evaluation <- jsonlite::fromJSON(evaluation_path, simplifyVector = FALSE)
  shared <- c("row_id", "predictions_sha256", "dataset_sha256", "split_sha256")
  fields <- c(shared, "metrics_sha256", "training_manifest_sha256")
  for (field in fields) {
    value <- if (is.list(evaluation)) evaluation[[field]] else NULL
    if (!is_single_string(value)) {
      stop(sprintf("Soubor %s: %s musí být jeden neprázdný řetězec.", evaluation_path, field))
    }
  }
  paths <- c(metrics_sha256 = metrics_path, training_manifest_sha256 = manifest_path)
  for (field in names(paths)) {
    if (!identical(unname(tools::sha256sum(paths[[field]])), evaluation[[field]])) {
      stop(sprintf(
        "Soubor %s: %s neodpovídá manifestu evaluace %s.",
        paths[[field]], field, evaluation_path
      ))
    }
  }
  for (field in shared) {
    if (!identical(evaluation[[field]], manifest[[field]])) {
      stop(sprintf(
        "Soubor %s: %s neodpovídá manifestu tréninku %s.",
        evaluation_path, field, manifest_path
      ))
    }
  }
  invisible(NULL)
}

write_sidecar <- function(output, input, evaluation, manifest, row_id) {
  path <- paste0(output, ".meta.json")
  provenance <- list(
    input = list(path = input, sha256 = unname(tools::sha256sum(input))),
    evaluation_manifest = list(path = evaluation, sha256 = unname(tools::sha256sum(evaluation))),
    run_manifest = list(
      path = manifest, sha256 = unname(tools::sha256sum(manifest)),
      row_id = row_id
    )
  )
  jsonlite::write_json(provenance, path, auto_unbox = TRUE, pretty = TRUE)
  invisible(path)
}

SCREENING_COLUMNS <- c(
  fold = "BIGINT", vectorizer = "VARCHAR", family = "VARCHAR", diagnosis = "VARCHAR",
  test_notes = "BIGINT", test_positives = "BIGINT", threshold = "DOUBLE",
  converged = "BOOLEAN", precision = "DOUBLE", recall = "DOUBLE", f1 = "DOUBLE",
  accuracy = "DOUBLE", roc_auc = "DOUBLE", pr_auc = "DOUBLE"
)

PREDICTION_COLUMNS <- c(
  fold = "BIGINT", vectorizer = "VARCHAR", family = "VARCHAR", diagnosis = "VARCHAR",
  note_id = "VARCHAR", label = "BOOLEAN", predicted = "BOOLEAN", score = "DOUBLE"
)

FAMILY_LABELS <- c(
  complement_nb = "Complement NB", logreg = "Logistická regrese",
  linear_svm = "Lineární SVM", nb_svm = "NB-SVM", lightgbm_lsa = "LightGBM nad LSA"
)

SCREENING_DIAGNOSES <- c("diabetes", "copd", "ckd", "pneumonia", "atrial_fibrillation")

screening_paths <- function() {
  directory <- Sys.getenv("MEDIPARSE_SCREENING", project_path("results", "screening"))
  predictions <- Sys.getenv(
    "MEDIPARSE_SCREENING_PREDICTIONS", project_path("resources", "mimic", "screening")
  )
  list(
    results = file.path(directory, "classical.parquet"),
    manifest = file.path(directory, "classical_manifest.json"),
    predictions = file.path(predictions, "classical_predictions.parquet")
  )
}

read_screening_results <- function(path) read_contract_table(path, SCREENING_COLUMNS)

screening_predictions_source <- function(con, path) {
  ensure_data_access(dirname(path))
  source <- sprintf("read_parquet(%s)", DBI::dbQuoteString(con, path))
  check_contract(con, source, PREDICTION_COLUMNS)
  source
}

read_screening_manifest <- function(paths, config_path) {
  ensure_data_access(dirname(paths$predictions))
  manifest <- jsonlite::read_json(paths$manifest, simplifyVector = TRUE)
  inputs <- c(
    results_sha256 = paths$results, predictions_sha256 = paths$predictions,
    config_sha256 = config_path
  )
  for (field in names(inputs)) {
    path <- inputs[[field]]
    if (!file.exists(path) || !scalar_text(manifest[[field]]) ||
        !identical(unname(tools::sha256sum(path)), manifest[[field]])) {
      stop("Soubor ", path, ": ", field, " neodpovídá manifestu screeningu.", call. = FALSE)
    }
  }
  manifest
}

check_screening_grid <- function(results, config) {
  vectorizers <- vapply(config$vectorizers, `[[`, character(1), "name")
  expected <- expand.grid(
    fold = seq_len(config$folds) - 1L, vectorizer = vectorizers,
    family = unlist(config$families), diagnosis = SCREENING_DIAGNOSES,
    stringsAsFactors = FALSE
  )
  keys <- names(expected)
  actual <- results[keys]
  joined <- merge(actual, expected, by = keys)
  if (nrow(actual) != nrow(expected) || anyDuplicated(actual) ||
      nrow(joined) != nrow(expected)) {
    stop("Výsledky screeningu neobsahují přesnou mřížku configu.", call. = FALSE)
  }
  if (any(!actual$family %in% names(FAMILY_LABELS))) {
    stop("Neznámá rodina screeningu.", call. = FALSE)
  }
  invisible(TRUE)
}

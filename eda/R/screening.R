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

screening_fold_metrics <- function(results) {
  stats::aggregate(
    results[c("f1", "pr_auc")], results[c("fold", "family", "vectorizer")], mean
  )
}

screening_summary <- function(results) {
  folds <- screening_fold_metrics(results)
  groups <- split(folds, interaction(folds$family, folds$vectorizer, drop = TRUE))
  summary <- do.call(rbind, lapply(groups, function(group) {
    cells <- results[results$family == group$family[1] & results$vectorizer == group$vectorizer[1], ]
    data.frame(
      family = group$family[1], vectorizer = group$vectorizer[1],
      mean = mean(group$f1), sd = stats::sd(group$f1),
      se = stats::sd(group$f1) / sqrt(nrow(group)), pr_auc = mean(group$pr_auc),
      nonconverged = sum(!cells$converged), cells = nrow(cells)
    )
  }))
  summary <- summary[order(summary$family, -summary$mean, summary$vectorizer), ]
  summary$best <- !duplicated(summary$family)
  rownames(summary) <- NULL
  summary
}

screening_selection <- function(summary) {
  best <- summary[summary$best, ]
  best <- best[order(-best$mean, best$family), ]
  winner <- best[1, ]
  threshold <- winner$mean - winner$se
  band <- best[best$mean >= threshold, ]
  comparison <- band[band$family != winner$family, ]
  fallback <- nrow(comparison) == 0L && nrow(best) > 1L
  if (fallback) comparison <- best[2, ]
  list(winner = winner, threshold = threshold, band = band, comparison = comparison, fallback = fallback)
}

screening_pair_counts <- function(con, source, winner, other) {
  subset <- function(model) sprintf(
    "SELECT fold, diagnosis, note_id, label, predicted = label AS correct FROM %s WHERE family = %s AND vectorizer = %s",
    source, DBI::dbQuoteString(con, model$family), DBI::dbQuoteString(con, model$vectorizer)
  )
  counts <- DBI::dbGetQuery(con, sprintf("
    WITH a AS (%s), b AS (%s)
    SELECT fold, diagnosis, count(*) AS n,
      count(a.note_id) AS n_a, count(b.note_id) AS n_b,
      count(DISTINCT a.note_id) AS unique_a, count(DISTINCT b.note_id) AS unique_b,
      count(*) FILTER (WHERE a.label != b.label) AS different_labels,
      count(*) FILTER (WHERE a.correct AND b.correct) AS n11,
      count(*) FILTER (WHERE a.correct AND NOT b.correct) AS n10,
      count(*) FILTER (WHERE NOT a.correct AND b.correct) AS n01,
      count(*) FILTER (WHERE NOT a.correct AND NOT b.correct) AS n00
    FROM a FULL OUTER JOIN b USING (fold, diagnosis, note_id)
    GROUP BY fold, diagnosis ORDER BY fold, diagnosis
  ", subset(winner), subset(other)))
  if (nrow(counts) == 0L || any(
    counts$n_a != counts$n | counts$n_b != counts$n |
      counts$unique_a != counts$n | counts$unique_b != counts$n | counts$different_labels > 0
  )) {
    stop("Predikce dvojice screeningu se nepárují úplně a jednoznačně se stejnými labely.", call. = FALSE)
  }
  counts
}

screening_phi <- function(counts) {
  n11 <- as.numeric(counts$n11)
  n10 <- as.numeric(counts$n10)
  n01 <- as.numeric(counts$n01)
  n00 <- as.numeric(counts$n00)
  denominator <- sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
  phi <- rep(NA_real_, length(denominator))
  defined <- denominator > 0
  phi[defined] <- (n11 * n00 - n01 * n10)[defined] / denominator[defined]
  phi
}

error_correlation <- function(con, source, winner, others) {
  rows <- lapply(seq_len(nrow(others)), function(i) {
    other <- others[i, ]
    counts <- screening_pair_counts(con, source, winner, other)
    phi <- screening_phi(counts)
    data.frame(
      family = other$family, vectorizer = other$vectorizer,
      phi = if (all(is.na(phi))) NA_real_ else mean(phi, na.rm = TRUE),
      defined = sum(!is.na(phi)), cells = length(phi)
    )
  })
  if (length(rows) == 0L) return(data.frame(
    family = character(), vectorizer = character(), phi = numeric(), defined = integer(), cells = integer()
  ))
  do.call(rbind, rows)
}

ensemble_needed <- function(phi, comparison) {
  values <- phi$phi[phi$family %in% comparison$family]
  if (all(is.na(values))) return(NA)
  any(values < 0.5, na.rm = TRUE)
}

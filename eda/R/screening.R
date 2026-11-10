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
  summary <- summary[order(summary$family, -summary$mean, summary$vectorizer, method = "radix"), ]
  summary$best <- !duplicated(summary$family)
  rownames(summary) <- NULL
  summary
}

screening_selection <- function(summary) {
  best <- summary[summary$best, ]
  best <- best[order(-best$mean, best$family, method = "radix"), ]
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
  # Úplné spojení umožní kontrolovat chybějící i duplicitní zprávy obou stran;
  # vnitřní spojení by chybějící protějšky skrylo.
  counts <- DBI::dbGetQuery(con, sprintf("
    WITH a AS (%s), b AS (%s)
    SELECT fold, diagnosis, count(*) AS n,
      count(DISTINCT a.note_id) AS unique_a, count(DISTINCT b.note_id) AS unique_b,
      count(*) FILTER (WHERE a.label != b.label) AS different_labels,
      count(*) FILTER (WHERE a.correct AND b.correct) AS n11,
      count(*) FILTER (WHERE a.correct AND NOT b.correct) AS n10,
      count(*) FILTER (WHERE NOT a.correct AND b.correct) AS n01,
      count(*) FILTER (WHERE NOT a.correct AND NOT b.correct) AS n00
    FROM a FULL OUTER JOIN b USING (fold, diagnosis, note_id)
    GROUP BY fold, diagnosis ORDER BY fold, diagnosis
  ", subset(winner), subset(other)))
  complete <- counts$unique_a == counts$n & counts$unique_b == counts$n &
    counts$different_labels == 0 & counts$n11 + counts$n10 + counts$n01 + counts$n00 == counts$n
  if (nrow(counts) == 0L || any(!complete)) {
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

screening_number <- function(x) {
  out <- formatC(x, format = "f", digits = 3, decimal.mark = ",")
  out[is.na(x)] <- "–"
  out
}

screening_model_text <- function(models) {
  paste0(unname(FAMILY_LABELS[models$family]), " (", models$vectorizer, ")")
}

screening_grid_table <- function(summary) {
  data.frame(
    rodina = unname(FAMILY_LABELS[summary$family]), vektorizér = summary$vectorizer,
    macro = paste(screening_number(summary$mean), "±", screening_number(summary$sd)),
    pr_auc = screening_number(summary$pr_auc),
    konvergence = paste(summary$nonconverged, "z", summary$cells),
    nejlepší = ifelse(summary$best, "ano", "ne")
  )
}

screening_diagnosis_table <- function(results, summary, candidates) {
  best <- summary[summary$best, c("family", "vectorizer")]
  rows <- merge(results, best, by = c("family", "vectorizer"))
  means <- stats::aggregate(rows["f1"], rows[c("family", "diagnosis")], mean)
  data.frame(
    rodina = unname(FAMILY_LABELS[means$family]),
    diagnóza = candidate_labels(means$diagnosis, candidates), f1 = screening_number(means$f1)
  )
}

screening_phi_table <- function(phi, selection) {
  data.frame(
    rodina = unname(FAMILY_LABELS[phi$family]), vektorizér = phi$vectorizer,
    phi = screening_number(phi$phi), buňky = paste(phi$defined, "z", phi$cells),
    pásmo = ifelse(phi$family %in% selection$band$family, "ano", "ne"),
    rozhoduje = ifelse(phi$family %in% selection$comparison$family, "ano", "ne")
  )
}

screening_plot <- function(results, summary, selection) {
  best <- summary[summary$best, ]
  best <- best[order(best$mean, best$family), ]
  folds <- merge(screening_fold_metrics(results), best[c("family", "vectorizer")])
  labels <- setNames(screening_model_text(best), best$family)
  folds$rodina <- factor(labels[folds$family], levels = labels)
  best$rodina <- factor(labels[best$family], levels = labels)
  ggplot2::ggplot(folds, ggplot2::aes(x = f1, y = rodina)) +
    ggplot2::geom_vline(xintercept = selection$threshold, linetype = "dashed") +
    ggplot2::geom_point(colour = "grey50", size = 1.8) +
    ggplot2::geom_point(data = best, ggplot2::aes(x = mean), shape = 18, size = 3) +
    ggplot2::scale_x_continuous(labels = screening_number) +
    ggplot2::labs(x = "Macro-F1", y = NULL) + theme_thesis()
}

screening_provenance_table <- function(manifest) {
  items <- c(
    "Dataset SHA-256" = manifest$dataset_sha256,
    "Split SHA-256" = manifest$split_sha256,
    "Seed a foldy původního splitu" = paste(manifest$split_seed, manifest$split_folds, sep = "; "),
    "Config SHA-256" = manifest$config_sha256,
    "Výsledky SHA-256" = manifest$results_sha256,
    "Predikce SHA-256" = manifest$predictions_sha256,
    "Seed screeningu" = as.character(manifest$seed),
    "Vývojové zprávy" = format_number(manifest$development_notes),
    "Vývojoví pacienti" = format_number(manifest$development_patients),
    "Předzpracování" = manifest$text_preprocessing,
    "Strip diagnostických sekcí" = if (is.null(manifest$diagnostic_strip)) "ne" else "ano",
    "Revize kódu screeningu" = manifest$environment$source$commit,
    "Rozpracovaný strom" = if (manifest$environment$source$dirty) "ano" else "ne",
    unlist(manifest$environment$packages)
  )
  provenance_table(as.list(items))
}

screening_vectorizers_text <- function(config) {
  labels <- vapply(config$vectorizers, function(spec) {
    analyzer <- switch(spec$analyzer, word = "slovní", char_wb = "znakové")
    location <- if (spec$analyzer == "char_wb") " uvnitř slov" else ""
    sprintf("%s: %s n-gramy %d–%d%s", spec$name, analyzer, spec$ngram_range[[1]], spec$ngram_range[[2]], location)
  }, character(1))
  join_labels(labels)
}

screening_selection_text <- function(selection) {
  single <- nrow(selection$band) == 1L
  verb <- if (single) "postupuje" else "postupují"
  target <- if (single) "této rodiny s uvedeným vektorizérem" else "těchto rodin s uvedenými vektorizéry"
  sprintf(
    paste0(
      "Vítězem je %s s průměrnou macro-F1 %s. Hranice pásma 1 SE je %s; do S39j %s %s. ",
      "Ladění S39j se týká %s; screening neurčuje laděné hyperparametry."
    ),
    screening_model_text(selection$winner), screening_number(selection$winner$mean),
    screening_number(selection$threshold), verb, join_labels(screening_model_text(selection$band)), target
  )
}

screening_ensemble_text <- function(phi, selection) {
  comparison <- join_labels(screening_model_text(selection$comparison))
  rule <- if (selection$fallback) {
    paste0("V pásmu zůstal jen vítěz; rozhoduje srovnání s druhou nejlepší rodinou: ", comparison, ". ")
  } else {
    paste0("O ensemblu rozhoduje srovnání vítěze s rodinami pásma: ", comparison, ". ")
  }
  needed <- ensemble_needed(phi, selection$comparison)
  if (is.na(needed)) return(paste0(rule,
    "Pro rozhodující dvojice není φ k dispozici; nejde rozhodnout o podtasku ensemblu a výsledek patří do diskuse."))
  if (needed) return(paste0(rule,
    "Alespoň jedna rozhodující dvojice má průměr φ < 0,5; před S39j má vzniknout podtask ensemblu."))
  paste0(rule, "Žádná rozhodující dvojice nemá průměr φ < 0,5; podtask ensemblu nevzniká a výsledek patří do diskuse.")
}

STAGE_TRIAL_COLUMNS <- c(
  trial = "BIGINT", fold = "BIGINT", diagnosis = "VARCHAR", branch = "VARCHAR",
  representation = "VARCHAR", family = "VARCHAR", default = "BOOLEAN", params = "VARCHAR",
  value = "DOUBLE", seconds = "DOUBLE", max_f1 = "DOUBLE", converged = "BOOLEAN",
  n_features = "BIGINT"
)
STAGE_PREDICTION_COLUMNS <- c(
  trial = "BIGINT", fold = "BIGINT", diagnosis = "VARCHAR", note_id = "VARCHAR",
  label = "BOOLEAN", predicted = "BOOLEAN"
)

FAMILY_LABELS <- c(
  complement_nb = "Complement NB", logreg = "Logistická regrese",
  linear_svm = "Lineární SVM", nb_svm = "NB-SVM", lightgbm_lsa = "LightGBM nad LSA"
)

STAGE_DIAGNOSES <- c("diabetes", "copd", "ckd", "pneumonia", "atrial_fibrillation")

stage_trial_paths <- function(stage) {
  directory <- Sys.getenv("MEDIPARSE_SCREENING", project_path("results", "screening"))
  predictions <- Sys.getenv(
    "MEDIPARSE_SCREENING_PREDICTIONS", project_path("resources", "mimic", "screening")
  )
  list(
    results = file.path(directory, paste0(stage, "_trials.parquet")),
    manifest = file.path(directory, paste0(stage, "_trials_manifest.json")),
    predictions = file.path(predictions, paste0(stage, "_trial_predictions.parquet"))
  )
}

read_stage_trials <- function(path) read_contract_table(path, STAGE_TRIAL_COLUMNS)

stage_predictions_source <- function(con, path) {
  ensure_data_access(dirname(path))
  source <- sprintf("read_parquet(%s)", DBI::dbQuoteString(con, path))
  check_contract(con, source, STAGE_PREDICTION_COLUMNS)
  source
}

read_stage_manifest <- function(paths, config_path) {
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
      stop("Soubor ", path, ": ", field, " neodpovídá manifestu ladění stupně.", call. = FALSE)
    }
  }
  manifest
}

stage_pair_counts <- function(con, source, winner, other) {
  subset <- function(model) sprintf(
    "SELECT fold, diagnosis, note_id, label, predicted = label AS correct FROM %s WHERE trial = %d",
    source, model$trial
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

stage_phi <- function(counts) {
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

stage_error_correlation <- function(con, source, winner, others) {
  rows <- lapply(seq_len(nrow(others)), function(i) {
    other <- others[i, ]
    counts <- stage_pair_counts(con, source, winner, other)
    phi <- stage_phi(counts)
    data.frame(
      family = other$family, trial = other$trial,
      phi = if (all(is.na(phi))) NA_real_ else mean(phi, na.rm = TRUE),
      defined = sum(!is.na(phi)), cells = length(phi)
    )
  })
  if (length(rows) == 0L) return(data.frame(
    family = character(), trial = integer(), phi = numeric(), defined = integer(), cells = integer()
  ))
  do.call(rbind, rows)
}

stage_ensemble_needed <- function(phi, second) {
  values <- phi$phi[phi$trial %in% second$trial]
  if (all(is.na(values))) return(NA)
  any(values < 0.5, na.rm = TRUE)
}

check_stage_trials <- function(trials, manifest, config, candidates) {
  fail <- function() stop("Tabulka pokusů stupně není úplná nebo konzistentní.", call. = FALSE)
  if (!setequal(unique(trials$trial), seq_len(config$trials) - 1L) ||
      manifest$completed_trials != config$trials || manifest$trials != config$trials ||
      anyDuplicated(trials[c("trial", "fold", "diagnosis")])) fail()
  if (any(!trials$family %in% names(FAMILY_LABELS))) {
    stop("Neznámá rodina stupně.", call. = FALSE)
  }
  if (!all(STAGE_DIAGNOSES %in% names(candidates))) {
    stop("Diagnóza stupně chybí mezi kandidáty.", call. = FALSE)
  }
  expected <- expand.grid(fold = seq_len(config$folds) - 1L,
    diagnosis = STAGE_DIAGNOSES, stringsAsFactors = FALSE)
  constants <- c("branch", "representation", "family", "default", "params", "value", "seconds")
  for (group in split(trials, trials$trial)) {
    if (nrow(unique(group[constants])) != 1L ||
        nrow(group) != nrow(expected) ||
        nrow(merge(group, expected, by = c("fold", "diagnosis"))) != nrow(expected) ||
        any(group$branch != paste(group$representation, group$family, sep = "/"))) fail()
  }
  defaults <- unique(trials[trials$default, c("trial", "branch")])
  if (!nrow(defaults) || anyDuplicated(defaults$branch) ||
      !setequal(defaults$trial, seq_len(nrow(defaults)) - 1L) ||
      !setequal(defaults$branch, trials$branch)) fail()
  invisible(TRUE)
}

stage_trial_summary <- function(trials) {
  folds <- stats::aggregate(trials["max_f1"], trials[c("trial", "fold")], mean)
  summary <- do.call(rbind, lapply(split(trials, trials$trial), function(group) {
    values <- folds$max_f1[folds$trial == group$trial[1]]
    if (any(abs(group$value - mean(values)) > 1e-9)) {
      stop("Přepočtený cíl pokusu neodpovídá sloupci value.", call. = FALSE)
    }
    row <- group[1, c("trial", "branch", "representation", "family", "default", "value", "seconds", "params")]
    row$sd <- stats::sd(values)
    row$converged <- all(group$converged)
    row
  }))
  summary[order(summary$trial), ]
}

stage_selection <- function(summary, manifest) {
  ordered <- summary[order(-summary$value, summary$trial, method = "radix"), ]
  best <- ordered[!duplicated(ordered$family), ]
  if (nrow(best) == 1L) stop("stupeň má jedinou rodinu", call. = FALSE)
  if (best$trial[1] != manifest$best_trial) {
    stop("Vítěz neodpovídá manifestu.", call. = FALSE)
  }
  list(winner = best[1, ], family_best = best, second = best[2, ], others = best[-1, ])
}

space_value_text <- function(x) {
  if (is.null(x)) return("žádná")
  if (is.logical(x)) return(ifelse(x, "ano", "ne"))
  if (is.numeric(x)) return(format(x, decimal.mark = ",", big.mark = NBSP,
    scientific = FALSE, trim = TRUE))
  x
}

space_range_text <- function(param) {
  if (param$kind == "categorical") {
    return(paste(vapply(param$choices, space_value_text, character(1)), collapse = ", "))
  }
  range <- paste0(space_value_text(param$low), "–", space_value_text(param$high))
  if (isTRUE(param$log)) return(paste0(range, " (log)"))
  range
}

stage_component_space_table <- function(component, winner, active) {
  name <- component$name %||% component$family
  rows <- lapply(component$space$params, function(param) {
    value <- "–"
    if (name %in% active) {
      value <- space_value_text(winner[[paste(name, param$name, sep = ".")]])
    }
    data.frame(složka = name, parametr = param$name, rozsah = space_range_text(param),
      výchozí = space_value_text(component$default[[param$name]]), vítěz = value)
  })
  do.call(rbind, rows)
}

stage_space_table <- function(config, winner) {
  active <- strsplit(winner$branch, "/", fixed = TRUE)[[1]]
  rows <- lapply(c(config$representations, config$families),
    stage_component_space_table, winner = winner, active = active)
  do.call(rbind, rows)
}

stage_defaults_table <- function(summary) {
  rows <- summary[summary$default, ]
  data.frame(reprezentace = rows$representation, rodina = unname(FAMILY_LABELS[rows$family]),
    macro = paste(metric_text(rows$value), "±", metric_text(rows$sd)),
    konvergence = ifelse(rows$converged, "ano", "ne"), minuty = metric_text(rows$seconds / 60))
}

stage_family_table <- function(selection, summary) {
  rows <- selection$family_best
  defaults <- summary[summary$default, ]
  data.frame(rodina = unname(FAMILY_LABELS[rows$family]),
    pokus = paste(rows$trial, rows$representation),
    macro = paste(metric_text(rows$value), "±", metric_text(rows$sd)),
    výchozí = metric_text(defaults$value[match(rows$branch, defaults$branch)]),
    pokusy = vapply(rows$family, function(family) sum(summary$family == family), integer(1)))
}

stage_diagnosis_table <- function(trials, selection, candidates) {
  rows <- trials[trials$trial %in% selection$family_best$trial, ]
  means <- stats::aggregate(rows["max_f1"], rows[c("family", "diagnosis")], mean)
  data.frame(rodina = unname(FAMILY_LABELS[means$family]),
    diagnóza = candidate_labels(means$diagnosis, candidates), f1 = metric_text(means$max_f1))
}

stage_phi_table <- function(phi, selection) {
  data.frame(rodina = unname(FAMILY_LABELS[phi$family]), pokus = phi$trial,
    phi = metric_text(phi$phi), buňky = paste(phi$defined, "z", phi$cells),
    rozhoduje = ifelse(phi$trial == selection$second$trial, "ano", "ne"))
}

stage_history_plot <- function(summary, manifest) {
  summary <- summary[order(summary$trial), ]
  summary$best <- cummax(summary$value)
  ggplot2::ggplot(summary, ggplot2::aes(x = trial, y = value)) +
    ggplot2::geom_step(ggplot2::aes(y = best)) +
    ggplot2::geom_vline(xintercept = sum(summary$default) + manifest$random_startup_trials - 0.5,
      linetype = "dashed") +
    ggplot2::geom_point(ggplot2::aes(colour = family, shape = family)) +
    ggplot2::scale_colour_viridis_d(labels = FAMILY_LABELS) +
    ggplot2::scale_shape_discrete(labels = FAMILY_LABELS) +
    ggplot2::scale_y_continuous(labels = metric_text) +
    ggplot2::labs(x = "Číslo pokusu", y = "Macro-MaxF1", colour = "Rodina", shape = "Rodina") +
    theme_thesis()
}

stage_selection_text <- function(selection) {
  sprintf(paste0("Vítězem je %s (%s), pokus %d s cílem %s. Druhá rodina je %s (pokus %d). ",
    "Vítěz postupuje do matice a dál se ladí v každém foldu s rozpočtem tuning_trials podle D40."),
    FAMILY_LABELS[selection$winner$family], selection$winner$representation,
    selection$winner$trial, metric_text(selection$winner$value),
    FAMILY_LABELS[selection$second$family], selection$second$trial)
}

stage_ensemble_text <- function(phi, selection) {
  needed <- stage_ensemble_needed(phi, selection$second)
  if (is.na(needed)) return("Pro druhou rodinu není φ definováno; o tasku ensemblu nelze rozhodnout a výsledek patří do diskuse.")
  if (needed) return("φ vítěze a druhé rodiny je pod 0,5; má vzniknout task ensemblu.")
  "φ vítěze a druhé rodiny není pod 0,5; task ensemblu nevzniká, výsledek patří do diskuse."
}

stage_lsa_cap_text <- function(trials, config) {
  fits <- unique(trials[trials$family == "lightgbm_lsa", c("trial", "fold", "n_features")])
  if (!nrow(fits)) return("")
  sprintf("Strop slovníku LSA %s sloupců byl dosažen v %d z %d fitů; rozsah byl %s–%s.",
    space_value_text(config$lsa$max_features), sum(fits$n_features == config$lsa$max_features),
    nrow(fits), space_value_text(min(fits$n_features)), space_value_text(max(fits$n_features)))
}

stage_provenance_table <- function(manifest) {
  items <- c(
    "Dataset SHA-256" = manifest$dataset_sha256,
    "Split SHA-256" = manifest$split_sha256,
    "Seed a foldy původního splitu" = paste(manifest$split_seed, manifest$split_folds, sep = "; "),
    "Config SHA-256" = manifest$config_sha256,
    "Výsledky SHA-256" = manifest$results_sha256,
    "Predikce SHA-256" = manifest$predictions_sha256,
    "Seed ladění stupně" = as.character(manifest$seed),
    "Vývojové zprávy" = format_number(manifest$development_notes),
    "Vývojoví pacienti" = format_number(manifest$development_patients),
    "Předzpracování" = manifest$text_preprocessing,
    "Strip diagnostických sekcí" = if (is.null(manifest$diagnostic_strip)) "ne" else "ano",
    "Revize kódu ladění stupně" = manifest$environment$source$commit,
    "Rozpracovaný strom" = if (manifest$environment$source$dirty) "ano" else "ne",
    "Pokusy" = format_number(manifest$completed_trials),
    "Náhodné startovní pokusy" = format_number(manifest$random_startup_trials),
    "Sampler" = manifest$sampler, "Pruner" = manifest$pruner,
    "Vzorek zpráv" = format_number(manifest$sample_notes), "Vzorek pacientů" = format_number(manifest$sample_patients),
    "Doba běhu v minutách" = format_number(round(manifest$seconds / 60)),
    unlist(manifest$environment$packages)
  )
  provenance_table(as.list(items))
}

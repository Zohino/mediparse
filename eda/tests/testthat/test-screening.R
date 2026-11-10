source("../../R/screening.R")

screening_fixture_paths <- function(directory = file.path(CONTRACT_FIXTURES, "screening")) {
  list(
    results = file.path(directory, "classical.parquet"),
    predictions = file.path(directory, "classical_predictions.parquet"),
    manifest = file.path(directory, "classical_manifest.json")
  )
}

screening_test_con <- function(env = parent.frame()) {
  con <- DBI::dbConnect(duckdb::duckdb())
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE), envir = env)
  con
}

testthat::test_that("Kontrakt: oba parquety fixture lze přečíst", {
  paths <- screening_fixture_paths()
  testthat::expect_equal(nrow(read_screening_results(paths$results)), 250L)
  con <- screening_test_con()
  source <- screening_predictions_source(con, paths$predictions)
  testthat::expect_equal(DBI::dbGetQuery(con, paste("SELECT count(*) n FROM", source))$n, 6000)
})

testthat::test_that("A6 lineage kontroluje všechny tři otisky a jmenuje soubor", {
  paths <- screening_fixture_paths()
  config <- config_path("screening_classical.json")
  testthat::expect_silent(read_screening_manifest(paths, config))
  original <- jsonlite::read_json(paths$manifest)
  directory <- withr::local_tempdir()
  for (field in c("results_sha256", "predictions_sha256", "config_sha256")) {
    manifest <- original
    value <- manifest[[field]]
    substr(value, 1, 1) <- if (substr(value, 1, 1) == "0") "1" else "0"
    manifest[[field]] <- value
    changed <- paths
    changed$manifest <- file.path(directory, "manifest.json")
    jsonlite::write_json(manifest, changed$manifest, auto_unbox = TRUE)
    name <- switch(field,
      results_sha256 = "classical.parquet", predictions_sha256 = "classical_predictions.parquet",
      config_sha256 = "screening_classical.json"
    )
    testthat::expect_error(read_screening_manifest(changed, config), name, fixed = TRUE)
  }
})

testthat::test_that("A7 stráž odmítne predikce bez značky před čtením", {
  withr::local_envvar(CODEX_THREAD_ID = "screening-test")
  directory <- withr::local_tempdir()
  con <- screening_test_con()
  paths <- screening_fixture_paths(directory)
  testthat::expect_error(screening_predictions_source(con, paths$predictions), "synthetic.json", fixed = TRUE)
  testthat::expect_error(read_screening_manifest(paths, "config.json"), "synthetic.json", fixed = TRUE)
  testthat::expect_type(screening_predictions_source(con, screening_fixture_paths()$predictions), "character")
})

testthat::test_that("A8 mřížka musí být úplná a bez duplicit", {
  config <- jsonlite::read_json(config_path("screening_classical.json"))
  results <- read_screening_results(screening_fixture_paths()$results)
  testthat::expect_silent(check_screening_grid(results, config))
  testthat::expect_error(check_screening_grid(results[-1, ], config), "mřížku")
  testthat::expect_error(check_screening_grid(rbind(results, results[1, ]), config), "mřížku")
  wrong <- results
  wrong$fold[1] <- config$folds
  testthat::expect_error(check_screening_grid(wrong, config), "mřížku")
})
screening_example <- function(b_c = 0.79) {
  rows <- expand.grid(fold = 0:2, diagnosis = c("x", "y"), vectorizer = c("w", "c"), family = c("a", "b", "d"))
  rows$f1 <- mapply(function(family, vectorizer, fold) {
    values <- list(a = list(w = c(0.80, 0.78, 0.82), c = rep(0.70, 3)),
      b = list(w = rep(0.60, 3), c = rep(b_c, 3)),
      d = list(w = rep(0.78, 3), c = rep(0.50, 3)))
    values[[family]][[vectorizer]][fold + 1L]
  }, as.character(rows$family), as.character(rows$vectorizer), rows$fold)
  rows$family <- as.character(rows$family)
  rows$vectorizer <- as.character(rows$vectorizer)
  rows$pr_auc <- 0.9
  rows$converged <- TRUE
  rows$converged[1] <- FALSE
  rows
}

testthat::test_that("A1 vítěz a pásmo používají SE přes vnější foldy", {
  summary <- screening_summary(screening_example())
  best <- summary[summary$best, ]
  testthat::expect_equal(setNames(best$vectorizer, best$family), c(a = "w", b = "c", d = "w"))
  a <- best[best$family == "a", ]
  testthat::expect_equal(a$mean, 0.80)
  testthat::expect_equal(a$sd, 0.02)
  testthat::expect_equal(a$se, 0.011547, tolerance = 1e-6)
  testthat::expect_equal(a$nonconverged, 1)
  selection <- screening_selection(summary)
  testthat::expect_equal(selection$threshold, 0.788453, tolerance = 1e-6)
  testthat::expect_equal(selection$winner$family, "a")
  testthat::expect_equal(selection$band$family, c("a", "b"))
  testthat::expect_equal(selection$comparison$family, "b")
  testthat::expect_false(selection$fallback)
})

testthat::test_that("A2 jediné rodině v pásmu se přiřadí druhá nejlepší", {
  selection <- screening_selection(screening_summary(screening_example(0.785)))
  testthat::expect_equal(selection$band$family, "a")
  testthat::expect_equal(selection$comparison$family, "b")
  testthat::expect_true(selection$fallback)
})

testthat::test_that("A3 shodu rodin i vektorizérů rozhoduje jméno vzestupně", {
  rows <- screening_example()
  rows <- rows[rows$family != "d", ]
  rows$f1 <- 0.8
  summary <- screening_summary(rows[nrow(rows):1, ])
  testthat::expect_equal(summary$vectorizer[summary$best], c("c", "c"))
  testthat::expect_equal(screening_selection(summary)$winner$family, "a")
})

phi_example <- function(counts, diagnosis) {
  correct_a <- rep(c(TRUE, TRUE, FALSE, FALSE), counts)
  correct_b <- rep(c(TRUE, FALSE, TRUE, FALSE), counts)
  labels <- rep(c(TRUE, FALSE), length.out = length(correct_a))
  rows <- data.frame(fold = 0, diagnosis = diagnosis, note_id = paste0("synthetic-", seq_along(labels)),
    label = labels, vectorizer = "w")
  a <- transform(rows, family = "a", predicted = ifelse(correct_a, labels, !labels))
  b <- transform(rows, family = "b", predicted = ifelse(correct_b, labels, !labels))
  rbind(a, b)
}

phi_models <- function() list(winner = data.frame(family = "a", vectorizer = "w"),
  others = data.frame(family = "b", vectorizer = "c"))

phi_from_rows <- function(rows, env = parent.frame()) {
  con <- screening_test_con(env)
  rows$vectorizer[rows$family == "b"] <- "c"
  DBI::dbWriteTable(con, "predictions", rows)
  models <- phi_models()
  error_correlation(con, "predictions", models$winner, models$others)
}

testthat::test_that("A4 φ je průměr správnosti obou tříd po buňkách", {
  rows <- rbind(phi_example(c(3, 1, 1, 3), "x"), phi_example(c(2, 2, 2, 2), "y"))
  phi <- phi_from_rows(rows)
  testthat::expect_equal(phi$phi, 0.25)
  testthat::expect_equal(phi$defined, 2L)
  testthat::expect_equal(phi$cells, 2L)
  testthat::expect_true(ensemble_needed(phi, phi_models()$others))
  descriptive <- data.frame(family = "d", phi = 0)
  testthat::expect_false(ensemble_needed(descriptive, phi_models()$others) %in% TRUE)
})

testthat::test_that("A5 hranice 0,5 je ostrá a nedefinované buňky se vynechají", {
  x <- phi_example(c(3, 1, 1, 3), "x")
  phi <- phi_from_rows(x)
  testthat::expect_equal(phi$phi, 0.5)
  testthat::expect_false(ensemble_needed(phi, phi_models()$others))
  undefined <- phi_example(c(4, 0, 0, 0), "z")
  mixed <- phi_from_rows(rbind(x, undefined))
  testthat::expect_equal(mixed$phi, 0.5)
  testthat::expect_equal(mixed$defined, 1L)
  testthat::expect_equal(mixed$cells, 2L)
  testthat::expect_false(ensemble_needed(mixed, phi_models()$others))
  unavailable <- phi_from_rows(undefined)
  testthat::expect_true(is.na(unavailable$phi))
  testthat::expect_equal(unavailable$defined, 0L)
  testthat::expect_true(is.na(ensemble_needed(unavailable, phi_models()$others)))
})

testthat::test_that("A8 párování odmítá chybějící zprávu, duplicitu i rozdílný label", {
  rows <- phi_example(c(3, 1, 1, 3), "x")
  testthat::expect_error(phi_from_rows(rows[-1, ]), "nepárují")
  testthat::expect_error(phi_from_rows(rbind(rows, rows[1, ])), "nepárují")
  rows$label[1] <- !rows$label[1]
  testthat::expect_error(phi_from_rows(rows), "nepárují")
})

testthat::test_that("Ensemble vyžaduje alespoň jednu rozhodující dvojici pod hranicí", {
  comparison <- data.frame(family = c("b", "d"))
  phi <- data.frame(family = c("b", "d"), phi = c(0.3, 0.7))
  testthat::expect_true(ensemble_needed(phi, comparison))
  phi$phi <- c(NA_real_, 0.7)
  testthat::expect_false(ensemble_needed(phi, comparison))
  phi$phi <- c(NA_real_, 0.3)
  testthat::expect_true(ensemble_needed(phi, comparison))
  phi$phi <- c(NA_real_, NA_real_)
  testthat::expect_true(is.na(ensemble_needed(phi, comparison)))
})

testthat::test_that("Macro-F1 i PR-AUC jsou průměry různých diagnóz", {
  rows <- expand.grid(fold = 0:2, diagnosis = c("x", "y"), family = c("a", "b"),
    vectorizer = "w", stringsAsFactors = FALSE)
  rows$f1 <- ifelse(rows$diagnosis == "x", 0.6, 0.8)
  rows$f1[rows$family == "b"] <- 0.75
  rows$pr_auc <- ifelse(rows$diagnosis == "x", 0.5, 0.9)
  rows$converged <- TRUE
  summary <- screening_summary(rows)
  testthat::expect_equal(summary$mean, c(0.7, 0.75))
  testthat::expect_equal(summary$pr_auc, c(0.7, 0.7))
  testthat::expect_equal(screening_selection(summary)$winner$family, "b")
})

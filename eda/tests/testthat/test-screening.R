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

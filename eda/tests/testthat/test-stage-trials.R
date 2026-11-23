stage_fixture_paths <- function(stage = "classical", directory = file.path(CONTRACT_FIXTURES, "stage_trials")) {
  withr::with_envvar(c(MEDIPARSE_SCREENING = directory,
    MEDIPARSE_SCREENING_PREDICTIONS = directory), stage_trial_paths(stage))
}

stage_test_con <- function(env = parent.frame()) {
  con <- DBI::dbConnect(duckdb::duckdb())
  withr::defer(DBI::dbDisconnect(con, shutdown = TRUE), envir = env)
  con
}

stage_config_files <- function() list.files(dirname(config_path("stage_classical.json")),
  "^stage_.*\\.json$", full.names = TRUE)

testthat::test_that("B1 všechny rodiny configů mají popisky", {
  for (path in stage_config_files()) {
    config <- jsonlite::read_json(path)
    testthat::expect_true(all(vapply(config$families, `[[`, character(1), "family") %in% names(FAMILY_LABELS)))
  }
})

testthat::test_that("B2 kontrakty všech stupňů jsou úplné a souhlasí s manifesty", {
  con <- stage_test_con()
  rows <- 0
  for (path in stage_config_files()) {
    stage <- sub("\\.json$", "", sub("^stage_", "", basename(path)))
    paths <- stage_fixture_paths(stage)
    config <- jsonlite::read_json(path)
    manifest <- read_stage_manifest(paths, path)
    trials <- read_stage_trials(paths$results)
    testthat::expect_silent(check_stage_trials(trials, manifest, config, read_candidates(candidates_path())))
    testthat::expect_equal(nrow(trials), config$trials * config$folds * 5)
    testthat::expect_equal(stage_selection(stage_trial_summary(trials), manifest)$winner$trial, manifest$best_trial)
    source <- stage_predictions_source(con, paths$predictions)
    rows <- rows + DBI::dbGetQuery(con, paste("SELECT count(*) n FROM", source))$n
  }
  testthat::expect_equal(rows, 91200)
  testthat::expect_equal(rows,
    jsonlite::read_json(file.path(CONTRACT_FIXTURES, "stage_trials", "synthetic.json"))$rows)
})

testthat::test_that("A6 lineage kontroluje všechny tři otisky a jmenuje soubor", {
  paths <- stage_fixture_paths()
  config <- config_path("stage_classical.json")
  testthat::expect_silent(read_stage_manifest(paths, config))
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
      results_sha256 = "classical_trials.parquet", predictions_sha256 = "classical_trial_predictions.parquet",
      config_sha256 = "stage_classical.json"
    )
    testthat::expect_error(read_stage_manifest(changed, config), name, fixed = TRUE)
  }
})

testthat::test_that("A7 stráž odmítne predikce bez značky před čtením", {
  withr::local_envvar(CODEX_THREAD_ID = "stage-test")
  directory <- withr::local_tempdir()
  con <- stage_test_con()
  paths <- stage_fixture_paths(directory = directory)
  testthat::expect_error(stage_predictions_source(con, paths$predictions), "synthetic.json", fixed = TRUE)
  testthat::expect_error(read_stage_manifest(paths, "config.json"), "synthetic.json", fixed = TRUE)
  testthat::expect_type(stage_predictions_source(con, stage_fixture_paths()$predictions), "character")
})

phi_example <- function(counts, diagnosis) {
  correct_a <- rep(c(TRUE, TRUE, FALSE, FALSE), counts)
  correct_b <- rep(c(TRUE, FALSE, TRUE, FALSE), counts)
  labels <- rep(c(TRUE, FALSE), length.out = length(correct_a))
  rows <- data.frame(fold = 0, diagnosis = diagnosis, note_id = paste0("synthetic-", seq_along(labels)),
    label = labels)
  a <- transform(rows, trial = 0, family = "a", predicted = ifelse(correct_a, labels, !labels))
  b <- transform(rows, trial = 1, family = "b", predicted = ifelse(correct_b, labels, !labels))
  rbind(a, b)
}

phi_models <- function() list(winner = data.frame(family = "a", trial = 0),
  others = data.frame(family = "b", trial = 1))

phi_from_rows <- function(rows, env = parent.frame()) {
  con <- stage_test_con(env)
  DBI::dbWriteTable(con, "predictions", rows)
  models <- phi_models()
  stage_error_correlation(con, "predictions", models$winner, models$others)
}

testthat::test_that("A4 φ je průměr správnosti obou tříd po buňkách", {
  rows <- rbind(phi_example(c(3, 1, 1, 3), "x"), phi_example(c(2, 2, 2, 2), "y"))
  phi <- phi_from_rows(rows)
  testthat::expect_equal(phi$phi, 0.25)
  testthat::expect_equal(phi$defined, 2L)
  testthat::expect_equal(phi$cells, 2L)
  testthat::expect_true(stage_ensemble_needed(phi, second = phi_models()$others))
  descriptive <- data.frame(family = "d", trial = 2, phi = 0)
  testthat::expect_false(stage_ensemble_needed(descriptive, phi_models()$others) %in% TRUE)
})

testthat::test_that("A5 hranice 0,5 je ostrá a nedefinované buňky se vynechají", {
  x <- phi_example(c(3, 1, 1, 3), "x")
  phi <- phi_from_rows(x)
  testthat::expect_equal(phi$phi, 0.5)
  testthat::expect_false(stage_ensemble_needed(phi, phi_models()$others))
  undefined <- phi_example(c(4, 0, 0, 0), "z")
  mixed <- phi_from_rows(rbind(x, undefined))
  testthat::expect_equal(mixed$phi, 0.5)
  testthat::expect_equal(mixed$defined, 1L)
  testthat::expect_equal(mixed$cells, 2L)
  testthat::expect_false(stage_ensemble_needed(mixed, phi_models()$others))
  unavailable <- phi_from_rows(undefined)
  testthat::expect_true(is.na(unavailable$phi))
  testthat::expect_equal(unavailable$defined, 0L)
  testthat::expect_true(is.na(stage_ensemble_needed(unavailable, phi_models()$others)))
})

testthat::test_that("A8 párování odmítá chybějící zprávu, duplicitu i rozdílný label", {
  rows <- phi_example(c(3, 1, 1, 3), "x")
  testthat::expect_error(phi_from_rows(rows[-1, ]), "nepárují")
  testthat::expect_error(phi_from_rows(rbind(rows, rows[1, ])), "nepárují")
  rows$label[1] <- !rows$label[1]
  testthat::expect_error(phi_from_rows(rows), "nepárují")
})

testthat::test_that("Párování odmítne NULL v pravdě i predikci", {
  for (column in c("label", "predicted")) {
    rows <- phi_example(c(3, 1, 1, 3), "x")
    rows[1, column] <- NA
    testthat::expect_error(phi_from_rows(rows), "nepárují")
  }
})

testthat::test_that("Čtení predikcí zapojuje deklarovaný typ BOOLEAN", {
  con <- stage_test_con()
  source <- stage_predictions_source(con, stage_fixture_paths()$predictions)
  directory <- withr::local_tempdir()
  file.copy(file.path(CONTRACT_FIXTURES, "stage_trials", "synthetic.json"), directory)
  path <- file.path(directory, "wrong-predictions.parquet")
  DBI::dbExecute(con, sprintf(
    "COPY (SELECT * REPLACE (CAST(predicted AS VARCHAR) AS predicted) FROM %s) TO %s (FORMAT parquet)",
    source, DBI::dbQuoteString(con, path)
  ))
  testthat::expect_error(stage_predictions_source(con, path), "sloupec predicted, očekávaný typ BOOLEAN, skutečný typ VARCHAR", fixed = TRUE)
})

stage_example <- function() {
  rows <- expand.grid(fold = 0:2, diagnosis = c("x", "y"), stringsAsFactors = FALSE)
  rows$max_f1 <- c(0.6, 0.7, 0.8, 0.8, 0.9, 1.0)
  transform(rows, trial = 0, branch = "w/a", representation = "w", family = "a",
    default = TRUE, params = "{}", value = 0.8, seconds = 10, converged = TRUE)
}

testthat::test_that("A1 a A3 souhrn ověřuje cíl a výběrovou SD foldů", {
  rows <- stage_example()
  summary <- stage_trial_summary(rows)
  testthat::expect_equal(summary$sd, 0.1)
  rows$value <- 0.8 + 1e-6
  testthat::expect_error(stage_trial_summary(rows), "cíl")
  rows$value <- 0.8 + 1e-12
  testthat::expect_silent(stage_trial_summary(rows))
})

testthat::test_that("A2 vítěze a shody určuje cíl a číslo pokusu", {
  summary <- data.frame(trial = 0:5, family = c("a", "b", "a", "b", "c", "c"),
    value = c(0.80, 0.78, 0.82, 0.82, 0.70, 0.70))
  selection <- stage_selection(summary, list(best_trial = 2))
  testthat::expect_equal(selection$winner$trial, 2L)
  testthat::expect_equal(selection$second$trial, 3L)
  testthat::expect_equal(selection$family_best$trial, c(2L, 3L, 4L))
  testthat::expect_equal(selection$others$trial, c(3L, 4L))
  testthat::expect_equal(selection$family_best$family, c("a", "b", "c"))
  testthat::expect_error(stage_selection(summary, list(best_trial = 3)), "manifest")
  summary$value <- 0.8
  testthat::expect_equal(stage_selection(summary, list(best_trial = 0))$winner$trial, 0L)
  testthat::expect_equal(stage_selection(summary, list(best_trial = 0))$second$trial, 1L)
  testthat::expect_error(stage_selection(summary[summary$family == "a", ], list(best_trial = 0)), "jedinou rodinu")
})

testthat::test_that("Shoda mezi rodinami nezávisí na locale ani jméně", {
  original <- Sys.getlocale("LC_COLLATE")
  withr::defer(Sys.setlocale("LC_COLLATE", original))
  for (locale in c(original, "C.UTF-8")) {
    Sys.setlocale("LC_COLLATE", locale)
    summary <- data.frame(trial = 0:1, family = c("á", "z"), value = 0.8)
    selection <- stage_selection(summary[2:1, ], list(best_trial = 0))
    testthat::expect_equal(selection$winner$trial, 0L)
    testthat::expect_equal(selection$second$family, "z")
  }
})

testthat::test_that("A8b tabulka vyžaduje přesné pokusy, buňky a konstantní metadata", {
  paths <- stage_fixture_paths()
  rows <- read_stage_trials(paths$results)
  manifest <- read_stage_manifest(paths, config_path("stage_classical.json"))
  config <- jsonlite::read_json(config_path("stage_classical.json"))
  candidates <- read_candidates(candidates_path())
  check <- function(rows) check_stage_trials(rows, manifest, config, candidates)
  testthat::expect_silent(check(rows))
  testthat::expect_error(check_stage_trials(rows, manifest, config,
    candidates[setdiff(names(candidates), "copd")]), "kandidát")
  testthat::expect_error(check(rows[-1, ]), "není úplná nebo konzistentní")
  testthat::expect_error(check(rbind(rows, rows[1, ])), "není úplná nebo konzistentní")
  testthat::expect_error(check(rows[rows$trial != 1, ]), "není úplná nebo konzistentní")
  for (field in c("trials", "completed_trials")) {
    wrong <- manifest
    wrong[[field]] <- manifest[[field]] + 1
    testthat::expect_error(check_stage_trials(rows, wrong, config, candidates),
      "není úplná nebo konzistentní")
  }
  mutations <- list(
    function(x) { x$fold[1] <- config$folds; x },
    function(x) { x$default[x$trial == 0] <- FALSE; x$default[x$trial == 10] <- TRUE; x },
    function(x) {
      x$representation[x$trial == 10] <- "new"
      x$branch[x$trial == 10] <- "new/complement_nb"
      x
    },
    function(x) { x$branch[x$trial == 10] <- "word/logreg"; x },
    function(x) { x$value[1] <- 0; x },
    function(x) { x$params[1] <- "{}"; x },
    function(x) { x$diagnosis[1] <- "heart_failure"; x }
  )
  for (mutate in mutations) {
    testthat::expect_error(check(mutate(rows)), "není úplná nebo konzistentní")
  }
  unknown <- rows
  unknown$family[unknown$trial == 10] <- "unknown"
  testthat::expect_error(check(unknown), "Neznámá rodina")
})

testthat::test_that("A9 strop LSA počítá fity místo diagnóz", {
  rows <- expand.grid(trial = 0:1, fold = 0:1, diagnosis = c("x", "y"))
  rows$family <- "lightgbm_lsa"
  rows$n_features <- rep(c(100, 50, 100, 100), 2)
  config <- list(lsa = list(max_features = 100))
  testthat::expect_match(stage_lsa_cap_text(rows, config), "3 z 4", fixed = TRUE)
  rows$family <- "logreg"
  testthat::expect_identical(stage_lsa_cap_text(rows, config), "")
})

testthat::test_that("A10 prostor ukazuje rozsahy, výchozí hodnoty a aktivní parametry", {
  config <- jsonlite::fromJSON('{
    "representations": [{"name":"word","space":{"params":[
      {"kind":"int","name":"min_df","low":1,"high":20,"log":true},
      {"kind":"categorical","name":"sublinear_tf","choices":[true,false]}]},
      "default":{"min_df":1,"sublinear_tf":true}}],
    "families":[{"family":"nb_svm","space":{"params":[
      {"kind":"float","name":"C","low":0.001,"high":100,"log":true}]},"default":{"C":1}},
      {"family":"logreg","space":{"params":[
      {"kind":"float","name":"C","low":0.001,"high":1000,"log":true}]},"default":{"C":1}}]
  }', simplifyVector = FALSE)
  winner <- list(branch = "word/nb_svm", word.min_df = 3, word.sublinear_tf = FALSE, nb_svm.C = 8.5)
  second <- list(branch = "word/logreg", word.min_df = 2, word.sublinear_tf = TRUE, logreg.C = 3)
  table <- stage_space_table(config, winner, second)
  testthat::expect_equal(table$složka, c("word", "word", "nb_svm", "logreg"))
  testthat::expect_equal(table$parametr, c("min_df", "sublinear_tf", "C", "C"))
  testthat::expect_equal(table$rozsah, c("1–20 (log)", "ano, ne", "0,001–100 (log)", "0,001–1 000 (log)"))
  testthat::expect_equal(table$výchozí, c("1", "ano", "1", "1"))
  testthat::expect_equal(table$nejlepší, c("3", "ne", "8,5", "–"))
  testthat::expect_equal(table$druhá, c("2", "ano", "–", "3"))
  testthat::expect_identical(space_value_text(8.5), "8,5")
  testthat::expect_identical(space_value_text(1000), paste0("1", NBSP, "000"))
  testthat::expect_identical(space_value_text(TRUE), "ano")
  testthat::expect_identical(space_value_text(NULL), "žádná")
  testthat::expect_identical(space_value_text("balanced"), "balanced")
})

testthat::test_that("Nedefinované rozhodující φ patří do diskuse", {
  phi <- data.frame(trial = 1, phi = NA_real_)
  selection <- list(second = data.frame(trial = 1))
  testthat::expect_match(stage_ensemble_text(phi, selection), "diskuse")
})

testthat::test_that("Provenance formátuje počty a dobu běhu v minutách", {
  manifest <- jsonlite::read_json(stage_fixture_paths()$manifest, simplifyVector = TRUE)
  manifest$completed_trials <- 1234
  manifest$random_startup_trials <- 2345
  manifest$sample_notes <- 3456
  manifest$sample_patients <- 4567
  manifest$seconds <- 21534.8123
  table <- stage_provenance_table(manifest)
  items <- c("Pokusy", "Náhodné startovní pokusy", "Vzorek zpráv",
    "Vzorek pacientů", "Doba běhu v minutách")
  testthat::expect_equal(table$Hodnota[match(items, table$Položka)],
    c(paste0("1", NBSP, "234"), paste0("2", NBSP, "345"),
      paste0("3", NBSP, "456"), paste0("4", NBSP, "567"), "359"))
})

testthat::test_that("Souhrn zachová parametry pokusu pro tabulku prostoru", {
  rows <- stage_example()
  rows$params <- '{"branch":"word/nb_svm","word.min_df":3,"nb_svm.C":8.5}'
  winner <- stage_trial_summary(rows)
  testthat::expect_equal(jsonlite::fromJSON(winner$params),
    list(branch = "word/nb_svm", word.min_df = 3, nb_svm.C = 8.5))
})


testthat::test_that("Provenance vypíše předzpracování po reprezentacích", {
  manifest <- jsonlite::read_json(stage_fixture_paths()$manifest, simplifyVector = TRUE)
  table <- stage_provenance_table(manifest)
  testthat::expect_equal(table$Hodnota[table$Položka == "Předzpracování"],
    "word: dictionary; char_wb: dictionary")
})

testthat::test_that("B3 meze respektují škálu, práh a aktivní složky", {
  config <- jsonlite::fromJSON('{
    "representations":[{"name":"w","space":{"params":[]}}],
    "families":[{"family":"fasttext","space":{"params":[
      {"name":"lr","kind":"float","low":0.01,"high":5,"log":true},
      {"name":"epoch","kind":"int","low":5,"high":100,"log":true},
      {"name":"word_ngrams","kind":"int","low":1,"high":2}]}},
      {"family":"multirescnn","space":{"params":[
      {"name":"dropout","kind":"float","low":0,"high":0.5},
      {"name":"filter_maps","kind":"int","low":25,"high":100,"log":true}]}}]
  }', simplifyVector = FALSE)
  rows <- data.frame(family = c("fasttext", "multirescnn"), branch = c("w/fasttext", "w/multirescnn"),
    params = c('{"fasttext.lr":4.8,"fasttext.epoch":100,"fasttext.word_ngrams":2}',
      '{"multirescnn.dropout":0.02,"multirescnn.filter_maps":50}'))
  boundary <- stage_boundary_params(config, rows)
  testthat::expect_equal(boundary, data.frame(rodina = c("fastText", "fastText", "MultiResCNN"),
    složka = c("fasttext", "fasttext", "multirescnn"), parametr = c("lr", "epoch", "dropout"),
    hodnota = c("4,8", "100", "0,02"), rozsah = c("0,01–5 (log)", "5–100 (log)", "0–0,5"),
    mez = c("horní", "horní", "dolní")))
  text <- stage_boundary_text(boundary)
  for (value in c("lr", "0,01–5 (log)", "horní mez", "dolní mez")) {
    testthat::expect_match(text, value, fixed = TRUE)
  }
  rows$params <- c('{"fasttext.lr":1.0,"fasttext.epoch":50,"fasttext.word_ngrams":1}',
    '{"multirescnn.dropout":0.03,"multirescnn.filter_maps":50}')
  testthat::expect_equal(nrow(stage_boundary_params(config, rows)), 0L)
  testthat::expect_match(stage_boundary_text(stage_boundary_params(config, rows)), "žádný")
  for (value in c(0.025, 0.475)) {
    rows$params[2] <- sprintf('{"multirescnn.dropout":%s,"multirescnn.filter_maps":50}', value)
    testthat::expect_equal(stage_boundary_params(config, rows)$mez, if (value == 0.025) "dolní" else "horní")
  }
  config <- jsonlite::fromJSON('{
    "representations":[{"name":"word2vec","space":{"params":[
      {"name":"vector_size","kind":"categorical","choices":[100,200]},
      {"name":"window","kind":"int","low":2,"high":10}]}}],
    "families":[{"family":"logreg","space":{"params":[
      {"name":"C","kind":"float","low":0.001,"high":1000,"log":true},
      {"name":"class_weight","kind":"categorical","choices":[null,"balanced"]}]}}]
  }', simplifyVector = FALSE)
  rows <- data.frame(family = "logreg", branch = "word2vec/logreg",
    params = '{"word2vec.window":10,"logreg.C":1}')
  testthat::expect_equal(stage_boundary_params(config, rows), data.frame(rodina = "Logistická regrese",
    složka = "word2vec", parametr = "window", hodnota = "10", rozsah = "2–10", mez = "horní"))
})

testthat::test_that("B4 reprezentace řadí cíl a shody rozhodne číslo pokusu", {
  summary <- data.frame(trial = 0:3, representation = c("a", "b", "a", "b"),
    value = c(0.80, 0.82, 0.85, 0.82), sd = c(0.02, 0.01, 0.01, 0.03))
  testthat::expect_equal(stage_representation_table(summary), data.frame(reprezentace = c("a", "b"),
    pokus = c(2L, 1L), macro = c("0,850 ± 0,010", "0,820 ± 0,010"), pokusy = c(2L, 2L)))
  summary$representation <- c("a", "b", "a", "a")
  testthat::expect_equal(stage_representation_table(summary)$pokusy, c(3L, 1L))
})

testthat::test_that("B5 kandidáti postupují podle D41 do S39j4", {
  selection <- list(winner = data.frame(family = "fasttext", representation = "word-tokens", trial = 7, value = 0.8123),
    second = data.frame(family = "multirescnn", representation = "word-tokens", trial = 3))
  text <- stage_selection_text(selection)
  for (value in c("fastText (word-tokens, pokus 7)", "MultiResCNN (word-tokens, pokus 3)", "D41", "S39j4")) {
    testthat::expect_match(text, value, fixed = TRUE)
  }
  testthat::expect_false(grepl("každém foldu", text, fixed = TRUE))
  selection <- list(winner = data.frame(family = "logreg", representation = "word", trial = 7),
    second = data.frame(family = "nb_svm", representation = "char_wb", trial = 3))
  testthat::expect_match(stage_selection_text(selection), "NB-SVM (char_wb, pokus 3)", fixed = TRUE)
})

testthat::test_that("B6 provenance uvádí procesy, encodery a GPU", {
  manifest <- jsonlite::read_json(stage_fixture_paths()$manifest, simplifyVector = TRUE)
  table <- stage_provenance_table(manifest)
  testthat::expect_equal(table$Hodnota[match(c("Procesy (--jobs)", "GPU"), table$Položka)], c("1", "žádná"))
  manifest$hardware$gpus <- data.frame(name = c("A", "B"))
  table <- stage_provenance_table(manifest)
  testthat::expect_equal(table$Hodnota[table$Položka == "GPU"], "A; B")
  manifest <- jsonlite::read_json(stage_fixture_paths("embeddings")$manifest, simplifyVector = TRUE)
  table <- stage_provenance_table(manifest)
  names <- c("mmbert-base-mean", "mmbert-base-mean-512", "mmbert-base-mean-dictionary")
  testthat::expect_equal(table$Hodnota[match(paste("Encoder", names), table$Položka)],
    paste0("jhu-clsp/mmBERT-base@c5955035435e2bf121cde7f3c8863ef52ff35d82, okno ",
      c(8192, 512, 8192), ", bfloat16, cuda"))
})

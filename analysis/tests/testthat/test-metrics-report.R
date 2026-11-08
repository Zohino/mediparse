CANDIDATES_FIXTURE <- "../../../config/diagnosis_candidates.json"

write_metrics_variant <- function(path, replacement) {
  write_contract_parquet(sprintf(
    "SELECT * REPLACE (%s) FROM read_parquet('%s')", replacement, METRICS_FIXTURE
  ), path)
}

previous_metrics_report <- function(dir) {
  dir.create(dir)
  paths <- file.path(dir, c("metrics.tex", "metrics.pdf", "metrics.tex.meta.json", "metrics.pdf.meta.json"))
  for (path in paths) writeLines("předchozí výstup", path)
  paths
}

test_that("tabulka metrik má česká čísla, přesné počty a metriky jen nad prahem stability", {
  dir <- withr::local_tempdir()
  report_with_chain(METRICS_FIXTURE, metrics_chain(dir), dir, CANDIDATES_FIXTURE)

  expect_identical(readLines(file.path(dir, "metrics.tex"), encoding = "UTF-8"), c(
    "\\begin{tabular}{lrrrrrrrr}",
    "\\toprule",
    "Diagnóza & Zprávy & Pozitivní & Přesnost & Úplnost & F1 & Správnost & ROC AUC & PR AUC\\\\",
    "\\midrule",
    "Diabetes mellitus & 40 & 12 & 0,500 & 0,250 & 0,750 & 0,875 & 0,900 & 0,800\\\\",
    "CHOPN & 40 & 10 & 0,600 & 0,300 & 0,400 & 0,850 & 0,700 & 0,550\\\\",
    "Chronické onemocnění ledvin & 40 & 8 & – & – & – & – & – & –\\\\",
    "Pneumonie & 40 & 20 & 1,000 & 0,000 & 0,000 & 0,500 & 0,500 & 0,500\\\\",
    "Fibrilace síní & 40 & 4 & – & – & – & – & – & –\\\\",
    "\\bottomrule",
    "\\end{tabular}"
  ))
})

test_that("tabulka zachová pět pozitivních zpráv a skryje jen nestabilní metriky", {
  metrics <- read_metrics(METRICS_FIXTURE)
  metrics$test_positives[1] <- 5
  labels <- diagnosis_labels(metrics$diagnosis, CANDIDATES_FIXTURE)

  table <- metrics_table(metrics, labels)

  expect_identical(table$Pozitivní[1], "5")
  expect_identical(
    unname(unlist(table[1, c("Přesnost", "Úplnost", "F1", "Správnost", "ROC AUC", "PR AUC")])),
    rep("–", 6)
  )
})

test_that("graf nese jen F1 a AUC zveřejnitelných diagnóz", {
  metrics <- read_metrics(METRICS_FIXTURE)
  labels <- diagnosis_labels(metrics$diagnosis, CANDIDATES_FIXTURE)
  actual <- metrics_figure(metrics, labels)$data
  expected <- data.frame(
    diagnoza = rep(c("Diabetes mellitus", "CHOPN", "Pneumonie"), each = 3),
    metrika = rep(c("F1", "ROC AUC", "PR AUC"), 3),
    hodnota = c(0.75, 0.9, 0.8, 0.4, 0.7, 0.55, 0, 0.5, 0.5)
  )
  expect_s3_class(actual$diagnoza, "factor")
  expect_identical(levels(actual$diagnoza), c("Pneumonie", "CHOPN", "Diabetes mellitus"))
  expect_identical(levels(actual$metrika), c("F1", "ROC AUC", "PR AUC"))
  expect_type(actual$hodnota, "double")
  triples <- function(df) apply(df, 1, paste, collapse = "|")
  expect_equal(nrow(actual), 9L)
  expect_setequal(triples(actual), triples(expected))
})

test_that("PDF metrik má šířku sazby a vložené Libertinus Serif", {
  dir <- withr::local_tempdir()
  report_with_chain(METRICS_FIXTURE, metrics_chain(dir), dir, CANDIDATES_FIXTURE)
  path <- file.path(dir, "metrics.pdf")
  info <- system2("pdfinfo", shQuote(path), stdout = TRUE)
  box <- grep("Page size", info, value = TRUE)
  size <- as.numeric(strsplit(trimws(sub("Page size:", "", box)), " +")[[1]][c(1, 3)])

  expect_equal(size, c(16.5, 7.5) / 2.54 * 72, tolerance = 1 / 467.7)
  expect_true(any(grepl("LibertinusSerif", system2("pdffonts", shQuote(path), stdout = TRUE))))
})

test_that("oba výstupy nesou ověřitelný odkaz na vstup a manifest běhu", {
  dir <- withr::local_tempdir()
  metrics <- write_metrics_variant(file.path(dir, "beh-42.parquet"), "'beh-42' AS row_id")
  chain <- metrics_chain(dir, "beh-42", metrics_path = metrics)
  out <- file.path(dir, "výstupy")
  previous_metrics_report(out)
  result <- withVisible(report_with_chain(metrics, chain, out, CANDIDATES_FIXTURE))
  files <- c("metrics.tex", "metrics.pdf", "metrics.tex.meta.json", "metrics.pdf.meta.json")
  expect_false(result$visible)
  expect_setequal(result$value, file.path(out, files))
  expect_setequal(list.files(out), files)
  for (output in c("metrics.tex", "metrics.pdf")) {
    sidecar <- jsonlite::fromJSON(file.path(out, paste0(output, ".meta.json")))
    expect_identical(sidecar, list(
      input = list(path = metrics, sha256 = coreutils_sha256(metrics)),
      evaluation_manifest = list(path = chain$evaluation, sha256 = coreutils_sha256(chain$evaluation)),
      run_manifest = list(path = chain$manifest, sha256 = coreutils_sha256(chain$manifest), row_id = "beh-42")
    ))
  }
})

test_that("graf jediné zveřejnitelné diagnózy má výšku 4,5 cm", {
  dir <- withr::local_tempdir()
  metrics <- write_metrics_variant(file.path(dir, "jedna-diagnoza.parquet"),
    "CASE WHEN diagnosis = 'diabetes' THEN test_positives ELSE 5::BIGINT END AS test_positives")
  report_with_chain(metrics, metrics_chain(dir, metrics_path = metrics), dir, CANDIDATES_FIXTURE)
  info <- system2("pdfinfo", shQuote(file.path(dir, "metrics.pdf")), stdout = TRUE)
  box <- grep("Page size", info, value = TRUE)
  height <- as.numeric(strsplit(trimws(sub("Page size:", "", box)), " +")[[1]][3])

  # Cairo ukládá rozměr v celých bodech; tolerance je jeden bod jako u šířky.
  expect_equal(height, 4.5 / 2.54 * 72, tolerance = 1 / (4.5 / 2.54 * 72))
})

test_that("cizí manifest se odmítne před zápisem a chyba jmenuje oba běhy", {
  dir <- withr::local_tempdir()
  out <- file.path(dir, "out")
  expect_error(
    report_with_chain(METRICS_FIXTURE, metrics_chain(dir, "jiny"), out, CANDIDATES_FIXTURE),
    "smoketest.*jiny"
  )
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
})

test_that("manifest musí mít row_id jako jeden řetězec", {
  dir <- withr::local_tempdir()
  out <- file.path(dir, "out")
  for (row_id in list(NULL, 1, c("smoketest", "jiny"), list("smoketest"), "")) {
    expect_error(
      report_with_chain(METRICS_FIXTURE, metrics_chain(dir, row_id), out, CANDIDATES_FIXTURE),
      "manifest běhu.json.*row_id.*řetězec"
    )
    expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
  }
})

test_that("více běhů v metrikách se odmítne před zápisem", {
  dir <- withr::local_tempdir()
  metrics <- write_metrics_variant(file.path(dir, "dva-behy.parquet"),
    "CASE WHEN diagnosis = 'diabetes' THEN 'jiny' ELSE row_id END AS row_id")
  out <- file.path(dir, "out")
  expect_error(
    report_with_chain(metrics, metrics_chain(dir, metrics_path = metrics), out, CANDIDATES_FIXTURE),
    "dva-behy.parquet.*právě jeden row_id"
  )
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
})

test_that("chybějící nebo prázdný český název diagnózy se odmítne před zápisem", {
  dir <- withr::local_tempdir()
  candidates <- jsonlite::fromJSON(CANDIDATES_FIXTURE, simplifyVector = FALSE)
  config <- file.path(dir, "candidates.json")
  out <- file.path(dir, "out")
  for (label in list(NULL, "")) {
    candidates$candidates$ckd$label_cs <- label
    jsonlite::write_json(candidates, config, auto_unbox = TRUE)
    expect_error(
      report_with_chain(METRICS_FIXTURE, metrics_chain(dir), out, config),
      "candidates.json.*ckd.*label_cs"
    )
    expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
  }
})

test_that("duplicitní diagnóza se odmítne českou chybou před zápisem", {
  dir <- withr::local_tempdir()
  metrics <- write_contract_parquet(sprintf(
    paste("SELECT * FROM read_parquet('%s') UNION ALL",
          "SELECT * FROM read_parquet('%s') WHERE diagnosis = 'diabetes'"),
    METRICS_FIXTURE, METRICS_FIXTURE
  ), file.path(dir, "duplicitni.parquet"))
  out <- file.path(dir, "out")
  expect_error(
    report_with_chain(metrics, metrics_chain(dir, metrics_path = metrics), out, CANDIDATES_FIXTURE),
    "duplicitni.parquet.*diagnóza diabetes.*víckrát"
  )
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
})

test_that("bez zveřejnitelné diagnózy se nezapíše ani tabulka", {
  dir <- withr::local_tempdir()
  metrics <- write_metrics_variant(file.path(dir, "male-skupiny.parquet"), "5::BIGINT AS test_positives")
  out <- file.path(dir, "out")
  expect_error(
    report_with_chain(metrics, metrics_chain(dir, metrics_path = metrics), out, CANDIDATES_FIXTURE),
    "male-skupiny.parquet.*zveřejniteln"
  )
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
})

test_that("chyba kontroly vstupu zachová všechny předchozí výstupy", {
  dir <- withr::local_tempdir()
  metrics <- write_metrics_variant(file.path(dir, "male-skupiny.parquet"), "5::BIGINT AS test_positives")
  out <- file.path(dir, "out")
  paths <- previous_metrics_report(out)

  expect_error(
    report_with_chain(metrics, metrics_chain(dir, metrics_path = metrics), out, CANDIDATES_FIXTURE),
    "male-skupiny.parquet.*zveřejniteln"
  )
  expect_identical(lapply(paths, readLines), rep(list("předchozí výstup"), 4))
})

test_that("selhání exportu PDF nenechá starý sidecar u nové tabulky", {
  dir <- withr::local_tempdir()
  out <- file.path(dir, "out")
  previous_metrics_report(out)
  font_environment <- environment(theme_thesis)
  previous_font <- get("THESIS_FONT", envir = font_environment)
  withr::defer(assign("THESIS_FONT", previous_font, envir = font_environment))
  assign("THESIS_FONT", "Neexistující písmo", envir = font_environment)

  expect_error(
    report_with_chain(METRICS_FIXTURE, metrics_chain(dir), out, CANDIDATES_FIXTURE),
    "install-libertinus.sh"
  )
  expect_match(readLines(file.path(out, "metrics.tex"))[1], "\\begin{tabular}", fixed = TRUE)
  expect_false(any(file.exists(file.path(out, c("metrics.pdf", "metrics.tex.meta.json", "metrics.pdf.meta.json")))))
})

test_that("skript při chybném počtu argumentů vypíše použití a selže", {
  withr::local_dir("../../..")
  output <- suppressWarnings(system2("Rscript", "analysis/scripts/metrics-report.R",
                                     stdout = TRUE, stderr = TRUE))
  expect_identical(attr(output, "status"), 1L)
  expect_true(any(grepl("Použití: Rscript analysis/scripts/metrics-report.R", output, fixed = TRUE)))
})

test_that("report přijme sdílené JSON fixture zapsané Pythonem", {
  out <- withr::local_tempdir()
  write_metrics_report(METRICS_FIXTURE, EVALUATION_MANIFEST_FIXTURE,
                       RUN_MANIFEST_FIXTURE, out, CANDIDATES_FIXTURE)
  expect_true(file.exists(file.path(out, "metrics.tex")))
  expect_true(file.exists(file.path(out, "metrics.pdf")))
})

test_that("A2: změněné metriky odmítne metrics_sha256 před prvním zápisem", {
  dir <- withr::local_tempdir()
  chain <- metrics_chain(dir)
  metrics <- write_metrics_variant(file.path(dir, "jine-metriky.parquet"), "0.125::DOUBLE AS f1")
  out <- file.path(dir, "out")

  expect_error(
    write_metrics_report(metrics, chain$evaluation, chain$manifest, out, CANDIDATES_FIXTURE),
    "jine-metriky.parquet.*metrics_sha256"
  )
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
})

for (field in c("predictions_sha256", "dataset_sha256", "split_sha256", "row_id")) {
  test_that(paste("A3: manifest evaluace odmítne cizí", field), {
    dir <- withr::local_tempdir()
    chain <- metrics_chain(dir)
    evaluation <- jsonlite::fromJSON(chain$evaluation, simplifyVector = FALSE)
    evaluation[[field]] <- if (field == "row_id") "jiny" else strrep("e", 64)
    jsonlite::write_json(evaluation, chain$evaluation, auto_unbox = TRUE)
    out <- file.path(dir, "out")

    expect_error(
      write_metrics_report(METRICS_FIXTURE, chain$evaluation, chain$manifest, out, CANDIDATES_FIXTURE),
      paste0("manifest evaluace.json.*", field)
    )
    expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
  })
}

test_that("A4: změnu jednoho bajtu tréninku odmítne training_manifest_sha256", {
  dir <- withr::local_tempdir()
  chain <- metrics_chain(dir)
  cat(" ", file = chain$manifest, append = TRUE)
  out <- file.path(dir, "out")

  expect_error(
    write_metrics_report(METRICS_FIXTURE, chain$evaluation, chain$manifest, out, CANDIDATES_FIXTURE),
    "manifest běhu.json.*training_manifest_sha256"
  )
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
})

test_that("každé čtené pole evaluace musí být jeden neprázdný řetězec", {
  dir <- withr::local_tempdir()
  chain <- metrics_chain(dir)
  original <- jsonlite::fromJSON(chain$evaluation, simplifyVector = FALSE)
  out <- file.path(dir, "out")
  fields <- c("row_id", "predictions_sha256", "dataset_sha256", "split_sha256",
              "metrics_sha256", "training_manifest_sha256")
  for (field in fields) {
    for (value in list(NULL, "", 1, c("a", "b"), list("a"))) {
      evaluation <- original
      evaluation[[field]] <- value
      jsonlite::write_json(evaluation, chain$evaluation, auto_unbox = TRUE)
      expect_error(
        write_metrics_report(METRICS_FIXTURE, chain$evaluation, chain$manifest, out, CANDIDATES_FIXTURE),
        paste0("manifest evaluace.json.*", field, ".*řetězec")
      )
      expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
    }
  }
})

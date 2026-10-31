METRICS_FIXTURE <- "../../../tests/fixtures/contracts/metrics.parquet"
CANDIDATES_FIXTURE <- "../../../config/diagnosis_candidates.json"

metrics_manifest <- function(dir, row_id = "smoketest") {
  path <- file.path(dir, "manifest běhu.json")
  jsonlite::write_json(list(row_id = row_id), path, auto_unbox = TRUE)
  path
}

write_metrics_variant <- function(path, replacement) {
  con <- DBI::dbConnect(duckdb::duckdb())
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE))
  DBI::dbExecute(con, sprintf(
    "COPY (SELECT * REPLACE (%s) FROM read_parquet(%s)) TO %s (FORMAT parquet)",
    replacement, DBI::dbQuoteString(con, METRICS_FIXTURE), DBI::dbQuoteString(con, path)
  ))
  path
}

test_that("tabulka metrik má česká čísla a potlačí malé skupiny i jejich metriky", {
  dir <- withr::local_tempdir()
  write_metrics_report(METRICS_FIXTURE, metrics_manifest(dir), dir, CANDIDATES_FIXTURE)

  expect_identical(readLines(file.path(dir, "metrics.tex"), encoding = "UTF-8"), c(
    "\\begin{tabular}{lrrrrrrrr}",
    "\\toprule",
    "Diagnóza & Zprávy & Pozitivní & Přesnost & Úplnost & F1 & Správnost & ROC AUC & PR AUC\\\\",
    "\\midrule",
    "Diabetes mellitus & 40 & 12 & 0,500 & 0,250 & 0,750 & 0,875 & 0,900 & 0,800\\\\",
    "CHOPN & 40 & 10 & 0,600 & 0,300 & 0,400 & 0,850 & 0,700 & 0,550\\\\",
    "Chronické onemocnění ledvin & 40 & – & – & – & – & – & – & –\\\\",
    "Pneumonie & 40 & 20 & 1,000 & 0,000 & 0,000 & 0,500 & 0,500 & 0,500\\\\",
    "Fibrilace síní & 40 & – & – & – & – & – & – & –\\\\",
    "\\bottomrule",
    "\\end{tabular}"
  ))
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
  write_metrics_report(METRICS_FIXTURE, metrics_manifest(dir), dir, CANDIDATES_FIXTURE)
  path <- file.path(dir, "metrics.pdf")
  info <- system2("pdfinfo", shQuote(path), stdout = TRUE)
  box <- grep("Page size", info, value = TRUE)
  size <- as.numeric(strsplit(trimws(sub("Page size:", "", box)), " +")[[1]][c(1, 3)])

  expect_equal(size, c(16.5, 7.5) / 2.54 * 72, tolerance = 1 / 467.7)
  expect_true(any(grepl("LibertinusSerif", system2("pdffonts", shQuote(path), stdout = TRUE))))
})

test_that("oba výstupy nesou ověřitelný odkaz na vstup a manifest běhu", {
  dir <- withr::local_tempdir()
  manifest <- metrics_manifest(dir)
  out <- file.path(dir, "výstupy")
  result <- withVisible(write_metrics_report(METRICS_FIXTURE, manifest, out, CANDIDATES_FIXTURE))
  files <- c("metrics.tex", "metrics.pdf", "metrics.tex.meta.json", "metrics.pdf.meta.json")
  expect_false(result$visible)
  expect_setequal(result$value, file.path(out, files))
  expect_setequal(list.files(out), files)
  coreutils_sha256 <- function(path) {
    strsplit(system2("sha256sum", shQuote(path), stdout = TRUE), " +")[[1]][1]
  }
  for (output in c("metrics.tex", "metrics.pdf")) {
    sidecar <- jsonlite::fromJSON(file.path(out, paste0(output, ".meta.json")))
    expect_identical(sidecar, list(
      input = list(path = METRICS_FIXTURE, sha256 = coreutils_sha256(METRICS_FIXTURE)),
      run_manifest = list(path = manifest, sha256 = coreutils_sha256(manifest), row_id = "smoketest")
    ))
  }
})

test_that("cizí manifest se odmítne před zápisem a chyba jmenuje oba běhy", {
  dir <- withr::local_tempdir()
  out <- file.path(dir, "out")
  expect_error(
    write_metrics_report(METRICS_FIXTURE, metrics_manifest(dir, "jiny"), out, CANDIDATES_FIXTURE),
    "smoketest.*jiny"
  )
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
})

test_that("manifest musí mít row_id jako jeden řetězec", {
  dir <- withr::local_tempdir()
  out <- file.path(dir, "out")
  for (row_id in list(NULL, 1, c("smoketest", "jiny"), list("smoketest"))) {
    expect_error(
      write_metrics_report(METRICS_FIXTURE, metrics_manifest(dir, row_id), out, CANDIDATES_FIXTURE),
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
    write_metrics_report(metrics, metrics_manifest(dir), out, CANDIDATES_FIXTURE),
    "dva-behy.parquet.*právě jeden row_id"
  )
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
})

test_that("chybějící český název diagnózy se odmítne před zápisem", {
  dir <- withr::local_tempdir()
  candidates <- jsonlite::fromJSON(CANDIDATES_FIXTURE, simplifyVector = FALSE)
  candidates$candidates$ckd$label_cs <- NULL
  config <- file.path(dir, "candidates.json")
  jsonlite::write_json(candidates, config, auto_unbox = TRUE)
  out <- file.path(dir, "out")
  expect_error(
    write_metrics_report(METRICS_FIXTURE, metrics_manifest(dir), out, config),
    "candidates.json.*ckd.*label_cs"
  )
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
})

test_that("bez zveřejnitelné diagnózy se nezapíše ani tabulka", {
  dir <- withr::local_tempdir()
  metrics <- write_metrics_variant(file.path(dir, "male-skupiny.parquet"), "5::BIGINT AS test_positives")
  out <- file.path(dir, "out")
  expect_error(
    write_metrics_report(metrics, metrics_manifest(dir), out, CANDIDATES_FIXTURE),
    "male-skupiny.parquet.*zveřejniteln"
  )
  expect_length(list.files(out, recursive = TRUE, all.files = TRUE), 0L)
})

test_that("skript při chybném počtu argumentů vypíše použití a selže", {
  withr::local_dir("../../..")
  output <- suppressWarnings(system2("Rscript", "analysis/scripts/metrics-report.R",
                                     stdout = TRUE, stderr = TRUE))
  expect_identical(attr(output, "status"), 1L)
  expect_true(any(grepl("Použití: Rscript analysis/scripts/metrics-report.R", output, fixed = TRUE)))
})

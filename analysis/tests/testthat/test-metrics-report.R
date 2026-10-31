METRICS_FIXTURE <- "../../../tests/fixtures/contracts/metrics.parquet"
CANDIDATES_FIXTURE <- "../../../config/diagnosis_candidates.json"

test_that("tabulka metrik má česká čísla a potlačí malé skupiny i jejich metriky", {
  metrics <- read_metrics(METRICS_FIXTURE)
  labels <- diagnosis_labels(metrics$diagnosis, CANDIDATES_FIXTURE)
  table <- metrics_table(metrics, labels)
  dir <- withr::local_tempdir()
  export_table(table, col.names = names(table), align = "lrrrrrrrr",
               label = "metrics", widths = NULL, dir = dir)

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
  metrics <- read_metrics(METRICS_FIXTURE)
  labels <- diagnosis_labels(metrics$diagnosis, CANDIDATES_FIXTURE)
  dir <- withr::local_tempdir()
  export_figure(metrics_figure(metrics, labels), height_cm = 7.5,
                label = "metrics", dir = dir)
  path <- file.path(dir, "metrics.pdf")
  info <- system2("pdfinfo", shQuote(path), stdout = TRUE)
  box <- grep("Page size", info, value = TRUE)
  size <- as.numeric(strsplit(trimws(sub("Page size:", "", box)), " +")[[1]][c(1, 3)])

  expect_equal(size, c(16.5, 7.5) / 2.54 * 72, tolerance = 1 / 467.7)
  expect_true(any(grepl("LibertinusSerif", system2("pdffonts", shQuote(path), stdout = TRUE))))
})

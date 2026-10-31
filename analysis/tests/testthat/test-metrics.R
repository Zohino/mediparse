test_that("metriky ze sdílené syntetické fixture zachovají schéma a hodnoty", {
  actual <- read_metrics(METRICS_FIXTURE)
  expected <- data.frame(
    row_id = rep("smoketest", 5),
    diagnosis = c("diabetes", "copd", "ckd", "pneumonia", "atrial_fibrillation"),
    test_notes = rep(40, 5),
    test_positives = c(12, 10, 8, 20, 4),
    precision = c(0.5, 0.6, 0.25, 1.0, 0.125),
    recall = c(0.25, 0.3, 0.125, 0.0, 1.0),
    f1 = c(0.75, 0.4, 0.5, 0.0, 0.2),
    accuracy = c(0.875, 0.85, 0.9, 0.5, 0.3),
    roc_auc = c(0.9, 0.7, 0.65, 0.5, 0.95),
    pr_auc = c(0.8, 0.55, 0.45, 0.5, 0.9)
  )

  expect_identical(actual, expected)
})

# Konzument má záměrně vlastní deklaraci nezávislou na SCHEMA v Pythonu;
# sdílená fixture zachytí změnu producenta, kterou by export schématu skryl.
METRICS_COLUMNS <- c(
  row_id = "VARCHAR", diagnosis = "VARCHAR", test_notes = "BIGINT",
  test_positives = "BIGINT", precision = "DOUBLE", recall = "DOUBLE",
  f1 = "DOUBLE", accuracy = "DOUBLE", roc_auc = "DOUBLE", pr_auc = "DOUBLE"
)

read_metrics <- function(path) read_contract_table(path, METRICS_COLUMNS)

METRIC_LABELS <- c(
  precision = "Přesnost", recall = "Úplnost", f1 = "F1",
  accuracy = "Správnost", roc_auc = "ROC AUC", pr_auc = "PR AUC"
)

diagnosis_labels <- function(keys, candidates_path) {
  candidates <- jsonlite::fromJSON(candidates_path, simplifyVector = FALSE)$candidates
  vapply(keys, function(key) {
    label <- candidates[[key]]$label_cs
    if (!is.character(label) || length(label) != 1L || is.na(label) || !nzchar(label)) {
      stop(sprintf("Soubor %s: diagnóza %s nemá label_cs.", candidates_path, key))
    }
    label
  }, character(1), USE.NAMES = FALSE)
}

publishable_metrics <- function(metrics) {
  !is.na(dua_binary(metrics$test_positives, metrics$test_notes))
}

metrics_table <- function(metrics, labels) {
  published <- publishable_metrics(metrics)
  out <- data.frame(
    Diagnóza = labels,
    Zprávy = format_number(metrics$test_notes),
    Pozitivní = ifelse(published, format_number(metrics$test_positives), "–"),
    check.names = FALSE
  )
  for (name in names(METRIC_LABELS)) {
    values <- metrics[[name]]
    out[[METRIC_LABELS[[name]]]] <- ifelse(
      published & !is.na(values),
      formatC(values, format = "f", digits = 3, decimal.mark = ","), "–"
    )
  }
  out
}

metrics_figure <- function(metrics, labels) {
  published <- publishable_metrics(metrics)
  labels <- labels[published]
  plotted <- c("f1", "roc_auc", "pr_auc")
  metric_labels <- unname(METRIC_LABELS[plotted])
  data <- data.frame(
    diagnoza = factor(rep(labels, each = 3), levels = rev(labels)),
    metrika = factor(rep(metric_labels, length(labels)), levels = metric_labels),
    hodnota = as.vector(t(as.matrix(metrics[published, plotted, drop = FALSE])))
  )
  ggplot2::ggplot(data, ggplot2::aes(x = hodnota, y = diagnoza, fill = metrika)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::scale_x_continuous(
      limits = c(0, 1), labels = function(x) format(x, decimal.mark = ",", trim = TRUE)
    ) +
    ggplot2::labs(x = "Hodnota", y = NULL, fill = NULL) +
    theme_thesis()
}

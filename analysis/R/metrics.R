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

is_single_string <- function(x) {
  is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x)
}

diagnosis_labels <- function(keys, candidates_path) {
  candidates <- jsonlite::fromJSON(candidates_path, simplifyVector = FALSE)$candidates
  vapply(keys, function(key) {
    label <- candidates[[key]]$label_cs
    if (!is_single_string(label)) {
      stop(sprintf("Soubor %s: diagnóza %s nemá label_cs.", candidates_path, key))
    }
    label
  }, character(1), USE.NAMES = FALSE)
}

publishable_metrics <- function(metrics) {
  # Práh hlídá skupiny vymezené diagnózou (pozitivní a negativní); buňky matice
  # záměn dělí výstup modelu, ne atribut pacienta (AGENTS.md, MIMIC a DUA).
  !is.na(dua_binary(metrics$test_positives, metrics$test_notes))
}

metrics_table <- function(metrics, labels) {
  published <- publishable_metrics(metrics)
  positives <- dua_binary(metrics$test_positives, metrics$test_notes)
  out <- data.frame(
    Diagnóza = labels,
    Zprávy = format_number(metrics$test_notes),
    Pozitivní = number_text(positives),
    check.names = FALSE
  )
  for (name in names(METRIC_LABELS)) {
    values <- metrics[[name]]
    out[[METRIC_LABELS[[name]]]] <- ifelse(
      published,
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

write_metrics_report <- function(metrics_path, manifest_path, out_dir, candidates_path) {
  metrics <- read_metrics(metrics_path)
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  row_id <- if (is.list(manifest)) manifest$row_id else NULL
  if (!is_single_string(row_id)) {
    stop(sprintf("Soubor %s: row_id manifestu musí být jeden řetězec.", manifest_path))
  }
  runs <- unique(metrics$row_id)
  if (length(runs) != 1L) {
    stop(sprintf("Soubor %s: metriky musí mít právě jeden row_id.", metrics_path))
  }
  if (runs != row_id) {
    stop(sprintf(
      "Soubor %s má row_id %s, ale manifest %s má row_id %s.",
      metrics_path, runs, manifest_path, row_id
    ))
  }
  duplicate <- anyDuplicated(metrics$diagnosis)
  if (duplicate > 0L) {
    stop(sprintf(
      "Soubor %s: diagnóza %s je v metrikách víckrát.",
      metrics_path, metrics$diagnosis[[duplicate]]
    ))
  }
  labels <- diagnosis_labels(metrics$diagnosis, candidates_path)
  published <- publishable_metrics(metrics)
  if (!any(published)) {
    stop(sprintf("Soubor %s: žádná diagnóza není zveřejnitelná.", metrics_path))
  }
  table <- metrics_table(metrics, labels)
  plot <- metrics_figure(metrics, labels)
  outputs <- file.path(out_dir, c("metrics.tex", "metrics.pdf"))
  sidecars <- paste0(outputs, ".meta.json")
  # Starý sidecar by při selhání exportu nepravdivě popisoval nový soubor.
  unlink(c(outputs, sidecars))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  # Návratový markdown patří Quarto; report potřebuje jen zapsaný fragment LaTeXu.
  export_table(table, col.names = names(table), align = "lrrrrrrrr",
               label = "metrics", widths = NULL, dir = out_dir)
  export_figure(plot, height_cm = 3 + 1.5 * sum(published), label = "metrics", dir = out_dir)
  for (output in outputs) {
    write_sidecar(output, input = metrics_path, manifest = manifest_path, row_id = row_id)
  }
  # Vypsání ggplotu v Rscriptu by otevřelo výchozí pdf() bez písma práce.
  invisible(c(outputs, sidecars))
}
